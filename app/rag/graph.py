"""The question flow as a LangGraph graph, compiled once; the state is new for every request.

load_history -> analyze -> (clarify | retrieve -> (generate -> format | no_answer))
-> save_history
"""

import asyncio
import json
import logging
import re
from collections.abc import Callable, Coroutine
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.vectorstores import VectorStore
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.core.config import Settings
from app.core.errors import RAGPipelineError, ServiceError
from app.memory.store import InMemorySessionStore
from app.rag.limiter import ModelCallLimiter
from app.rag.memory_nodes import load_history, save_history
from app.rag.prompts import (
    ANALYZE_SYSTEM,
    ANSWER_SYSTEM,
    ASSESS_SYSTEM,
    CLARIFY_SYSTEM,
    NO_ANSWER_TEXT,
)
from app.rag.state import GraphState, Hit, Source

logger = logging.getLogger(__name__)

NodeFn = Callable[[GraphState], Coroutine[Any, Any, dict[str, Any]]]

STATUS_ANSWERED = "answered"
STATUS_CLARIFICATION = "clarification_needed"
STATUS_NO_ANSWER = "no_answer"

NODE_NAMES = (
    "load_history",
    "analyze",
    "retrieve",
    "generate",
    "clarify",
    "no_answer",
    "format",
    "save_history",
)

# Greedy first (a question may contain braces), then the first object only.
_JSON_OBJECT = (re.compile(r"\{.*\}", re.DOTALL), re.compile(r"\{.*?\}", re.DOTALL))


def _text(content: Any) -> str:
    return content if isinstance(content, str) else str(content)


def _transcript(messages: list[BaseMessage]) -> str:
    lines = [
        f"{'User' if m.type == 'human' else 'Assistant'}: {_text(m.content)}" for m in messages
    ]
    return "\n".join(lines)


def _json_object(raw: str) -> dict[str, Any] | None:
    """The JSON object in a model reply, or None when there is none."""
    for pattern in _JSON_OBJECT:
        match = pattern.search(raw)
        if match is None:
            return None
        try:
            data = json.loads(match.group(0))
        except ValueError:
            continue
        return data if isinstance(data, dict) else None
    return None


def parse_clear(raw: str) -> bool:
    """Read the clarity decision; anything unparsable counts as clear so retrieval still runs."""
    value = (_json_object(raw) or {}).get("clear")
    return value if isinstance(value, bool) else True


def parse_analysis(raw: str, original: str) -> tuple[str, bool]:
    """Read (standalone question, clear) from the follow-up analysis.

    A reply without JSON is taken as the rewritten question. A missing or blank question falls
    back to the original one, and an unparsable decision counts as clear.
    """
    data = _json_object(raw)
    if data is None:
        return raw.strip() or original, True
    question = data.get("question")
    standalone = question.strip() if isinstance(question, str) and question.strip() else original
    clear = data.get("clear")
    return standalone, clear if isinstance(clear, bool) else True


def make_snippet(text: str, max_chars: int) -> str:
    """Collapse whitespace and cap the text at `max_chars` (ellipsis included)."""
    flat = " ".join(text.split())
    if len(flat) <= max_chars:
        return flat
    return flat[: max(max_chars - 3, 0)].rstrip() + "..."


def _source(hit: Hit, snippet_max: int) -> Source:
    meta = hit["document"].metadata
    return Source(
        product_id=str(meta.get("product_id", "")),
        product_name=str(meta.get("product_name", "")),
        section=str(meta.get("section", "")),
        source=str(meta.get("source", "")),
        chunk_id=str(meta.get("chunk_id", hit["document"].id or "")),
        score=hit["score"],
        snippet=make_snippet(hit["document"].page_content, snippet_max),
    )


def _guard(name: str, fn: NodeFn) -> NodeFn:
    """Let domain errors through unchanged; wrap anything else without leaking details."""

    async def wrapper(state: GraphState) -> dict[str, Any]:
        try:
            return await fn(state)
        except ServiceError:
            raise
        except Exception as exc:
            logger.exception("Unexpected failure in graph node %s", name)
            raise RAGPipelineError() from exc

    return wrapper


def build_graph(
    chat_model: BaseChatModel,
    vector_store: VectorStore,
    store: InMemorySessionStore,
    settings: Settings,
    limiter: ModelCallLimiter | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Compile the graph. Everything is passed in; no globals are read or written.

    Every chat call and the retrieval (query embedding + search) run inside one slot of the
    shared `limiter`; retries of the provider wrappers happen while the slot is held.
    """
    limit = limiter or ModelCallLimiter.from_settings(settings)
    top_k = settings.retrieval_top_k
    snippet_max = settings.snippet_max_chars
    threshold = settings.retrieval_relevance_threshold

    async def _chat(system: str, user: str) -> str:
        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        async with limit.slot():
            reply = await chat_model.ainvoke(messages)
        return _text(reply.content)

    async def _load(state: GraphState) -> dict[str, Any]:
        return await load_history(store, state)

    async def _save(state: GraphState) -> dict[str, Any]:
        return await save_history(store, state)

    async def analyze(state: GraphState) -> dict[str, Any]:
        """One LLM call: the clarity check, plus the follow-up rewrite when there is history."""
        question = state.get("question", "")
        history = state.get("messages", [])
        if history:
            prompt = f"Conversation:\n{_transcript(history)}\n\nFollow-up question: {question}"
            standalone, clear = parse_analysis(await _chat(ANALYZE_SYSTEM, prompt), question)
        else:
            standalone, clear = question, parse_clear(await _chat(ASSESS_SYSTEM, question))
        logger.info("Question assessed as %s", "clear" if clear else "unclear")
        return {
            "standalone_question": standalone,
            "status": "" if clear else STATUS_CLARIFICATION,
        }

    def route_analyze(state: GraphState) -> str:
        return "clarify" if state.get("status") == STATUS_CLARIFICATION else "retrieve"

    async def clarify(state: GraphState) -> dict[str, Any]:
        reply = await _chat(CLARIFY_SYSTEM, state["standalone_question"])
        return {"answer": reply.strip(), "status": STATUS_CLARIFICATION, "sources": []}

    async def retrieve(state: GraphState) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"k": top_k}
        product_id = state.get("product_id")
        if product_id:
            kwargs["filter"] = {"product_id": product_id}
        async with limit.slot():
            pairs = await asyncio.to_thread(
                vector_store.similarity_search_with_relevance_scores,
                state["standalone_question"],
                **kwargs,
            )
        # Only chunks at or above the threshold are kept; they are the context and the sources.
        hits = [Hit(document=doc, score=float(score)) for doc, score in pairs]
        return {"hits": [h for h in hits if h["score"] >= threshold]}

    def route_retrieve(state: GraphState) -> str:
        return "generate" if state.get("hits") else "no_answer"

    async def generate(state: GraphState) -> dict[str, Any]:
        context = "\n\n---\n\n".join(h["document"].page_content for h in state["hits"])
        reply = await _chat(ANSWER_SYSTEM.format(context=context), state["standalone_question"])
        return {"answer": reply.strip()}

    async def no_answer(state: GraphState) -> dict[str, Any]:
        return {"answer": NO_ANSWER_TEXT, "status": STATUS_NO_ANSWER, "sources": []}

    async def format_result(state: GraphState) -> dict[str, Any]:
        sources = [_source(h, snippet_max) for h in state["hits"]]
        return {"status": STATUS_ANSWERED, "sources": sources}

    nodes: dict[str, NodeFn] = {
        "load_history": _load,
        "analyze": analyze,
        "retrieve": retrieve,
        "generate": generate,
        "clarify": clarify,
        "no_answer": no_answer,
        "format": format_result,
        "save_history": _save,
    }

    graph = StateGraph(GraphState)
    for name, fn in nodes.items():
        graph.add_node(name, _guard(name, fn))  # type: ignore[call-overload]
    graph.add_edge(START, "load_history")
    graph.add_edge("load_history", "analyze")
    graph.add_conditional_edges(
        "analyze", route_analyze, {"clarify": "clarify", "retrieve": "retrieve"}
    )
    graph.add_conditional_edges(
        "retrieve", route_retrieve, {"generate": "generate", "no_answer": "no_answer"}
    )
    graph.add_edge("generate", "format")
    graph.add_edge("format", "save_history")
    graph.add_edge("clarify", "save_history")
    graph.add_edge("no_answer", "save_history")
    graph.add_edge("save_history", END)
    return graph.compile()
