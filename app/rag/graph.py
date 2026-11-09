"""The question flow as a LangGraph graph, compiled once; the state is new for every request.

load_history -> condense -> assess -> (clarify | retrieve -> (generate -> format | no_answer))
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
from app.rag.memory_nodes import load_history, save_history
from app.rag.prompts import (
    ANSWER_SYSTEM,
    ASSESS_SYSTEM,
    CLARIFY_SYSTEM,
    CONDENSE_SYSTEM,
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
    "condense",
    "assess",
    "retrieve",
    "generate",
    "clarify",
    "no_answer",
    "format",
    "save_history",
)


def _text(content: Any) -> str:
    return content if isinstance(content, str) else str(content)


def _transcript(messages: list[BaseMessage]) -> str:
    lines = [
        f"{'User' if m.type == 'human' else 'Assistant'}: {_text(m.content)}" for m in messages
    ]
    return "\n".join(lines)


def parse_clear(raw: str) -> bool:
    """Read the assess decision; anything unparsable counts as clear so retrieval still runs."""
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if match:
        try:
            value = json.loads(match.group(0)).get("clear")
        except (ValueError, AttributeError):
            return True
        if isinstance(value, bool):
            return value
    return True


def relevant_hits(state: GraphState, threshold: float) -> list[Hit]:
    """Hits whose score is at or above the threshold."""
    return [h for h in state.get("hits", []) if h["score"] >= threshold]


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
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Compile the graph. Everything is passed in; no globals are read or written."""
    default_top_k = settings.retrieval_top_k
    snippet_max = settings.snippet_max_chars
    threshold = settings.retrieval_relevance_threshold
    max_log = settings.log_question_max_chars

    async def _load(state: GraphState) -> dict[str, Any]:
        return await load_history(store, state)

    async def _save(state: GraphState) -> dict[str, Any]:
        return await save_history(store, state)

    async def condense(state: GraphState) -> dict[str, Any]:
        question = state.get("question", "")
        history = state.get("messages", [])
        if not history:
            return {"standalone_question": question}
        prompt = f"Conversation:\n{_transcript(history)}\n\nFollow-up question: {question}"
        reply = await chat_model.ainvoke(
            [SystemMessage(content=CONDENSE_SYSTEM), HumanMessage(content=prompt)]
        )
        return {"standalone_question": _text(reply.content).strip() or question}

    async def assess(state: GraphState) -> dict[str, Any]:
        reply = await chat_model.ainvoke(
            [
                SystemMessage(content=ASSESS_SYSTEM),
                HumanMessage(content=state["standalone_question"]),
            ]
        )
        clear = parse_clear(_text(reply.content))
        logger.info("Question assessed as %s", "clear" if clear else "unclear")
        return {"status": "" if clear else STATUS_CLARIFICATION}

    def route_assess(state: GraphState) -> str:
        return "clarify" if state.get("status") == STATUS_CLARIFICATION else "retrieve"

    async def clarify(state: GraphState) -> dict[str, Any]:
        reply = await chat_model.ainvoke(
            [
                SystemMessage(content=CLARIFY_SYSTEM),
                HumanMessage(content=state["standalone_question"]),
            ]
        )
        return {
            "answer": _text(reply.content).strip(),
            "status": STATUS_CLARIFICATION,
            "sources": [],
        }

    async def retrieve(state: GraphState) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"k": state.get("top_k") or default_top_k}
        product_id = state.get("product_id")
        if product_id:
            kwargs["filter"] = {"product_id": product_id}
        pairs = await asyncio.to_thread(
            vector_store.similarity_search_with_relevance_scores,
            state["standalone_question"],
            **kwargs,
        )
        return {"hits": [Hit(document=doc, score=float(score)) for doc, score in pairs]}

    def route_retrieve(state: GraphState) -> str:
        return "generate" if relevant_hits(state, threshold) else "no_answer"

    async def generate(state: GraphState) -> dict[str, Any]:
        used = relevant_hits(state, threshold)
        context = "\n\n---\n\n".join(h["document"].page_content for h in used)
        reply = await chat_model.ainvoke(
            [
                SystemMessage(content=ANSWER_SYSTEM.format(context=context)),
                HumanMessage(content=state["standalone_question"]),
            ]
        )
        return {"answer": _text(reply.content).strip()}

    async def no_answer(state: GraphState) -> dict[str, Any]:
        return {"answer": NO_ANSWER_TEXT, "status": STATUS_NO_ANSWER, "sources": []}

    async def format_result(state: GraphState) -> dict[str, Any]:
        sources = [_source(h, snippet_max) for h in relevant_hits(state, threshold)]
        question = state.get("question", "")
        logger.info(
            "Answered question %r using sources: %s",
            question[:max_log],
            [f"{s['product_id']}/{s['section']}/{s['chunk_id']}" for s in sources],
        )
        return {"status": STATUS_ANSWERED, "sources": sources}

    nodes: dict[str, NodeFn] = {
        "load_history": _load,
        "condense": condense,
        "assess": assess,
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
    graph.add_edge("load_history", "condense")
    graph.add_edge("condense", "assess")
    graph.add_conditional_edges(
        "assess", route_assess, {"clarify": "clarify", "retrieve": "retrieve"}
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
