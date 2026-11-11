import re
from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import BaseMessage

from app.core.config import Settings
from app.core.errors import (
    LLMRateLimitedError,
    LLMTimeoutError,
    LLMUnavailableError,
    RAGPipelineError,
    ServiceError,
)
from app.memory.store import InMemorySessionStore
from app.rag.graph import NODE_NAMES, build_graph, parse_analysis
from app.rag.prompts import NO_ANSWER_TEXT
from tests.fakes import (
    EmptyRetrievalVectorStore,
    FlakyChatModel,
    FlakyEmbeddings,
    RelevanceVectorStore,
    ScriptedChatModel,
)

ROOT = Path(__file__).resolve().parents[2]
QUESTION = "What is the noise level of the AP-120?"
FOLLOW_UP = "And what is its noise level?"


def responder(clear: str = '{"clear": true}', condensed: str = QUESTION) -> Any:
    def respond(messages: list[BaseMessage]) -> str:
        system = str(messages[0].content)
        if "rewrite follow-up" in system:
            return condensed
        if "check questions" in system:
            return clear
        if "clarifying" in system:
            return "Which product do you mean?"
        return "It is 24 dB."

    return respond


def scripted(**kwargs: str) -> ScriptedChatModel:
    return ScriptedChatModel(responder=responder(**kwargs), prompts=[])


def make_store(*texts: str) -> RelevanceVectorStore:
    vs = RelevanceVectorStore(DeterministicFakeEmbedding(size=64))
    vs.add_documents(
        [
            Document(
                id=f"c{i}",
                page_content=t,
                metadata={
                    "source": "ap-120.md",
                    "product_id": "AP-120",
                    "product_name": "PureAir Air Purifier",
                    "section": "Specifications",
                    "chunk_id": f"ap-120.md::0::{i}",
                },
            )
            for i, t in enumerate(texts)
        ]
    )
    return vs


def make(
    model: ScriptedChatModel | None = None,
) -> tuple[Any, ScriptedChatModel, RelevanceVectorStore, InMemorySessionStore]:
    vs = make_store(QUESTION, "unrelated text about chainsaw oil")
    model = model or scripted()
    store = InMemorySessionStore(20)
    settings = Settings(retrieval_top_k=2, retrieval_relevance_threshold=0.5)
    return build_graph(model, vs, store, settings), model, vs, store


async def test_successful_answer() -> None:
    graph, model, _, _ = make()
    out = await graph.ainvoke({"question": QUESTION})
    assert out["status"] == "answered"
    assert out["answer"] == "It is 24 dB."
    assert len(out["sources"]) == 1
    source = out["sources"][0]
    assert source.pop("score") >= 0.5
    assert source.pop("snippet") == QUESTION
    assert source == {
        "product_id": "AP-120",
        "product_name": "PureAir Air Purifier",
        "section": "Specifications",
        "source": "ap-120.md",
        "chunk_id": "ap-120.md::0::0",
    }
    assert model.calls_with("rewrite follow-up") == []


def test_graph_structure() -> None:
    graph, *_ = make()
    drawable = graph.get_graph()
    assert set(NODE_NAMES) <= set(drawable.nodes)
    edges = {(e.source, e.target) for e in drawable.edges}
    assert ("__start__", "load_history") in edges
    assert ("save_history", "__end__") in edges


async def test_graph_is_reusable_and_state_is_per_call() -> None:
    graph, *_ = make()
    first = await graph.ainvoke({"question": QUESTION})
    second = await graph.ainvoke({"question": "zzz unrelated"})
    assert first["status"] == "answered" and first["sources"]
    assert second["status"] == "no_answer"
    assert second["sources"] == []
    assert second["question"] == "zzz unrelated"


async def test_follow_up_is_condensed_and_saved() -> None:
    graph, model, vs, store = make()
    await graph.ainvoke({"question": QUESTION, "session_key": "session-1"})
    assert model.calls_with("rewrite follow-up") == []
    out = await graph.ainvoke({"question": FOLLOW_UP, "session_key": "session-1"})
    assert len(model.calls_with("rewrite follow-up")) == 1
    assert vs.queries[-1] == QUESTION
    assert out["standalone_question"] == QUESTION
    assert out["status"] == "answered"
    saved = await store.get_messages("session-1")
    assert [m.content for m in saved] == [QUESTION, "It is 24 dB.", FOLLOW_UP, "It is 24 dB."]


async def test_no_key_leaves_store_untouched() -> None:
    graph, _, _, store = make()
    await graph.ainvoke({"question": QUESTION})
    assert store._sessions == {}


async def test_sessions_are_isolated() -> None:
    graph, model, _, store = make()
    await graph.ainvoke({"question": QUESTION, "session_key": "session-a"})
    out = await graph.ainvoke({"question": FOLLOW_UP, "session_key": "session-b"})
    assert out["standalone_question"] == FOLLOW_UP
    assert model.calls_with("rewrite follow-up") == []
    assert len(await store.get_messages("session-a")) == 2
    assert len(await store.get_messages("session-b")) == 2


async def test_clarification_skips_retrieve_and_generate() -> None:
    model = scripted(clear='{"clear": false}')
    graph, _, vs, store = make(model)
    out = await graph.ainvoke({"question": "it is broken", "session_key": "session-1"})
    assert out["status"] == "clarification_needed"
    assert out["answer"] == "Which product do you mean?"
    assert vs.queries == []
    assert model.calls_with("Use ONLY the context") == []
    saved = await store.get_messages("session-1")
    assert saved[-1].content == "Which product do you mean?"


@pytest.mark.parametrize("raw", ["not json at all", "{broken", '{"clear": "maybe"}', ""])
async def test_unparsable_assess_defaults_to_clear(raw: str) -> None:
    graph, *_ = make(scripted(clear=raw))
    out = await graph.ainvoke({"question": QUESTION})
    assert out["status"] == "answered"


async def test_no_relevant_hit_skips_generation() -> None:
    graph, model, vs, _ = make()
    out = await graph.ainvoke({"question": "completely different topic"})
    assert out["status"] == "no_answer"
    assert out["answer"] == NO_ANSWER_TEXT
    assert out["sources"] == []
    assert vs.queries
    assert model.calls_with("Use ONLY the context") == []


async def test_empty_retrieval_skips_generation() -> None:
    vs = EmptyRetrievalVectorStore(DeterministicFakeEmbedding(size=64))
    model = scripted()
    graph = build_graph(model, vs, InMemorySessionStore(5), Settings())
    out = await graph.ainvoke({"question": QUESTION})
    assert out["status"] == "no_answer"
    assert out["answer"] == NO_ANSWER_TEXT
    assert out["sources"] == []
    assert vs.queries
    assert model.calls_with("Use ONLY the context") == []


async def test_threshold_is_configurable() -> None:
    vs = make_store(QUESTION)
    settings = Settings(retrieval_relevance_threshold=1.0)
    graph = build_graph(scripted(), vs, InMemorySessionStore(5), settings)
    assert (await graph.ainvoke({"question": QUESTION}))["status"] == "answered"
    assert (await graph.ainvoke({"question": "other"}))["status"] == "no_answer"


async def test_generate_prompt_restricts_to_context() -> None:
    graph, model, _, _ = make()
    await graph.ainvoke({"question": QUESTION})
    (call,) = model.calls_with("Use ONLY the context")
    system = str(call[0].content)
    assert QUESTION in system  # the retrieved chunk text
    assert "do not know" in system
    assert call[1].content == QUESTION


@pytest.mark.parametrize(
    "error", [LLMTimeoutError("t"), LLMUnavailableError("u"), LLMRateLimitedError("r")]
)
async def test_model_errors_propagate_and_do_not_save(error: ServiceError) -> None:
    store = InMemorySessionStore(10)
    graph = build_graph(
        FlakyChatModel(failures=[error], always=True), make_store(QUESTION), store, Settings()
    )
    with pytest.raises(type(error)) as info:
        await graph.ainvoke({"question": QUESTION, "session_key": "session-1"})
    assert info.value is error
    assert await store.get_messages("session-1") == []


async def test_embedding_error_propagates() -> None:
    error = LLMTimeoutError("embed")
    vs = RelevanceVectorStore(FlakyEmbeddings([error], always=True))
    graph = build_graph(scripted(), vs, InMemorySessionStore(10), Settings())
    with pytest.raises(LLMTimeoutError):
        await graph.ainvoke({"question": QUESTION})


async def test_unexpected_errors_are_wrapped_without_leaking() -> None:
    graph = build_graph(
        FlakyChatModel(failures=[RuntimeError("secret-token-123")], always=True),
        make_store(QUESTION),
        InMemorySessionStore(10),
        Settings(),
    )
    with pytest.raises(RAGPipelineError) as info:
        await graph.ainvoke({"question": QUESTION})
    assert "secret-token-123" not in str(info.value)
    assert isinstance(info.value, ServiceError)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            '{"question": "How loud is the AP-120?", "clear": true}',
            ("How loud is the AP-120?", True),
        ),
        ('Sure: {"question": "Which filter?", "clear": false}', ("Which filter?", False)),
        ('{"question": "Is {x} ok?", "clear": true}', ("Is {x} ok?", True)),
        ('{"clear": false}', ("orig", False)),
        ('{"question": "  ", "clear": "maybe"}', ("orig", True)),
        ("How loud is the AP-120?", ("How loud is the AP-120?", True)),
        ("", ("orig", True)),
    ],
)
def test_parse_analysis(raw: str, expected: tuple[str, bool]) -> None:
    assert parse_analysis(raw, "orig") == expected


async def test_follow_up_uses_one_call_for_rewrite_and_clarity() -> None:
    model = scripted(condensed='{"question": "Which AP-120 filter?", "clear": false}')
    graph, _, vs, _ = make(model)
    await graph.ainvoke({"question": QUESTION, "session_key": "session-1"})
    calls_before = len(model.prompts)
    out = await graph.ainvoke({"question": "Which one?", "session_key": "session-1"})
    assert out["status"] == "clarification_needed"
    assert out["standalone_question"] == "Which AP-120 filter?"
    assert model.calls_with("check questions")[1:] == []  # no separate clarity call
    assert len(model.prompts) - calls_before == 2  # analyze + clarify
    assert len(vs.queries) == 1


def test_readme_diagram_matches_graph() -> None:
    graph, *_ = make()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```mermaid\n(.*?)```", readme, re.DOTALL)
    names = [n for n in graph.get_graph().nodes if not n.startswith("__")]
    assert any(all(n in block for n in names) for block in blocks)


def test_rag_package_has_no_provider_imports() -> None:
    banned = re.compile(
        r"^\s*(?:from|import)\s+(?:langchain_google_genai|langchain_ollama"
        r"|langchain_chroma|ollama|chromadb|google)\b",
        re.MULTILINE,
    )
    for path in (ROOT / "app" / "rag").glob("*.py"):
        assert not banned.search(path.read_text(encoding="utf-8")), path
