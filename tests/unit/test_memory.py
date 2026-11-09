import asyncio
from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.vectorstores import InMemoryVectorStore
from pydantic import ValidationError

from app.api.schemas import AskRequest, AskResponse
from app.core.config import Settings
from app.main import create_app
from app.memory.store import InMemorySessionStore, get_session_store, reset_session_store
from app.rag.memory_nodes import load_history, save_history


def msgs(n: int) -> list[BaseMessage]:
    return [HumanMessage(content=f"m{i}") for i in range(n)]


async def test_unknown_key_returns_empty() -> None:
    assert await InMemorySessionStore(10).get_messages("nope") == []


async def test_save_then_read_and_copy_independent() -> None:
    store = InMemorySessionStore(10)
    original = msgs(2)
    await store.save_messages("a", original)
    original.append(HumanMessage(content="x"))
    got = await store.get_messages("a")
    assert [m.content for m in got] == ["m0", "m1"]
    got.clear()
    assert len(await store.get_messages("a")) == 2


async def test_sessions_isolated_and_save_replaces() -> None:
    store = InMemorySessionStore(10)
    await store.save_messages("a", msgs(3))
    await store.save_messages("b", msgs(1))
    await store.save_messages("a", msgs(2))
    assert len(await store.get_messages("a")) == 2
    assert len(await store.get_messages("b")) == 1


async def test_limit_keeps_most_recent() -> None:
    store = InMemorySessionStore(3)
    await store.save_messages("a", msgs(5))
    assert [m.content for m in await store.get_messages("a")] == ["m2", "m3", "m4"]


async def test_concurrent_access_loses_nothing() -> None:
    store = InMemorySessionStore(10)
    keys = [f"key-{i}" for i in range(50)]
    await asyncio.gather(*(store.save_messages(k, msgs(2)) for k in keys))
    results = await asyncio.gather(*(store.get_messages(k) for k in keys))
    assert all(len(r) == 2 for r in results)


@pytest.fixture(autouse=True)
def _fresh_singleton() -> Iterator[None]:
    reset_session_store()
    yield
    reset_session_store()


def test_singleton_store() -> None:
    assert get_session_store() is get_session_store()


def test_app_state_uses_the_process_singleton() -> None:
    app = create_app(
        Settings(history_max_messages=7),
        chat_model=object(),  # type: ignore[arg-type]
        embeddings=DeterministicFakeEmbedding(size=8),
        vector_store=InMemoryVectorStore(DeterministicFakeEmbedding(size=8)),
    )
    with TestClient(app):
        assert app.state.session_store is get_session_store()
        assert app.state.session_store.max_messages == 7


def test_history_max_messages_validation() -> None:
    assert Settings(history_max_messages=5).history_max_messages == 5
    with pytest.raises(ValidationError):
        Settings(history_max_messages=0)


class SpyStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__(10)
        self.calls = 0

    async def get_messages(self, key: str) -> list[BaseMessage]:
        self.calls += 1
        return await super().get_messages(key)

    async def save_messages(
        self, key: str, messages: "list[BaseMessage] | tuple[BaseMessage, ...]"
    ) -> None:
        self.calls += 1
        await super().save_messages(key, messages)


@pytest.mark.parametrize("key", [None, ""])
async def test_nodes_skip_store_without_key(key: str | None) -> None:
    store = SpyStore()
    state = {"session_key": key, "question": "q", "answer": "a"}
    await load_history(store, state)
    await save_history(store, state)
    assert store.calls == 0


async def test_follow_up_gets_history() -> None:
    store = SpyStore()
    first = {"session_key": "session-1", "question": "q1", "answer": "a1"}
    first.update(await load_history(store, first))  # type: ignore[arg-type]
    await save_history(store, first)  # type: ignore[arg-type]
    second = {"session_key": "session-1", "question": "q2", "answer": "a2"}
    loaded = await load_history(store, second)  # type: ignore[arg-type]
    assert [m.content for m in loaded["messages"]] == ["q1", "a1"]


async def test_clarification_is_last_message() -> None:
    store = InMemorySessionStore(10)
    state = {"session_key": "session-1", "question": "it broke", "answer": "Which product?"}
    await save_history(store, state)  # type: ignore[arg-type]
    loaded = await load_history(store, {"session_key": "session-1"})
    assert isinstance(loaded["messages"][-1], AIMessage)
    assert loaded["messages"][-1].content == "Which product?"


def test_schema_strips_and_echoes() -> None:
    req = AskRequest(question="q", session_key="  abcdefgh  ")
    assert req.session_key == "abcdefgh"
    assert AskRequest(question="q").session_key is None
    assert AskResponse(answer="a").session_key is None


@pytest.mark.parametrize("key", ["short", "x" * 65, "bad key!!"])
def test_bad_session_key_gives_422(key: str) -> None:
    app = FastAPI()

    @app.post("/probe")
    async def probe(body: AskRequest) -> AskResponse:
        return AskResponse(answer="ok", session_key=body.session_key)

    client = TestClient(app)
    assert client.post("/probe", json={"question": "q", "session_key": key}).status_code == 422
    ok = client.post("/probe", json={"question": "q", "session_key": "good-key_1"})
    assert ok.json()["session_key"] == "good-key_1"
