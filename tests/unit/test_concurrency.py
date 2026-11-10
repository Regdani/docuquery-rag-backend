"""Model-call limiter: blocking calls stay off the event loop, peak concurrency is bounded."""

import asyncio
import time
from typing import Any

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from pydantic import ValidationError

from app.core.config import Settings
from app.core.errors import LLMTimeoutError, ServerBusyError
from app.memory.store import InMemorySessionStore
from app.rag.graph import build_graph
from app.rag.limiter import ModelCallLimiter
from tests.fakes import BlockingVectorStore, SlowChatModel
from tests.unit.test_graph import QUESTION, make_store, responder


def make_graph(model: SlowChatModel, limiter: ModelCallLimiter, store: Any = None) -> Any:
    settings = Settings(retrieval_top_k=2, retrieval_relevance_threshold=0.5)
    vs = store or make_store(QUESTION)
    return build_graph(model, vs, InMemorySessionStore(20), settings, limiter)


async def test_blocking_search_does_not_stall_the_event_loop() -> None:
    vs = BlockingVectorStore(DeterministicFakeEmbedding(size=64), block_seconds=0.3)
    vs.add_texts([QUESTION])
    graph = make_graph(SlowChatModel(responder=responder()), ModelCallLimiter(4, 5.0), vs)
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    task = asyncio.create_task(ticker())
    started = time.perf_counter()
    await graph.ainvoke({"question": QUESTION})
    elapsed = time.perf_counter() - started
    task.cancel()
    assert elapsed >= 0.3
    # The loop kept running while the search blocked its worker thread.
    assert ticks >= 10


async def test_peak_model_concurrency_is_bounded_and_all_requests_succeed() -> None:
    limit = 2
    model = SlowChatModel(responder=responder(), delay=0.05)
    graph = make_graph(model, ModelCallLimiter(limit, 10.0))
    results = await asyncio.gather(
        *(graph.ainvoke({"question": QUESTION, "session_key": f"s{i}"}) for i in range(8))
    )
    assert all(r["status"] == "answered" for r in results)
    assert 1 <= model.peak <= limit
    assert model.calls == 16  # assess + generate per request


async def test_retrieval_counts_against_the_same_limit() -> None:
    vs = BlockingVectorStore(DeterministicFakeEmbedding(size=64), block_seconds=0.05)
    vs.add_texts([QUESTION])
    graph = make_graph(SlowChatModel(responder=responder()), ModelCallLimiter(1, 10.0), vs)
    await asyncio.gather(*(graph.ainvoke({"question": QUESTION}) for _ in range(4)))
    assert vs.peak == 1


async def test_wait_limit_raises_server_busy_and_slot_is_released() -> None:
    limiter = ModelCallLimiter(1, 0.1)
    async with limiter.slot():
        started = time.perf_counter()
        with pytest.raises(ServerBusyError) as info:
            async with limiter.slot():
                pytest.fail("slot must not be granted")
        assert time.perf_counter() - started < 1.0
    assert info.value.code == "SERVER_BUSY"
    assert info.value.status_code == 503
    assert info.value.headers == {"Retry-After": "1"}
    async with limiter.slot():  # free again
        pass


async def test_slot_is_released_after_a_failed_model_call() -> None:
    limiter = ModelCallLimiter(1, 0.1)
    failing = SlowChatModel(responder=responder(), error=LLMTimeoutError("slow"))
    with pytest.raises(LLMTimeoutError):
        await make_graph(failing, limiter).ainvoke({"question": QUESTION})
    ok = make_graph(SlowChatModel(responder=responder()), limiter)
    assert (await ok.ainvoke({"question": QUESTION}))["status"] == "answered"


async def test_graph_request_fails_fast_when_saturated() -> None:
    limiter = ModelCallLimiter(1, 0.1)
    gate = asyncio.Event()
    graph = make_graph(SlowChatModel(responder=responder(), gate=gate), limiter)
    first = asyncio.create_task(graph.ainvoke({"question": QUESTION}))
    await asyncio.sleep(0.05)
    with pytest.raises(ServerBusyError):
        await graph.ainvoke({"question": QUESTION})
    gate.set()
    assert (await first)["status"] == "answered"


def test_limiter_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        ModelCallLimiter(0, 1.0)
    with pytest.raises(ValueError):
        ModelCallLimiter(1, 0)


@pytest.mark.parametrize(
    "env", [{"MAX_CONCURRENT_REQUESTS": "0"}, {"CONCURRENCY_WAIT_SECONDS": "0"}]
)
def test_settings_reject_invalid_limits(
    env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        Settings()


def test_settings_defaults() -> None:
    s = Settings()
    assert s.max_concurrent_requests == 8
    assert s.concurrency_wait_seconds == 5.0
