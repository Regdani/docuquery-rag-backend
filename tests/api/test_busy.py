"""Overload behaviour over HTTP: fast 503 SERVER_BUSY, header, logging, slot release."""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from app.core.config import Settings
from app.core.errors import LLMTimeoutError
from app.main import create_app
from tests.api.test_ask import QUESTION, ROOT, responder
from tests.fakes import RelevanceVectorStore, SlowChatModel

URL = "/api/v1/ask"
WAIT = 0.3


async def run_client(tmp_path: Path, chat: SlowChatModel) -> AsyncIterator[httpx.AsyncClient]:
    settings = Settings(
        data_dir=str(ROOT / "data" / "sample"),
        index_dir=str(tmp_path / "index"),
        retrieval_relevance_threshold=0.0,
        max_concurrent_requests=1,
        concurrency_wait_seconds=WAIT,
    )
    emb = DeterministicFakeEmbedding(size=64)
    app = create_app(
        settings, chat_model=chat, embeddings=emb, vector_store=RelevanceVectorStore(emb)
    )
    async with app.router.lifespan_context(app):
        for _ in range(200):
            if app.state.index_status.state != "indexing":
                break
            await asyncio.sleep(0.05)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


async def test_saturated_limiter_returns_503_fast(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    gate = asyncio.Event()
    chat = SlowChatModel(responder=responder(), gate=gate)
    async for client in run_client(tmp_path, chat):
        first = asyncio.create_task(client.post(URL, json={"question": QUESTION}))
        await asyncio.sleep(0.1)  # the first request now holds the only slot
        with caplog.at_level(logging.WARNING):
            started = time.perf_counter()
            resp = await client.post(URL, json={"question": QUESTION})
            elapsed = time.perf_counter() - started
        assert resp.status_code == 503
        body: dict[str, Any] = resp.json()
        assert body["code"] == "SERVER_BUSY"
        assert body["request_id"] and resp.headers["x-request-id"] == body["request_id"]
        assert set(body) == {"detail", "code", "request_id"}
        assert resp.headers["retry-after"] == "1"
        assert "Traceback" not in resp.text
        # Arrives after about the wait limit, far before the model timeout.
        assert WAIT * 0.8 <= elapsed < 3.0

        records = [r for r in caplog.records if "SERVER_BUSY" in r.getMessage()]
        assert len(records) == 1
        assert records[0].levelno == logging.WARNING
        assert f"request_id={body['request_id']}" in records[0].getMessage()
        assert records[0].exc_info is None

        gate.set()
        assert (await first).status_code == 200
        again = await client.post(URL, json={"question": QUESTION})
        assert again.status_code == 200  # the slot was released


async def test_slot_is_released_after_model_timeout(tmp_path: Path) -> None:
    chat = SlowChatModel(responder=responder(), error=LLMTimeoutError("model timed out"))
    async for client in run_client(tmp_path, chat):
        bad = await client.post(URL, json={"question": QUESTION})
        assert bad.status_code == 504
        chat.error = None
        ok = await client.post(URL, json={"question": QUESTION})
        assert ok.status_code == 200
