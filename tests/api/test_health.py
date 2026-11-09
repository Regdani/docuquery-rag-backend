import asyncio
import logging
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models import BaseChatModel, FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.core.config import Settings
from app.main import create_app
from app.rag.index_status import IndexStatus
from tests.fakes import FlakyChatModel


class SlowChatModel(BaseChatModel):
    delay: float = 5.0

    @property
    def _llm_type(self) -> str:
        return "slow"

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        raise NotImplementedError

    async def _agenerate(
        self, messages: list[BaseMessage], *args: Any, **kwargs: Any
    ) -> ChatResult:
        await asyncio.sleep(self.delay)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="late"))])


def make_client(status: IndexStatus, chat: BaseChatModel | None = None) -> TestClient:
    app = create_app(Settings(health_llm_timeout_seconds=0.2))
    app.state.index_status = status
    app.state.chat_model = chat or FakeListChatModel(responses=["pong"])
    return TestClient(app)  # no lifespan: no indexing, no providers


READY = IndexStatus(
    state="ready", document_count=3, chunk_count=17, embedding_model="nomic-embed-text"
)
INDEXING = IndexStatus(state="indexing", embedding_model="nomic-embed-text")
FAILED = IndexStatus(state="failed", error="Embedding provider unreachable.")
EMPTY = IndexStatus(state="ready", embedding_model="nomic-embed-text")


@pytest.mark.parametrize("status", [READY, INDEXING, FAILED, EMPTY])
def test_liveness_always_200(status: IndexStatus) -> None:
    app = create_app(Settings())
    app.state.index_status = status  # no chat_model on state: liveness must not need it
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["X-Request-ID"]


def test_ready_200_with_all_fields() -> None:
    response = make_client(READY).get("/health/ready", headers={"X-Request-ID": "abc123"})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "abc123"
    body = response.json()
    assert body == {
        "state": "ready",
        "document_count": 3,
        "chunk_count": 17,
        "embedding_model": "nomic-embed-text",
        "last_indexed_at": READY.timestamp.isoformat().replace("+00:00", "Z"),
        "error": None,
    }


def test_ready_503_while_indexing() -> None:
    response = make_client(INDEXING).get("/health/ready")
    assert response.status_code == 503
    assert response.json()["state"] == "indexing"
    assert "code" not in response.json()


def test_ready_503_when_failed() -> None:
    response = make_client(FAILED).get("/health/ready")
    assert response.status_code == 503
    assert response.json()["state"] == "failed"
    assert response.json()["error"] == "Embedding provider unreachable."


def test_ready_503_with_zero_chunks() -> None:
    response = make_client(EMPTY).get("/health/ready")
    assert response.status_code == 503
    assert response.json()["state"] == "ready"
    assert response.json()["chunk_count"] == 0


def test_llm_check_off_by_default() -> None:
    chat = FlakyChatModel(always=True, failures=[RuntimeError("boom")])
    response = make_client(READY, chat).get("/health/ready")
    assert "llm" not in response.json()
    assert chat.calls == 0


def test_llm_check_ok() -> None:
    response = make_client(READY).get("/health/ready", params={"check_llm": "true"})
    assert response.status_code == 200
    assert response.json()["llm"] == {"status": "ok"}


def test_llm_check_failure_keeps_status_and_hides_details(
    caplog: pytest.LogCaptureFixture,
) -> None:
    chat = FlakyChatModel(always=True, failures=[RuntimeError("secret internal text")])
    with caplog.at_level(logging.WARNING):
        response = make_client(READY, chat).get("/health/ready", params={"check_llm": "true"})
    assert response.status_code == 200
    assert response.json()["state"] == "ready"
    assert response.json()["llm"] == {"status": "unreachable"}
    assert "secret internal text" not in response.text
    assert any(
        r.levelno == logging.WARNING and "LLM health check" in r.message for r in caplog.records
    )


def test_llm_check_timeout(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        response = make_client(READY, SlowChatModel()).get(
            "/health/ready", params={"check_llm": "true"}
        )
    assert response.status_code == 200
    assert response.json()["llm"] == {"status": "timeout"}
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_llm_check_does_not_make_unready_state_ready() -> None:
    response = make_client(INDEXING).get("/health/ready", params={"check_llm": "true"})
    assert response.status_code == 503
    assert response.json()["llm"] == {"status": "ok"}


def test_openapi_documents_health_routes() -> None:
    spec = make_client(READY).get("/openapi.json").json()
    for path in ("/health", "/health/ready"):
        assert spec["paths"][path]["get"]["summary"]
    ready_op = spec["paths"]["/health/ready"]["get"]
    assert set(ready_op["responses"]["503"]["content"]["application/json"]["examples"]) >= {
        "indexing",
        "failed",
    }
