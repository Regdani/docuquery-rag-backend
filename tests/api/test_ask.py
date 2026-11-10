import logging
import re
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from app.api.deps import get_qa_service
from app.core.config import Settings
from app.main import create_app
from app.rag.index_status import IndexStatus
from app.services.qa_service import QAResult
from tests.fakes import RelevanceVectorStore, ScriptedChatModel

ROOT = Path(__file__).resolve().parents[2]
QUESTION = "How often should I descale the KV-200 coffee maker?"


def responder(clear: str = '{"clear": true}') -> Any:
    def respond(messages: list[BaseMessage]) -> str:
        system = str(messages[0].content)
        if "rewrite follow-up" in system:
            return QUESTION
        if "check questions" in system:
            return clear
        if "clarifying" in system:
            return "Which appliance do you mean?"
        return "Every 60 days."

    return respond


def scripted(**kwargs: str) -> ScriptedChatModel:
    return ScriptedChatModel(responder=responder(**kwargs), prompts=[])


def make_client(
    tmp_path: Path, chat: BaseChatModel | None = None, threshold: float = 0.0, **extra: Any
) -> TestClient:
    settings = Settings(
        data_dir=str(ROOT / "data" / "sample"),
        index_dir=str(tmp_path / "index"),
        retrieval_relevance_threshold=threshold,
        **extra,
    )
    emb = DeterministicFakeEmbedding(size=64)
    app = create_app(
        settings,
        chat_model=chat or scripted(),
        embeddings=emb,
        vector_store=RelevanceVectorStore(emb),
    )
    return TestClient(app)


def wait_ready(client: TestClient) -> None:
    for _ in range(200):
        if client.app.state.index_status.state != "indexing":  # type: ignore[attr-defined]
            return
        time.sleep(0.05)
    raise AssertionError("indexing did not finish")


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    with make_client(tmp_path) as c:
        wait_ready(c)
        yield c


def ask(client: TestClient, **body: Any) -> Any:
    body.setdefault("question", QUESTION)
    return client.post("/api/v1/ask", json=body)


def test_graph_and_service_built_once_on_app_state(client: TestClient) -> None:
    state = client.app.state  # type: ignore[attr-defined]
    graph, service = state.graph, state.qa_service
    ask(client)
    ask(client)
    assert state.graph is graph and state.qa_service is service


def test_success_with_sources(client: TestClient) -> None:
    resp = ask(client)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "answered"
    assert body["answer"] == "Every 60 days."
    assert re.fullmatch(r"[0-9a-f]{32}", body["session_key"])  # generated
    assert 1 <= len(body["sources"]) <= 4
    for s in body["sources"]:
        assert set(s) == {"product_id", "product_name", "section", "chunk_id", "score", "snippet"}
        assert 0.0 <= s["score"] <= 1.0


def test_retrieval_top_k_comes_from_settings(tmp_path: Path) -> None:
    with make_client(tmp_path, retrieval_top_k=2) as c:
        wait_ready(c)
        assert len(ask(c).json()["sources"]) == 2


def test_top_k_in_request_body_is_ignored(client: TestClient) -> None:
    resp = ask(client, top_k=1)
    assert resp.status_code == 200
    assert len(resp.json()["sources"]) == 4  # server default, not the body value


def test_snippet_is_capped(tmp_path: Path) -> None:
    with make_client(tmp_path, snippet_max_chars=40) as c:
        wait_ready(c)
        sources = ask(c).json()["sources"]
    assert sources and all(0 < len(s["snippet"]) <= 40 for s in sources)


def test_product_filter(tmp_path: Path) -> None:
    with make_client(tmp_path, retrieval_top_k=20) as c:
        wait_ready(c)
        everything = {s["product_id"] for s in ask(c).json()["sources"]}
        scoped = ask(c, product_id="KV-200").json()
    assert len(everything) > 1
    assert scoped["sources"]
    assert {s["product_id"] for s in scoped["sources"]} == {"KV-200"}


def test_unknown_product_is_404(client: TestClient) -> None:
    resp = ask(client, product_id="KV-999")
    assert resp.status_code == 404
    body = resp.json()
    assert body["code"] == "PRODUCT_NOT_FOUND"
    assert "KV-999" in body["detail"]
    assert body["request_id"]


def test_clarification_needed(tmp_path: Path) -> None:
    with make_client(tmp_path, scripted(clear='{"clear": false}')) as c:
        wait_ready(c)
        body = ask(c, question="How do I clean it?").json()
    assert body["status"] == "clarification_needed"
    assert body["answer"] == "Which appliance do you mean?"
    assert body["sources"] == []


def test_no_answer(tmp_path: Path) -> None:
    with make_client(tmp_path, threshold=1.0) as c:
        wait_ready(c)
        resp = ask(c, question="zzz qqq")
    body = resp.json()
    assert body["status"] == "no_answer"
    assert body["sources"] == []


def test_follow_up_uses_history(tmp_path: Path) -> None:
    chat = scripted()
    with make_client(tmp_path, chat) as c:
        wait_ready(c)
        key = "ask-follow-up-01"
        first = ask(c, session_key=key)
        assert chat.calls_with("rewrite follow-up") == []
        second = ask(c, question="And the KV-300?", session_key=key)
    assert first.json()["session_key"] == key
    assert second.json()["session_key"] == key
    assert len(chat.calls_with("rewrite follow-up")) == 1


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"question": ""},
        {"question": "   \n "},
        {"question": "x", "session_key": "short"},
    ],
)
def test_invalid_input_is_422(client: TestClient, body: dict[str, Any]) -> None:
    assert client.post("/api/v1/ask", json=body).status_code == 422


def test_empty_optionals_become_none_and_question_is_stripped(client: TestClient) -> None:
    resp = ask(client, question="  " + QUESTION + "  ", session_key="  ", product_id="")
    assert resp.status_code == 200
    assert re.fullmatch(r"[0-9a-f]{32}", resp.json()["session_key"])  # blank key = new session


def test_session_key_is_generated_per_new_conversation(client: TestClient) -> None:
    first = ask(client).json()["session_key"]
    second = ask(client).json()["session_key"]
    assert first != second


def test_generated_session_key_continues_the_conversation(tmp_path: Path) -> None:
    chat = scripted()
    with make_client(tmp_path, chat) as c:
        wait_ready(c)
        key = ask(c).json()["session_key"]
        assert chat.calls_with("rewrite follow-up") == []
        second = ask(c, question="And the KV-300?", session_key=key)
    assert second.json()["session_key"] == key
    assert len(chat.calls_with("rewrite follow-up")) == 1


def test_request_without_key_ignores_earlier_history(tmp_path: Path) -> None:
    chat = scripted()
    with make_client(tmp_path, chat) as c:
        wait_ready(c)
        ask(c)
        ask(c, question="And the KV-300?")  # no key: a new conversation, no rewrite
    assert chat.calls_with("rewrite follow-up") == []


@pytest.mark.parametrize(
    "status", [IndexStatus(state="indexing"), IndexStatus(state="failed", error="boom")]
)
def test_503_until_index_ready(client: TestClient, status: IndexStatus) -> None:
    client.app.state.index_status = status  # type: ignore[attr-defined]
    resp = ask(client)
    assert resp.status_code == 503
    assert resp.json()["code"] == "INDEX_NOT_READY"


def test_request_id_header_body_and_logs(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        resp = client.post(
            "/api/v1/ask", json={"question": QUESTION}, headers={"X-Request-ID": "req-abc-123"}
        )
        generated = ask(client)
    assert resp.json()["request_id"] == "req-abc-123"
    assert resp.headers["x-request-id"] == "req-abc-123"
    lines = [r.getMessage() for r in caplog.records if "request_id=req-abc-123" in r.getMessage()]
    assert any("source product_id=" in m and "chunk_id=" in m and "score=" in m for m in lines)
    assert generated.json()["request_id"] and generated.json()["request_id"] != "req-abc-123"


def test_question_logged_truncated(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with make_client(tmp_path, log_question_max_chars=10) as c:
        wait_ready(c)
        with caplog.at_level(logging.INFO, logger="app.services.qa_service"):
            ask(c)
    text = " ".join(r.getMessage() for r in caplog.records)
    assert QUESTION[:10] in text and QUESTION[:11] not in text


def test_clarification_and_no_answer_are_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with make_client(tmp_path, scripted(clear='{"clear": false}')) as c:
        wait_ready(c)
        with caplog.at_level(logging.INFO, logger="app.services.qa_service"):
            ask(c)
    assert any("clarification_needed" in r.getMessage() for r in caplog.records)
    caplog.clear()
    with make_client(tmp_path / "b", threshold=1.0) as c:
        wait_ready(c)
        with caplog.at_level(logging.INFO, logger="app.services.qa_service"):
            ask(c)
    assert any("no_answer" in r.getMessage() for r in caplog.records)


class FakeService:
    async def ask(
        self,
        question: str,
        *,
        session_key: str | None,
        product_id: str | None,
        request_id: str,
    ) -> QAResult:
        return QAResult(
            answer=f"fake:{question}", status="no_answer", session_key=session_key or "fake-key"
        )


def test_dependency_override_with_fake_service(client: TestClient) -> None:
    client.app.dependency_overrides[get_qa_service] = FakeService  # type: ignore[attr-defined]
    body = ask(client, question="hello").json()
    assert body["answer"] == "fake:hello"
    assert body["status"] == "no_answer"


def test_openapi_documents_ask(client: TestClient) -> None:
    schema = client.app.openapi()  # type: ignore[attr-defined]
    op = schema["paths"]["/api/v1/ask"]["post"]
    examples = op["requestBody"]["content"]["application/json"]["examples"]
    assert {"successful_question", "follow_up", "clarification", "product_scoped"} <= set(examples)
    assert "session_key" in examples["follow_up"]["value"]
    assert "product_id" in examples["product_scoped"]["value"]
    ok = op["responses"]["200"]["content"]["application/json"]["examples"]
    assert {"answered", "clarification_needed", "no_answer"} <= set(ok)
    assert {"404", "422", "502", "503", "504"} <= set(op["responses"])
    assert "ErrorResponse" in str(op["responses"]["404"])
    fields = schema["components"]["schemas"]["AskRequest"]["properties"]
    assert all("description" in f and "examples" in f for f in fields.values())
