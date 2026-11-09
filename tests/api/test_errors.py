"""Unified error responses, validation, check order and error logging."""

import logging
import re
from pathlib import Path
from typing import Any

import httpx
import openai
import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding

from app.api.deps import get_qa_service
from app.core.config import Settings
from app.core.errors import LLMRateLimitedError, LLMTimeoutError, LLMUnavailableError
from app.main import create_app
from app.providers.resilience import ResilientChatModel
from app.rag.index_status import IndexStatus
from tests.api.test_ask import QUESTION, make_client, scripted, wait_ready
from tests.fakes import (
    EmptyRetrievalVectorStore,
    FailingQueryEmbeddings,
    FlakyChatModel,
    RelevanceVectorStore,
)

URL = "/api/v1/ask"
SECRET = "secret internals"


def assert_unified(resp: Any, status: int, code: str) -> dict[str, Any]:
    assert resp.status_code == status
    body = resp.json()
    assert body["code"] == code
    assert isinstance(body["detail"], str) and body["detail"]
    assert body["request_id"] and resp.headers["x-request-id"] == body["request_id"]
    assert "Traceback" not in resp.text
    return dict(body)


@pytest.fixture
def client(tmp_path: Path) -> Any:
    with make_client(tmp_path) as c:
        wait_ready(c)
        yield c


def raw(client: TestClient, content: str) -> Any:
    return client.post(URL, content=content, headers={"Content-Type": "application/json"})


# --- 422 -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "field"),
    [
        ({}, "question"),
        ({"question": 123}, "question"),
        ({"question": "x", "top_k": "x"}, "top_k"),
        ({"question": ""}, "question"),
        ({"question": "   \n "}, "question"),
        ({"question": "x" * 1001}, "question"),
        ({"question": "x", "session_key": "short"}, "session_key"),
        ({"question": "x", "session_key": "k" * 65}, "session_key"),
        ({"question": "x", "session_key": "bad key with spaces!"}, "session_key"),
        ({"question": "x", "top_k": 0}, "top_k"),
        ({"question": "x", "top_k": 21}, "top_k"),
    ],
)
def test_invalid_requests_are_422_in_unified_shape(
    client: TestClient, body: dict[str, Any], field: str
) -> None:
    resp = client.post(URL, json=body)
    out = assert_unified(resp, 422, "VALIDATION_ERROR")
    assert any(e["field"] == field and e["message"] for e in out["errors"])
    assert all(set(e) == {"field", "message", "type"} for e in out["errors"])
    assert "bad key with spaces!" not in resp.text and "x" * 50 not in resp.text


def test_malformed_json_is_422(client: TestClient) -> None:
    out = assert_unified(raw(client, '{"question": '), 422, "VALIDATION_ERROR")
    assert out["errors"]


def test_max_length_boundary_comes_from_settings(tmp_path: Path) -> None:
    with make_client(tmp_path, max_question_length=25) as c:
        wait_ready(c)
        assert c.post(URL, json={"question": "q" * 25}).status_code == 200
        out = assert_unified(c.post(URL, json={"question": "q" * 26}), 422, "VALIDATION_ERROR")
    assert out["errors"][0]["field"] == "question"


def test_question_of_exactly_default_limit_is_accepted(client: TestClient) -> None:
    assert client.post(URL, json={"question": "q" * 1000}).status_code == 200
    assert client.post(URL, json={"question": "q" * 1001}).status_code == 422


@pytest.mark.parametrize(
    "status", [IndexStatus(state="indexing"), IndexStatus(state="failed", error="boom")]
)
def test_validation_runs_before_index_guard(client: TestClient, status: IndexStatus) -> None:
    client.app.state.index_status = status  # type: ignore[attr-defined]
    assert_unified(client.post(URL, json={"question": ""}), 422, "VALIDATION_ERROR")
    assert_unified(client.post(URL, json={"question": "x" * 1001}), 422, "VALIDATION_ERROR")
    assert_unified(client.post(URL, json={"question": QUESTION}), 503, "INDEX_NOT_READY")


# --- clarification / no answer ---------------------------------------------------------------


def test_clarification_is_200_and_not_logged_as_problem(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with make_client(tmp_path, scripted(clear='{"clear": false}')) as c:
        wait_ready(c)
        with caplog.at_level(logging.INFO):
            resp = c.post(URL, json={"question": "My machine doesn't work."})
    body = resp.json()
    assert resp.status_code == 200
    assert body["status"] == "clarification_needed"
    assert body["answer"] == "Which appliance do you mean?"
    assert body["sources"] == []
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_clarification_follow_up_uses_stored_history(tmp_path: Path) -> None:
    chat = scripted(clear='{"clear": false}')
    with make_client(tmp_path, chat) as c:
        wait_ready(c)
        key = "clarify-session-1"
        first = c.post(URL, json={"question": "It doesn't work.", "session_key": key})
        assert first.json()["status"] == "clarification_needed"
        assert chat.calls_with("rewrite follow-up") == []
        c.post(URL, json={"question": "The KV-200.", "session_key": key})
    rewrite = chat.calls_with("rewrite follow-up")
    assert len(rewrite) == 1
    prompt = str(rewrite[0][1].content)
    assert "It doesn't work." in prompt and "Which appliance do you mean?" in prompt


def test_no_answer_is_200(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with make_client(tmp_path, threshold=1.0) as c:
        wait_ready(c)
        with caplog.at_level(logging.INFO):
            resp = c.post(URL, json={"question": "zzz qqq"})
    body = resp.json()
    assert resp.status_code == 200 and body["status"] == "no_answer"
    assert body["answer"] and body["sources"] == []
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


# --- model errors ----------------------------------------------------------------------------


def with_failure(tmp_path: Path, error: Exception) -> Any:
    chat = FlakyChatModel(failures=[error], always=True)
    with make_client(tmp_path, chat) as c:
        wait_ready(c)
        return c.post(URL, json={"question": QUESTION})


def test_timeout_is_504(tmp_path: Path) -> None:
    assert_unified(with_failure(tmp_path, LLMTimeoutError("t")), 504, "LLM_TIMEOUT")


def test_unavailable_is_502(tmp_path: Path) -> None:
    assert_unified(with_failure(tmp_path, LLMUnavailableError("down")), 502, "LLM_UNAVAILABLE")


def test_rate_limited_is_503_with_default_retry_after(tmp_path: Path) -> None:
    resp = with_failure(tmp_path, LLMRateLimitedError("slow"))
    assert_unified(resp, 503, "LLM_RATE_LIMITED")
    assert resp.headers["retry-after"] == "5"


def test_rate_limited_uses_provider_retry_after(tmp_path: Path) -> None:
    resp = with_failure(tmp_path, LLMRateLimitedError("slow", retry_after=12.2))
    assert resp.status_code == 503 and resp.headers["retry-after"] == "13"


def test_rate_limited_default_comes_from_settings(tmp_path: Path) -> None:
    chat = FlakyChatModel(failures=[LLMRateLimitedError("slow")], always=True)
    with make_client(tmp_path, chat, rate_limit_retry_after_seconds=9) as c:
        wait_ready(c)
        resp = c.post(URL, json={"question": QUESTION})
    assert resp.headers["retry-after"] == "9"


def test_vendor_rate_limit_after_retries_end_to_end(tmp_path: Path) -> None:
    exc = openai.RateLimitError(
        "slow",
        response=httpx.Response(
            429, request=httpx.Request("POST", "https://x.test"), headers={"Retry-After": "3"}
        ),
        body=None,
    )
    inner = FlakyChatModel(failures=[exc], always=True)
    chat = ResilientChatModel(inner=inner, provider="azure", rate_limit_retries=1)
    with make_client(tmp_path, chat) as c:
        wait_ready(c)
        resp = c.post(URL, json={"question": QUESTION})
    assert_unified(resp, 503, "LLM_RATE_LIMITED")
    assert resp.headers["retry-after"] == "3" and inner.calls == 2


def test_index_not_ready_is_503(client: TestClient) -> None:
    client.app.state.index_status = IndexStatus(state="indexing")  # type: ignore[attr-defined]
    assert_unified(client.post(URL, json={"question": QUESTION}), 503, "INDEX_NOT_READY")


def test_empty_answer_is_502(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    chat = FlakyChatModel(reply="   ")
    with make_client(tmp_path, chat) as c:
        wait_ready(c)
        with caplog.at_level(logging.WARNING):
            resp = c.post(URL, json={"question": QUESTION})
    assert_unified(resp, 502, "LLM_EMPTY_RESPONSE")
    assert [r for r in caplog.records if r.levelno == logging.ERROR]


def test_unexpected_graph_exception_is_502_without_details(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        resp = with_failure(tmp_path, RuntimeError(SECRET))
    assert_unified(resp, 502, "RAG_PIPELINE_ERROR")
    assert SECRET not in resp.text and "RuntimeError" not in resp.text
    errors = [r for r in caplog.records if r.levelno == logging.ERROR and r.exc_info]
    assert errors and any(SECRET in str(r.exc_info[1]) for r in errors if r.exc_info)


class Boom:
    async def ask(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(SECRET)


def test_catch_all_is_500_internal_error(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with make_client(tmp_path) as c:
        wait_ready(c)
        c.app.dependency_overrides[get_qa_service] = Boom  # type: ignore[attr-defined]
        quiet = TestClient(c.app, raise_server_exceptions=False)
        with caplog.at_level(logging.WARNING):
            resp = quiet.post(URL, json={"question": QUESTION}, headers={"X-Request-ID": "boom-1"})
    body = assert_unified(resp, 500, "INTERNAL_ERROR")
    assert body["request_id"] == "boom-1"
    assert SECRET not in resp.text
    assert any(
        r.levelno == logging.ERROR and r.exc_info and "boom-1" in r.getMessage()
        for r in caplog.records
    )


# --- Starlette HTTP errors and request ids ---------------------------------------------------


def test_404_and_405_use_unified_body(client: TestClient) -> None:
    assert_unified(client.get("/nope"), 404, "NOT_FOUND")
    resp = client.get(URL)
    assert_unified(resp, 405, "METHOD_NOT_ALLOWED")
    assert "POST" in resp.headers["allow"]


def test_request_id_is_reused_or_generated_on_every_path(client: TestClient) -> None:
    for resp in (
        client.post(URL, json={}, headers={"X-Request-ID": "rid-1"}),
        client.get("/nope", headers={"X-Request-ID": "rid-1"}),
        client.post(URL, json={"question": QUESTION}, headers={"X-Request-ID": "rid-1"}),
    ):
        assert resp.headers["x-request-id"] == "rid-1"
        assert resp.json()["request_id"] == "rid-1"
    generated = client.post(URL, json={})
    assert generated.json()["request_id"] == generated.headers["x-request-id"]


# --- logging ---------------------------------------------------------------------------------


def test_4xx_logged_as_warning_with_code_field_and_request_id(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        client.post(URL, json={"question": ""}, headers={"X-Request-ID": "log-422"})
        client.get("/nope", headers={"X-Request-ID": "log-404"})
        client.post(
            URL, json={"question": "x", "product_id": "KV-999"}, headers={"X-Request-ID": "log-p"}
        )
        client.app.state.index_status = IndexStatus(state="indexing")  # type: ignore[attr-defined]
        client.post(URL, json={"question": "x"}, headers={"X-Request-ID": "log-503"})
    by_id: dict[str, logging.LogRecord] = {}
    for rec in caplog.records:
        for rid in ("log-422", "log-404", "log-p", "log-503"):
            if f"request_id={rid}" in rec.getMessage():
                by_id[rid] = rec
    expected = {
        "log-422": ("VALIDATION_ERROR", "fields=[question]"),
        "log-404": ("NOT_FOUND", ""),
        "log-p": ("PRODUCT_NOT_FOUND", ""),
    }
    for rid, parts in expected.items():
        assert by_id[rid].levelno == logging.WARNING
        assert all(p in by_id[rid].getMessage() for p in parts)
    assert "INDEX_NOT_READY" in by_id["log-503"].getMessage()


def test_5xx_logged_as_error_with_trace_only_in_log(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        resp = with_failure(tmp_path, LLMTimeoutError("t"))
    records = [r for r in caplog.records if "LLM_TIMEOUT" in r.getMessage()]
    assert records and all(r.levelno == logging.ERROR and r.exc_info for r in records)
    assert "Traceback" not in resp.text


def test_long_question_is_only_logged_truncated(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    long_question = "abcdefghij" * 200
    with make_client(tmp_path, log_question_max_chars=10, max_question_length=3000) as c:
        wait_ready(c)
        with caplog.at_level(logging.DEBUG):
            c.post(URL, json={"question": long_question})
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "abcdefghij" in text and "abcdefghijabc" not in text

    caplog.clear()
    with make_client(tmp_path / "b") as c:
        wait_ready(c)
        with caplog.at_level(logging.DEBUG):
            c.post(URL, json={"question": long_question * 2})  # 4000 chars: 422
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "abcdefghijabc" not in text


# --- OpenAPI ---------------------------------------------------------------------------------


def _codes(response: dict[str, Any]) -> set[str]:
    examples = response["content"]["application/json"]["examples"]
    return {e["value"]["code"] for e in examples.values()}


def test_openapi_documents_error_responses(client: TestClient) -> None:
    schema = client.app.openapi()  # type: ignore[attr-defined]
    responses = schema["paths"][URL]["post"]["responses"]
    assert {"404", "422", "502", "503", "504"} <= set(responses)
    assert "429" not in responses
    ex422 = responses["422"]["content"]["application/json"]["examples"]["validation"]["value"]
    assert ex422["code"] == "VALIDATION_ERROR" and isinstance(ex422["detail"], str)
    assert ex422["errors"][0]["field"] == "question"
    assert {"LLM_UNAVAILABLE", "LLM_EMPTY_RESPONSE"} <= _codes(responses["502"])
    assert {"INDEX_NOT_READY", "LLM_RATE_LIMITED"} <= _codes(responses["503"])
    assert "Retry-After" in responses["503"]["headers"]
    assert _codes(responses["504"]) == {"LLM_TIMEOUT"}
    props = schema["components"]["schemas"]["ErrorResponse"]["properties"]
    assert {"detail", "code", "request_id", "errors"} <= set(props)
    assert not re.search(r"\bT\d{2}\b|#\d+", str(schema))


# --- empty retrieval and embedding failures --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]


def sample_settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=str(ROOT / "data" / "sample"), index_dir=str(tmp_path / "index"))


def test_empty_retrieval_is_no_answer_without_generation(tmp_path: Path) -> None:
    chat = scripted()
    emb = DeterministicFakeEmbedding(size=64)
    app = create_app(
        sample_settings(tmp_path),
        chat_model=chat,
        embeddings=emb,
        vector_store=EmptyRetrievalVectorStore(emb),
    )
    with TestClient(app) as c:
        wait_ready(c)
        resp = c.post(URL, json={"question": QUESTION})
    body = resp.json()
    assert resp.status_code == 200 and body["status"] == "no_answer"
    assert body["answer"] and body["sources"] == []
    assert chat.calls_with("Use ONLY the context") == []


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (LLMTimeoutError("embed timeout"), 504, "LLM_TIMEOUT"),
        (LLMUnavailableError("embed down"), 502, "LLM_UNAVAILABLE"),
    ],
)
def test_embedding_failure_on_ask_is_unified_error(
    tmp_path: Path, error: Exception, status: int, code: str
) -> None:
    emb = FailingQueryEmbeddings(error)
    app = create_app(
        sample_settings(tmp_path),
        chat_model=scripted(),
        embeddings=emb,
        vector_store=RelevanceVectorStore(emb),
    )
    with TestClient(app) as c:
        wait_ready(c)
        resp = c.post(URL, json={"question": QUESTION})
    assert_unified(resp, status, code)
