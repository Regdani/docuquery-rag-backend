import logging
from typing import Any

import httpx
import pytest
from google.genai import errors as genai_errors

from app.core.config import Settings
from app.core.errors import (
    LLMRateLimitedError,
    LLMTimeoutError,
    LLMUnavailableError,
    ProviderConfigError,
    ServiceError,
)
from app.providers.factory import create_chat_model, create_embeddings
from app.providers.resilience import ResilientChatModel, ResilientEmbeddings, RetryPolicy
from tests.fakes import FlakyChatModel, FlakyEmbeddings

SECRET = "sk-SECRET-0123456789"
REQ = httpx.Request("POST", "https://example.test/v1")


def gemini_status(status: int, headers: dict[str, str] | None = None) -> genai_errors.APIError:
    response = httpx.Response(status, request=REQ, headers=headers)
    return genai_errors.APIError(status, {"error": {"message": f"boom {SECRET}"}}, response)


FAILURES: list[tuple[str, Any, type[ServiceError]]] = [
    ("gemini-429", gemini_status(429), LLMRateLimitedError),
    ("gemini-500", gemini_status(500), LLMUnavailableError),
    ("gemini-503", gemini_status(503), LLMUnavailableError),
    ("gemini-504", gemini_status(504), LLMTimeoutError),
    ("httpx-timeout", httpx.ReadTimeout("slow", request=REQ), LLMTimeoutError),
    ("httpx-conn", httpx.ConnectError("down", request=REQ), LLMUnavailableError),
    ("builtin-timeout", TimeoutError("t"), LLMTimeoutError),
    ("builtin-conn", ConnectionError("c"), LLMUnavailableError),
]
IDS = [f[0] for f in FAILURES]


def chat(failures: list[Any], retries: int = 2, always: bool = False) -> tuple[Any, Any]:
    inner = FlakyChatModel(failures=failures, always=always)
    wrapper = ResilientChatModel(inner=inner, policy=RetryPolicy("gemini", retries, 0.0))
    return wrapper, inner


def test_error_classes_have_distinct_codes_and_statuses() -> None:
    errors = [LLMTimeoutError("a"), LLMUnavailableError("b"), LLMRateLimitedError("c")]
    assert all(isinstance(e, ServiceError) for e in errors)
    assert [(e.code, e.status_code) for e in errors] == [
        ("LLM_TIMEOUT", 504),
        ("LLM_UNAVAILABLE", 502),
        ("LLM_RATE_LIMITED", 503),
    ]


@pytest.mark.parametrize(("name", "exc", "expected"), FAILURES, ids=IDS)
def test_chat_failures_translated(name: str, exc: Any, expected: type[ServiceError]) -> None:
    wrapper, _ = chat([exc], retries=0)
    with pytest.raises(expected) as info:
        wrapper.invoke("hi")
    assert SECRET not in str(info.value)


@pytest.mark.parametrize(("name", "exc", "expected"), FAILURES, ids=IDS)
def test_embedding_failures_translated(name: str, exc: Any, expected: type[ServiceError]) -> None:
    emb = ResilientEmbeddings(FlakyEmbeddings([exc]), RetryPolicy("gemini", 0, 0.0))
    with pytest.raises(expected):
        emb.embed_query("q")
    emb = ResilientEmbeddings(FlakyEmbeddings([exc]), RetryPolicy("gemini", 0, 0.0))
    with pytest.raises(expected):
        emb.embed_documents(["d"])


@pytest.mark.parametrize("exc", [ValueError("bug"), KeyError("k"), gemini_status(400)])
def test_unknown_errors_not_swallowed(exc: Exception) -> None:
    wrapper, _ = chat([exc])
    with pytest.raises(type(exc)):
        wrapper.invoke("hi")
    emb = ResilientEmbeddings(FlakyEmbeddings([exc]), RetryPolicy("gemini", 2, 0.0))
    with pytest.raises(type(exc)):
        emb.embed_query("q")


def test_rate_limit_retries_are_bounded() -> None:
    wrapper, inner = chat([gemini_status(429)], retries=3, always=True)
    with pytest.raises(LLMRateLimitedError):
        wrapper.invoke("hi")
    assert inner.calls == 4  # one attempt + at most max_retries retries


def test_rate_limit_then_success() -> None:
    wrapper, inner = chat([gemini_status(429)])
    assert wrapper.invoke("hi").content == "ok"
    assert inner.calls == 2


def test_other_failures_are_not_retried() -> None:
    wrapper, inner = chat([httpx.ConnectError("down", request=REQ)], always=True)
    with pytest.raises(LLMUnavailableError):
        wrapper.invoke("hi")
    assert inner.calls == 1


async def test_async_chat_translation_and_retry() -> None:
    wrapper, inner = chat([gemini_status(429)])
    assert (await wrapper.ainvoke("hi")).content == "ok"
    wrapper, inner = chat([gemini_status(429)], retries=1, always=True)
    with pytest.raises(LLMRateLimitedError):
        await wrapper.ainvoke("hi")
    assert inner.calls == 2


def test_embeddings_retry_bounded_and_recovering() -> None:
    always = FlakyEmbeddings([gemini_status(429)], always=True)
    with pytest.raises(LLMRateLimitedError):
        ResilientEmbeddings(always, RetryPolicy("gemini", 2, 0.0)).embed_query("q")
    assert always.calls == 3
    once = FlakyEmbeddings([gemini_status(429)])
    assert ResilientEmbeddings(once, RetryPolicy("gemini", 2, 0.0)).embed_query("q") == [1.0, 0.0]
    assert once.calls == 2


async def test_async_embeddings_translation() -> None:
    emb = ResilientEmbeddings(FlakyEmbeddings([TimeoutError()]), RetryPolicy("gemini", 0, 0.0))
    with pytest.raises(LLMTimeoutError):
        await emb.aembed_query("q")


def test_backoff_is_exponential_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("app.providers.resilience.time.sleep", sleeps.append)
    emb = FlakyEmbeddings([gemini_status(429)], always=True)
    with pytest.raises(LLMRateLimitedError):
        ResilientEmbeddings(emb, RetryPolicy("gemini", 3, 0.5)).embed_query("q")
    assert sleeps == [0.5, 1.0, 2.0]


def test_factory_wires_retry_settings_for_cloud_and_local() -> None:
    s = Settings(
        _env_file=None,  # type: ignore[call-arg]
        llm_provider="gemini",
        embedding_provider="gemini",
        embedding_model="gemini-embedding-001",
        google_api_key=SECRET,
        max_retries=4,
        retry_backoff_seconds=0.25,
    )
    model = create_chat_model(s)
    assert isinstance(model, ResilientChatModel)
    assert (model.policy.max_retries, model.policy.backoff_seconds) == (4, 0.25)
    emb = create_embeddings(s)
    assert isinstance(emb, ResilientEmbeddings)
    policy = emb._policy
    assert (policy.max_retries, policy.backoff_seconds) == (4, 0.25)
    local = Settings(_env_file=None, llm_provider="ollama")  # type: ignore[call-arg]
    local_chat = create_chat_model(local)  # ollama: translation only, never a 429 retry
    assert isinstance(local_chat, ResilientChatModel)
    assert local_chat.policy.max_retries == 0


def test_secret_not_in_logs_or_errors(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    messages: list[str] = []
    for _, exc, _ in FAILURES:
        wrapper, _ = chat([exc], retries=1, always=True)
        with pytest.raises(ServiceError) as info:
            wrapper.invoke("hi")
        messages.append(str(info.value))
    configs = [
        Settings(_env_file=None, llm_provider="gemini"),  # type: ignore[call-arg]
        Settings(_env_file=None, llm_provider="nope", google_api_key=SECRET),  # type: ignore[call-arg]
    ]
    for s in configs:
        with pytest.raises(ProviderConfigError) as cfg:
            create_chat_model(s)
        messages.append(str(cfg.value))
    assert not any(SECRET in m for m in messages)
    assert SECRET not in caplog.text
    assert any("rate limited" in r.message for r in caplog.records)


def test_retry_after_header_is_carried_into_rate_limit_error() -> None:
    exc = gemini_status(429, headers={"Retry-After": "7"})
    wrapper, _ = chat([exc], retries=0)
    with pytest.raises(LLMRateLimitedError) as info:
        wrapper.invoke("hi")
    assert info.value.headers == {"Retry-After": "7"}


def test_missing_or_bad_retry_after_leaves_headers_empty() -> None:
    bad = gemini_status(429, headers={"Retry-After": "Wed, 21 Oct"})
    for exc in (gemini_status(429), bad):
        wrapper, _ = chat([exc], retries=0)
        with pytest.raises(LLMRateLimitedError) as info:
            wrapper.invoke("hi")
        assert info.value.headers is None
