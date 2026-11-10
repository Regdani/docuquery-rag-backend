import pytest
from pydantic import ValidationError

from app.core.config import Settings

KEYS = [
    "LLM_PROVIDER",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "OLLAMA_BASE_URL",
    "RETRIEVAL_TOP_K",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)


def make() -> Settings:
    return Settings(_env_file=None)  # type: ignore[call-arg]


def test_providers_and_embedding_model_have_no_defaults() -> None:
    s = make()
    assert (s.llm_provider, s.embedding_provider, s.embedding_model) == (None, None, "")
    assert s.ollama_chat_model == "llama3.2:3b"


def test_embedding_provider_does_not_follow_llm_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert make().embedding_provider is None


def test_blank_providers_are_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "   ")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "")
    s = make()
    assert (s.llm_provider, s.embedding_provider) == (None, None)


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", " Gemini ")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "OLLAMA")
    monkeypatch.setenv("EMBEDDING_MODEL", "mxbai-embed-large")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://example:1234")
    s = make()
    assert (s.llm_provider, s.embedding_provider) == ("gemini", "ollama")
    assert s.embedding_model == "mxbai-embed-large"
    assert s.ollama_base_url == "http://example:1234"


def test_retrieval_top_k_default() -> None:
    assert make().retrieval_top_k == 4


@pytest.mark.parametrize("value", ["1", "20"])
def test_retrieval_top_k_accepts_bounds(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("RETRIEVAL_TOP_K", value)
    assert make().retrieval_top_k == int(value)


@pytest.mark.parametrize("value", ["0", "-1", "21", "abc"])
def test_retrieval_top_k_rejects_invalid(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("RETRIEVAL_TOP_K", value)
    with pytest.raises(ValidationError, match="retrieval_top_k"):
        make()
