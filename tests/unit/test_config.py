import pytest
from pydantic import ValidationError

from app.core.config import Settings

KEYS = ["LLM_PROVIDER", "EMBEDDING_PROVIDER", "OLLAMA_BASE_URL", "RETRIEVAL_TOP_K"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)


def make() -> Settings:
    return Settings(_env_file=None)  # type: ignore[call-arg]


def test_defaults() -> None:
    s = make()
    assert s.llm_provider == "ollama"
    assert s.embedding_provider == "ollama"
    assert s.ollama_chat_model == "llama3.2:3b"
    assert s.ollama_embedding_model == "nomic-embed-text"


def test_embedding_provider_follows_llm_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert make().embedding_provider == "gemini"


def test_empty_embedding_provider_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "azure")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "")
    assert make().embedding_provider == "azure"


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "azure")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://example:1234")
    s = make()
    assert (s.llm_provider, s.embedding_provider) == ("azure", "ollama")
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
