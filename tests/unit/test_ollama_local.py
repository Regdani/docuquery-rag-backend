"""Free local mode: provider switching by env only, keyless ingest and local failure handling."""

import re
from pathlib import Path
from typing import Any

import httpx
import ollama
import pytest

from app.core.config import Settings
from app.core.errors import LLMTimeoutError, LLMUnavailableError
from app.ingest import main as ingest_main
from app.providers import factory
from app.providers.factory import (
    create_chat_model,
    create_embeddings,
    embedding_model_name,
    index_collection_name,
)
from app.providers.resilience import unwrap
from app.rag.indexing import run_indexing
from tests.fakes import (
    DeterministicFakeEmbedding,
    FlakyChatModel,
    FlakyEmbeddings,
    InMemoryVectorStore,
)

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "data" / "sample"
BASE_URL = "http://ollama.test:11434"
ENV_VARS = (
    "LLM_PROVIDER",
    "EMBEDDING_PROVIDER",
    "GOOGLE_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_CHAT_DEPLOYMENT",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    "OLLAMA_BASE_URL",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ENV_VARS:
        monkeypatch.delenv(key, raising=False)


def settings(**kwargs: Any) -> Settings:
    return Settings(_env_file=None, **kwargs)  # type: ignore[call-arg]


def use_embeddings(monkeypatch: pytest.MonkeyPatch, embeddings: Any) -> None:
    """Replace the vendor OllamaEmbeddings client by a fake; the factory wrapper stays."""
    monkeypatch.setitem(factory._EMBEDDINGS, "ollama", lambda s: embeddings)


# --- same code path, only env vars change ---


@pytest.mark.parametrize(
    ("llm", "emb", "chat_cls", "emb_cls"),
    [
        ("ollama", None, "ChatOllama", "OllamaEmbeddings"),
        ("azure", "ollama", "AzureChatOpenAI", "OllamaEmbeddings"),
        ("gemini", "ollama", "ChatGoogleGenerativeAI", "OllamaEmbeddings"),
        ("ollama", "azure", "ChatOllama", "AzureOpenAIEmbeddings"),
        ("gemini", None, "ChatGoogleGenerativeAI", "GoogleGenerativeAIEmbeddings"),
    ],
)
def test_providers_switch_by_env_only(
    monkeypatch: pytest.MonkeyPatch, llm: str, emb: str | None, chat_cls: str, emb_cls: str
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", llm)
    if emb:
        monkeypatch.setenv("EMBEDDING_PROVIDER", emb)
    monkeypatch.setenv("GOOGLE_API_KEY", "g-key")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "a-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "chat")
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "emb")
    s = settings()  # same call every time; only the environment differs
    assert type(unwrap(create_chat_model(s))).__name__ == chat_cls
    assert type(unwrap(create_embeddings(s))).__name__ == emb_cls


def test_ollama_defaults_need_no_key() -> None:
    s = settings()
    assert (s.ollama_base_url, s.ollama_chat_model, s.ollama_embedding_model) == (
        "http://localhost:11434",
        "llama3.2:3b",
        "nomic-embed-text",
    )
    assert type(unwrap(create_chat_model(s))).__name__ == "ChatOllama"
    assert type(unwrap(create_embeddings(s))).__name__ == "OllamaEmbeddings"


def test_no_provider_branching_outside_factory() -> None:
    pattern = re.compile(r"(llm_provider|embedding_provider|\bprovider)\s*(==|!=|in\b)")
    offenders = []
    for path in (ROOT / "app").rglob("*.py"):
        rel = path.relative_to(ROOT / "app").as_posix()
        if rel in ("providers/factory.py", "providers/resilience.py"):
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{rel}:{number}: {line.strip()}")
    assert offenders == []


# --- ingest with no cloud keys ---


def test_ingest_over_sample_without_keys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    use_embeddings(monkeypatch, DeterministicFakeEmbedding(size=16))
    s = settings(llm_provider="ollama", data_dir=str(SAMPLE), index_dir=str(tmp_path))
    assert s.google_api_key == "" and s.azure_openai_api_key == ""
    store = InMemoryVectorStore(create_embeddings(s))
    collection = index_collection_name(s)
    assert collection == "manuals-ollama-nomic-embed-text"
    status = run_indexing(s, store, embedding_model_name(s), collection)
    assert status.state == "ready"
    assert status.document_count == len(list(SAMPLE.glob("*.md")))
    assert status.chunk_count > 0
    assert status.embedding_model == "ollama:nomic-embed-text"


# --- local failures ---

NOT_FOUND = ollama.ResponseError("model 'nomic-embed-text' not found", 404)
FAILURES: list[tuple[str, BaseException, list[str]]] = [
    ("refused", ConnectionError("Failed to connect to Ollama"), [BASE_URL, "ollama serve"]),
    ("transport", httpx.ConnectError("refused"), [BASE_URL, "ollama serve"]),
    ("missing-model", NOT_FOUND, [BASE_URL, "nomic-embed-text", "ollama pull nomic-embed-text"]),
    ("timeout", httpx.ReadTimeout("slow"), [BASE_URL, "in time"]),
]


@pytest.mark.parametrize(("name", "failure", "expected"), FAILURES, ids=[f[0] for f in FAILURES])
def test_ingest_failure_is_clear(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    name: str,
    failure: BaseException,
    expected: list[str],
) -> None:
    use_embeddings(monkeypatch, FlakyEmbeddings([failure], always=True))
    s = settings(ollama_base_url=BASE_URL, data_dir=str(SAMPLE), index_dir=str(tmp_path))
    store = InMemoryVectorStore(create_embeddings(s))
    status = run_indexing(s, store, embedding_model_name(s), index_collection_name(s))
    assert status.state == "failed"
    assert status.error is not None
    for part in expected:
        assert part in status.error
    assert "Traceback" not in status.error


def test_ingest_cli_reports_ollama_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    use_embeddings(monkeypatch, FlakyEmbeddings([ConnectionError("refused")], always=True))
    s = settings(ollama_base_url=BASE_URL, data_dir=str(SAMPLE), index_dir=str(tmp_path))
    monkeypatch.setattr("app.ingest.get_settings", lambda: s)
    monkeypatch.setattr("app.ingest.create_vector_store", lambda st, e: InMemoryVectorStore(e))
    assert ingest_main() == 1
    err = capsys.readouterr().err
    assert BASE_URL in err and "Traceback" not in err


@pytest.mark.parametrize(
    ("failure", "error_cls", "code", "status"),
    [
        (ConnectionError("refused"), LLMUnavailableError, "LLM_UNAVAILABLE", 502),
        (NOT_FOUND, LLMUnavailableError, "LLM_UNAVAILABLE", 502),
        (httpx.ReadTimeout("slow"), LLMTimeoutError, "LLM_TIMEOUT", 504),
    ],
)
def test_chat_failures_map_to_domain_errors(
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
    error_cls: type[Exception],
    code: str,
    status: int,
) -> None:
    fake = FlakyChatModel(failures=[failure], always=True)
    monkeypatch.setitem(factory._CHAT, "ollama", lambda s: fake)
    model = create_chat_model(settings(ollama_base_url=BASE_URL))
    with pytest.raises(error_cls) as exc_info:
        model.invoke("hi")
    err: Any = exc_info.value
    assert (err.code, err.status_code) == (code, status)
    assert BASE_URL in err.message
    assert fake.calls == 1  # no retries locally


def test_non_ollama_errors_pass_through(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FlakyChatModel(failures=[ValueError("bug")], always=True)
    monkeypatch.setitem(factory._CHAT, "ollama", lambda s: fake)
    with pytest.raises(ValueError, match="bug"):
        create_chat_model(settings()).invoke("hi")
