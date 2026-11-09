import ast
import logging
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.errors import ProviderConfigError, ServiceError
from app.main import create_app
from app.providers import factory
from app.providers.factory import (
    SUPPORTED_PROVIDERS,
    create_chat_model,
    create_embeddings,
    create_vector_store,
)
from tests.fakes import DeterministicFakeEmbedding, FakeListChatModel, InMemoryVectorStore

AZURE: dict[str, Any] = {
    "azure_openai_api_key": "secret-azure-key",
    "azure_openai_endpoint": "https://example.openai.azure.com",
    "azure_openai_chat_deployment": "chat-dep",
    "azure_openai_embedding_deployment": "emb-dep",
}
GEMINI: dict[str, Any] = {"google_api_key": "secret-google-key"}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("LLM_PROVIDER", "EMBEDDING_PROVIDER", "GOOGLE_API_KEY", "OLLAMA_BASE_URL"):
        monkeypatch.delenv(key, raising=False)


def make(**kwargs: Any) -> Settings:
    return Settings(_env_file=None, **kwargs)  # type: ignore[call-arg]


def test_supported_providers() -> None:
    assert SUPPORTED_PROVIDERS == ("azure", "gemini", "ollama")


def test_default_and_inheritance_and_normalization() -> None:
    assert make().llm_provider == "ollama"
    assert make(llm_provider="gemini").embedding_provider == "gemini"
    s = make(llm_provider=" Azure ", embedding_provider=" OLLAMA ")
    assert (s.llm_provider, s.embedding_provider) == ("azure", "ollama")


def test_service_error_pattern() -> None:
    err = ProviderConfigError("boom")
    assert isinstance(err, ServiceError)
    assert (err.message, err.code, err.status_code) == ("boom", "provider_config_error", 500)


@pytest.mark.parametrize(
    ("provider", "extra", "chat_cls", "emb_cls"),
    [
        ("azure", AZURE, "AzureChatOpenAI", "AzureOpenAIEmbeddings"),
        ("gemini", GEMINI, "ChatGoogleGenerativeAI", "GoogleGenerativeAIEmbeddings"),
        ("ollama", {}, "ChatOllama", "OllamaEmbeddings"),
    ],
)
def test_provider_classes(
    provider: str, extra: dict[str, Any], chat_cls: str, emb_cls: str
) -> None:
    s = make(llm_provider=provider, **extra)
    assert type(create_chat_model(s)).__name__ == chat_cls
    assert type(create_embeddings(s)).__name__ == emb_cls


def test_timeouts_and_retries_applied() -> None:
    s = make(llm_provider="azure", llm_timeout_seconds=12.0, max_retries=5, **AZURE)
    model = create_chat_model(s)
    assert model.request_timeout == 12.0  # type: ignore[attr-defined]
    assert model.max_retries == 5  # type: ignore[attr-defined]

    ae = create_embeddings(make(llm_provider="azure", embedding_timeout_seconds=4.0, **AZURE))
    assert ae.request_timeout == 4.0  # type: ignore[attr-defined]

    g = create_chat_model(make(llm_provider="gemini", llm_timeout_seconds=7.0, **GEMINI))
    assert g.timeout == 7.0  # type: ignore[attr-defined]

    o = create_chat_model(make(llm_timeout_seconds=9.0))
    assert o.client_kwargs == {"timeout": 9.0}  # type: ignore[attr-defined]
    e = create_embeddings(make(embedding_timeout_seconds=3.0))
    assert e.client_kwargs == {"timeout": 3.0}  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("provider", "extra", "missing"),
    [
        ("azure", {}, ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"]),
        ("gemini", {}, ["GOOGLE_API_KEY"]),
    ],
)
def test_missing_variables_named(provider: str, extra: dict[str, Any], missing: list[str]) -> None:
    s = make(llm_provider=provider, **extra)
    with pytest.raises(ProviderConfigError) as chat_exc:
        create_chat_model(s)
    with pytest.raises(ProviderConfigError) as emb_exc:
        create_embeddings(s)
    for name in missing:
        assert name in str(chat_exc.value)
        assert name in str(emb_exc.value)
    assert "AZURE_OPENAI_CHAT_DEPLOYMENT" in str(chat_exc.value) or provider != "azure"
    assert "AZURE_OPENAI_EMBEDDING_DEPLOYMENT" in str(emb_exc.value) or provider != "azure"


def test_azure_missing_only_deployment() -> None:
    s = make(llm_provider="azure", **{**AZURE, "azure_openai_chat_deployment": ""})
    with pytest.raises(ProviderConfigError, match="AZURE_OPENAI_CHAT_DEPLOYMENT"):
        create_chat_model(s)
    create_embeddings(s)  # embedding deployment is set, so this still works


def test_unknown_llm_provider_lists_supported() -> None:
    with pytest.raises(ProviderConfigError) as exc:
        create_chat_model(make(llm_provider="mock"))
    assert "azure, gemini, ollama" in str(exc.value)
    assert "LLM_PROVIDER" in str(exc.value)


def test_unknown_embedding_provider() -> None:
    with pytest.raises(ProviderConfigError) as exc:
        create_embeddings(make(embedding_provider="nope"))
    assert "azure, gemini, ollama" in str(exc.value)
    assert "EMBEDDING_PROVIDER" in str(exc.value)


def test_incomplete_embedding_provider() -> None:
    s = make(llm_provider="ollama", embedding_provider="gemini")
    create_chat_model(s)  # chat side is fine
    with pytest.raises(ProviderConfigError, match="GOOGLE_API_KEY"):
        create_embeddings(s)


def test_mixed_setup() -> None:
    s = make(llm_provider="azure", embedding_provider="ollama", **AZURE)
    assert type(create_chat_model(s)).__name__ == "AzureChatOpenAI"
    assert type(create_embeddings(s)).__name__ == "OllamaEmbeddings"


def test_missing_package_raises_provider_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "langchain_ollama":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ProviderConfigError, match="langchain-ollama"):
        create_chat_model(make())


def test_vector_store_is_chroma_without_network(tmp_path: Path) -> None:
    s = make(index_dir=str(tmp_path / "index"), chroma_collection="test_coll")
    store = create_vector_store(s, DeterministicFakeEmbedding(size=8))
    assert type(store).__name__ == "Chroma"
    assert (tmp_path / "index").exists()
    store.add_texts(["a drill manual"])
    assert store.similarity_search("drill", k=1)[0].page_content == "a drill manual"


def test_fakes_available() -> None:
    assert FakeListChatModel(responses=["hi"]).invoke("x").content == "hi"
    store = InMemoryVectorStore(DeterministicFakeEmbedding(size=4))
    store.add_texts(["x"])
    assert len(store.similarity_search("x", k=1)) == 1


def test_lifespan_builds_once_and_logs(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    s = make(
        llm_provider="azure",
        embedding_provider="ollama",
        index_dir=str(tmp_path / "index"),
        **AZURE,
    )
    with caplog.at_level(logging.INFO, logger="app.main"):
        with TestClient(create_app(s)) as client:
            state = client.app.state  # type: ignore[attr-defined]
            assert type(state.chat_model).__name__ == "AzureChatOpenAI"
            assert type(state.embeddings).__name__ == "OllamaEmbeddings"
            assert type(state.vector_store).__name__ == "Chroma"
    assert "LLM_PROVIDER=azure client=AzureChatOpenAI" in caplog.text
    assert "EMBEDDING_PROVIDER=ollama client=OllamaEmbeddings" in caplog.text
    assert "secret-azure-key" not in caplog.text


def test_lifespan_fails_on_invalid_config() -> None:
    app = create_app(make(llm_provider="gemini"))
    with pytest.raises(ProviderConfigError), TestClient(app):
        pass


def test_provider_imports_confined_to_providers_package() -> None:
    forbidden = {
        "langchain_openai",
        "langchain_ollama",
        "langchain_chroma",
        "langchain_google_genai",
        "openai",
        "ollama",
        "chromadb",
    }
    app_dir = Path(factory.__file__).resolve().parents[1]
    offenders = []
    for path in app_dir.rglob("*.py"):
        if "providers" in path.relative_to(app_dir).parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            offenders += [f"{path.name}: {n}" for n in names if n.split(".")[0] in forbidden]
    assert offenders == []
