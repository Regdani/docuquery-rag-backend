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
from app.providers.resilience import unwrap
from tests.fakes import DeterministicFakeEmbedding, FakeListChatModel, InMemoryVectorStore

AZURE: dict[str, Any] = {
    "azure_openai_api_key": "secret-azure-key",
    "azure_openai_endpoint": "https://example.openai.azure.com",
    "azure_openai_chat_deployment": "chat-dep",
}
GEMINI: dict[str, Any] = {"google_api_key": "secret-google-key"}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "LLM_PROVIDER",
        "EMBEDDING_PROVIDER",
        "EMBEDDING_MODEL",
        "GOOGLE_API_KEY",
        "OLLAMA_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)


def make(**kwargs: Any) -> Settings:
    return Settings(_env_file=None, **kwargs)  # type: ignore[call-arg]


def both(provider: str, **kwargs: Any) -> Settings:
    """Chat and embeddings from the same provider, with an explicit embedding model."""
    kwargs.setdefault("embedding_model", "emb-model")
    return make(llm_provider=provider, embedding_provider=provider, **kwargs)


def test_supported_providers() -> None:
    assert SUPPORTED_PROVIDERS == ("azure", "gemini", "ollama")


def test_no_default_no_inheritance_and_normalization() -> None:
    assert make().llm_provider is None
    assert make(llm_provider="gemini").embedding_provider is None
    s = make(llm_provider=" Azure ", embedding_provider=" OLLAMA ")
    assert (s.llm_provider, s.embedding_provider) == ("azure", "ollama")


def test_service_error_pattern() -> None:
    err = ProviderConfigError("boom")
    assert isinstance(err, ServiceError)
    assert (err.message, err.code, err.status_code) == ("boom", "provider_config_error", 500)


def test_provider_defaults() -> None:
    s = make()
    assert s.azure_openai_api_version
    assert s.gemini_chat_model == "gemini-3.1-flash-lite"


def test_azure_all_missing_vars_named() -> None:
    s = make(llm_provider="azure", embedding_provider="azure")
    with pytest.raises(ProviderConfigError) as chat_exc:
        create_chat_model(s)
    with pytest.raises(ProviderConfigError) as emb_exc:
        create_embeddings(s)
    for name in ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_CHAT_DEPLOYMENT"):
        assert name in str(chat_exc.value)
    for name in ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "EMBEDDING_MODEL"):
        assert name in str(emb_exc.value)


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
    s = both(provider, **extra)
    assert type(unwrap(create_chat_model(s))).__name__ == chat_cls
    assert type(unwrap(create_embeddings(s))).__name__ == emb_cls


def test_timeouts_and_retries_applied() -> None:
    s = make(llm_provider="azure", llm_timeout_seconds=12.0, max_retries=5, **AZURE)
    model = unwrap(create_chat_model(s))
    assert model.request_timeout == 12.0  # type: ignore[attr-defined]
    assert model.max_retries == 5  # type: ignore[attr-defined]

    ae = unwrap(create_embeddings(both("azure", embedding_timeout_seconds=4.0, **AZURE)))
    assert ae.request_timeout == 4.0  # type: ignore[attr-defined]

    g = unwrap(
        create_chat_model(
            make(llm_provider="gemini", llm_timeout_seconds=7.0, max_retries=3, **GEMINI)
        )
    )
    assert g.timeout == 7.0  # type: ignore[attr-defined]
    assert g.max_retries == 3  # type: ignore[attr-defined]
    ge = unwrap(create_embeddings(both("gemini", embedding_timeout_seconds=6.0, **GEMINI)))
    assert ge.client_args == {"timeout": 6.0}  # type: ignore[attr-defined]
    assert ge.client._api_client._httpx_client.timeout.read == 6.0  # type: ignore[attr-defined]

    o = unwrap(create_chat_model(both("ollama", llm_timeout_seconds=9.0)))
    assert o.client_kwargs == {"timeout": 9.0}  # type: ignore[attr-defined]
    e = unwrap(create_embeddings(both("ollama", embedding_timeout_seconds=3.0)))
    assert e.client_kwargs == {"timeout": 3.0}  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("provider", "extra", "missing"),
    [
        ("azure", {}, ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"]),
        ("gemini", {}, ["GOOGLE_API_KEY"]),
    ],
)
def test_missing_variables_named(provider: str, extra: dict[str, Any], missing: list[str]) -> None:
    s = make(llm_provider=provider, embedding_provider=provider, **extra)
    with pytest.raises(ProviderConfigError) as chat_exc:
        create_chat_model(s)
    with pytest.raises(ProviderConfigError) as emb_exc:
        create_embeddings(s)
    for name in missing:
        assert name in str(chat_exc.value)
        assert name in str(emb_exc.value)
    assert "AZURE_OPENAI_CHAT_DEPLOYMENT" in str(chat_exc.value) or provider != "azure"
    assert "EMBEDDING_MODEL" in str(emb_exc.value)


def test_azure_missing_only_deployment() -> None:
    s = both("azure", **{**AZURE, "azure_openai_chat_deployment": ""})
    with pytest.raises(ProviderConfigError, match="AZURE_OPENAI_CHAT_DEPLOYMENT"):
        create_chat_model(s)
    create_embeddings(s)  # embedding deployment is set, so this still works


def test_unknown_llm_provider_lists_supported() -> None:
    with pytest.raises(ProviderConfigError) as exc:
        create_chat_model(make(llm_provider="mock"))
    assert "azure, gemini, ollama" in str(exc.value)
    assert "LLM_PROVIDER" in str(exc.value)


@pytest.mark.parametrize(
    "call", [create_chat_model, create_embeddings, factory.embedding_model_name]
)
def test_missing_provider_is_named(call: Any) -> None:
    variable = "LLM_PROVIDER" if call is create_chat_model else "EMBEDDING_PROVIDER"
    with pytest.raises(ProviderConfigError, match=f"{variable} is not set") as exc:
        call(make())
    assert "azure, gemini, ollama" in str(exc.value)


@pytest.mark.parametrize("provider", SUPPORTED_PROVIDERS)
def test_missing_embedding_model_is_named(provider: str) -> None:
    s = make(embedding_provider=provider, **AZURE, **GEMINI)
    with pytest.raises(ProviderConfigError, match="EMBEDDING_MODEL"):
        create_embeddings(s)
    with pytest.raises(ProviderConfigError, match="EMBEDDING_MODEL"):
        factory.embedding_model_name(s)


def test_embedding_model_is_used_by_every_provider() -> None:
    ollama = unwrap(create_embeddings(both("ollama", embedding_model="m1")))
    assert ollama.model == "m1"  # type: ignore[attr-defined]
    gemini = unwrap(create_embeddings(both("gemini", embedding_model="m2", **GEMINI)))
    assert gemini.model.endswith("m2")  # type: ignore[attr-defined]
    azure = unwrap(create_embeddings(both("azure", embedding_model="m3", **AZURE)))
    assert azure.deployment == "m3"  # type: ignore[attr-defined]
    assert factory.embedding_model_name(both("gemini", embedding_model="m2")) == "gemini:m2"


def test_unknown_embedding_provider() -> None:
    with pytest.raises(ProviderConfigError) as exc:
        create_embeddings(make(embedding_provider="nope"))
    assert "azure, gemini, ollama" in str(exc.value)
    assert "EMBEDDING_PROVIDER" in str(exc.value)


def test_incomplete_embedding_provider() -> None:
    s = make(llm_provider="ollama", embedding_provider="gemini", embedding_model="emb")
    create_chat_model(s)  # chat side is fine
    with pytest.raises(ProviderConfigError, match="GOOGLE_API_KEY"):
        create_embeddings(s)


def test_mixed_setup() -> None:
    s = make(llm_provider="azure", embedding_provider="ollama", embedding_model="emb", **AZURE)
    assert type(unwrap(create_chat_model(s))).__name__ == "AzureChatOpenAI"
    assert type(unwrap(create_embeddings(s))).__name__ == "OllamaEmbeddings"


def test_missing_package_raises_provider_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "langchain_ollama":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ProviderConfigError, match="langchain-ollama"):
        create_chat_model(make(llm_provider="ollama"))


def test_vector_store_is_chroma_without_network(tmp_path: Path) -> None:
    s = both("ollama", index_dir=str(tmp_path / "index"), chroma_collection="test_coll")
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
        embedding_model="nomic-embed-text",
        index_dir=str(tmp_path / "index"),
        data_dir=str(tmp_path / "missing"),
        **AZURE,
    )
    with caplog.at_level(logging.INFO, logger="app.main"):
        with TestClient(create_app(s)) as client:
            state = client.app.state  # type: ignore[attr-defined]
            assert type(unwrap(state.chat_model)).__name__ == "AzureChatOpenAI"
            assert type(unwrap(state.embeddings)).__name__ == "OllamaEmbeddings"
            assert type(state.vector_store).__name__ == "Chroma"
    assert "LLM_PROVIDER=azure client=AzureChatOpenAI" in caplog.text
    assert "EMBEDDING_PROVIDER=ollama client=OllamaEmbeddings" in caplog.text
    assert "secret-azure-key" not in caplog.text


def test_lifespan_fails_on_invalid_config() -> None:
    app = create_app(make(llm_provider="gemini"))
    with pytest.raises(ProviderConfigError), TestClient(app):
        pass


def test_lifespan_fails_without_providers() -> None:
    with pytest.raises(ProviderConfigError, match="LLM_PROVIDER is not set"):
        with TestClient(create_app(make())):
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
        "google",
        "httpx",
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
