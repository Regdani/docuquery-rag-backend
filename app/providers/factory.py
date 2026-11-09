"""Provider factory: builds chat model, embeddings and vector store from Settings.

Provider packages are imported lazily inside the `_create_*` functions. Missing configuration
or an unknown provider raises ProviderConfigError; there is never a fallback to another provider.
"""

import re
from collections.abc import Callable, Sequence

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from pydantic import SecretStr

from app.core.config import Settings
from app.core.errors import ProviderConfigError
from app.providers.resilience import ResilientChatModel, ResilientEmbeddings, RetryPolicy, unwrap

SUPPORTED_PROVIDERS = ("azure", "gemini", "ollama")


def _require(settings: Settings, provider: str, fields: Sequence[str]) -> None:
    """Raise ProviderConfigError naming every missing env var among `fields`."""
    missing = [name.upper() for name in fields if not str(getattr(settings, name, "")).strip()]
    if missing:
        raise ProviderConfigError(
            f"Provider '{provider}' is missing required settings: {', '.join(missing)}"
        )


def _import_error(provider: str, package: str, exc: ImportError) -> ProviderConfigError:
    return ProviderConfigError(f"Provider '{provider}' needs the '{package}' package: {exc}")


def _unknown(kind: str, value: str | None) -> ProviderConfigError:
    return ProviderConfigError(
        f"Unknown {kind} '{value}'. Supported values: {', '.join(SUPPORTED_PROVIDERS)}"
    )


# --- chat models ---


def _create_azure_chat(settings: Settings) -> BaseChatModel:
    _require(
        settings,
        "azure",
        ["azure_openai_api_key", "azure_openai_endpoint", "azure_openai_chat_deployment"],
    )
    try:
        from langchain_openai import AzureChatOpenAI
    except ImportError as exc:
        raise _import_error("azure", "langchain-openai", exc) from exc
    return AzureChatOpenAI(
        azure_deployment=settings.azure_openai_chat_deployment,
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.max_retries,
    )


def _create_gemini_chat(settings: Settings) -> BaseChatModel:
    _require(settings, "gemini", ["google_api_key"])
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise _import_error("gemini", "langchain-google-genai", exc) from exc
    return ChatGoogleGenerativeAI(
        model=settings.gemini_chat_model,
        google_api_key=settings.google_api_key,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.max_retries,
    )


def _create_ollama_chat(settings: Settings) -> BaseChatModel:
    _require(settings, "ollama", ["ollama_base_url", "ollama_chat_model"])
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise _import_error("ollama", "langchain-ollama", exc) from exc
    return ChatOllama(
        model=settings.ollama_chat_model,
        base_url=settings.ollama_base_url,
        client_kwargs={"timeout": settings.llm_timeout_seconds},
    )


# --- embeddings ---


def _create_azure_embeddings(settings: Settings) -> Embeddings:
    _require(
        settings,
        "azure",
        ["azure_openai_api_key", "azure_openai_endpoint", "azure_openai_embedding_deployment"],
    )
    try:
        from langchain_openai import AzureOpenAIEmbeddings
    except ImportError as exc:
        raise _import_error("azure", "langchain-openai", exc) from exc
    return AzureOpenAIEmbeddings(
        deployment=settings.azure_openai_embedding_deployment,
        azure_endpoint=settings.azure_openai_endpoint,
        openai_api_key=SecretStr(settings.azure_openai_api_key),
        openai_api_version=settings.azure_openai_api_version,
        request_timeout=settings.embedding_timeout_seconds,
        max_retries=settings.max_retries,
    )


def _create_gemini_embeddings(settings: Settings) -> Embeddings:
    _require(settings, "gemini", ["google_api_key"])
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
    except ImportError as exc:
        raise _import_error("gemini", "langchain-google-genai", exc) from exc
    return GoogleGenerativeAIEmbeddings(
        model=settings.gemini_embedding_model,
        google_api_key=settings.google_api_key,
        # The embeddings client has no max_retries; retries come from the resilience wrapper.
        client_args={"timeout": settings.embedding_timeout_seconds},
    )


def _create_ollama_embeddings(settings: Settings) -> Embeddings:
    _require(settings, "ollama", ["ollama_base_url", "ollama_embedding_model"])
    try:
        from langchain_ollama import OllamaEmbeddings
    except ImportError as exc:
        raise _import_error("ollama", "langchain-ollama", exc) from exc
    return OllamaEmbeddings(
        model=settings.ollama_embedding_model,
        base_url=settings.ollama_base_url,
        client_kwargs={"timeout": settings.embedding_timeout_seconds},
    )


# Cloud providers get error translation and bounded 429 retry; ollama is used as is.
_CLOUD = frozenset({"azure", "gemini"})
_CHAT: dict[str, Callable[[Settings], BaseChatModel]] = {
    "azure": _create_azure_chat,
    "gemini": _create_gemini_chat,
    "ollama": _create_ollama_chat,
}
_EMBEDDINGS: dict[str, Callable[[Settings], Embeddings]] = {
    "azure": _create_azure_embeddings,
    "gemini": _create_gemini_embeddings,
    "ollama": _create_ollama_embeddings,
}


def create_chat_model(settings: Settings) -> BaseChatModel:
    """Build the chat model for `settings.llm_provider`."""
    builder = _CHAT.get(settings.llm_provider)
    if builder is None:
        raise _unknown("LLM_PROVIDER", settings.llm_provider)
    model = builder(settings)
    if settings.llm_provider in _CLOUD:
        return ResilientChatModel(
            inner=model,
            provider=settings.llm_provider,
            rate_limit_retries=settings.max_retries,
            backoff_seconds=settings.retry_backoff_seconds,
        )
    return model


def create_embeddings(settings: Settings) -> Embeddings:
    """Build the embeddings for `settings.embedding_provider` (defaults to the LLM provider)."""
    provider = settings.embedding_provider or settings.llm_provider
    builder = _EMBEDDINGS.get(provider)
    if builder is None:
        raise _unknown("EMBEDDING_PROVIDER", provider)
    embeddings = builder(settings)
    if provider in _CLOUD:
        policy = RetryPolicy(provider, settings.max_retries, settings.retry_backoff_seconds)
        return ResilientEmbeddings(embeddings, policy)
    return embeddings


def client_name(client: object) -> str:
    """Class name of the vendor client behind any resilient wrapper (for logging)."""
    return type(unwrap(client)).__name__


def embedding_model_name(settings: Settings) -> str:
    """Name of the embedding model/deployment the configured embedding provider uses."""
    provider = settings.embedding_provider or settings.llm_provider
    names = {
        "azure": settings.azure_openai_embedding_deployment,
        "gemini": settings.gemini_embedding_model,
        "ollama": settings.ollama_embedding_model,
    }
    if provider not in names:
        raise _unknown("EMBEDDING_PROVIDER", provider)
    return f"{provider}:{names[provider]}"


def index_collection_name(settings: Settings) -> str:
    """Collection name = `chroma_collection` + sanitized embedding model, so indexes never mix."""
    suffix = re.sub(r"[^a-zA-Z0-9]+", "-", embedding_model_name(settings)).strip("-").lower()
    return f"{settings.chroma_collection}-{suffix}"[:512].strip("-")


def create_vector_store(settings: Settings, embeddings: Embeddings) -> VectorStore:
    """Build the persistent Chroma vector store (no network access with local embeddings)."""
    try:
        from langchain_chroma import Chroma
    except ImportError as exc:
        raise _import_error("chroma", "langchain-chroma", exc) from exc
    return Chroma(
        collection_name=index_collection_name(settings),
        embedding_function=embeddings,
        persist_directory=settings.index_dir,
    )
