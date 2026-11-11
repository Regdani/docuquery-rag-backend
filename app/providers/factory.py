"""Provider factory: builds chat model, embeddings and vector store from Settings.

Provider packages are imported lazily inside the `_create_*` functions. Missing configuration
or an unknown provider raises ProviderConfigError; there is never a fallback to another provider.
"""

import re
from collections.abc import Callable, Sequence
from typing import Literal

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from app.core.config import Settings
from app.core.errors import ProviderConfigError
from app.providers.resilience import (
    ResilientChatModel,
    ResilientEmbeddings,
    RetryPolicy,
    ollama_translator,
    unwrap,
)

SUPPORTED_PROVIDERS = ("gemini", "ollama")
# Cosine distance: Chroma then reports relevance as cosine similarity (0..1), whether or not
# the embedding model returns normalized vectors.
_DISTANCE: Literal["cosine"] = "cosine"


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
    if value is None:
        return ProviderConfigError(
            f"{kind} is not set. Supported values: {', '.join(SUPPORTED_PROVIDERS)}"
        )
    return ProviderConfigError(
        f"Unknown {kind} '{value}'. Supported values: {', '.join(SUPPORTED_PROVIDERS)}"
    )


# --- chat models ---


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
        # 1 = a single attempt (0 would mean the SDK default). Retries come only from the
        # resilience wrapper, so they do not multiply.
        max_retries=1,
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


def _create_gemini_embeddings(settings: Settings) -> Embeddings:
    _require(settings, "gemini", ["google_api_key", "embedding_model"])
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
    except ImportError as exc:
        raise _import_error("gemini", "langchain-google-genai", exc) from exc
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
        # The embeddings client has no max_retries; retries come from the resilience wrapper.
        client_args={"timeout": settings.embedding_timeout_seconds},
    )


def _create_ollama_embeddings(settings: Settings) -> Embeddings:
    _require(settings, "ollama", ["ollama_base_url", "embedding_model"])
    try:
        from langchain_ollama import OllamaEmbeddings
    except ImportError as exc:
        raise _import_error("ollama", "langchain-ollama", exc) from exc
    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
        client_kwargs={"timeout": settings.embedding_timeout_seconds},
    )


_CHAT: dict[str, Callable[[Settings], BaseChatModel]] = {
    "gemini": _create_gemini_chat,
    "ollama": _create_ollama_chat,
}
_EMBEDDINGS: dict[str, Callable[[Settings], Embeddings]] = {
    "gemini": _create_gemini_embeddings,
    "ollama": _create_ollama_embeddings,
}


def _policy(settings: Settings, provider: str, model: str) -> RetryPolicy:
    """Gemini: error translation and bounded 429 retry. Ollama: translation only (no 429s)."""
    if provider == "ollama":
        return RetryPolicy("ollama", 0, 0.0, ollama_translator(settings.ollama_base_url, model))
    return RetryPolicy(provider, settings.max_retries, settings.retry_backoff_seconds)


def create_chat_model(settings: Settings) -> BaseChatModel:
    """Build the chat model for `settings.llm_provider`, wrapped for error translation."""
    provider = settings.llm_provider
    builder = _CHAT.get(provider) if provider else None
    if provider is None or builder is None:
        raise _unknown("LLM_PROVIDER", provider)
    return ResilientChatModel(
        inner=builder(settings),
        policy=_policy(settings, provider, settings.ollama_chat_model),
    )


def create_embeddings(settings: Settings) -> Embeddings:
    """Build the embeddings for `settings.embedding_provider` (required, no fallback)."""
    provider = settings.embedding_provider
    builder = _EMBEDDINGS.get(provider) if provider else None
    if provider is None or builder is None:
        raise _unknown("EMBEDDING_PROVIDER", provider)
    return ResilientEmbeddings(
        builder(settings), _policy(settings, provider, settings.embedding_model)
    )


def client_name(client: object) -> str:
    """Class name of the vendor client behind any resilient wrapper (for logging)."""
    return type(unwrap(client)).__name__


def embedding_model_name(settings: Settings) -> str:
    """`<provider>:<model>` of the configured embeddings; it also names the index collection."""
    provider = settings.embedding_provider
    if provider not in _EMBEDDINGS:
        raise _unknown("EMBEDDING_PROVIDER", provider)
    _require(settings, provider, ["embedding_model"])
    return f"{provider}:{settings.embedding_model.strip()}"


def index_collection_name(settings: Settings) -> str:
    """Collection name = `chroma_collection` + sanitized embedding model, so indexes never mix."""
    suffix = re.sub(r"[^a-zA-Z0-9]+", "-", embedding_model_name(settings)).strip("-").lower()
    return f"{settings.chroma_collection}-{suffix}"[:512].strip("-")


def create_vector_store(settings: Settings, embeddings: Embeddings) -> VectorStore:
    """Build the persistent Chroma vector store (no network access with local embeddings)."""
    try:
        from chromadb.api.collection_configuration import (
            CreateCollectionConfiguration,
            CreateHNSWConfiguration,
        )
        from langchain_chroma import Chroma
    except ImportError as exc:
        raise _import_error("chroma", "langchain-chroma", exc) from exc
    store = Chroma(
        collection_name=index_collection_name(settings),
        embedding_function=embeddings,
        persist_directory=settings.index_dir,
        collection_configuration=CreateCollectionConfiguration(
            hnsw=CreateHNSWConfiguration(space=_DISTANCE)
        ),
    )
    hnsw = store._collection.configuration.get("hnsw") or {}
    if hnsw.get("space") != _DISTANCE:
        # An existing collection keeps the distance it was created with; an index built with
        # another distance is dropped here and rebuilt by the indexing step.
        store.reset_collection()
    return store
