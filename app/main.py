"""FastAPI application factory."""

import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from app.api.handlers import register_exception_handlers
from app.core.config import Settings, get_settings
from app.memory.store import get_session_store
from app.providers.factory import (
    client_name,
    create_chat_model,
    create_embeddings,
    create_vector_store,
    embedding_model_name,
    index_collection_name,
)
from app.rag.index_status import IndexStatus
from app.rag.indexing import run_indexing

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    chat_model: BaseChatModel | None = None,
    embeddings: Embeddings | None = None,
    vector_store: VectorStore | None = None,
) -> FastAPI:
    """Build the FastAPI app. Providers are created once in the lifespan.

    Pre-built providers can be passed in (tests inject fakes this way); otherwise the
    provider factory builds them from settings. Indexing runs in a background task so the
    `indexing` state is observable; `app.state.index_status` is replaced when it finishes.
    """
    resolved = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        chat = chat_model or create_chat_model(resolved)
        logger.info("LLM_PROVIDER=%s client=%s", resolved.llm_provider, client_name(chat))
        emb = embeddings or create_embeddings(resolved)
        logger.info(
            "EMBEDDING_PROVIDER=%s client=%s", resolved.embedding_provider, client_name(emb)
        )
        model_name = embedding_model_name(resolved)
        store = vector_store or create_vector_store(resolved, emb)
        session_store = get_session_store(resolved)
        app.state.session_store = session_store
        logger.info(
            "Conversation history is in memory only, lost on restart; max %d messages per session",
            resolved.history_max_messages,
        )
        app.state.chat_model = chat
        app.state.embeddings = emb
        app.state.vector_store = store
        app.state.index_status = IndexStatus(state="indexing", embedding_model=model_name)

        async def build_index() -> None:
            app.state.index_status = await asyncio.to_thread(
                run_indexing, resolved, store, model_name, index_collection_name(resolved)
            )

        task = asyncio.create_task(build_index())
        app.state.index_task = task
        yield
        if not task.done():
            with contextlib.suppress(asyncio.CancelledError):
                await task

    app = FastAPI(title="DocuQuery RAG backend", version="0.1.0", lifespan=lifespan)
    app.state.settings = resolved
    app.state.index_status = IndexStatus(state="indexing")
    register_exception_handlers(app)

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {"service": "docuquery"}

    return app


app = create_app()
