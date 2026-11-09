"""FastAPI application factory."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import Settings, get_settings
from app.providers.factory import create_chat_model, create_embeddings, create_vector_store

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI app. Providers are created once in the lifespan."""
    resolved = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        chat_model = create_chat_model(resolved)
        logger.info("LLM_PROVIDER=%s client=%s", resolved.llm_provider, type(chat_model).__name__)
        embeddings = create_embeddings(resolved)
        logger.info(
            "EMBEDDING_PROVIDER=%s client=%s",
            resolved.embedding_provider,
            type(embeddings).__name__,
        )
        app.state.chat_model = chat_model
        app.state.embeddings = embeddings
        app.state.vector_store = create_vector_store(resolved, embeddings)
        yield

    app = FastAPI(title="DocuQuery RAG backend", version="0.1.0", lifespan=lifespan)
    app.state.settings = resolved

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {"service": "docuquery"}

    return app


app = create_app()
