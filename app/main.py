"""FastAPI application factory."""

from fastapi import FastAPI

from app.core.config import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI app."""
    app = FastAPI(title="DocuQuery RAG backend", version="0.1.0")
    app.state.settings = settings or get_settings()

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {"service": "docuquery"}

    return app


app = create_app()
