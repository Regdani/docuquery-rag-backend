"""Real cloud calls. Skipped when the credentials are absent; run with `-m integration`.

Both tests use the provider for chat and embeddings. The embedding model comes from
EMBEDDING_MODEL when EMBEDDING_PROVIDER names the same provider; otherwise Gemini falls back
to `gemini-embedding-001` and Azure skips (a deployment name cannot be guessed).
"""

import pytest

from app.core.config import Settings
from app.providers.factory import create_chat_model, create_embeddings

pytestmark = pytest.mark.integration

AZURE_VARS = ("azure_openai_api_key", "azure_openai_endpoint", "azure_openai_chat_deployment")
DEFAULT_EMBEDDING_MODEL = {"azure": "", "gemini": "gemini-embedding-001"}


def _settings(provider: str) -> Settings:
    configured = Settings()
    model = (
        configured.embedding_model
        if configured.embedding_provider == provider and configured.embedding_model
        else DEFAULT_EMBEDDING_MODEL[provider]
    )
    return Settings(llm_provider=provider, embedding_provider=provider, embedding_model=model)


def _check(settings: Settings) -> None:
    answer = create_chat_model(settings).invoke("Reply with the single word: pong")
    assert str(answer.content).strip()
    vector = create_embeddings(settings).embed_query("How do I descale the coffee maker?")
    assert len(vector) > 0
    assert all(isinstance(x, float) for x in vector)


def test_azure_chat_and_embedding() -> None:
    settings = _settings("azure")
    missing = [name.upper() for name in AZURE_VARS if not getattr(settings, name)]
    if not settings.embedding_model:
        missing.append("EMBEDDING_MODEL (with EMBEDDING_PROVIDER=azure)")
    if missing:
        pytest.skip(f"Azure settings missing: {', '.join(missing)}")
    _check(settings)


def test_gemini_chat_and_embedding() -> None:
    # Also verifies that the configured embedding model is available for the key.
    settings = _settings("gemini")
    if not settings.google_api_key:
        pytest.skip("GOOGLE_API_KEY is not set")
    _check(settings)
