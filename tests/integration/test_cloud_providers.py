"""Real cloud calls. Skipped when the credentials are absent; run with `-m integration`."""

import pytest

from app.core.config import Settings
from app.providers.factory import create_chat_model, create_embeddings

pytestmark = pytest.mark.integration

AZURE_VARS = (
    "azure_openai_api_key",
    "azure_openai_endpoint",
    "azure_openai_chat_deployment",
    "azure_openai_embedding_deployment",
)


def _settings(provider: str) -> Settings:
    return Settings(llm_provider=provider)


def _check(provider: str) -> None:
    settings = _settings(provider)
    answer = create_chat_model(settings).invoke("Reply with the single word: pong")
    assert str(answer.content).strip()
    vector = create_embeddings(settings).embed_query("How do I descale the coffee maker?")
    assert len(vector) > 0
    assert all(isinstance(x, float) for x in vector)


def test_azure_chat_and_embedding() -> None:
    settings = _settings("azure")
    missing = [name.upper() for name in AZURE_VARS if not getattr(settings, name)]
    if missing:
        pytest.skip(f"Azure credentials missing: {', '.join(missing)}")
    _check("azure")


def test_gemini_chat_and_embedding() -> None:
    # Also verifies that the default embedding model (gemini-embedding-001) is available.
    settings = _settings("gemini")
    if not settings.google_api_key:
        pytest.skip("GOOGLE_API_KEY is not set")
    assert settings.gemini_embedding_model == "gemini-embedding-001"
    _check("gemini")
