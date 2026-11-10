"""Real Gemini calls. Skipped when GOOGLE_API_KEY is absent; run with `-m integration`.

The test uses Gemini for chat and embeddings. The embedding model comes from EMBEDDING_MODEL
when EMBEDDING_PROVIDER=gemini, otherwise from the recommended `gemini-embedding-001`.
"""

import pytest

from app.core.config import Settings
from app.providers.factory import create_chat_model, create_embeddings

pytestmark = pytest.mark.integration

RECOMMENDED_EMBEDDING_MODEL = "gemini-embedding-001"


def _settings() -> Settings:
    configured = Settings()
    model = (
        configured.embedding_model
        if configured.embedding_provider == "gemini" and configured.embedding_model
        else RECOMMENDED_EMBEDDING_MODEL
    )
    return Settings(llm_provider="gemini", embedding_provider="gemini", embedding_model=model)


def test_gemini_chat_and_embedding() -> None:
    # Also verifies that the configured embedding model is available for the key.
    settings = _settings()
    if not settings.google_api_key:
        pytest.skip("GOOGLE_API_KEY is not set")
    answer = create_chat_model(settings).invoke("Reply with the single word: pong")
    assert str(answer.content).strip()
    vector = create_embeddings(settings).embed_query("How do I descale the coffee maker?")
    assert len(vector) > 0
    assert all(isinstance(x, float) for x in vector)
