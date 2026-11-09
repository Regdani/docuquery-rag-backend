"""Real Ollama calls. Skipped when Ollama is unreachable or a model is not pulled.

Run with `uv run pytest -m integration -k ollama`.
"""

from pathlib import Path

import httpx
import pytest

from app.core.config import Settings
from app.providers.factory import (
    create_chat_model,
    create_embeddings,
    create_vector_store,
    embedding_model_name,
    index_collection_name,
)
from app.rag.indexing import run_indexing

pytestmark = pytest.mark.integration

SAMPLE = Path(__file__).resolve().parents[2] / "data" / "sample"


@pytest.fixture(scope="module")
def ollama_settings() -> Settings:
    settings = Settings(_env_file=None, llm_provider="ollama")  # type: ignore[call-arg]
    base = settings.ollama_base_url.rstrip("/")
    try:
        response = httpx.get(f"{base}/api/tags", timeout=3.0)
        response.raise_for_status()
        pulled = {str(m.get("name", "")) for m in response.json().get("models", [])}
    except (httpx.HTTPError, ValueError) as exc:
        pytest.skip(f"Ollama is not reachable at {base} ({type(exc).__name__})")
    for model in (settings.ollama_chat_model, settings.ollama_embedding_model):
        if not any(name == model or name.split(":")[0] == model for name in pulled):
            pytest.skip(f"Ollama model '{model}' is not pulled (run: ollama pull {model})")
    return settings


def test_ollama_chat(ollama_settings: Settings) -> None:
    answer = create_chat_model(ollama_settings).invoke("Reply with the single word: pong")
    assert str(answer.content).strip()


def test_ollama_embed_query(ollama_settings: Settings) -> None:
    vector = create_embeddings(ollama_settings).embed_query("How do I descale the coffee maker?")
    assert len(vector) > 0
    assert all(isinstance(x, float) for x in vector)


def test_ollama_index_and_retrieve(ollama_settings: Settings, tmp_path: Path) -> None:
    data_dir = tmp_path / "docs"
    data_dir.mkdir()
    manual = "kv-200-espresso-coffee-maker.md"
    (data_dir / manual).write_text((SAMPLE / manual).read_text(encoding="utf-8"), encoding="utf-8")
    settings = ollama_settings.model_copy(
        update={"data_dir": str(data_dir), "index_dir": str(tmp_path / "index")}
    )
    store = create_vector_store(settings, create_embeddings(settings))
    status = run_indexing(
        settings, store, embedding_model_name(settings), index_collection_name(settings)
    )
    assert status.state == "ready", status.error
    hits = store.similarity_search("How often should I descale the coffee maker?", k=4)
    assert hits
    assert any(h.metadata.get("section") == "Maintenance" for h in hits)
    assert all(h.metadata.get("product_id") == "KV-200" for h in hits)
