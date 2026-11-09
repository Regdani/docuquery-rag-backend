import logging
import time
from pathlib import Path
from typing import Annotated, Any

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.api.deps import require_index_ready
from app.core.config import Settings
from app.core.errors import IndexNotReadyError, ServiceError
from app.ingest import main as ingest_main
from app.main import create_app
from app.providers.factory import index_collection_name
from app.rag.chunking import chunk_document, chunk_documents
from app.rag.index_status import IndexStatus
from app.rag.indexing import run_indexing
from app.rag.loader import load_documents, parse_manual
from tests.fakes import DeterministicFakeEmbedding, InMemoryVectorStore

SAMPLE = Path(__file__).resolve().parents[2] / "data" / "sample"
MODEL = "ollama:test-model"


class CountingEmbedding(DeterministicFakeEmbedding):
    """Deterministic fake embedding that counts embedded texts."""

    calls: int = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += len(texts)
        return super().embed_documents(texts)


def make(tmp_path: Path, **kwargs: Any) -> Settings:
    kwargs.setdefault("data_dir", str(tmp_path / "docs"))
    kwargs.setdefault("index_dir", str(tmp_path / "index"))
    return Settings(_env_file=None, **kwargs)  # type: ignore[call-arg]


def write_doc(directory: Path, name: str, product: str, sections: dict[str, str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    body = "".join(f"## {title}\n{text}\n\n" for title, text in sections.items())
    (directory / name).write_text(f"# {product} – Test Product\n\n{body}", encoding="utf-8")


def run(settings: Settings, store: InMemoryVectorStore) -> IndexStatus:
    return run_indexing(settings, store, MODEL, "manuals-test")


# --- chunking and metadata (AC6, AC7) ---


def test_chunk_metadata_on_sample_document() -> None:
    docs = {d.source: d for d in load_documents(SAMPLE)}
    doc = docs["kv-200-espresso-coffee-maker.md"]
    assert (doc.product_id, doc.product_name) == ("KV-200", "Aroma Brew Espresso Coffee Maker")
    chunks = chunk_document(doc, 800, 100)
    assert {c.metadata["section"] for c in chunks} >= {"Safety", "Maintenance", "Usage"}
    for c in chunks:
        assert c.metadata["source"] == "kv-200-espresso-coffee-maker.md"
        assert c.metadata["product_id"] == "KV-200"
        assert c.metadata["product_name"] == "Aroma Brew Espresso Coffee Maker"
        assert c.metadata["chunk_id"] == c.id
    all_ids = [c.id for c in chunk_documents(list(docs.values()), 800, 100)]
    assert len(all_ids) == len(set(all_ids))
    assert all_ids == [c.id for c in chunk_documents(list(docs.values()), 800, 100)]


def test_sections_are_never_merged(tmp_path: Path) -> None:
    write_doc(tmp_path, "a.md", "AA-1", {"One": "short text", "Two": "other short text"})
    doc = load_documents(tmp_path)[0]
    chunks = chunk_document(doc, 5000, 0)
    assert [c.metadata["section"] for c in chunks] == ["One", "Two"]
    assert "other short" not in chunks[0].page_content


def test_chunk_settings_change_chunks(tmp_path: Path) -> None:
    write_doc(tmp_path, "a.md", "AA-1", {"One": " ".join(f"word{i}" for i in range(120))})
    doc = load_documents(tmp_path)[0]
    big = chunk_document(doc, 2000, 0)
    small = chunk_document(doc, 100, 0)
    overlapped = chunk_document(doc, 100, 40)
    assert len(big) == 1
    assert len(small) > 1
    assert len(overlapped) > len(small)


def test_parse_manual_requires_heading() -> None:
    with pytest.raises(ValueError, match="heading"):
        parse_manual("x.md", "no heading\n## Safety\ntext")


# --- collection name (AC8) ---


def test_collection_name_includes_sanitized_model(tmp_path: Path) -> None:
    a = index_collection_name(make(tmp_path, ollama_embedding_model="nomic-embed-text"))
    b = index_collection_name(make(tmp_path, ollama_embedding_model="mxbai/embed:large"))
    assert a != b
    assert a.startswith("manuals-") and "nomic-embed-text" in a
    assert all(ch.isalnum() or ch in "-_." for ch in b)


# --- reuse and rebuild (AC4, AC5) ---


def test_unchanged_index_is_reused_without_embedding_calls(tmp_path: Path) -> None:
    write_doc(tmp_path / "docs", "a.md", "AA-1", {"One": "first", "Two": "second"})
    settings = make(tmp_path)
    emb = CountingEmbedding(size=8)
    store = InMemoryVectorStore(emb)
    first = run(settings, store)
    assert (first.state, first.chunk_count, first.reused) == ("ready", 2, False)
    assert emb.calls == 2
    second = run(settings, store)
    assert (second.state, second.reused) == ("ready", True)
    assert emb.calls == 2  # no new embedding calls


def test_changed_documents_rebuild_without_stale_chunks(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    write_doc(docs, "a.md", "AA-1", {"One": "first", "Two": "second", "Three": "third"})
    write_doc(docs, "b.md", "BB-2", {"One": "bee"})
    settings = make(tmp_path)
    store = InMemoryVectorStore(CountingEmbedding(size=8))
    assert run(settings, store).chunk_count == 4
    assert len(store.store) == 4

    (docs / "b.md").unlink()
    write_doc(docs, "a.md", "AA-1", {"One": "changed"})
    status = run(settings, store)
    assert (status.chunk_count, status.reused) == (1, False)
    assert len(store.store) == 1


def test_changed_chunk_settings_rebuild(tmp_path: Path) -> None:
    write_doc(tmp_path / "docs", "a.md", "AA-1", {"One": " ".join(["word"] * 100)})
    store = InMemoryVectorStore(CountingEmbedding(size=8))
    first = run(make(tmp_path, chunk_size=2000, chunk_overlap=0), store)
    second = run(make(tmp_path, chunk_size=100, chunk_overlap=0), store)
    assert not second.reused
    assert second.chunk_count > first.chunk_count
    assert len(store.store) == second.chunk_count


def test_missing_chunks_in_store_trigger_rebuild(tmp_path: Path) -> None:
    write_doc(tmp_path / "docs", "a.md", "AA-1", {"One": "first"})
    settings = make(tmp_path)
    run(settings, InMemoryVectorStore(CountingEmbedding(size=8)))
    fresh = InMemoryVectorStore(CountingEmbedding(size=8))
    status = run(settings, fresh)  # fingerprint matches but the store is empty
    assert not status.reused
    assert len(fresh.store) == 1


# --- failure handling (AC9, AC10) ---


def test_invalid_file_logged_and_skipped(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    docs = tmp_path / "docs"
    write_doc(docs, "good.md", "AA-1", {"One": "first"})
    (docs / "bad.md").write_text("no title here\n## One\ntext", encoding="utf-8")
    (docs / "binary.md").write_bytes(b"\xff\xfe\x00\x80")
    store = InMemoryVectorStore(CountingEmbedding(size=8))
    with caplog.at_level(logging.INFO):
        status = run(make(tmp_path), store)
    assert (status.state, status.document_count) == ("ready", 1)
    assert "Skipping bad.md" in caplog.text and "Skipping binary.md" in caplog.text
    assert "Loaded document good.md (product AA-1)" in caplog.text
    assert "into 1 chunks" in caplog.text and "s" in caplog.text.split("chunks in")[-1]


@pytest.mark.parametrize("kind", ["missing", "empty"])
def test_no_documents_sets_failed(tmp_path: Path, kind: str) -> None:
    if kind == "empty":
        (tmp_path / "docs").mkdir()
    status = run(make(tmp_path), InMemoryVectorStore(CountingEmbedding(size=8)))
    assert status.state == "failed"
    assert status.error and "No usable documents" in status.error


def test_embedding_error_sets_failed(tmp_path: Path) -> None:
    class Boom(DeterministicFakeEmbedding):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("embedding service down")

    write_doc(tmp_path / "docs", "a.md", "AA-1", {"One": "first"})
    status = run(make(tmp_path), InMemoryVectorStore(Boom(size=8)))
    assert status.state == "failed"
    assert status.error and "embedding service down" in status.error


# --- lifespan, state and guard (AC1, AC2, AC3, AC10) ---


def wait_for(client: TestClient, states: tuple[str, ...]) -> IndexStatus:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        status: IndexStatus = client.app.state.index_status  # type: ignore[attr-defined]
        if status.state in states:
            return status
        time.sleep(0.02)
    raise AssertionError("indexing did not finish")


def guarded_app(settings: Settings) -> FastAPI:
    app = create_app(
        settings,
        chat_model=object(),  # type: ignore[arg-type]
        embeddings=CountingEmbedding(size=8),
        vector_store=InMemoryVectorStore(CountingEmbedding(size=8)),
    )

    @app.get("/guarded")
    async def guarded(
        status: Annotated[IndexStatus, Depends(require_index_ready)],
    ) -> dict[str, int]:
        return {"chunks": status.chunk_count}

    return app


def test_lifespan_indexes_data_dir_and_guard_opens(tmp_path: Path) -> None:
    write_doc(tmp_path / "docs", "a.md", "AA-1", {"One": "first", "Two": "second"})
    with TestClient(guarded_app(make(tmp_path))) as client:
        status = wait_for(client, ("ready", "failed"))
        assert status.state == "ready"
        assert (status.document_count, status.chunk_count) == (1, 2)
        assert status.embedding_model == "ollama:nomic-embed-text"
        assert status.error is None and status.timestamp
        assert client.get("/guarded").json() == {"chunks": 2}


def test_guard_returns_503_while_indexing(tmp_path: Path) -> None:
    client = TestClient(guarded_app(make(tmp_path)))  # lifespan not started: still indexing
    assert client.app.state.index_status.state == "indexing"  # type: ignore[attr-defined]
    res = client.get("/guarded", headers={"x-request-id": "rid-1"})
    assert res.status_code == 503
    assert res.json() == {
        "detail": "The document index is still being built; retry shortly.",
        "code": "INDEX_NOT_READY",
        "request_id": "rid-1",
    }


@pytest.mark.parametrize("kind", ["missing", "empty"])
def test_failed_index_app_starts_and_guard_returns_503(tmp_path: Path, kind: str) -> None:
    if kind == "empty":
        (tmp_path / "docs").mkdir()
    with TestClient(guarded_app(make(tmp_path))) as client:
        assert client.get("/").status_code == 200
        status = wait_for(client, ("ready", "failed"))
        assert status.state == "failed" and status.error
        res = client.get("/guarded")
        assert res.status_code == 503
        body = res.json()
        assert body["code"] == "INDEX_NOT_READY"
        assert status.error in body["detail"]
        assert body["request_id"]


def test_index_not_ready_error_shape() -> None:
    err = IndexNotReadyError("x")
    assert isinstance(err, ServiceError)
    assert (err.code, err.status_code) == ("INDEX_NOT_READY", 503)


# --- CLI (AC11) ---


def test_ingest_cli_success_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write_doc(tmp_path / "docs", "a.md", "AA-1", {"One": "first"})
    settings = make(tmp_path)
    store = InMemoryVectorStore(CountingEmbedding(size=8))
    monkeypatch.setattr("app.ingest.get_settings", lambda: settings)
    monkeypatch.setattr("app.ingest.create_embeddings", lambda s: CountingEmbedding(size=8))
    monkeypatch.setattr("app.ingest.create_vector_store", lambda s, e: store)
    assert ingest_main() == 0
    assert "1 documents, 1 chunks" in capsys.readouterr().out

    (tmp_path / "docs" / "a.md").unlink()
    assert ingest_main() == 1
    assert "No usable documents" in capsys.readouterr().err
