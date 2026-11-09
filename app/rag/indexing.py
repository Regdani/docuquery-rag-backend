"""The shared indexing pipeline used by the lifespan and by `python -m app.ingest`."""

import hashlib
import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path

from langchain_core.vectorstores import VectorStore

from app.core.config import Settings
from app.core.errors import ServiceError
from app.rag.chunking import chunk_documents
from app.rag.index_status import IndexStatus
from app.rag.loader import ManualDocument, load_documents

logger = logging.getLogger(__name__)


def compute_fingerprint(
    docs: Sequence[ManualDocument], embedding_model: str, chunk_size: int, chunk_overlap: int
) -> str:
    """SHA-256 over sorted file names and contents, embedding model and chunk parameters."""
    digest = hashlib.sha256()
    digest.update(json.dumps([embedding_model, chunk_size, chunk_overlap]).encode())
    for doc in sorted(docs, key=lambda d: d.source):
        digest.update(b"\0" + doc.source.encode() + b"\0" + doc.content.encode())
    return digest.hexdigest()


def _state_path(settings: Settings, collection: str) -> Path:
    return Path(settings.index_dir) / f"{collection}.state.json"


def _read_state(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_state(path: Path, fingerprint: str, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fingerprint": fingerprint, "ids": ids}), encoding="utf-8")


def _failed(model: str, message: str) -> IndexStatus:
    logger.error("Indexing failed: %s", message)
    return IndexStatus(state="failed", embedding_model=model, error=message)


def run_indexing(
    settings: Settings, vector_store: VectorStore, embedding_model: str, collection: str
) -> IndexStatus:
    """Load, chunk, embed and persist the manuals. Never raises: failures become `failed`.

    The persisted index is reused (no embedding calls) when the fingerprint matches and all
    chunks are still present; otherwise the previously stored chunks are deleted first so no
    stale chunks or duplicates remain.
    """
    started = time.perf_counter()
    try:
        docs = load_documents(settings.data_dir)
        if not docs:
            return _failed(
                embedding_model,
                f"No usable documents found in data directory '{settings.data_dir}'",
            )
        chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            return _failed(embedding_model, "The documents contain no indexable sections")
        ids = [str(c.metadata["chunk_id"]) for c in chunks]
        fingerprint = compute_fingerprint(
            docs, embedding_model, settings.chunk_size, settings.chunk_overlap
        )
        state_path = _state_path(settings, collection)
        state = _read_state(state_path)

        reused = state.get("fingerprint") == fingerprint and len(
            vector_store.get_by_ids(ids)
        ) == len(ids)
        if reused:
            logger.info("Index unchanged, reusing %d persisted chunks", len(chunks))
        else:
            old_ids = state.get("ids")
            if isinstance(old_ids, list) and old_ids:
                vector_store.delete(ids=[str(i) for i in old_ids])
            vector_store.delete(ids=ids)
            vector_store.add_documents(chunks, ids=ids)
            _write_state(state_path, fingerprint, ids)
    except Exception as exc:  # noqa: BLE001 - any failure must end up as a failed status
        if isinstance(exc, ServiceError):
            return _failed(embedding_model, exc.message)
        return _failed(embedding_model, f"{type(exc).__name__}: {exc}")

    logger.info(
        "Indexed %d documents (%s) into %d chunks in %.2fs",
        len(docs),
        ", ".join(d.product_id for d in docs),
        len(chunks),
        time.perf_counter() - started,
    )
    return IndexStatus(
        state="ready",
        document_count=len(docs),
        chunk_count=len(chunks),
        embedding_model=embedding_model,
        reused=reused,
        product_ids=sorted({d.product_id for d in docs}),
    )
