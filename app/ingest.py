"""CLI: `python -m app.ingest` runs the same indexing pipeline as the application startup."""

import sys

from app.core.config import get_settings
from app.core.errors import ServiceError
from app.core.logging import configure_logging
from app.providers.factory import (
    create_embeddings,
    create_vector_store,
    embedding_model_name,
    index_collection_name,
)
from app.rag.indexing import run_indexing


def main() -> int:
    """Build the index; return 0 on success and 1 when indexing failed."""
    settings = get_settings()
    configure_logging(settings.log_level)
    try:
        embeddings = create_embeddings(settings)
        store = create_vector_store(settings, embeddings)
        model_name = embedding_model_name(settings)
        collection = index_collection_name(settings)
    except ServiceError as exc:
        print(f"Indexing failed: {exc.message}", file=sys.stderr)
        return 1
    status = run_indexing(settings, store, model_name, collection)
    if status.state != "ready":
        print(f"Indexing failed: {status.error}", file=sys.stderr)
        return 1
    mode = "reused persisted index" if status.reused else "built index"
    print(
        f"Indexing ready ({mode}): {status.document_count} documents, "
        f"{status.chunk_count} chunks, embedding model {status.embedding_model}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
