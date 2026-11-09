"""Queryable indexing status, kept on `app.state.index_status`."""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

IndexState = Literal["indexing", "ready", "failed"]


class IndexStatus(BaseModel):
    """Snapshot of the indexing pipeline; replaced (never mutated) when the state changes."""

    state: IndexState
    document_count: int = 0
    chunk_count: int = 0
    embedding_model: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    error: str | None = None
    reused: bool = Field(default=False, description="True when the persisted index was reused.")
    product_ids: list[str] = Field(
        default_factory=list, description="Product IDs present in the index."
    )
