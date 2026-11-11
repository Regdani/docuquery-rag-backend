"""The single typed state of the question graph; a fresh one is created for every request."""

from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class Hit(TypedDict):
    """One retrieved chunk with its relevance score (0..1, higher is better)."""

    document: Document
    score: float


class Source(TypedDict):
    """A chunk that was used for the answer, taken from the chunk metadata."""

    product_id: str
    product_name: str
    section: str
    source: str
    chunk_id: str
    score: float
    snippet: str


class GraphState(TypedDict, total=False):
    """Graph state. Only `question` (and optionally `session_key`) is supplied by the caller."""

    session_key: str | None
    product_id: str | None  # restrict retrieval to this product
    messages: list[BaseMessage]  # prior history for the session
    question: str  # original question as asked
    standalone_question: str
    hits: list[Hit]  # retrieved chunks at or above the relevance threshold
    answer: str
    sources: list[Source]
    status: str  # answered | clarification_needed | no_answer
