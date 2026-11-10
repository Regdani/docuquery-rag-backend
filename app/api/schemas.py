"""Request and response models for the Q&A API."""

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, Field, StringConstraints

SessionKey = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_-]+$"
    ),
]


def _blank_to_none(value: Any) -> Any:
    if isinstance(value, str) and not value.strip():
        return None
    return value


OptionalText = BeforeValidator(_blank_to_none)

_SESSION_KEY_DESCRIPTION = (
    "Optional client-chosen key (8-64 characters: letters, digits, '-' and '_') that links "
    "follow-up questions. Without it the server starts a new conversation and returns a "
    "generated key in the response; send that key with the next question to continue. "
    "History is stored in memory only and is lost when the server restarts; the key is not "
    "authentication. An empty string counts as absent."
)

AnswerStatus = Literal["answered", "clarification_needed", "no_answer"]


class AskRequest(BaseModel):
    """A question about the product manuals."""

    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)] = Field(
        description=(
            "Natural-language question (whitespace is trimmed; blank is invalid). "
            "Longer than the server's MAX_QUESTION_LENGTH (default 1000 characters) is invalid."
        ),
        examples=["How do I descale the KV-200 coffee maker?"],
    )
    session_key: Annotated[SessionKey | None, OptionalText] = Field(
        default=None, description=_SESSION_KEY_DESCRIPTION, examples=["demo-session-01"]
    )
    product_id: Annotated[str | None, OptionalText] = Field(
        default=None,
        description=(
            "Optional product ID (for example KV-200); restricts the search to that product's "
            "manual. An ID that is not in the index returns 404 PRODUCT_NOT_FOUND. "
            "An empty string counts as absent."
        ),
        examples=["KV-200"],
    )


class Source(BaseModel):
    """A manual chunk that was used for the answer."""

    product_id: str = Field(description="Product ID of the manual.", examples=["KV-200"])
    product_name: str = Field(description="Product name.", examples=["Espresso Coffee Maker"])
    section: str = Field(
        description="Manual section the chunk comes from.", examples=["Maintenance"]
    )
    chunk_id: str = Field(
        description="Stable ID of the chunk.", examples=["kv-200-espresso-coffee-maker.md::3::0"]
    )
    score: float = Field(
        description="Relevance score from 0 to 1 (higher is better).", examples=[0.82]
    )
    snippet: str = Field(
        description="Short excerpt of the chunk text (length capped).",
        examples=["Descale the machine every 60 days or after 200 brew cycles."],
    )


class AskResponse(BaseModel):
    """Answer to a question."""

    answer: str = Field(
        description="Generated answer, clarifying question or fixed no-information reply.",
        examples=["Descale the KV-200 every 60 days or after 200 brew cycles."],
    )
    status: AnswerStatus = Field(
        description=(
            "answered: grounded answer with sources; clarification_needed: the question is "
            "unclear and the answer is a clarifying question; no_answer: nothing relevant found."
        ),
        examples=["answered"],
    )
    sources: list[Source] = Field(
        description="Chunks used for the answer; empty unless status is answered."
    )
    session_key: str = Field(
        description=(
            "The session key from the request, or a newly generated one when the request had "
            "none. Send it with the next question to continue the conversation."
        ),
        examples=["demo-session-01"],
    )
    request_id: str = Field(
        description=(
            "Request ID (the X-Request-ID header if sent, otherwise generated); also in the logs."
        ),
        examples=["3f2b8c1e9d7a4e6f8a1b2c3d4e5f6a7b"],
    )


class ValidationIssue(BaseModel):
    """One problem found in the request (the submitted value is never echoed)."""

    field: str = Field(description="Name of the offending field.", examples=["question"])
    message: str = Field(
        description="What is wrong.", examples=["String should have at least 1 character"]
    )
    type: str = Field(description="Validation error type.", examples=["string_too_short"])


class ErrorResponse(BaseModel):
    """Unified error body."""

    detail: str = Field(
        description="Human-readable message.",
        examples=["The document index is still being built; retry shortly."],
    )
    code: str = Field(description="Machine-readable error code.", examples=["INDEX_NOT_READY"])
    request_id: str = Field(
        description="Request ID for log correlation.", examples=["3f2b8c1e9d7a4e6f"]
    )
    errors: list[ValidationIssue] | None = Field(
        default=None, description="Validation problems (422 only)."
    )


class HealthResponse(BaseModel):
    """Liveness body."""

    status: Literal["ok"] = Field(description="Always `ok` while the process is serving.")


class LLMCheck(BaseModel):
    """Result of the optional chat-model reachability check."""

    status: Literal["ok", "unreachable", "timeout"] = Field(
        description="`ok`, `unreachable` (the call failed) or `timeout` (no answer in time)."
    )


class ReadinessResponse(BaseModel):
    """Readiness body: the startup indexing state, plus the optional LLM check."""

    state: Literal["indexing", "ready", "failed"] = Field(description="Startup indexing state.")
    document_count: int = Field(description="Documents indexed.")
    chunk_count: int = Field(description="Chunks in the index.")
    embedding_model: str = Field(description="Embedding model used for the index.")
    last_indexed_at: datetime = Field(description="Time of the last indexing status change (UTC).")
    error: str | None = Field(default=None, description="Indexing error; null unless failed.")
    llm: LLMCheck | None = Field(
        default=None,
        description="Present only with `check_llm=true`; informational, never changes the status.",
    )
