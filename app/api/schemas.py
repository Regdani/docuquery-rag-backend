"""Request and response models for the Q&A API."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, Field, StringConstraints

SessionKey = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_-]+$"
    ),
]

TOP_K_MIN = 1
TOP_K_MAX = 20


def _blank_to_none(value: Any) -> Any:
    if isinstance(value, str) and not value.strip():
        return None
    return value


OptionalText = BeforeValidator(_blank_to_none)

_SESSION_KEY_DESCRIPTION = (
    "Optional client-chosen key (8-64 characters: letters, digits, '-' and '_') that links "
    "follow-up questions. Without it no history is kept. History is stored in memory only "
    "and is lost when the server restarts; the key is not authentication. "
    "An empty string counts as absent."
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
    top_k: int | None = Field(
        default=None,
        ge=TOP_K_MIN,
        le=TOP_K_MAX,
        description=(
            f"Number of chunks to retrieve for this request only ({TOP_K_MIN}-{TOP_K_MAX}). "
            "Defaults to the server setting."
        ),
        examples=[4],
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
    session_key: str | None = Field(
        default=None,
        description="The session key from the request, or null when absent.",
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
