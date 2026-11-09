"""Request and response models for the Q&A API."""

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

SessionKey = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_-]+$"
    ),
]

_SESSION_KEY_DESCRIPTION = (
    "Optional client-chosen key (8-64 characters: letters, digits, '-' and '_') that links "
    "follow-up questions. Without it no history is kept. History is stored in memory only "
    "and is lost when the server restarts; the key is not authentication."
)


class AskRequest(BaseModel):
    """A question about the product manuals."""

    question: str = Field(
        min_length=1,
        description="Natural-language question.",
        examples=["How do I descale the kettle?"],
    )
    session_key: SessionKey | None = Field(
        default=None, description=_SESSION_KEY_DESCRIPTION, examples=["demo-session-01"]
    )


class AskResponse(BaseModel):
    """Answer to a question."""

    answer: str = Field(description="Generated answer or clarifying question.")
    session_key: str | None = Field(
        default=None, description="The session key from the request, or null when absent."
    )
