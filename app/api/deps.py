"""FastAPI dependencies that read shared objects from `app.state`."""

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from langchain_core.language_models import BaseChatModel

from app.core.errors import IndexNotReadyError
from app.rag.index_status import IndexStatus
from app.rag.limiter import ModelCallLimiter
from app.services.qa_service import QAServiceProtocol


def get_index_status(request: Request) -> IndexStatus:
    """Return the current index status object."""
    status: IndexStatus = request.app.state.index_status
    return status


def require_index_ready(request: Request) -> IndexStatus:
    """Guard: raise IndexNotReadyError (503) until indexing has succeeded."""
    status = get_index_status(request)
    if status.state == "ready":
        return status
    if status.state == "indexing":
        raise IndexNotReadyError("The document index is still being built; retry shortly.")
    raise IndexNotReadyError(f"The document index is unavailable: {status.error}")


def validate_question_length(request: Request, question: str) -> None:
    """Raise a 422 when the question is longer than `Settings.max_question_length`."""
    limit: int = request.app.state.settings.max_question_length
    if len(question) > limit:
        raise RequestValidationError(
            [
                {
                    "type": "string_too_long",
                    "loc": ("body", "question"),
                    "msg": f"String should have at most {limit} characters",
                }
            ]
        )


def get_qa_service(request: Request) -> QAServiceProtocol:
    """Return the QAService created at startup (override in tests)."""
    service: QAServiceProtocol = request.app.state.qa_service
    return service


def get_chat_model(request: Request) -> BaseChatModel:
    """Return the chat model created at startup (override in tests)."""
    model: BaseChatModel = request.app.state.chat_model
    return model


def get_limiter(request: Request) -> ModelCallLimiter:
    """Return the model-call limiter created at startup."""
    limiter: ModelCallLimiter = request.app.state.limiter
    return limiter
