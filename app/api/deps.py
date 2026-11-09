"""FastAPI dependencies that read shared objects from `app.state`."""

from fastapi import Request

from app.core.errors import IndexNotReadyError
from app.memory.store import InMemorySessionStore
from app.rag.index_status import IndexStatus


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


def get_store(request: Request) -> InMemorySessionStore:
    """Return the session store created at startup (override in tests)."""
    store: InMemorySessionStore = request.app.state.session_store
    return store
