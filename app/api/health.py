"""Liveness and readiness endpoints (root level, outside `/api/v1`)."""

import asyncio
import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse
from langchain_core.language_models import BaseChatModel

from app.api.deps import get_chat_model, get_index_status
from app.api.schemas import HealthResponse, LLMCheck, ReadinessResponse
from app.rag.index_status import IndexStatus

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

_TS = "2025-01-15T09:30:00Z"


def _body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "state": "ready",
        "document_count": 6,
        "chunk_count": 42,
        "embedding_model": "nomic-embed-text",
        "last_indexed_at": _TS,
        "error": None,
    }
    body.update(overrides)
    return body


_EMPTY = {"document_count": 0, "chunk_count": 0}

_READINESS_RESPONSES: dict[int | str, dict[str, Any]] = {
    200: {
        "description": "The index is built and contains at least one chunk.",
        "content": {
            "application/json": {
                "examples": {
                    "ready": {"summary": "Ready", "value": _body()},
                    "ready_with_llm_check": {
                        "summary": "Ready, with check_llm=true",
                        "value": _body(llm={"status": "ok"}),
                    },
                }
            }
        },
    },
    503: {
        "model": ReadinessResponse,
        "description": (
            "Not ready: indexing is running, indexing failed, or the index has no chunks. "
            "The body has the same shape as the 200 body (not the error body)."
        ),
        "content": {
            "application/json": {
                "examples": {
                    "indexing": {
                        "summary": "Still indexing",
                        "value": _body(state="indexing", **_EMPTY),
                    },
                    "failed": {
                        "summary": "Indexing failed",
                        "value": _body(
                            state="failed", error="Embedding provider unreachable.", **_EMPTY
                        ),
                    },
                    "empty_index": {
                        "summary": "Ready but no chunks",
                        "value": _body(**_EMPTY),
                    },
                    "llm_unreachable": {
                        "summary": "Indexing, with check_llm=true and an unreachable model",
                        "value": _body(state="indexing", llm={"status": "unreachable"}, **_EMPTY),
                    },
                }
            }
        },
    },
}


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness",
    responses={200: {"content": {"application/json": {"example": {"status": "ok"}}}}},
)
async def health() -> HealthResponse:
    """Return 200 whenever the process is serving, whatever the index state.

    It does not touch the index, the model or any provider.
    """
    return HealthResponse(status="ok")


async def _check_llm(chat: BaseChatModel, timeout: float) -> LLMCheck:
    try:
        await asyncio.wait_for(chat.ainvoke("ping"), timeout=timeout)
    except TimeoutError:
        logger.warning("LLM health check timed out after %.1fs", timeout)
        return LLMCheck(status="timeout")
    except Exception as exc:
        logger.warning("LLM health check failed (%s)", type(exc).__name__)
        return LLMCheck(status="unreachable")
    return LLMCheck(status="ok")


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Readiness (indexing status)",
    responses=_READINESS_RESPONSES,
)
async def ready(
    request: Request,
    status: Annotated[IndexStatus, Depends(get_index_status)],
    check_llm: Annotated[
        bool,
        Query(
            description=(
                "Also send one minimal request to the chat model, bounded by the "
                "HEALTH_LLM_TIMEOUT_SECONDS setting. Informational: the result is reported in "
                "`llm` and never changes the status code. Off by default (it costs a few tokens)."
            )
        ),
    ] = False,
) -> JSONResponse:
    """Report the startup indexing state: 200 only when `ready` with at least one chunk."""
    llm: LLMCheck | None = None
    if check_llm:
        timeout: float = request.app.state.settings.health_llm_timeout_seconds
        llm = await _check_llm(get_chat_model(request), timeout)
    body = ReadinessResponse(
        state=status.state,
        document_count=status.document_count,
        chunk_count=status.chunk_count,
        embedding_model=status.embedding_model,
        last_indexed_at=status.timestamp,
        error=status.error,
        llm=llm,
    )
    is_ready = status.state == "ready" and status.chunk_count >= 1
    content = body.model_dump(mode="json", exclude=None if llm else {"llm"})
    return JSONResponse(status_code=200 if is_ready else 503, content=content)
