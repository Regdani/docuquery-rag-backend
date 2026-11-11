"""Application logic for questions: calls the compiled graph and maps its result."""

import asyncio
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.core.errors import LLMEmptyResponseError, LLMTimeoutError, ProductNotFoundError
from app.rag.index_status import IndexStatus
from app.rag.state import GraphState, Source

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QAResult:
    """What the API needs from one question."""

    answer: str
    status: str
    session_key: str
    sources: list[Source] = field(default_factory=list)


def new_session_key() -> str:
    """Random, unguessable key that satisfies the request schema (32 hex characters)."""
    return uuid.uuid4().hex


class QAServiceProtocol(Protocol):
    """The interface routes depend on; tests inject fakes that satisfy it."""

    async def ask(
        self,
        question: str,
        *,
        session_key: str | None,
        product_id: str | None,
        request_id: str,
    ) -> QAResult: ...


class QAService:
    """Runs the question graph (built once at startup) for one request at a time."""

    def __init__(
        self,
        graph: Any,
        index_status: Callable[[], IndexStatus],
        log_question_max_chars: int,
        request_timeout_seconds: float,
    ) -> None:
        self._graph = graph
        self._index_status = index_status
        self._max_log = log_question_max_chars
        self._timeout = request_timeout_seconds

    def _resolve_product(self, product_id: str) -> str:
        for known in self._index_status().product_ids:
            if known.casefold() == product_id.casefold():
                return known
        raise ProductNotFoundError(product_id)

    async def ask(
        self,
        question: str,
        *,
        session_key: str | None,
        product_id: str | None,
        request_id: str,
    ) -> QAResult:
        resolved = self._resolve_product(product_id) if product_id else None
        # No key means a new conversation: generate one so the client can continue it.
        key = session_key or new_session_key()
        state: GraphState = {
            "question": question,
            "session_key": key,
            "product_id": resolved,
        }
        try:
            # One deadline for the whole flow, so per-call timeouts and retries cannot add up.
            async with asyncio.timeout(self._timeout):
                out = await self._graph.ainvoke(state)
        except TimeoutError:
            raise LLMTimeoutError(
                f"The question was not answered within {self._timeout:g} seconds."
            ) from None
        status = str(out["status"])
        answer = str(out["answer"])
        if status == "answered" and not answer.strip():
            logger.error("ask request_id=%s model returned an empty answer", request_id)
            raise LLMEmptyResponseError()
        sources: list[Source] = list(out.get("sources", [])) if status == "answered" else []
        logger.info(
            "ask request_id=%s status=%s question=%r sources=%d",
            request_id,
            status,
            question[: self._max_log],
            len(sources),
        )
        for s in sources:
            logger.info(
                "ask request_id=%s source product_id=%s section=%r chunk_id=%s score=%.3f",
                request_id,
                s["product_id"],
                s["section"],
                s["chunk_id"],
                s["score"],
            )
        return QAResult(answer=answer, status=status, session_key=key, sources=sources)
