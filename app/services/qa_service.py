"""Application logic for questions: calls the compiled graph and maps its result."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.core.errors import LLMEmptyResponseError, ProductNotFoundError
from app.rag.index_status import IndexStatus
from app.rag.state import GraphState, Source

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QAResult:
    """What the API needs from one question."""

    answer: str
    status: str
    sources: list[Source] = field(default_factory=list)
    session_key: str | None = None


class QAServiceProtocol(Protocol):
    """The interface routes depend on; tests inject fakes that satisfy it."""

    async def ask(
        self,
        question: str,
        *,
        session_key: str | None,
        top_k: int | None,
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
    ) -> None:
        self._graph = graph
        self._index_status = index_status
        self._max_log = log_question_max_chars

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
        top_k: int | None,
        product_id: str | None,
        request_id: str,
    ) -> QAResult:
        resolved = self._resolve_product(product_id) if product_id else None
        state: GraphState = {
            "question": question,
            "session_key": session_key,
            "top_k": top_k,
            "product_id": resolved,
        }
        out = await self._graph.ainvoke(state)
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
        if status != "answered":
            logger.info("ask request_id=%s ended with %s (no sources)", request_id, status)
        for s in sources:
            logger.info(
                "ask request_id=%s source product_id=%s section=%r chunk_id=%s score=%.3f",
                request_id,
                s["product_id"],
                s["section"],
                s["chunk_id"],
                s["score"],
            )
        return QAResult(answer=answer, status=status, sources=sources, session_key=session_key)
