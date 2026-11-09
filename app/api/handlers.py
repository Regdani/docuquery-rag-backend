"""Exception handler mapping ServiceError to the unified error body."""

import logging
import uuid
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from app.core.errors import ServiceError

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    """Register the request-id middleware and the ServiceError handler.

    Error body: `{"detail", "code", "request_id"}`.
    """

    @app.middleware("http")
    async def request_id_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(ServiceError)
    async def service_error_handler(request: Request, exc: ServiceError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None) or uuid.uuid4().hex
        if exc.status_code >= 500:
            logger.error("%s (%s) request_id=%s", exc.message, exc.code, request_id, exc_info=exc)
        else:
            logger.warning("%s (%s) request_id=%s", exc.message, exc.code, request_id)
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message, "code": exc.code, "request_id": request_id},
        )
