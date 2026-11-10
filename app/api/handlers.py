"""Exception handlers mapping every error to the unified error body.

Body: `{"detail", "code", "request_id"}` (422 adds `errors`). 4xx is logged at WARNING, 5xx at
ERROR with the stack trace; the response never contains a trace or exception text.
"""

import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.errors import LLMRateLimitedError, ServiceError

logger = logging.getLogger(__name__)

INTERNAL_MESSAGE = "An unexpected error occurred."


def _request_id(request: Request) -> str:
    request_id: str | None = getattr(request.state, "request_id", None)
    if not request_id:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
    return request_id


def _respond(
    request: Request,
    status_code: int,
    detail: str,
    code: str,
    headers: dict[str, str] | None = None,
    errors: list[dict[str, str]] | None = None,
) -> JSONResponse:
    request_id = _request_id(request)
    content: dict[str, Any] = {"detail": detail, "code": code, "request_id": request_id}
    if errors is not None:
        content["errors"] = errors
    return JSONResponse(
        status_code=status_code,
        content=content,
        headers={**(headers or {}), "X-Request-ID": request_id},
    )


def _log(
    status_code: int, message: str, code: str, request_id: str, exc: BaseException | None = None
) -> None:
    if status_code >= 500:
        logger.error("%s (%s) request_id=%s", message, code, request_id, exc_info=exc)
    else:
        logger.warning("%s (%s) request_id=%s", message, code, request_id)


def _validation_issues(exc: RequestValidationError) -> list[dict[str, str]]:
    """Field, message and type only; the submitted values are never copied."""
    issues: list[dict[str, str]] = []
    for err in exc.errors():
        parts = [str(p) for p in err.get("loc", ()) if p != "body"]
        issues.append(
            {
                "field": ".".join(parts) or "body",
                "message": str(err.get("msg", "Invalid value")),
                "type": str(err.get("type", "value_error")),
            }
        )
    return issues


def register_exception_handlers(app: FastAPI) -> None:
    """Register the request-id middleware and one handler per exception type."""

    @app.middleware("http")
    async def request_id_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request.state.request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        path = request.url.path
        # Health probes run every few seconds; keep them out of the INFO log.
        level = logging.DEBUG if path.startswith("/health") else logging.INFO
        logger.log(
            level,
            "request method=%s path=%s status=%d duration_ms=%.1f request_id=%s",
            request.method,
            path,
            response.status_code,
            (time.perf_counter() - started) * 1000,
            request.state.request_id,
        )
        return response

    @app.exception_handler(ServiceError)
    async def service_error_handler(request: Request, exc: ServiceError) -> JSONResponse:
        headers = dict(exc.headers or {})
        if isinstance(exc, LLMRateLimitedError) and "Retry-After" not in headers:
            default = request.app.state.settings.rate_limit_retry_after_seconds
            headers["Retry-After"] = str(default)
        _log(exc.status_code, exc.message, exc.code, _request_id(request), exc)
        return _respond(request, exc.status_code, exc.message, exc.code, headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        issues = _validation_issues(exc)
        fields = ", ".join(sorted({i["field"] for i in issues}))
        detail = "Invalid request: " + "; ".join(f"{i['field']}: {i['message']}" for i in issues)
        logger.warning(
            "Invalid request (VALIDATION_ERROR) fields=[%s] request_id=%s",
            fields,
            _request_id(request),
        )
        return _respond(request, 422, detail, "VALIDATION_ERROR", errors=issues)

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        try:
            code = HTTPStatus(exc.status_code).name
        except ValueError:
            code = "HTTP_ERROR"
        detail = str(exc.detail)
        _log(exc.status_code, detail, code, _request_id(request), exc)
        return _respond(request, exc.status_code, detail, code, dict(exc.headers or {}))

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        _log(500, INTERNAL_MESSAGE, "INTERNAL_ERROR", _request_id(request), exc)
        return _respond(request, 500, INTERNAL_MESSAGE, "INTERNAL_ERROR")
