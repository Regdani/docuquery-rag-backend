"""Exception handler mapping ServiceError to the unified error body."""

import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.errors import ServiceError

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    """Register the ServiceError handler: `{"detail", "code", "request_id"}`."""

    @app.exception_handler(ServiceError)
    async def service_error_handler(request: Request, exc: ServiceError) -> JSONResponse:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        if exc.status_code >= 500:
            logger.error("%s (%s) request_id=%s", exc.message, exc.code, request_id, exc_info=exc)
        else:
            logger.warning("%s (%s) request_id=%s", exc.message, exc.code, request_id)
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message, "code": exc.code, "request_id": request_id},
        )
