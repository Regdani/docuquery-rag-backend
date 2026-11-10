"""Application errors. One handler maps these to the unified error body."""

import math


class ServiceError(Exception):
    """Base class for errors that carry a machine-readable code, an HTTP status and headers."""

    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_ERROR",
        status_code: int = 500,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.headers = headers


class IndexNotReadyError(ServiceError):
    """Raised while the document index is still being built or failed to build."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="INDEX_NOT_READY", status_code=503)


class ProductNotFoundError(ServiceError):
    """Raised when a question is scoped to a product ID that is not in the index."""

    def __init__(self, product_id: str) -> None:
        super().__init__(
            f"Unknown product_id '{product_id}'; it is not in the document index.",
            code="PRODUCT_NOT_FOUND",
            status_code=404,
        )


class ProviderConfigError(ServiceError):
    """Raised at startup when a provider is unknown, incomplete or not installed."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="provider_config_error", status_code=500)


class LLMTimeoutError(ServiceError):
    """Raised when the chat or embedding provider did not answer within the timeout."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="LLM_TIMEOUT", status_code=504)


class LLMUnavailableError(ServiceError):
    """Raised when the provider is unreachable or fails with a server-side (5xx) error."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="LLM_UNAVAILABLE", status_code=502)


class LLMRateLimitedError(ServiceError):
    """Raised when the provider keeps answering HTTP 429 after the bounded retries.

    `retry_after` (seconds, from the provider's Retry-After header) becomes the response header;
    when it is None the API handler uses the configured default.
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        headers = None
        if retry_after is not None:
            headers = {"Retry-After": str(max(1, math.ceil(retry_after)))}
        super().__init__(message, code="LLM_RATE_LIMITED", status_code=503, headers=headers)


class ServerBusyError(ServiceError):
    """Raised when no model slot became free within the wait limit (fast 503 with Retry-After)."""

    def __init__(self, message: str, retry_after: int = 1) -> None:
        super().__init__(
            message,
            code="SERVER_BUSY",
            status_code=503,
            headers={"Retry-After": str(max(1, retry_after))},
        )


class LLMEmptyResponseError(ServiceError):
    """Raised when the model returned an empty answer."""

    def __init__(self, message: str = "The model returned an empty answer.") -> None:
        super().__init__(message, code="LLM_EMPTY_RESPONSE", status_code=502)


class RAGPipelineError(ServiceError):
    """Raised when a question-graph step fails unexpectedly; the message never leaks details."""

    def __init__(self, message: str = "The question could not be processed.") -> None:
        super().__init__(message, code="RAG_PIPELINE_ERROR", status_code=502)
