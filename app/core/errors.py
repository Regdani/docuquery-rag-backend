"""Application errors. A single handler (added later) maps these to the unified error body."""


class ServiceError(Exception):
    """Base class for errors that carry a machine-readable code and an HTTP status."""

    def __init__(self, message: str, code: str = "internal_error", status_code: int = 500) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


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
        super().__init__(message, code="llm_timeout", status_code=504)


class LLMUnavailableError(ServiceError):
    """Raised when the provider is unreachable or fails with a server-side (5xx) error."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="llm_unavailable", status_code=503)


class LLMRateLimitedError(ServiceError):
    """Raised when the provider keeps answering HTTP 429 after the bounded retries."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="llm_rate_limited", status_code=429)


class RAGPipelineError(ServiceError):
    """Raised when a question-graph step fails unexpectedly; the message never leaks details."""

    def __init__(self, message: str = "The question could not be processed.") -> None:
        super().__init__(message, code="rag_pipeline_error", status_code=500)
