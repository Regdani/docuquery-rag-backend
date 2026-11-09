"""Application errors. A single handler (added later) maps these to the unified error body."""


class ServiceError(Exception):
    """Base class for errors that carry a machine-readable code and an HTTP status."""

    def __init__(self, message: str, code: str = "internal_error", status_code: int = 500) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


class ProviderConfigError(ServiceError):
    """Raised at startup when a provider is unknown, incomplete or not installed."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="provider_config_error", status_code=500)
