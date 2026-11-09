"""Error translation and bounded 429 retry around cloud chat models and embeddings.

Ollama gets the same wrappers with its own translator (clear messages, no retries).

Vendor exception types are imported lazily here (and nowhere else). Provider failures are
mapped to LLMTimeoutError / LLMUnavailableError / LLMRateLimitedError; anything unrecognised
(programming errors, bad input) is re-raised untouched. Domain error messages only carry the
provider name and the exception class name, never the vendor message, so secrets cannot leak.

Retry note: the vendor clients keep their own `max_retries`; this layer adds a bounded retry of
HTTP 429 with exponential backoff on top of it and then raises LLMRateLimitedError.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, TypeVar

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult

from app.core.errors import (
    LLMRateLimitedError,
    LLMTimeoutError,
    LLMUnavailableError,
    ServiceError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _vendor_types(module: str, names: Sequence[str]) -> tuple[type[BaseException], ...]:
    """Lazily resolve exception classes; a missing optional package yields nothing."""
    try:
        mod = __import__(module, fromlist=list(names))
    except ImportError:
        return ()
    return tuple(getattr(mod, n) for n in names if hasattr(mod, n))


def _retry_after(exc: BaseException) -> float | None:
    """Seconds from the provider's Retry-After header (numeric form only), if present."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    try:
        value = headers.get("retry-after") if headers is not None else None
        seconds = float(value) if value is not None else None
    except (TypeError, ValueError, AttributeError):
        return None
    return seconds if seconds is not None and seconds >= 0 else None


def _from_status(
    provider: str, status: int, name: str, exc: BaseException | None = None
) -> ServiceError | None:
    if status == 429:
        retry_after = _retry_after(exc) if exc is not None else None
        return LLMRateLimitedError(f"{provider} rate limit exceeded ({name})", retry_after)
    if status in (408, 504):
        return LLMTimeoutError(f"{provider} request timed out ({name})")
    if status >= 500:
        return LLMUnavailableError(f"{provider} service error {status} ({name})")
    return None


def translate_error(provider: str, exc: BaseException) -> ServiceError | None:
    """Map a vendor/transport exception to a domain error, or None if it is not an LLM failure."""
    if isinstance(exc, ServiceError):
        return exc
    name = type(exc).__name__

    # OpenAI / Azure OpenAI (APITimeoutError subclasses APIConnectionError: check it first)
    timeouts = _vendor_types("openai", ["APITimeoutError"])
    if timeouts and isinstance(exc, timeouts):
        return LLMTimeoutError(f"{provider} request timed out ({name})")
    conn = _vendor_types("openai", ["APIConnectionError"])
    if conn and isinstance(exc, conn):
        return LLMUnavailableError(f"{provider} is unreachable ({name})")
    status_types = _vendor_types("openai", ["APIStatusError"]) + _vendor_types(
        "google.genai.errors", ["APIError"]
    )
    if status_types and isinstance(exc, status_types):
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        if isinstance(status, int):
            return _from_status(provider, status, name, exc)
        return None

    # Transport level (Gemini uses httpx; also builtin timeouts and connection errors)
    http_timeout = _vendor_types("httpx", ["TimeoutException"])
    if isinstance(exc, TimeoutError) or (http_timeout and isinstance(exc, http_timeout)):
        return LLMTimeoutError(f"{provider} request timed out ({name})")
    http_transport = _vendor_types("httpx", ["TransportError"])
    if isinstance(exc, ConnectionError) or (http_transport and isinstance(exc, http_transport)):
        return LLMUnavailableError(f"{provider} is unreachable ({name})")
    return None


Translator = Callable[[BaseException], ServiceError | None]


def ollama_translator(base_url: str, model: str) -> Translator:
    """Error translation for a local Ollama server (no keys, no 429s).

    Messages name the base URL and the model and say how to fix it; the vendor message is never
    included.
    """

    def translate(exc: BaseException) -> ServiceError | None:
        if isinstance(exc, ServiceError):
            return exc
        name = type(exc).__name__
        http_timeout = _vendor_types("httpx", ["TimeoutException"])
        if isinstance(exc, TimeoutError) or (http_timeout and isinstance(exc, http_timeout)):
            return LLMTimeoutError(
                f"Ollama at {base_url} did not answer in time (model '{model}', {name})"
            )
        http_transport = _vendor_types("httpx", ["TransportError"])
        if isinstance(exc, ConnectionError) or (http_transport and isinstance(exc, http_transport)):
            return LLMUnavailableError(
                f"Ollama is not reachable at {base_url}. Start it with 'ollama serve' "
                f"or check OLLAMA_BASE_URL ({name})"
            )
        response_errors = _vendor_types("ollama", ["ResponseError"])
        if response_errors and isinstance(exc, response_errors):
            status = getattr(exc, "status_code", -1)
            if status == 404:
                return LLMUnavailableError(
                    f"Ollama model '{model}' was not found at {base_url}. Run 'ollama pull {model}'"
                )
            if status == 408 or status == 504:
                return LLMTimeoutError(
                    f"Ollama at {base_url} did not answer in time (model '{model}', {name})"
                )
            if isinstance(status, int) and status >= 500:
                return LLMUnavailableError(
                    f"Ollama at {base_url} failed with status {status} (model '{model}')"
                )
        return None

    return translate


class RetryPolicy:
    """Bounded retry of rate-limited calls: at most `max_retries` retries, exponential backoff."""

    def __init__(
        self,
        provider: str,
        max_retries: int,
        backoff_seconds: float,
        translator: Translator | None = None,
    ) -> None:
        self.provider = provider
        self._translator = translator
        self.max_retries = max(0, max_retries)
        self.backoff_seconds = max(0.0, backoff_seconds)

    def delay(self, attempt: int) -> float:
        return float(self.backoff_seconds * (2**attempt))

    def _handle(self, exc: Exception, attempt: int) -> float:
        """Return the delay before the next attempt, or raise the final error."""
        domain = self._translator(exc) if self._translator else translate_error(self.provider, exc)
        if domain is None:
            raise exc
        if isinstance(domain, LLMRateLimitedError) and attempt < self.max_retries:
            delay = self.delay(attempt)
            logger.warning(
                "%s rate limited, retry %d/%d in %.2fs",
                self.provider,
                attempt + 1,
                self.max_retries,
                delay,
            )
            return delay
        if domain is exc:
            raise exc
        raise domain from exc

    def call(self, fn: Callable[[], T]) -> T:
        attempt = 0
        while True:
            try:
                return fn()
            except Exception as exc:
                delay = self._handle(exc, attempt)
            time.sleep(delay)
            attempt += 1

    async def acall(self, fn: Callable[[], Awaitable[T]]) -> T:
        attempt = 0
        while True:
            try:
                return await fn()
            except Exception as exc:
                delay = self._handle(exc, attempt)
            await asyncio.sleep(delay)
            attempt += 1


class ResilientChatModel(BaseChatModel):
    """Delegates to `inner`, translating provider failures and retrying HTTP 429."""

    inner: BaseChatModel
    provider: str
    rate_limit_retries: int = 0
    backoff_seconds: float = 0.0
    translator: Translator | None = None

    @property
    def _llm_type(self) -> str:
        return f"resilient-{self.inner._llm_type}"

    def _policy(self) -> RetryPolicy:
        return RetryPolicy(
            self.provider, self.rate_limit_retries, self.backoff_seconds, self.translator
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._policy().call(
            lambda: self.inner._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return await self._policy().acall(
            lambda: self.inner._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        )


class ResilientEmbeddings(Embeddings):
    """Delegates to `inner`, translating provider failures and retrying HTTP 429."""

    def __init__(self, inner: Embeddings, policy: RetryPolicy) -> None:
        self.inner = inner
        self._policy = policy

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._policy.call(lambda: self.inner.embed_documents(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._policy.call(lambda: self.inner.embed_query(text))

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._policy.acall(lambda: self.inner.aembed_documents(texts))

    async def aembed_query(self, text: str) -> list[float]:
        return await self._policy.acall(lambda: self.inner.aembed_query(text))


def unwrap(client: object) -> object:
    """The underlying vendor client of a resilient wrapper (or the object itself)."""
    if isinstance(client, ResilientChatModel | ResilientEmbeddings):
        return client.inner
    return client
