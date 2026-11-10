"""Limit on concurrent model calls (chat and query embedding) shared by all requests."""

import asyncio
import math
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from app.core.config import Settings
from app.core.errors import ServerBusyError


class ModelCallLimiter:
    """An `asyncio.Semaphore` plus a wait limit. Created once at startup, shared read-only.

    A caller that cannot get a slot within `wait_seconds` gets ServerBusyError (HTTP 503) at
    once instead of queueing behind slow model calls. The slot is released on success, on error
    and on cancellation, so a failed or timed-out call never leaks a slot.
    """

    def __init__(self, max_concurrent: int, wait_seconds: float) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")
        if wait_seconds <= 0:
            raise ValueError("wait_seconds must be greater than 0")
        self.max_concurrent = max_concurrent
        self.wait_seconds = wait_seconds
        self._semaphore = asyncio.Semaphore(max_concurrent)

    @classmethod
    def from_settings(cls, settings: Settings) -> "ModelCallLimiter":
        return cls(settings.max_concurrent_requests, settings.concurrency_wait_seconds)

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Hold one model slot for the duration of the block."""
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=self.wait_seconds)
        except TimeoutError:
            raise ServerBusyError(
                "The server is busy: no model slot became free in time; retry shortly.",
                retry_after=max(1, math.ceil(self.wait_seconds)),
            ) from None
        try:
            yield
        finally:
            self._semaphore.release()
