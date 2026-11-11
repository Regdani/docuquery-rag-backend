"""In-memory conversation history keyed by ``session_key``.

A process-local dict guarded by an ``asyncio.Lock``. It is not shared between workers or
instances and is lost on restart (see the README "Conversation history" section).
"""

import asyncio
from collections.abc import Sequence

from langchain_core.messages import BaseMessage


class InMemorySessionStore:
    """Map of session key to chat messages; each session keeps the latest N messages."""

    def __init__(self, max_messages: int) -> None:
        if max_messages < 1:
            raise ValueError("max_messages must be at least 1")
        self._max_messages = max_messages
        self._sessions: dict[str, list[BaseMessage]] = {}
        self._lock = asyncio.Lock()

    @property
    def max_messages(self) -> int:
        return self._max_messages

    async def get_messages(self, key: str) -> list[BaseMessage]:
        """Return a copy of the stored transcript, or an empty list for an unknown key."""
        async with self._lock:
            return list(self._sessions.get(key, []))

    async def save_messages(self, key: str, messages: Sequence[BaseMessage]) -> None:
        """Replace the transcript for `key`, keeping only the most recent N messages."""
        async with self._lock:
            self._sessions[key] = list(messages)[-self._max_messages :]
