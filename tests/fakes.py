"""Test-only doubles built on langchain-core."""

from typing import Any

from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_core.language_models import BaseChatModel, FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.vectorstores import InMemoryVectorStore

__all__ = [
    "DeterministicFakeEmbedding",
    "FakeListChatModel",
    "FlakyChatModel",
    "FlakyEmbeddings",
    "InMemoryVectorStore",
]


class FlakyChatModel(BaseChatModel):
    """Raises the queued failures in order, then answers `reply`.

    With `always=True` the first failure is raised on every call.
    """

    failures: list[Any] = []
    always: bool = False
    reply: str = "ok"
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "flaky"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls += 1
        if self.failures:
            raise self.failures[0] if self.always else self.failures.pop(0)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])


class FlakyEmbeddings(Embeddings):
    """Raises the queued failures in order, then returns a fixed vector."""

    def __init__(self, failures: list[BaseException] | None = None, always: bool = False) -> None:
        self.failures = list(failures or [])
        self.always = always
        self.calls = 0

    def _maybe_fail(self) -> None:
        self.calls += 1
        if self.failures:
            raise self.failures[0] if self.always else self.failures.pop(0)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._maybe_fail()
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        self._maybe_fail()
        return [1.0, 0.0]
