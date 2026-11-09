"""Test-only doubles built on langchain-core."""

from collections.abc import Callable
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
    "RelevanceVectorStore",
    "ScriptedChatModel",
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


class ScriptedChatModel(BaseChatModel):
    """Answers via `responder(messages)` and records every call's messages in `prompts`."""

    responder: Any = None
    prompts: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.prompts.append(list(messages))
        reply = self.responder(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])

    def calls_with(self, marker: str) -> list[list[BaseMessage]]:
        """Calls whose system prompt contains `marker`."""
        return [p for p in self.prompts if marker in str(p[0].content)]


class RelevanceVectorStore(InMemoryVectorStore):
    """In-memory store with relevance scores (cosine clipped to 0..1) that logs its queries."""

    def __init__(self, embedding: Embeddings) -> None:
        super().__init__(embedding)
        self.queries: list[str] = []

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        return lambda score: max(0.0, min(1.0, score))

    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs: Any) -> Any:
        self.queries.append(query)
        flt = kwargs.pop("filter", None)
        if isinstance(flt, dict):  # Chroma-style metadata filter used by the application
            wanted = dict(flt)
            kwargs["filter"] = lambda doc: all(doc.metadata.get(n) == v for n, v in wanted.items())
        elif flt is not None:
            kwargs["filter"] = flt
        return super().similarity_search_with_score(query, k=k, **kwargs)
