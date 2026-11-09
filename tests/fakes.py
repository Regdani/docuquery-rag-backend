"""Test-only doubles built on langchain-core."""

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel
from langchain_core.vectorstores import InMemoryVectorStore

__all__ = ["DeterministicFakeEmbedding", "FakeListChatModel", "InMemoryVectorStore"]
