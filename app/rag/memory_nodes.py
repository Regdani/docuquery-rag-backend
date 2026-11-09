"""Graph-ready history nodes: load the transcript first, save it last.

State shape (provisional): `session_key`, `question`, `answer`, `messages` (prior history).
Both nodes skip the store entirely when no session key is given.
"""

from typing import Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.memory.store import InMemorySessionStore


class HistoryState(TypedDict, total=False):
    session_key: str | None
    question: str
    answer: str
    messages: list[BaseMessage]


async def load_history(store: InMemorySessionStore, state: HistoryState) -> dict[str, Any]:
    """Put the stored messages for the session into the state."""
    key = state.get("session_key")
    if not key:
        return {"messages": []}
    return {"messages": await store.get_messages(key)}


async def save_history(store: InMemorySessionStore, state: HistoryState) -> dict[str, Any]:
    """Store history + the new user message + the assistant reply (answer or clarification)."""
    key = state.get("session_key")
    if not key:
        return {}
    messages = [
        *state.get("messages", []),
        HumanMessage(content=state.get("question", "")),
        AIMessage(content=state.get("answer", "")),
    ]
    await store.save_messages(key, messages)
    return {"messages": messages}
