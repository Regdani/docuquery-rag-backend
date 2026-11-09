"""Graph-ready history nodes: load the transcript first, save it last.

They work on the shared `GraphState` (`session_key`, `question`, `answer`,
`messages`). Both nodes skip the store entirely when no session key is given.
"""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from app.memory.store import InMemorySessionStore
from app.rag.state import GraphState


async def load_history(store: InMemorySessionStore, state: GraphState) -> dict[str, Any]:
    """Put the stored messages for the session into the state."""
    key = state.get("session_key")
    if not key:
        return {"messages": []}
    return {"messages": await store.get_messages(key)}


async def save_history(store: InMemorySessionStore, state: GraphState) -> dict[str, Any]:
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
