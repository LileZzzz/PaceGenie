from __future__ import annotations

from typing import Type, TypeVar

from langchain_core.messages import BaseMessage

from agent.state import AgentState

M = TypeVar("M", bound=BaseMessage)


def get_last_message(state: AgentState, message_type: Type[M]) -> M | None:
    """Return the most recent message of the given type, or None if not found."""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, message_type):
            return msg
    return None
