from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    garmin_data: dict | None
    retrieved_context: str | None
    reflection_count: int
