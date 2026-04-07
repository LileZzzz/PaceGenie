from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    retrieved_context: str | None
    reflection_count: int
    semantic_reflection_enabled: bool  # True → _should_reflect uses LLM-as-judge
    last_critique: str                 # REVISE reason from judge → used by reflect_on_answer
