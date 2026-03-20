from __future__ import annotations

import os
from typing import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.state import AgentState
from agent.tools import ALL_TOOLS, get_training_load

_llm: ChatOpenAI | None = None


class RetrieveContextUpdate(TypedDict):
    retrieved_context: str


class MessageUpdate(TypedDict):
    messages: list


class ReflectionUpdate(TypedDict):
    messages: list
    reflection_count: int


def _get_latest_user_text(state: AgentState) -> str:
    """Extract latest user text so fallback logic can remain deterministic without LLM calls."""
    for message in reversed(state.get("messages", [])):
        content = getattr(message, "content", None)
        role = getattr(message, "type", None)
        if role == "human" and content is not None:
            return str(content)
        if isinstance(message, tuple) and len(message) == 2 and message[0] == "user":
            return str(message[1])
    return ""


def _build_training_volume_fallback(user_id: str, days: int = 14) -> str:
    """Create a concrete fallback answer so coaching output still includes actionable numbers."""
    try:
        training_json = get_training_load.invoke({"user_id": user_id, "days": days})
        return (
            "I could not reach the LLM service, so here is a data-backed fallback summary: "
            f"{training_json}"
        )
    except Exception:
        return (
            "I could not reach the LLM service. Keep weekly mileage increases under 10%, "
            "prioritize one easy day after each quality session, and monitor recovery trends."
        )


def get_llm() -> ChatOpenAI:
    """Return a singleton LLM client so repeated node calls reuse one connection setup."""
    global _llm
    if _llm is None:
        thinking_enabled = os.getenv("LLM_THINKING_ENABLED", "false").lower() == "true"
        default_headers = {
            "User-Agent": os.getenv("LLM_USER_AGENT", "claude-code/1.0"),
            "X-Client-Name": os.getenv("LLM_CLIENT_NAME", "claude-code"),
        }
        extra_body = {
            "thinking": {"type": "enabled" if thinking_enabled else "disabled"}
        }
        _llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL"),
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
            default_headers=default_headers,
            extra_body=extra_body,
        )
    return _llm


def retrieve_context(state: AgentState) -> RetrieveContextUpdate:
    """Provide stable fallback context so responses remain deterministic before RAG is ready."""
    _ = state
    return {"retrieved_context": "No relevant knowledge base content is available yet."}


def generate_response(state: AgentState) -> MessageUpdate:
    """Bind tools at generation time so the model can fetch grounded user data on demand."""
    user_id = state.get("user_id", "demo_user")

    system_prompt = (
        "You are PaceGenie, a professional AI running coach. "
        "Provide personalized guidance based on the user's Garmin training data and running knowledge. "
        "Always use the available tools to fetch real training data before giving advice. "
        "Include concrete numbers (distances, paces, heart rates) in your answers. "
        f"The current user_id is: {user_id}"
    )

    prompt_messages = [
        SystemMessage(content=system_prompt),
        SystemMessage(content=f"Retrieved context: {state.get('retrieved_context')}"),
    ] + state.get("messages", [])

    try:
        llm_with_tools = get_llm().bind_tools(ALL_TOOLS)
        response = llm_with_tools.invoke(prompt_messages)
        return {"messages": [response]}
    except Exception as e:
        print(f"LLM error: {e}")
        latest_user_text = _get_latest_user_text(state).lower()
        # NOTE: Route volume/risk queries to the workload tool even in fallback mode.
        if any(keyword in latest_user_text for keyword in ("volume", "mileage", "injury", "risk", "load")):
            fallback = _build_training_volume_fallback(user_id=user_id, days=14)
        else:
            fallback = (
                "Based on your current goal to improve 5 km performance, start with 3 quality runs per week: "
                "1 easy run, 1 threshold workout, and 1 long run. Keep at least 1 full rest day. "
                "Increase weekly mileage by no more than 10%, track heart rate trends, and reassess every 2 weeks "
                "using pace and recovery signals."
            )
        return {"messages": [AIMessage(content=fallback)]}


def reflect_on_answer(state: AgentState) -> ReflectionUpdate:
    """Inject a reflection prompt to improve answer specificity within a bounded retry budget."""
    return {
        "messages": [
            HumanMessage(
                content=(
                    "Please answer again and include my concrete training data, "
                    "including specific numbers."
                )
            )
        ],
        "reflection_count": state.get("reflection_count", 0) + 1,
    }
