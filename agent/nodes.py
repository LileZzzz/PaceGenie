from __future__ import annotations

import os
from typing import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.state import AgentState
from agent.tools import ALL_TOOLS, get_training_load
from rag.retriever import get_retriever

_llm: ChatOpenAI | None = None


class RetrieveContextUpdate(TypedDict):
    retrieved_context: str


class MessageUpdate(TypedDict):
    messages: list


class ReflectionUpdate(TypedDict):
    messages: list
    reflection_count: int


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


def _extract_last_user_query(state: AgentState) -> str:
    """Pull the most recent HumanMessage text to use as the retrieval query.

    Using the last user message (not a summary) keeps retrieval grounded in
    exactly what the user just asked, avoiding drift from earlier turns.
    """
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
        # LangGraph also accepts (role, content) tuples at graph.invoke time.
        if isinstance(msg, tuple) and msg[0] == "user":
            return str(msg[1])
    return ""


def retrieve_context(state: AgentState) -> RetrieveContextUpdate:
    """Pre-fetch knowledge-base context before the LLM generates a response.

    Running retrieval as a dedicated node keeps generate_response focused on
    reasoning, and lets us swap retrieval strategies without touching LLM logic.
    """
    query = _extract_last_user_query(state)
    if not query:
        return {"retrieved_context": "No user query found."}

    try:
        chunks = get_retriever().hybrid_search(query, top_k=3)
        if not chunks:
            return {"retrieved_context": "No relevant knowledge found."}
        lines = "\n".join(f"{i + 1}. {chunk}" for i, chunk in enumerate(chunks))
        return {"retrieved_context": f"Related knowledge:\n{lines}"}
    except Exception as e:
        print(f"[nodes] retrieve_context error: {e}")
        return {"retrieved_context": "Knowledge base temporarily unavailable."}


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
        print(f"[nodes] LLM error: {e}")
        return {
            "messages": [
                AIMessage(content="I'm temporarily unavailable. Please try again.")
            ]
        }


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
