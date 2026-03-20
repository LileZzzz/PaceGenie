from __future__ import annotations

import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.state import AgentState
from agent.tools import ALL_TOOLS

_llm: ChatOpenAI | None = None


def get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL"),
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
            default_headers={
                "User-Agent": "claude-code/1.0",
                "X-Client-Name": "claude-code",
            },
        )
    return _llm


def retrieve_context(state: AgentState) -> dict:
    """Provide a temporary retrieval placeholder until hybrid RAG is implemented."""
    return {"retrieved_context": "No relevant knowledge base content is available yet."}


def generate_response(state: AgentState) -> dict:
    """Generate an answer with tools bound so the LLM can fetch real training data."""
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
        fallback = (
            "Based on your current goal to improve 5 km performance, start with 3 quality runs per week: "
            "1 easy run, 1 threshold workout, and 1 long run. Keep at least 1 full rest day. "
            "Increase weekly mileage by no more than 10%, track heart rate trends, and reassess every 2 weeks "
            "using pace and recovery signals."
        )
        return {"messages": [AIMessage(content=fallback)]}


def reflect_on_answer(state: AgentState) -> dict:
    """Add a corrective message prompting the LLM to include concrete data."""
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
