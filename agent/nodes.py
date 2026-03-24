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
        "You MUST follow these two rules on EVERY response:\n\n"
        "RULE 1 — Training data questions: If the user asks about their own runs, "
        "mileage, pace, heart rate, injury risk, or race history, you MUST call the "
        "appropriate Garmin tool (get_training_load, get_recent_runs, get_race_history) "
        "to fetch real data. Never guess or fabricate numbers.\n\n"
        "RULE 2 — Knowledge questions: If the user asks about running coaching topics "
        "(pace zones, E/M/T/I/R pace, threshold training, injury prevention, ITBS, "
        "runner's knee, race preparation, taper, 80/20 rule, etc.) you MUST call "
        "search_knowledge to retrieve from the knowledge base. "
        "Do NOT answer from your own internal knowledge — always use the tool.\n\n"
        "Always include concrete numbers in your final answer. "
        f"Current user_id: {user_id}"
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
    """Inject a targeted self-criticism prompt based on which quality gate failed.

    Checking conditions here (not relying on graph routing flags) keeps the node
    self-contained and makes the critique specific enough for the LLM to act on.
    """
    latest_text = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            latest_text = str(msg.content)
            break

    too_short = len(latest_text) < 100
    no_numbers = not any(c.isdigit() for c in latest_text)

    issues: list[str] = []
    if too_short:
        issues.append("Your previous answer was too brief (under 100 characters).")
    if no_numbers:
        issues.append(
            "Your previous answer did not cite any concrete numbers from the user's "
            "training data. You must include actual mileage (km), pace (min/km), "
            "and heart rate (bpm) figures pulled from the training tools."
        )

    critique = " ".join(issues)
    prompt = (
        f"{critique} "
        "Please call the relevant training data tools first, then give a detailed "
        "answer with specific numbers."
    )

    return {
        "messages": [HumanMessage(content=prompt)],
        "reflection_count": state.get("reflection_count", 0) + 1,
    }
