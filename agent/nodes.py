from __future__ import annotations

import logging
import os
import threading
from typing import TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.state import AgentState
from agent.tools import ALL_TOOLS
from agent.utils import get_last_message
from rag.retriever import get_retriever

logger = logging.getLogger(__name__)

_llm: ChatOpenAI | None = None
_llm_lock = threading.Lock()
_llm_with_tools: ChatOpenAI | None = None
_llm_with_tools_lock = threading.Lock()


class RetrieveContextUpdate(TypedDict):
    retrieved_context: str


class MessageUpdate(TypedDict):
    messages: list[BaseMessage]


class ReflectionUpdate(TypedDict):
    messages: list[BaseMessage]
    reflection_count: int


def get_llm() -> ChatOpenAI:
    """Return a singleton LLM client so repeated node calls reuse one connection setup."""
    global _llm
    if _llm is not None:
        return _llm
    with _llm_lock:
        if _llm is None:  # re-check inside lock
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


def get_llm_with_tools() -> ChatOpenAI:
    """Return a singleton LLM pre-bound to all tools; avoids re-serializing schemas per call."""
    global _llm_with_tools
    if _llm_with_tools is not None:
        return _llm_with_tools
    with _llm_with_tools_lock:
        if _llm_with_tools is None:
            _llm_with_tools = get_llm().bind_tools(ALL_TOOLS)
    return _llm_with_tools


def _extract_last_user_query(state: AgentState) -> str:
    """Pull the most recent HumanMessage text to use as the retrieval query.

    Using the last user message keeps retrieval grounded in exactly what the
    user just asked, avoiding drift from earlier turns. Tuples are also checked
    because LangGraph accepts (role, content) pairs at graph.invoke time.
    """
    msg = get_last_message(state, HumanMessage)
    if msg:
        return str(msg.content)
    # Fallback: LangGraph also accepts (role, content) tuples at invoke time.
    for m in reversed(state.get("messages", [])):
        if isinstance(m, tuple) and m[0] == "user":
            return str(m[1])
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
        logger.exception("[nodes] retrieve_context error: %s", e)
        return {"retrieved_context": "Knowledge base temporarily unavailable."}


def generate_response(state: AgentState) -> MessageUpdate:
    """Bind tools at generation time so the model can fetch grounded user data on demand."""
    user_id = state.get("user_id", "demo_user")

    system_prompt = (
        "You are PaceGenie, a professional AI running coach. "
        "You MUST follow these rules on EVERY response:\n\n"
        "RULE 1 — Training data questions: If the user asks about their own runs, "
        "mileage, pace, heart rate, injury risk, or race history, you MUST call the "
        "appropriate Garmin tool. Use get_recent_runs for individual session details, "
        "get_training_load for 14-day load and injury risk, get_weekly_trend for "
        "multi-week volume trends (4-8+ weeks), and get_race_history for personal bests. "
        "Never guess or fabricate numbers.\n\n"
        "RULE 2 — Race time predictions: If the user asks 'can I run sub-X?', "
        "'what is my predicted time?', or 'what pace should I target for a race?', "
        "you MUST call get_pace_prediction with the target distance in km.\n\n"
        "RULE 3 — Knowledge questions: If the user asks about running coaching topics "
        "(pace zones, threshold training, VO2max intervals, injury prevention, "
        "nutrition, recovery, marathon plans, heart rate training, 80/20 rule, etc.) "
        "you MUST call search_knowledge to retrieve from the knowledge base. "
        "Do NOT answer from your own internal knowledge — always use the tool.\n\n"
        "Always include concrete numbers in your final answer. "
        f"Current user_id: {user_id}"
    )

    prompt_messages = [
        SystemMessage(content=system_prompt),
        SystemMessage(content=f"Retrieved context: {state.get('retrieved_context')}"),
    ] + state.get("messages", [])

    try:
        response = get_llm_with_tools().invoke(prompt_messages)
        return {"messages": [response]}
    except Exception as e:
        logger.exception("[nodes] LLM error: %s", e)
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
    msg = get_last_message(state, AIMessage)
    latest_text = str(msg.content) if msg and msg.content else ""

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
