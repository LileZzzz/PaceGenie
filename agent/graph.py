import threading
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from agent.nodes import generate_response, reflect_on_answer, retrieve_context
from agent.state import AgentState
from agent.tools import ALL_TOOLS


if TYPE_CHECKING:
    from agent.config import AgentConfig

_memory = MemorySaver()
_graph: CompiledStateGraph | None = None
_graph_lock = threading.Lock()


def _get_latest_ai_text(state: AgentState) -> str:
    """Return the most recent AIMessage content as plain text."""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            return str(msg.content)
    return ""


def _should_reflect(state: AgentState) -> str:
    """Decide whether the answer needs a reflection pass.

    Two modes:
    - Rule-based (default): triggers when reply < 100 chars OR has no digits.
      Used by baseline, no-rag, no-reflection configs.
    - Semantic (LLM-as-judge): makes a lightweight LLM call to evaluate whether
      the reply actually answered the question and cited user data.
      Used by semantic-reflect config. Inspired by Reflexion (Shinn et al., 2023).
    """
    if state.get("reflection_count", 0) >= 2:
        return "end"

    if not state.get("semantic_reflection_enabled", False):
        # Rule-based fallback — unchanged for baseline/no-rag/no-reflection
        latest_text = _get_latest_ai_text(state)
        too_short = len(latest_text) < 100
        no_numbers = not any(c.isdigit() for c in latest_text)
        return "reflect" if (too_short or no_numbers) else "end"

    # Semantic path — LLM-as-judge
    from langchain_core.messages import SystemMessage
    from agent.nodes import get_llm, _extract_last_user_query

    answer = _get_latest_ai_text(state)
    question = _extract_last_user_query(state)
    if not answer or not question:
        return "end"

    judge_prompt = (
        f"User asked: {question}\n\n"
        f"Assistant answered:\n{answer}\n\n"
        "Evaluate this response against 3 criteria:\n"
        "1. Does it directly answer the specific question asked?\n"
        "2. Does it cite ≥2 specific numbers from the user's training data (km, min/km, bpm)?\n"
        "3. Does it reference the actual data returned by the tools that were called?\n\n"
        "Reply with exactly one of:\n"
        "APPROVE\n"
        "REVISE: [one-line reason explaining what is missing]"
    )
    verdict = str(get_llm().invoke([SystemMessage(content=judge_prompt)]).content)
    if "APPROVE" in verdict.upper():
        return "end"

    # Store critique so reflect_on_answer can use it directly
    state["last_critique"] = verdict
    return "reflect"


def route_after_generate(state: AgentState) -> str:
    """Route to tool execution if the LLM issued tool calls, otherwise check reflection."""
    messages = state.get("messages", [])
    if not messages:
        return "end"

    last_message = messages[-1]

    # NOTE: LangChain AIMessage carries tool_calls, while tuple messages do not.
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return _should_reflect(state)


def get_graph() -> CompiledStateGraph:
    """Return a singleton compiled graph to preserve memory and avoid rebuild overhead."""
    global _graph
    if _graph is not None:
        return _graph
    with _graph_lock:
        if _graph is not None:  # re-check inside lock
            return _graph

        graph = StateGraph(AgentState)
        tool_node = ToolNode(ALL_TOOLS)

        graph.add_node("retrieve_context", retrieve_context)
        graph.add_node("generate_response", generate_response)
        graph.add_node("tools", tool_node)
        graph.add_node("reflect_on_answer", reflect_on_answer)

        graph.set_entry_point("retrieve_context")
        graph.add_edge("retrieve_context", "generate_response")

        graph.add_conditional_edges(
            "generate_response",
            route_after_generate,
            {
                "tools": "tools",
                "reflect": "reflect_on_answer",
                "end": END,
            },
        )

        graph.add_edge("tools", "generate_response")
        graph.add_edge("reflect_on_answer", "generate_response")

        _graph = graph.compile(checkpointer=_memory)
    return _graph


def _route_tools_only(state: AgentState) -> str:
    """Route to tools if LLM issued tool calls, otherwise end.

    Used by build_graph() when reflection_enabled=False — avoids referencing
    a 'reflect_on_answer' node that hasn't been registered in the graph.
    """
    messages = state.get("messages", [])
    last = messages[-1] if messages else None
    if last and hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"


def build_graph(config: "AgentConfig | None" = None) -> CompiledStateGraph:
    """Create a fresh, non-cached graph for evaluation with the given config.

    Unlike get_graph(), this never caches — each call compiles a new graph.
    Used exclusively by evaluation scripts; the production API calls get_graph().

    No MemorySaver is attached so each graph.invoke() call is fully stateless
    (no cross-example memory bleed during eval runs).
    """
    from agent.config import AgentConfig as _AgentConfig
    from agent.nodes import make_generate_response

    if config is None:
        config = _AgentConfig()

    graph = StateGraph(AgentState)

    # Filter tools based on config — no-rag removes search_knowledge entirely
    # from BOTH the ToolNode AND the LLM's bound tool schema so it can't call it
    if config.knowledge_tools_enabled:
        tools_list = ALL_TOOLS
    else:
        tools_list = [t for t in ALL_TOOLS if getattr(t, "name", "") != "search_knowledge"]

    tool_node = ToolNode(tools_list)
    generate_node = make_generate_response(config.system_prompt_variant, tools=tools_list)

    if config.rag_enabled:
        graph.add_node("retrieve_context", retrieve_context)
        graph.add_node("generate_response", generate_node)
        graph.set_entry_point("retrieve_context")
        graph.add_edge("retrieve_context", "generate_response")
    else:
        graph.add_node("generate_response", generate_node)
        graph.set_entry_point("generate_response")

    graph.add_node("tools", tool_node)

    if config.reflection_enabled:
        graph.add_node("reflect_on_answer", reflect_on_answer)
        graph.add_conditional_edges(
            "generate_response",
            route_after_generate,
            {"tools": "tools", "reflect": "reflect_on_answer", "end": END},
        )
        graph.add_edge("reflect_on_answer", "generate_response")
    else:
        graph.add_conditional_edges(
            "generate_response",
            _route_tools_only,
            {"tools": "tools", "end": END},
        )

    graph.add_edge("tools", "generate_response")
    return graph.compile()  # No checkpointer — eval graphs are stateless
