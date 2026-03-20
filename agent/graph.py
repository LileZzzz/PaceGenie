from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from agent.nodes import generate_response, reflect_on_answer, retrieve_context
from agent.state import AgentState
from agent.tools import ALL_TOOLS


def _get_latest_assistant_text(state: AgentState) -> str:
    """Read the latest assistant content across tuple and LangChain message formats."""
    for message in reversed(state.get("messages", [])):
        content = getattr(message, "content", None)
        if content is not None:
            return str(content)
        if (
            isinstance(message, tuple)
            and len(message) == 2
            and message[0] == "assistant"
        ):
            return str(message[1])
    return ""


def _should_reflect(state: AgentState) -> str:
    """Decide whether the answer needs a reflection pass."""
    if state.get("reflection_count", 0) >= 2:
        return "end"

    latest_text = _get_latest_assistant_text(state)
    too_short = len(latest_text) < 100
    no_numbers = not any(char.isdigit() for char in latest_text)
    if too_short or no_numbers:
        return "reflect"
    return "end"


def route_after_generate(state: AgentState) -> str:
    """Route to tool execution if the LLM issued tool calls, otherwise check reflection."""
    messages = state.get("messages", [])
    if not messages:
        return "end"

    last_message = messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return _should_reflect(state)


def build_graph() -> object:
    """Build the PaceGenie ReAct agent with tool calling, reflection, and memory."""
    memory = MemorySaver()
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

    return graph.compile(checkpointer=memory)
