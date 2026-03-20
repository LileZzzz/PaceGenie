from langgraph.graph import END, StateGraph

from agent.state import AgentState


def generate_response(state: AgentState) -> dict:
    """Return a placeholder assistant response for initial graph validation."""
    return {
        "messages": [
            ("assistant", "Hello"),
        ]
    }


def build_graph() -> object:
    """Build the minimal PaceGenie graph for Day 1 setup."""
    graph = StateGraph(AgentState)
    graph.add_node("generate_response", generate_response)
    graph.set_entry_point("generate_response")
    graph.add_edge("generate_response", END)
    return graph.compile()
