"""End-to-end test script for the PaceGenie agent.

Covers three routing paths:
  1. Garmin tool path  -- get_training_load is called for volume questions
  2. RAG tool path     -- search_knowledge is called for coaching/knowledge questions
  3. Memory path       -- second turn references context from the first turn

Run:  uv run python test_graph.py
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage, ToolMessage

from agent.graph import build_graph

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEPARATOR = "=" * 60


def _print_messages(messages: list, label: str) -> None:
    """Print every message in a turn with role, tool names, and content preview."""
    print(f"\n{SEPARATOR}")
    print(f"TURN: {label}")
    print(SEPARATOR)
    for i, msg in enumerate(messages):
        role = getattr(msg, "type", "unknown")
        content = str(getattr(msg, "content", ""))
        tool_calls = getattr(msg, "tool_calls", [])

        print(f"\n  [{i + 1}] {role.upper()}")
        if tool_calls:
            names = [t["name"] for t in tool_calls]
            print(f"       tool_calls : {names}")
        if content:
            preview = content[:300].replace("\n", " ")
            print(f"       content    : {preview}")


def _assert_tool_called(messages: list, tool_name: str, label: str) -> None:
    """Fail with a clear message if tool_name was not called in this turn."""
    called = [
        t["name"]
        for msg in messages
        if isinstance(msg, AIMessage)
        for t in getattr(msg, "tool_calls", [])
    ]
    if tool_name not in called:
        print(f"  [FAIL] '{label}': expected {tool_name} to be called, got {called}")
    else:
        print(f"  [PASS] '{label}': {tool_name} was called")


def _assert_tool_result_contains(messages: list, keyword: str, label: str) -> None:
    """Fail if no ToolMessage content contains the keyword."""
    contents = [msg.content for msg in messages if isinstance(msg, ToolMessage)]
    if any(keyword.lower() in c.lower() for c in contents):
        print(f"  [PASS] '{label}': ToolMessage contains '{keyword}'")
    else:
        print(f"  [FAIL] '{label}': '{keyword}' not found in tool results")
        for c in contents:
            print(f"         tool result preview: {c[:120]}")


def _assert_final_answer_contains(messages: list, keyword: str, label: str) -> None:
    """Fail if the last AIMessage does not mention keyword."""
    final = next(
        (
            msg
            for msg in reversed(messages)
            if isinstance(msg, AIMessage) and msg.content
        ),
        None,
    )
    if final and keyword.lower() in final.content.lower():
        print(f"  [PASS] '{label}': final answer contains '{keyword}'")
    else:
        answer = final.content[:120] if final else "(no answer)"
        print(f"  [FAIL] '{label}': '{keyword}' not in final answer")
        print(f"         answer preview: {answer}")


def _invoke(graph, query: str, thread_id: str) -> list:
    """Invoke the graph and return the full messages list."""
    result = graph.invoke(
        {
            "messages": [("user", query)],
            "user_id": "demo_user",
            "garmin_data": None,
            "retrieved_context": None,
            "reflection_count": 0,
        },
        config={"configurable": {"thread_id": thread_id}},
    )
    return result["messages"]


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def test_garmin_tool_path(graph) -> None:
    """Agent should call get_training_load for load/injury-risk questions."""
    query = "Has my weekly mileage been increasing too fast? Am I at injury risk?"
    messages = _invoke(graph, query, thread_id="test-garmin")
    _print_messages(messages, query)

    print("\n  --- assertions ---")
    _assert_tool_called(messages, "get_training_load", "get_training_load called")
    _assert_tool_result_contains(messages, "total_km", "total_km in tool result")
    _assert_final_answer_contains(messages, "km", "answer mentions km")


def test_rag_tool_path(graph) -> None:
    """Agent should call search_knowledge for coaching/knowledge questions."""
    query = "I have knee pain after my long run. What should I do?"
    messages = _invoke(graph, query, thread_id="test-rag")
    _print_messages(messages, query)

    print("\n  --- assertions ---")
    _assert_tool_called(messages, "search_knowledge", "rag tool called")
    _assert_tool_result_contains(messages, "injury_prevention", "injury doc retrieved")
    _assert_final_answer_contains(messages, "rest", "answer mentions rest")


def test_threshold_training_rag(graph) -> None:
    """Agent should retrieve pace_zones.md content for threshold training questions."""
    query = "Explain threshold training and when I should do tempo runs."
    messages = _invoke(graph, query, thread_id="test-threshold")
    _print_messages(messages, query)

    print("\n  --- assertions ---")
    _assert_tool_called(messages, "search_knowledge", "rag tool called")
    _assert_tool_result_contains(messages, "pace_zones", "pace zones doc retrieved")
    _assert_final_answer_contains(messages, "threshold", "answer mentions threshold")


def test_memory_across_turns(graph) -> None:
    """Second turn should reference context established in the first turn."""
    thread = "test-memory"

    q1 = "How is my recent training volume?"
    messages1 = _invoke(graph, q1, thread_id=thread)
    _print_messages(messages1, f"Turn 1: {q1}")

    q2 = "Based on my volume, what pace zone should my easy runs be in?"
    messages2 = _invoke(graph, q2, thread_id=thread)
    _print_messages(messages2, f"Turn 2: {q2}")

    print("\n  --- assertions ---")
    _assert_tool_called(
        messages1, "get_training_load", "turn 1 calls training load tool"
    )
    # Turn 2 may call search_knowledge to look up pace zones
    called_t2 = [
        t["name"]
        for msg in messages2
        if isinstance(msg, AIMessage)
        for t in getattr(msg, "tool_calls", [])
    ]
    if called_t2:
        print(f"  [INFO] Turn 2 called: {called_t2}")
    else:
        print("  [INFO] Turn 2 answered from memory (no tool call)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all test cases and print a summary."""
    graph = build_graph()

    print("\n" + "=" * 60)
    print("PaceGenie Agent Test Suite")
    print("=" * 60)

    test_garmin_tool_path(graph)
    test_rag_tool_path(graph)
    test_threshold_training_rag(graph)
    test_memory_across_turns(graph)

    print(f"\n{SEPARATOR}")
    print("Done. Check [PASS] / [FAIL] lines above.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
