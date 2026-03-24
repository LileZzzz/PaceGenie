"""End-to-end integration test for PaceGenie agent.

Tests four capability areas:
  1. Garmin tool path     -- get_training_load / get_recent_runs called correctly
  2. RAG tool path        -- search_knowledge called for coaching questions
  3. Memory across turns  -- second turn references first-turn context
  4. 10-question suite    -- broad coverage; records which questions trigger Reflection

Run:  uv run python test_graph.py
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

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


def _count_reflections(messages: list) -> int:
    """Count how many reflection prompts were injected in this invoke result.

    Reflection prompts are HumanMessages added by reflect_on_answer that contain
    'Please call the relevant training data tools'.
    """
    return sum(
        1
        for msg in messages
        if isinstance(msg, HumanMessage)
        and "Please call the relevant training data tools" in str(msg.content)
    )


def _final_answer(messages: list) -> str:
    """Return the last non-empty AIMessage content."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return str(msg.content)
    return "(no answer)"


def _invoke(graph, query: str, thread_id: str) -> list:
    """Invoke the graph and return the full messages list."""
    result = graph.invoke(
        {
            "messages": [("user", query)],
            "user_id": "demo_user",
            "retrieved_context": None,
            "reflection_count": 0,
        },
        config={"configurable": {"thread_id": thread_id}},
    )
    return result["messages"]


# ---------------------------------------------------------------------------
# Task 1 & 2 -- Tool routing smoke tests
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


# ---------------------------------------------------------------------------
# Task 3 -- Memory across turns (Chinese queries)
# ---------------------------------------------------------------------------


def test_memory_across_turns(graph) -> None:
    """Second turn should reference the volume data established in the first turn.

    Uses a fresh thread so previous test state does not pollute memory.
    """
    thread = "test-memory-zh"

    q1 = "What was my weekly running volume recently?"
    messages1 = _invoke(graph, q1, thread_id=thread)
    _print_messages(messages1, f"Turn 1: {q1}")

    q2 = "How does that compare to the previous week?"
    messages2 = _invoke(graph, q2, thread_id=thread)
    _print_messages(messages2, f"Turn 2: {q2}")

    print("\n  --- assertions ---")
    _assert_tool_called(messages1, "get_training_load", "Turn 1 calls training load")

    # Turn 2 should reference data; check final answer contains a number
    answer2 = _final_answer(messages2)
    has_number = any(c.isdigit() for c in answer2)
    if has_number:
        print("  [PASS] 'memory': Turn 2 answer contains numeric data")
    else:
        print("  [FAIL] 'memory': Turn 2 answer has no numbers (memory may be broken)")
        print(f"         preview: {answer2[:200]}")

    called_t2 = [
        t["name"]
        for msg in messages2
        if isinstance(msg, AIMessage)
        for t in getattr(msg, "tool_calls", [])
    ]
    print(f"  [INFO] Turn 2 tools called: {called_t2 or '(none -- answered from memory)'}")


# ---------------------------------------------------------------------------
# Task 4 -- 10-question integration suite with Reflection tracking
# ---------------------------------------------------------------------------

# (question, thread_id, expected_tool, expected_keyword_in_answer)
INTEGRATION_QUESTIONS: list[tuple[str, str, str | None, str | None]] = [
    (
        "What is my total running volume over the last 7 days in km?",
        "q01",
        "get_training_load",
        "km",
    ),
    (
        "Show me my last 5 runs with distance and pace for each.",
        "q02",
        "get_recent_runs",
        "km",
    ),
    (
        "Has my weekly mileage increased by more than 10%? Am I at injury risk?",
        "q03",
        "get_training_load",
        "km",
    ),
    (
        "What is Easy pace (E pace) and how slow should I run?",
        "q04",
        "search_knowledge",
        "pace",
    ),
    (
        "What is lactate threshold training and when should I do tempo runs (T pace)?",
        "q05",
        "search_knowledge",
        "threshold",
    ),
    (
        "How do I prevent and treat runner's knee?",
        "q06",
        "search_knowledge",
        "knee",
    ),
    (
        "I am preparing for a half marathon. How should I taper in the final week?",
        "q07",
        "search_knowledge",
        "race",
    ),
    (
        "What races have I run in the past and what was my best finishing time?",
        "q08",
        "get_race_history",
        "time",
    ),
    (
        "Based on my current training load, which pace zone should most of my runs be in?",
        "q09",
        "get_training_load",
        None,
    ),
    (
        "What causes iliotibial band syndrome (ITBS) and how do I recover from it?",
        "q10",
        "search_knowledge",
        "itbs",
    ),
]


def test_integration_suite(graph) -> None:
    """Run 10 questions, check tool routing and answer quality, track reflections."""
    total = len(INTEGRATION_QUESTIONS)
    passed = 0
    reflection_triggers: list[str] = []

    print(f"\n{SEPARATOR}")
    print(f"10-QUESTION INTEGRATION SUITE  ({total} questions)")
    print(SEPARATOR)

    for q, thread_id, expected_tool, expected_keyword in INTEGRATION_QUESTIONS:
        messages = _invoke(graph, q, thread_id=thread_id)
        ref_count = _count_reflections(messages)
        answer = _final_answer(messages)
        has_numbers = any(c.isdigit() for c in answer)

        # Tool routing check
        tool_ok = True
        if expected_tool:
            called = [
                t["name"]
                for msg in messages
                if isinstance(msg, AIMessage)
                for t in getattr(msg, "tool_calls", [])
            ]
            tool_ok = expected_tool in called

        # Keyword check
        keyword_ok = True
        if expected_keyword:
            keyword_ok = expected_keyword.lower() in answer.lower()

        ok = tool_ok and keyword_ok and has_numbers
        status = "[PASS]" if ok else "[FAIL]"
        if ok:
            passed += 1
        if ref_count > 0:
            reflection_triggers.append(thread_id)

        print(
            f"\n  {status} {thread_id}: {q[:50]}"
            f"\n          tool={'OK' if tool_ok else 'MISS'} ({expected_tool})"
            f"  keyword={'OK' if keyword_ok else 'MISS'} ({expected_keyword})"
            f"  numbers={'YES' if has_numbers else 'NO'}"
            f"  reflections={ref_count}"
        )
        if not ok:
            print(f"          answer preview: {answer[:150]}")

    print(f"\n  --- Summary ---")
    print(f"  Passed         : {passed}/{total}")
    print(f"  Reflections triggered : {len(reflection_triggers)} questions -> {reflection_triggers}")

    if passed >= 9:
        print("  [PASS] Acceptance criterion met: >=9/10 correct")
    else:
        print(f"  [FAIL] Acceptance criterion not met: need 9/10, got {passed}/10")

    if len(reflection_triggers) >= 3:
        print("  [PASS] Reflection triggered >=3 times")
    else:
        print(f"  [WARN] Reflection triggered only {len(reflection_triggers)} time(s) (target: >=3)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all test cases and print a summary."""
    graph = build_graph()

    print("\n" + SEPARATOR)
    print("PaceGenie Agent Test Suite")
    print(SEPARATOR)

    print("\n>>> Task 1-2: Tool routing smoke tests")
    test_garmin_tool_path(graph)
    test_rag_tool_path(graph)

    print("\n>>> Task 3: Memory across turns")
    test_memory_across_turns(graph)

    print("\n>>> Task 4: 10-question integration suite")
    test_integration_suite(graph)

    print(f"\n{SEPARATOR}")
    print("Done. Check [PASS] / [FAIL] lines above.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
