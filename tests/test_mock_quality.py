"""Mock data quality validation for PaceGenie agent.

Runs 5 targeted questions to verify that the agent can produce
personalised, data-grounded answers from data/mock_garmin.json.

Evaluation checklist (assessed manually after running):
  [ ] Cites concrete numbers (km, pace, heart rate)
  [ ] Answer length > 150 characters
  [ ] References injury_history or personal_bests where relevant

Run:  uv run pytest tests/test_mock_quality.py -v -s
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import AIMessage

from agent.graph import get_graph

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MOCK_DATA_PATH = Path(__file__).parent.parent / "data" / "mock_garmin.json"
SEPARATOR = "=" * 70

QUESTIONS: list[tuple[str, str]] = [
    (
        "quality-test-1",
        "How has my training volume been over the last 30 days? Am I at risk of injury?",
    ),
    (
        "quality-test-2",
        "I have a history of knee injuries. What should I watch out for in my current training?",
    ),
    (
        "quality-test-3",
        "I want to finish a half marathon in under 110 minutes on May 10th. "
        "Based on my current fitness and personal bests, is that realistic?",
    ),
    (
        "quality-test-4",
        "Is my training intensity distribution reasonable? "
        "Am I doing enough easy running versus hard sessions?",
    ),
    (
        "quality-test-5",
        "How did my training go last week, and how should I structure this week?",
    ),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_garmin_data() -> dict:
    """Read mock Garmin data to verify it is present and well-formed."""
    return json.loads(MOCK_DATA_PATH.read_text(encoding="utf-8"))


def _final_answer(messages: list) -> str:
    """Extract the last non-empty AIMessage content."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return str(msg.content)
    return "(no answer)"


def _tools_called(messages: list) -> list[str]:
    """Collect all tool names called in this turn."""
    return [
        t["name"]
        for msg in messages
        if isinstance(msg, AIMessage)
        for t in getattr(msg, "tool_calls", [])
    ]


def _quality_flags(answer: str) -> dict[str, bool]:
    """Run lightweight quality checks on the answer text."""
    return {
        "has_numbers": any(c.isdigit() for c in answer),
        "length_ok": len(answer) > 150,
        "mentions_km": "km" in answer.lower(),
        "mentions_pace": any(w in answer.lower() for w in ["pace", "min/km", ":"]),
        "mentions_injury": any(
            w in answer.lower()
            for w in ["knee", "injury", "itbs", "tendon", "pain", "risk"]
        ),
        "mentions_goal": any(
            w in answer.lower()
            for w in ["110", "half marathon", "goal", "pb", "personal best", "113"]
        ),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_mock_data_present() -> None:
    """Verify mock Garmin data file exists and has runs."""
    data = _load_garmin_data()
    assert "recent_runs" in data, "mock_garmin.json missing recent_runs"
    assert len(data["recent_runs"]) >= 16, "Expected at least 16 runs in mock data"
    print(f"\nGarmin data loaded: {len(data['recent_runs'])} runs")


def test_quality_suite() -> None:
    """Run 5 quality-test questions and check all produce data-grounded answers."""
    graph = get_graph()
    garmin_data = _load_garmin_data()

    print(f"\n{SEPARATOR}")
    print("PaceGenie Mock Data Quality Validation")
    print(f"Garmin data loaded: {len(garmin_data.get('recent_runs', []))} runs")
    print(SEPARATOR)

    all_ok = True

    for thread_id, question in QUESTIONS:
        print(f"\n{SEPARATOR}")
        print(f"Q ({thread_id}): {question}")
        print(SEPARATOR)

        result = graph.invoke(
            {
                "messages": [("user", question)],
                "user_id": "demo_user",
                "retrieved_context": None,
                "reflection_count": 0,
            },
            config={"configurable": {"thread_id": thread_id}},
        )

        messages = result["messages"]
        answer = _final_answer(messages)
        tools = _tools_called(messages)
        flags = _quality_flags(answer)

        print(f"\nTools called : {tools}")
        print(f"Answer length: {len(answer)} chars")
        print()
        print("--- ANSWER ---")
        print(answer)
        print()
        print("--- QUALITY FLAGS ---")
        for flag, ok in flags.items():
            mark = "[OK]" if ok else "[--]"
            print(f"  {mark} {flag}")

        if not flags["has_numbers"] or not flags["length_ok"]:
            all_ok = False

    assert all_ok, "One or more answers failed basic quality checks (no numbers or too short)"
