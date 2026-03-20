from dotenv import load_dotenv

load_dotenv()

from agent.graph import build_graph


def main() -> None:
    """Run two sequential turns to validate memory and tool-routing behavior."""
    graph = build_graph()

    first_result = graph.invoke(
        {
            "messages": [
                ("user", "How is my recent training volume? Any injury risk?")
            ],
            "user_id": "demo_user",
            "garmin_data": None,
            "retrieved_context": None,
            "reflection_count": 0,
        },
        config={"configurable": {"thread_id": "test-1"}},
    )

    print("=" * 60)
    print("Query: How is my recent training volume? Any injury risk?")
    print("=" * 60)
    for msg in first_result["messages"]:
        role = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", str(msg))
        if content:
            print(f"[{role}] {content[:500]}")
    print()

    second_result = graph.invoke(
        {
            "messages": [
                ("user", "Can you build a one-week plan based on your previous advice?")
            ],
            "user_id": "demo_user",
            "garmin_data": None,
            "retrieved_context": None,
            "reflection_count": 0,
        },
        config={"configurable": {"thread_id": "test-1"}},
    )

    print("=" * 60)
    print("Query: Can you build a one-week plan based on your previous advice?")
    print("=" * 60)
    for msg in second_result["messages"]:
        role = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", str(msg))
        if content:
            print(f"[{role}] {content[:500]}")


if __name__ == "__main__":
    main()
