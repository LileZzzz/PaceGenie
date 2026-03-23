from agent.graph import build_graph


def print_messages(messages: list, query: str) -> None:
    """Print all messages with clear role and tool_calls info."""
    print("=" * 60)
    print(f"Query: {query}")
    print("=" * 60)
    for i, msg in enumerate(messages):
        role = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", "")
        tool_calls = getattr(msg, "tool_calls", [])

        print(f"\n--- message {i+1} [{role}] ---")
        if tool_calls:
            print(f"  tool_calls: {[t['name'] for t in tool_calls]}")
        if content:
            print(f"  content: {content[:300]}")
        if not tool_calls and not content:
            print("  (empty)")
    print()


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
    print_messages(
        first_result["messages"], "How is my recent training volume? Any injury risk?"
    )

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
    print_messages(
        second_result["messages"],
        "Can you build a one-week plan based on your previous advice?",
    )


if __name__ == "__main__":
    main()
