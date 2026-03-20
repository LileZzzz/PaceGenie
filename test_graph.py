from agent.graph import build_graph


def main() -> None:
    graph = build_graph()
    result = graph.invoke(
        {
            "messages": [("user", "test")],
            "user_id": "test",
            "garmin_data": None,
            "retrieved_context": None,
            "reflection_count": 0,
        },
        config={"configurable": {"thread_id": "test-1"}},
    )
    print(result)


if __name__ == "__main__":
    main()
