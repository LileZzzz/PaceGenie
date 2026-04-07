"""Phase 1 — Run agent, save replies to cache.

Runs N questions through the agent and saves every reply + context to a JSONL
file. Re-run only when the agent changes (new prompt, new config, new tools).

Cache lives at: evaluation/cache/<config>_<n>q.jsonl

Usage:
    uv run python evaluation/run_agent.py --config baseline --n 20
    uv run python evaluation/run_agent.py --config no-rag --n 20
    uv run python evaluation/run_agent.py --config semantic-reflect --n 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import AIMessage, ToolMessage

from agent.config import CONFIG_MAP, BASELINE
from agent.graph import build_graph
from evaluation.langsmith_eval import QUESTIONS, _extract_context

CACHE_DIR = Path(__file__).parent / "cache"


def run_one(question: str, graph, semantic_reflection: bool = False) -> dict:
    result = graph.invoke(
        {
            "messages": [("user", question)],
            "user_id": "demo_user",
            "retrieved_context": None,
            "reflection_count": 0,
            "semantic_reflection_enabled": semantic_reflection,
            "last_critique": "",
        },
        config={},
    )
    context = _extract_context(result)
    reply = ""
    tools_called = []
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tools_called.append(tc.get("name", "?"))
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            reply = str(msg.content)
            break
    return {
        "reply": reply,
        "context": context,
        "tools_called": tools_called,
        "reflection_count": result.get("reflection_count", 0),
    }


def cache_path(config_key: str, n: int) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"{config_key}_{n}q.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agent and save replies to cache")
    parser.add_argument("--config", choices=list(CONFIG_MAP.keys()), default="baseline")
    parser.add_argument("--n", type=int, default=20, help="Number of questions (default: 20)")
    args = parser.parse_args()

    agent_config, version_tag = CONFIG_MAP[args.config]
    graph = build_graph(agent_config)
    questions = QUESTIONS[:args.n]
    out_path = cache_path(args.config, args.n)

    print(f"Phase 1 — Agent Run")
    print(f"Config  : {args.config} ({version_tag})")
    print(f"Questions: {len(questions)}")
    print(f"Cache   : {out_path}")
    print("=" * 50)

    with open(out_path, "w") as f:
        for i, (qid, question) in enumerate(questions, 1):
            print(f"[{i:2d}/{len(questions)}] {question[:65]}", flush=True)
            data = run_one(question, graph, semantic_reflection=agent_config.semantic_reflection)
            entry = {
                "qid": qid,
                "question": question,
                "reply": data["reply"],
                "context": data["context"],
                "tools_called": data["tools_called"],
                "reflection_count": data["reflection_count"],
                "config": args.config,
                "version": version_tag,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            f.write(json.dumps(entry) + "\n")
            f.flush()  # write immediately — safe to Ctrl+C and resume
            tools = ", ".join(data["tools_called"]) or "none"
            print(f"       tools={tools}  reflect={data['reflection_count']}  chars={len(data['reply'])}", flush=True)

    print(f"\nSaved {len(questions)} entries → {out_path}")
    print("Run judges next:")
    print(f"  uv run python evaluation/run_judges.py --config {args.config} --n {args.n} --langsmith")


if __name__ == "__main__":
    main()
