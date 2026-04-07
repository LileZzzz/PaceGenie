"""Pilot evaluation — runs N questions locally with full verbose output.

Shows:
  - The full chat interaction (question → tool calls → answer)
  - Exactly what context each judge receives
  - Each judge's score

Cache flags to avoid re-running the slow agent:
  --save-cache   Run agent, save replies+context to JSON, then score
  --from-cache   Skip agent entirely, load from saved JSON, score only

Usage:
    # First run — agent + judges (~2.5 min per question)
    uv run python evaluation/pilot_test.py --n 3 --save-cache

    # Re-score from cache — judges only (~1.5 min per question)
    uv run python evaluation/pilot_test.py --from-cache

    # Different config, reuse same cache
    uv run python evaluation/pilot_test.py --config no-rag --save-cache
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import AIMessage, ToolMessage

from agent.config import CONFIG_MAP, BASELINE
from agent.graph import build_graph
from evaluation.langsmith_eval import (
    QUESTIONS,
    _extract_context,
    hallucination_judge,
    relevance_judge,
    personalization_judge,
)

SEP = "─" * 60
CACHE_DIR = Path(__file__).parent / "cache"


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------
def run_one(question: str, graph) -> dict:
    result = graph.invoke(
        {
            "messages": [("user", question)],
            "user_id": "demo_user",
            "retrieved_context": None,
            "reflection_count": 0,
        },
        config={},
    )
    context = _extract_context(result)
    reply = ""
    tool_calls_made = []
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_made.append(tc.get("name", "?"))
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            reply = str(msg.content)
            break
    return {
        "reply": reply,
        "context": context,
        "tools_called": tool_calls_made,
        "reflection_count": result.get("reflection_count", 0),
    }


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------
def score_one(question: str, reply: str, context: str) -> dict[str, dict]:
    """Run all 3 judges once, return results keyed by metric name."""
    inputs = {"question": question}
    outputs = {"reply": reply, "context": context}
    results = {}
    for judge_fn, name in [
        (hallucination_judge,   "hallucination"),
        (relevance_judge,       "answer_relevance"),
        (personalization_judge, "personalization"),
    ]:
        try:
            results[name] = judge_fn(inputs, outputs)
        except Exception as e:
            results[name] = {"key": name, "score": 0.0, "comment": f"ERROR: {e}"}
    return results


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
def print_entry(question: str, data: dict, scores: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"QUESTION: {question}")
    print(SEP)
    tools = ", ".join(data["tools_called"]) if data["tools_called"] else "none"
    print(f"TOOLS CALLED: {tools}")
    print(f"REFLECTIONS : {data['reflection_count']}")
    print(SEP)
    print("AGENT ANSWER:")
    print(data["reply"] or "(empty)")
    print(SEP)
    print("JUDGE CONTEXT (what judges see):")
    print(data["context"])
    print(SEP)
    print("JUDGE SCORES:")
    for name, res in scores.items():
        score = res["score"]
        comment = res.get("comment", "")
        bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
        suffix = f"  ({score * 5:.1f}/5.0)" if name == "personalization" else ""
        print(f"  {name:<22} {bar}  {score:.2f}{suffix}  {comment}")


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def cache_path(config_key: str, n: int) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"{config_key}_{n}q.json"


def save_cache(path: Path, entries: list[dict]) -> None:
    with open(path, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"\nCache saved → {path}")


def load_cache(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot evaluation with caching")
    parser.add_argument("--config", choices=list(CONFIG_MAP.keys()), default=None)
    parser.add_argument("--n", type=int, default=3, help="Number of questions (default: 3)")
    parser.add_argument("--save-cache", action="store_true",
                        help="Run agent and save replies to cache JSON")
    parser.add_argument("--from-cache", action="store_true",
                        help="Skip agent, load replies from cache, run judges only")
    args = parser.parse_args()

    config_key = args.config or "baseline"
    c_path = cache_path(config_key, args.n)

    questions = QUESTIONS[:args.n]

    # ---------------------------------------------------------------------------
    # Phase 1 — get agent replies (from cache or live)
    # ---------------------------------------------------------------------------
    if args.from_cache:
        if not c_path.exists():
            print(f"No cache found at {c_path}. Run with --save-cache first.")
            sys.exit(1)
        entries = load_cache(c_path)
        print(f"Loaded {len(entries)} cached replies from {c_path}")
        print("Skipping agent — running judges only\n")
    else:
        if args.config:
            agent_config, version_tag = CONFIG_MAP[args.config]
            graph = build_graph(agent_config)
            print(f"Config : {args.config} ({version_tag})")
        else:
            graph = build_graph(BASELINE)
            print("Config : baseline (default)")

        print(f"Running agent for {len(questions)} questions...\n")
        entries = []
        for i, (qid, question) in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}] {question[:70]}", flush=True)
            data = run_one(question, graph)
            entries.append({"qid": qid, "question": question, **data})
            print(f"  → {len(data['reply'])} chars, tools: {data['tools_called']}", flush=True)

        if args.save_cache:
            save_cache(c_path, entries)

    # ---------------------------------------------------------------------------
    # Phase 2 — score and display
    # ---------------------------------------------------------------------------
    print(f"\nScoring {len(entries)} replies with 3 judges...\n")
    all_scores: dict[str, list[float]] = {"hallucination": [], "answer_relevance": [], "personalization": []}

    for entry in entries:
        scores = score_one(entry["question"], entry["reply"], entry["context"])
        print_entry(entry["question"], entry, scores)
        for name, res in scores.items():
            all_scores[name].append(res["score"])

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"PILOT SUMMARY ({len(entries)} questions, config: {config_key})")
    print(SEP)
    for key, vals in all_scores.items():
        if vals:
            avg = sum(vals) / len(vals)
            if key == "personalization":
                print(f"  {key:<28} avg = {avg:.2f}  ({avg * 5:.1f}/5.0)")
            else:
                print(f"  {key:<28} avg = {avg:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
