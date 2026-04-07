"""Run all ablation configs sequentially — Phase 1 (agent) then Phase 2 (judges).

Orchestrates the full two-phase pipeline for all 4 configs:
  1. run_agent.py   → generates and caches replies (slow, LLM-heavy)
  2. run_judges.py  → scores cached replies, optionally pushes to LangSmith

Use the LangSmith Compare view to see side-by-side scores for all configs.

Usage:
    # Full run (~40 min) — agent + judges for all 4 configs
    uv run python evaluation/run_ablation.py

    # Judges only (skip agent — use existing caches)
    uv run python evaluation/run_ablation.py --judges-only

    # Push scores to LangSmith
    uv run python evaluation/run_ablation.py --langsmith

    # Limit to N questions per config (faster for testing)
    uv run python evaluation/run_ablation.py --n 5

Each config takes ~10 min for 20 questions (agent only).
View results at: smith.langchain.com → PaceGenie project → Experiments
Select all 4 rows and click Compare.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import argparse

from agent.config import CONFIG_MAP
from evaluation.langsmith_eval import (
    QUESTIONS,
    hallucination_judge,
    relevance_judge,
    personalization_judge,
    ensure_dataset,
    DATASET_NAME,
)
from evaluation.run_agent import run_one, cache_path as agent_cache_path
from evaluation.run_judges import load_cache, score_entry, push_to_langsmith

from agent.graph import build_graph

ABLATION_ORDER = ["baseline", "no-rag", "no-reflection", "semantic-reflect"]
CACHE_DIR = Path(__file__).parent / "cache"


# ---------------------------------------------------------------------------
# Phase 1 — Agent
# ---------------------------------------------------------------------------
def run_agent_phase(config_key: str, n: int) -> Path:
    """Run agent for N questions, save to JSONL. Returns cache path."""
    from datetime import datetime, timezone

    agent_config, version_tag = CONFIG_MAP[config_key]
    graph = build_graph(agent_config)
    questions = QUESTIONS[:n]
    out_path = agent_cache_path(config_key, n)

    print(f"  Phase 1 — agent ({config_key}, {version_tag})")
    with open(out_path, "w") as f:
        for i, (qid, question) in enumerate(questions, 1):
            print(f"    [{i:2d}/{n}] {question[:60]}", flush=True)
            data = run_one(question, graph, semantic_reflection=agent_config.semantic_reflection)
            entry = {
                "qid": qid,
                "question": question,
                "reply": data["reply"],
                "context": data["context"],
                "tools_called": data["tools_called"],
                "reflection_count": data["reflection_count"],
                "config": config_key,
                "version": version_tag,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            f.write(json.dumps(entry) + "\n")
            f.flush()
            tools = ", ".join(data["tools_called"]) or "none"
            print(f"         tools={tools}  reflect={data['reflection_count']}  chars={len(data['reply'])}", flush=True)

    print(f"  Saved {n} entries → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Phase 2 — Judges
# ---------------------------------------------------------------------------
def run_judges_phase(
    config_key: str,
    n: int,
    langsmith: bool,
) -> tuple[float, float, float]:
    """Score cached replies. Returns (avg_hallucination, avg_relevance, avg_personalization_0_1)."""
    _, version_tag = CONFIG_MAP[config_key]
    cache_file = CACHE_DIR / f"{config_key}_{n}q.jsonl"

    if not cache_file.exists():
        print(f"  Cache missing: {cache_file} — skipping judges")
        return 0.0, 0.0, 0.0

    cache = load_cache(cache_file)
    print(f"  Phase 2 — judges ({config_key}, {len(cache)} entries)")

    h_scores, r_scores, p_scores = [], [], []
    for entry in cache:
        scores = score_entry(entry)
        h_scores.append(scores["hallucination"]["score"])
        r_scores.append(scores["answer_relevance"]["score"])
        p_scores.append(scores["personalization"]["score"])

    h_avg = sum(h_scores) / len(h_scores) if h_scores else 0.0
    r_avg = sum(r_scores) / len(r_scores) if r_scores else 0.0
    p_avg = sum(p_scores) / len(p_scores) if p_scores else 0.0

    print(f"    hallucination    : {h_avg:.2f}")
    print(f"    answer_relevance : {r_avg:.2f}")
    print(f"    personalization  : {p_avg * 5:.1f}/5.0")

    if langsmith:
        url = push_to_langsmith(cache, config_key, version_tag)
        if url:
            print(f"    LangSmith: {url}")

    return h_avg, r_avg, p_avg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Full ablation evaluation pipeline")
    parser.add_argument("--n", type=int, default=20, help="Questions per config (default: 20)")
    parser.add_argument("--judges-only", action="store_true",
                        help="Skip Phase 1 (agent) — use existing caches")
    parser.add_argument("--langsmith", action="store_true",
                        help="Push judge scores to LangSmith")
    args = parser.parse_args()

    model_name = os.getenv("LLM_MODEL", "kimi-k2.5")

    print("PaceGenie — Ablation Evaluation Suite")
    print("=" * 60)
    print(f"Configs    : {', '.join(ABLATION_ORDER)}")
    print(f"Questions  : {args.n} per config")
    print(f"Model      : {model_name}")
    print(f"Phase 1    : {'SKIP (--judges-only)' if args.judges_only else 'run agent'}")
    print(f"LangSmith  : {'yes' if args.langsmith else 'no'}")
    print("=" * 60)

    # (config_key, version_tag, h, rel, per_0_1)
    summary: list[tuple[str, str, float, float, float]] = []

    for config_key in ABLATION_ORDER:
        _, version_tag = CONFIG_MAP[config_key]
        print(f"\n[{config_key}] ({version_tag})")
        print("-" * 50)

        # Phase 1
        if not args.judges_only:
            run_agent_phase(config_key, args.n)

        # Phase 2
        h, rel, per = run_judges_phase(config_key, args.n, args.langsmith)
        summary.append((config_key, version_tag, h, rel, per))

    # ---------------------------------------------------------------------------
    # Comparison table
    # ---------------------------------------------------------------------------
    print(f"\n{'=' * 68}")
    print("ABLATION COMPARISON TABLE")
    print(f"{'=' * 68}")
    print(f"{'Config':<18}  {'Version':<20}  {'Hallu':>6}  {'Relev':>6}  {'Pers/5':>7}")
    print(f"{'-' * 68}")
    for config_key, version_tag, h, rel, per in summary:
        print(f"{config_key:<18}  {version_tag:<20}  {h:>6.2f}  {rel:>6.2f}  {per * 5:>7.1f}")
    print(f"{'=' * 68}")

    if len(summary) >= 2:
        best = max(summary, key=lambda r: r[4])
        baseline = next((r for r in summary if r[0] == "baseline"), None)
        if baseline and best[0] != "baseline":
            improvement = (best[4] - baseline[4]) * 5
            print(f"\nBest config : {best[0]} ({best[1]})")
            print(f"Improvement : +{improvement:.1f}/5.0 personalization vs baseline")
            print(
                f'\nResume bullet →'
                f'\n  "4-configuration ablation study: personalization score improved from '
                f'{baseline[4] * 5:.1f}/5.0 (baseline) to {best[4] * 5:.1f}/5.0 ({best[0]}) '
                f'on {args.n} Garmin-grounded questions"'
            )
        elif baseline:
            print(f"\nBaseline personalization: {baseline[4] * 5:.1f}/5.0")

    if args.langsmith:
        print(f"\nView all experiments: smith.langchain.com → PaceGenie → Experiments")
        print("Select all rows and click Compare for side-by-side view.")


if __name__ == "__main__":
    main()
