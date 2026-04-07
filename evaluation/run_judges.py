"""Phase 2 — Load cached replies, run judges, optionally push to LangSmith.

Reads the JSONL cache written by run_agent.py and scores every reply with
all 3 judges. No agent calls — pure LLM-as-Judge scoring.

Optional --langsmith flag pushes the scores to a LangSmith experiment by
using a cached-replay target function (reads from the same JSONL instead
of calling the agent again).

Cache lives at: evaluation/cache/<config>_<n>q.jsonl

Usage:
    # Score locally only (fast, no LangSmith)
    uv run python evaluation/run_judges.py --config baseline --n 20

    # Score and push to LangSmith
    uv run python evaluation/run_judges.py --config baseline --n 20 --langsmith

    # After run_agent.py finished for all configs:
    uv run python evaluation/run_judges.py --config no-rag --n 20 --langsmith
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from evaluation.langsmith_eval import (
    DATASET_NAME,
    QUESTIONS,
    ensure_dataset,
    hallucination_judge,
    relevance_judge,
    personalization_judge,
)

CACHE_DIR = Path(__file__).parent / "cache"
SEP = "─" * 60


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------
def load_cache(path: Path) -> list[dict]:
    entries = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed line %d in %s: %s", lineno, path, e)
    return entries


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_entry(entry: dict) -> dict[str, dict]:
    inputs = {"question": entry["question"]}
    outputs = {"reply": entry["reply"], "context": entry.get("context", "")}
    results: dict[str, dict] = {}
    for judge_fn, name in [
        (hallucination_judge, "hallucination"),
        (relevance_judge, "answer_relevance"),
        (personalization_judge, "personalization"),
    ]:
        try:
            results[name] = judge_fn(inputs, outputs)
        except Exception as e:
            results[name] = {"key": name, "score": 0.0, "comment": f"ERROR: {e}"}
    return results


# ---------------------------------------------------------------------------
# LangSmith push — cached replay target (never calls the agent)
# ---------------------------------------------------------------------------
def _make_cached_target(cache: list[dict]):
    """Return a evaluate()-compatible target that replays answers from cache."""
    lookup = {entry["qid"]: entry for entry in cache}

    def _replay(inputs: dict) -> dict:
        # LangSmith example inputs include session_id == qid
        qid = inputs.get("session_id", "")
        entry = lookup.get(qid)
        if entry:
            return {"reply": entry["reply"], "context": entry.get("context", "")}
        return {"reply": "", "context": ""}

    return _replay


def push_to_langsmith(cache: list[dict], config_key: str, version_tag: str) -> str | None:
    """Push cached scores to LangSmith. Returns results URL or None on error."""
    try:
        from langsmith import Client
        from langsmith.evaluation import evaluate
    except ImportError:
        print("  langsmith not installed — skipping LangSmith push")
        return None

    if not os.getenv("LANGSMITH_API_KEY"):
        print("  LANGSMITH_API_KEY not set — skipping LangSmith push")
        return None

    client = Client()
    ensure_dataset(client)

    experiment_prefix = f"pacegenie-{version_tag}"
    print(f"  Pushing to LangSmith: {experiment_prefix}")

    target = _make_cached_target(cache)
    results = evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[hallucination_judge, relevance_judge, personalization_judge],
        experiment_prefix=experiment_prefix,
        metadata={
            "model": os.getenv("LLM_MODEL", "kimi-k2.5"),
            "version": version_tag,
            "ablation_config": config_key,
            "description": "scored from cache via run_judges.py",
        },
        max_concurrency=1,
    )
    return results.url


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def _bar(score: float, width: int = 10) -> str:
    filled = int(round(score * width))
    return "█" * filled + "░" * (width - filled)


def print_entry_scores(entry: dict, scores: dict[str, dict]) -> None:
    print(f"\n{'=' * 60}")
    print(f"[{entry['qid']}] {entry['question'][:70]}")
    tools = ", ".join(entry.get("tools_called", [])) or "none"
    print(f"tools={tools}  reflect={entry.get('reflection_count', 0)}  chars={len(entry.get('reply', ''))}")
    print(SEP)
    print("SCORES:")
    for name, res in scores.items():
        score = res["score"]
        comment = res.get("comment", "")
        suffix = f"  ({score * 5:.1f}/5.0)" if name == "personalization" else ""
        print(f"  {name:<22} {_bar(score)}  {score:.2f}{suffix}  {comment}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    from agent.config import CONFIG_MAP

    parser = argparse.ArgumentParser(description="Run judges against cached agent replies")
    parser.add_argument("--config", choices=list(CONFIG_MAP.keys()), default="baseline")
    parser.add_argument("--n", type=int, default=20, help="Number of questions (default: 20)")
    parser.add_argument("--langsmith", action="store_true",
                        help="Push scores to LangSmith as a new experiment")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-question scores (default: summary only)")
    args = parser.parse_args()

    _, version_tag = CONFIG_MAP[args.config]
    cache_path = CACHE_DIR / f"{args.config}_{args.n}q.jsonl"

    if not cache_path.exists():
        print(f"Cache not found: {cache_path}")
        print(f"Run agent first:")
        print(f"  uv run python evaluation/run_agent.py --config {args.config} --n {args.n}")
        sys.exit(1)

    cache = load_cache(cache_path)
    print(f"Phase 2 — Judge Run")
    print(f"Config  : {args.config} ({version_tag})")
    print(f"Loaded  : {len(cache)} entries from {cache_path}")
    print(f"LangSmith: {'yes' if args.langsmith else 'no (local only)'}")
    print("=" * 50)

    # Score all entries
    all_scores: dict[str, list[float]] = {
        "hallucination": [],
        "answer_relevance": [],
        "personalization": [],
    }
    scored_entries: list[tuple[dict, dict]] = []

    for i, entry in enumerate(cache, 1):
        print(f"[{i:2d}/{len(cache)}] judging {entry['qid']}...", end=" ", flush=True)
        scores = score_entry(entry)
        scored_entries.append((entry, scores))

        h = scores["hallucination"]["score"]
        r = scores["answer_relevance"]["score"]
        p = scores["personalization"]["score"]
        all_scores["hallucination"].append(h)
        all_scores["answer_relevance"].append(r)
        all_scores["personalization"].append(p)
        print(f"hallu={h:.0f}  relev={r:.2f}  pers={p * 5:.1f}/5", flush=True)

    # Verbose per-entry display
    if args.verbose:
        for entry, scores in scored_entries:
            print_entry_scores(entry, scores)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"SUMMARY  config={args.config}  n={len(cache)}")
    print(SEP)
    for key, vals in all_scores.items():
        if vals:
            avg = sum(vals) / len(vals)
            bar = _bar(avg)
            if key == "personalization":
                print(f"  {key:<22} {bar}  {avg:.2f}  ({avg * 5:.1f}/5.0)")
            else:
                print(f"  {key:<22} {bar}  {avg:.2f}")
    print("=" * 60)

    # Resume bullet
    if all_scores["personalization"]:
        p_avg = sum(all_scores["personalization"]) / len(all_scores["personalization"]) * 5
        h_avg = sum(all_scores["hallucination"]) / len(all_scores["hallucination"])
        print(f'\nResume bullet →')
        print(f'  "PaceGenie {args.config}: Personalization {p_avg:.1f}/5.0, '
              f'Hallucination {h_avg:.0%} grounded ({len(cache)} Garmin-grounded questions)"')

    # Push to LangSmith
    if args.langsmith:
        print(f"\nPushing to LangSmith...")
        url = push_to_langsmith(cache, args.config, version_tag)
        if url:
            print(f"Results: {url}")


if __name__ == "__main__":
    main()
