"""Retrieval evaluation metrics for PaceGenie RAG pipeline.

Run:  uv run python evaluation/metrics.py

Outputs MRR@5 for vector-only vs hybrid search, plus per-category breakdown.
Results should be pasted into docs/METRICS.md under the experiment log section.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, TypedDict

TEST_QUERIES_PATH = Path(__file__).parent / "test_queries.json"


# ---------------------------------------------------------------------------
# TypedDicts
# ---------------------------------------------------------------------------


class QueryRecord(TypedDict):
    id: str
    query: str
    relevant_chunk_ids: list[str]
    category: str


class MrrResult(TypedDict):
    vector_mrr: float
    hybrid_mrr: float
    improvement_pct: float
    n_queries: int


class CategoryResult(TypedDict):
    category: str
    vector_mrr: float
    hybrid_mrr: float
    n_queries: int


# ---------------------------------------------------------------------------
# Core MRR computation
# ---------------------------------------------------------------------------


def _chunk_key(source: str, chunk_id: int) -> str:
    """Format a (source, chunk_id) pair as the canonical identifier string.

    Using 'source:chunk_id' matches the format in test_queries.json so the
    comparison in compute_mrr_at_k works without extra mapping.
    """
    return f"{source}:{chunk_id}"


def compute_mrr_at_k(
    queries: list[QueryRecord],
    retrieval_fn: Callable[[str], list[str]],
    k: int = 5,
) -> float:
    """Compute Mean Reciprocal Rank at k over the given queries.

    retrieval_fn must return a list of chunk identifier strings in ranked order.
    MRR@k = mean of (1 / rank) for the first relevant chunk in the top-k results;
    0 if no relevant chunk appears in top-k.
    """
    scores: list[float] = []
    for q in queries:
        retrieved = retrieval_fn(q["query"])[:k]
        relevant = set(q["relevant_chunk_ids"])
        score = 0.0
        for rank, chunk_id in enumerate(retrieved, start=1):
            if chunk_id in relevant:
                score = 1.0 / rank
                break
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Retrieval wrappers that return chunk ID strings (not text)
# ---------------------------------------------------------------------------


def _make_vector_fn(retriever) -> Callable[[str], list[str]]:
    """Wrap retriever._vector_only_ranked to return chunk ID strings."""
    def fn(query: str) -> list[str]:
        chunks = retriever._vector_only_ranked(query, top_k=5)
        return [_chunk_key(c["source"], c["chunk_id"]) for c in chunks]
    return fn


def _make_hybrid_fn(retriever) -> Callable[[str], list[str]]:
    """Wrap retriever._hybrid_ranked to return chunk ID strings."""
    def fn(query: str) -> list[str]:
        chunks = retriever._hybrid_ranked(query, top_k=5)
        return [_chunk_key(c["source"], c["chunk_id"]) for c in chunks]
    return fn


# ---------------------------------------------------------------------------
# Comparison runner
# ---------------------------------------------------------------------------


class PerQueryScores(TypedDict):
    query_id: str
    category: str
    vector_score: float
    hybrid_score: float


def compute_all_scores(
    queries: list[QueryRecord],
    retriever,
    k: int = 5,
) -> list[PerQueryScores]:
    """Embed every query exactly once and record vector + hybrid scores.

    Centralising all embed calls here means main() can derive both overall
    MRR and per-category MRR from the same list without re-embedding anything.
    """
    vector_fn = _make_vector_fn(retriever)
    hybrid_fn = _make_hybrid_fn(retriever)
    scores: list[PerQueryScores] = []

    for q in queries:
        relevant = set(q["relevant_chunk_ids"])

        v_retrieved = vector_fn(q["query"])[:k]
        v_score = next(
            (1.0 / r for r, c in enumerate(v_retrieved, 1) if c in relevant), 0.0
        )

        h_retrieved = hybrid_fn(q["query"])[:k]
        h_score = next(
            (1.0 / r for r, c in enumerate(h_retrieved, 1) if c in relevant), 0.0
        )

        scores.append(
            PerQueryScores(
                query_id=q["id"],
                category=q["category"],
                vector_score=v_score,
                hybrid_score=h_score,
            )
        )

    return scores


def _mrr_from_scores(scores: list[PerQueryScores]) -> tuple[float, float]:
    """Return (vector_mrr, hybrid_mrr) for the given score list."""
    n = len(scores)
    if n == 0:
        return 0.0, 0.0
    v = sum(s["vector_score"] for s in scores) / n
    h = sum(s["hybrid_score"] for s in scores) / n
    return v, h


def run_mrr_comparison(scores: list[PerQueryScores]) -> MrrResult:
    """Aggregate pre-computed per-query scores into an overall MrrResult.

    Accepts scores from compute_all_scores so no extra embed calls are made.
    """
    vector_mrr, hybrid_mrr = _mrr_from_scores(scores)
    improvement = (
        (hybrid_mrr - vector_mrr) / vector_mrr * 100 if vector_mrr > 0 else 0.0
    )
    return MrrResult(
        vector_mrr=round(vector_mrr, 4),
        hybrid_mrr=round(hybrid_mrr, 4),
        improvement_pct=round(improvement, 1),
        n_queries=len(scores),
    )


def run_category_breakdown(scores: list[PerQueryScores]) -> list[CategoryResult]:
    """Group pre-computed scores by category and return per-category MRR.

    Accepts scores from compute_all_scores so no extra embed calls are made.
    """
    categories = sorted({s["category"] for s in scores})
    results: list[CategoryResult] = []
    for cat in categories:
        cat_scores = [s for s in scores if s["category"] == cat]
        v_mrr, h_mrr = _mrr_from_scores(cat_scores)
        results.append(
            CategoryResult(
                category=cat,
                vector_mrr=round(v_mrr, 4),
                hybrid_mrr=round(h_mrr, 4),
                n_queries=len(cat_scores),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Load test queries, run MRR comparison, print results for METRICS.md."""
    from dotenv import load_dotenv
    load_dotenv()

    from rag.retriever import get_retriever

    queries: list[QueryRecord] = json.loads(TEST_QUERIES_PATH.read_text())
    retriever = get_retriever()

    print(f"\n{'=' * 55}")
    print("PaceGenie RAG -- MRR@5 Evaluation")
    print(f"{'=' * 55}")
    print(f"Test queries : {len(queries)}")

    # Embed every query exactly once; both overall and category results
    # are derived from this single pass to avoid hitting the rate limit twice.
    scores = compute_all_scores(queries, retriever)

    overall = run_mrr_comparison(scores)
    print(f"\nVector-only MRR@5 : {overall['vector_mrr']}")
    print(f"Hybrid      MRR@5 : {overall['hybrid_mrr']}")
    print(f"Improvement       : {overall['improvement_pct']:+.1f}%")

    print(f"\n{'--- Per-category breakdown ---'}")
    breakdown = run_category_breakdown(scores)
    for r in breakdown:
        delta = round(r["hybrid_mrr"] - r["vector_mrr"], 4)
        sign = "+" if delta >= 0 else ""
        print(
            f"  {r['category']:<18}  "
            f"vector={r['vector_mrr']:.4f}  "
            f"hybrid={r['hybrid_mrr']:.4f}  "
            f"delta={sign}{delta:.4f}  "
            f"(n={r['n_queries']})"
        )

    print(f"\n{'=' * 55}")
    print("Paste these numbers into docs/METRICS.md experiment log.")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
