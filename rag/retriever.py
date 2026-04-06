"""Hybrid retriever: vector similarity (0.6) + BM25 keyword search (0.4).

Hybrid search outperforms pure vector search for short, keyword-heavy queries
(e.g. "IT band pain") and pure semantic queries alike, because BM25 anchors
on exact terms while the vector component handles paraphrase and synonyms.
Results are fused with Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

from typing import TypedDict

import logging

from rank_bm25 import BM25Okapi
from sqlalchemy import text
from sqlalchemy.orm import Session

from rag.embeddings import embed_query, get_db_engine

logger = logging.getLogger(__name__)

VECTOR_WEIGHT: float = 0.6
BM25_WEIGHT: float = 0.4
RRF_K: int = 60  # standard RRF constant; reduces sensitivity to rank position


class RetrievedChunk(TypedDict):
    """One search result returned to the caller."""

    source: str
    chunk_id: int
    content: str
    score: float


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def _fetch_all_chunks() -> list[RetrievedChunk]:
    """Load every chunk from the DB for BM25 corpus construction.

    BM25 needs the full corpus at query time; we load it fresh each call so
    new ingestions are immediately searchable without restarting the process.
    """
    try:
        engine = get_db_engine()
        with Session(engine) as session:
            rows = session.execute(
                text("SELECT source, chunk_id, content FROM knowledge_chunks ORDER BY id")
            ).fetchall()
        return [
            RetrievedChunk(source=r.source, chunk_id=r.chunk_id,
                           content=r.content, score=0.0)
            for r in rows
        ]
    except Exception as e:
        logger.exception("[retriever] _fetch_all_chunks error: %s", e)
        return []


def _vector_search(query_vec: list[float], top_k: int) -> list[RetrievedChunk]:
    """Return top_k chunks ordered by cosine similarity to query_vec."""
    try:
        engine = get_db_engine()
        with Session(engine) as session:
            rows = session.execute(
                text(
                    "SELECT source, chunk_id, content, "
                    "1 - (embedding <=> CAST(:vec AS vector)) AS score "
                    "FROM knowledge_chunks "
                    "ORDER BY embedding <=> CAST(:vec AS vector) "
                    "LIMIT :k"
                ),
                {"vec": str(query_vec), "k": top_k},
            ).fetchall()
        return [
            RetrievedChunk(source=r.source, chunk_id=r.chunk_id,
                           content=r.content, score=float(r.score))
            for r in rows
        ]
    except Exception as e:
        logger.exception("[retriever] _vector_search error: %s", e)
        return []


# ---------------------------------------------------------------------------
# BM25 helpers
# ---------------------------------------------------------------------------


def _tokenize(text_str: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for CJK and Latin text.

    BM25Okapi expects pre-tokenized lists; this covers Chinese + English
    without requiring a heavy NLP dependency.
    """
    import re
    # Split on whitespace and common punctuation; keep CJK characters as tokens.
    tokens = re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", text_str.lower())
    return tokens


def _bm25_search(
    query: str, corpus: list[RetrievedChunk], top_k: int
) -> list[RetrievedChunk]:
    """Score and rank the corpus against the query using BM25Okapi."""
    if not corpus:
        return []
    tokenized_corpus = [_tokenize(c["content"]) for c in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(_tokenize(query))
    scored = sorted(
        zip(scores, corpus), key=lambda x: x[0], reverse=True
    )[:top_k]
    return [
        RetrievedChunk(
            source=chunk["source"],
            chunk_id=chunk["chunk_id"],
            content=chunk["content"],
            score=float(score),
        )
        for score, chunk in scored
    ]


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------


def _rrf_fuse(
    vector_results: list[RetrievedChunk],
    bm25_results: list[RetrievedChunk],
    top_k: int,
) -> list[RetrievedChunk]:
    """Combine two ranked lists via Reciprocal Rank Fusion.

    RRF(d) = sum_over_lists( weight / (k + rank(d)) )
    Using separate weights per list to reflect that vector search is slightly
    more reliable for paraphrase-heavy queries in this domain.
    """
    rrf_scores: dict[tuple[str, int], float] = {}
    chunk_map: dict[tuple[str, int], RetrievedChunk] = {}

    for rank, chunk in enumerate(vector_results, start=1):
        key = (chunk["source"], chunk["chunk_id"])
        rrf_scores[key] = rrf_scores.get(key, 0.0) + VECTOR_WEIGHT / (RRF_K + rank)
        chunk_map[key] = chunk

    for rank, chunk in enumerate(bm25_results, start=1):
        key = (chunk["source"], chunk["chunk_id"])
        rrf_scores[key] = rrf_scores.get(key, 0.0) + BM25_WEIGHT / (RRF_K + rank)
        chunk_map[key] = chunk

    sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)[:top_k]
    return [
        RetrievedChunk(
            source=chunk_map[k]["source"],
            chunk_id=chunk_map[k]["chunk_id"],
            content=chunk_map[k]["content"],
            score=rrf_scores[k],
        )
        for k in sorted_keys
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class HybridRetriever:
    """Hybrid vector + BM25 retriever backed by pgvector.

    Exposes three public methods:
      vector_only_search -- pgvector cosine similarity only (ablation baseline)
      hybrid_search      -- vector + BM25 + RRF (production path, for nodes.py)
      search             -- same as hybrid_search but returns full RetrievedChunk
                           objects including source metadata (for tools.py)

    Instantiate once via get_retriever() to avoid re-creating the SQLAlchemy
    engine on every search call.
    """

    def vector_only_search(self, query: str, top_k: int = 5) -> list[str]:
        """Return chunk texts ranked by pgvector cosine similarity only.

        Use this as the ablation baseline to compare against hybrid_search;
        not used in the production agent path.
        """
        query_vec = embed_query(query)
        results = _vector_search(query_vec, top_k)
        return [c["content"] for c in results]

    def hybrid_search(self, query: str, top_k: int = 5) -> list[str]:
        """Return chunk texts ranked by vector + BM25 fused with RRF.

        Use for agent context injection in retrieve_context; returns plain
        text so the caller does not need to know about RetrievedChunk.
        """
        query_vec = embed_query(query)
        corpus = _fetch_all_chunks()
        vec_results = _vector_search(query_vec, top_k=top_k * 2)
        bm25_results = _bm25_search(query, corpus, top_k=top_k * 2)
        fused = _rrf_fuse(vec_results, bm25_results, top_k=top_k)
        return [c["content"] for c in fused]

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Return full RetrievedChunk objects (source + chunk_id + content + score).

        Used by tools.py search_knowledge so the LLM response can cite the
        source document; for plain text use hybrid_search instead.
        """
        query_vec = embed_query(query)
        corpus = _fetch_all_chunks()
        vec_results = _vector_search(query_vec, top_k=top_k * 2)
        bm25_results = _bm25_search(query, corpus, top_k=top_k * 2)
        return _rrf_fuse(vec_results, bm25_results, top_k=top_k)

    # ------------------------------------------------------------------
    # Evaluation helpers (used exclusively by evaluation/metrics.py)
    # ------------------------------------------------------------------

    def _vector_only_ranked(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Return ranked RetrievedChunk objects from vector-only search.

        Separated from vector_only_search so metrics.py can read chunk IDs
        without duplicating retrieval logic.
        """
        query_vec = embed_query(query)
        return _vector_search(query_vec, top_k)

    def _hybrid_ranked(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Return ranked RetrievedChunk objects from hybrid search.

        Separated from hybrid_search so metrics.py can read chunk IDs without
        duplicating retrieval logic.
        """
        query_vec = embed_query(query)
        corpus = _fetch_all_chunks()
        vec_results = _vector_search(query_vec, top_k=top_k * 2)
        bm25_results = _bm25_search(query, corpus, top_k=top_k * 2)
        return _rrf_fuse(vec_results, bm25_results, top_k=top_k)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

# Owned here so tools.py and any future caller share the same instance without
# each managing their own lazy-init boilerplate.
_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    """Return the singleton HybridRetriever, initializing on first use.

    Lazy so importing rag.retriever never touches the DB at import time.
    """
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
