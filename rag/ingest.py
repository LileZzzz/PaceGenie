"""Script for loading knowledge-base Markdown files into pgvector.

Run directly:  python -m rag.ingest
The script reads all .md files from data/knowledge/, splits each into chunks,
embeds them via Google Gemini (gemini-embedding-2-preview), and stores the
results in the knowledge_chunks table.
A quick search test at the end verifies that retrieval is working.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer, String, Text, text
from sqlalchemy.orm import DeclarativeBase, Session

from rag.embeddings import EMBEDDING_DIM, embed_texts, get_db_engine, make_splitter
from rag.retriever import get_retriever

KNOWLEDGE_DIR = Path(__file__).parent.parent / "data" / "knowledge"


# ---------------------------------------------------------------------------
# ORM model
# ---------------------------------------------------------------------------


class _Base(DeclarativeBase):
    pass


class KnowledgeChunk(_Base):
    """One text chunk from a knowledge-base document, stored with its vector."""

    __tablename__ = "knowledge_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String(255), nullable=False)
    chunk_id = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(EMBEDDING_DIM), nullable=False)


# ---------------------------------------------------------------------------
# TypedDicts for internal data passing (no bare dicts)
# ---------------------------------------------------------------------------


class ChunkRecord(TypedDict):
    """In-memory representation of one chunk before it is persisted."""

    source: str
    chunk_id: int
    content: str
    embedding: list[float]


# ---------------------------------------------------------------------------
# Private helpers (each <= 30 lines)
# ---------------------------------------------------------------------------


def _read_markdown_files(directory: Path) -> dict[str, str]:
    """Read every .md file in directory and return {filename: content}.

    Sorted alphabetically so ingestion order is deterministic across runs.
    """
    return {
        md_file.name: md_file.read_text(encoding="utf-8")
        for md_file in sorted(directory.glob("*.md"))
    }


def _chunk_document(filename: str, content: str) -> list[ChunkRecord]:
    """Split a document into text chunks and embed them.

    Separating chunking from DB writes lets us batch embedding calls per file,
    which is cheaper than one OpenAI call per chunk.
    """
    splitter = make_splitter(chunk_size=500, overlap=50)
    chunks = splitter.split_text(content)
    embeddings = embed_texts(chunks)
    return [
        ChunkRecord(
            source=filename,
            chunk_id=idx,
            content=chunk,
            embedding=emb,
        )
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]


def _ensure_schema(session: Session) -> None:
    """Enable the pgvector extension and create the table if it does not exist.

    Idempotent: safe to call on every ingest run.
    """
    session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    session.commit()
    _Base.metadata.create_all(get_db_engine())


def _upsert_chunks(session: Session, records: list[ChunkRecord]) -> None:
    """Delete old rows for a source file and insert fresh chunks.

    Full replace (delete + insert) rather than UPDATE avoids stale chunks
    when the document is edited and chunk boundaries shift.
    """
    if not records:
        return
    source = records[0]["source"]
    session.execute(
        text("DELETE FROM knowledge_chunks WHERE source = :source"),
        {"source": source},
    )
    for record in records:
        session.add(
            KnowledgeChunk(
                source=record["source"],
                chunk_id=record["chunk_id"],
                content=record["content"],
                embedding=record["embedding"],
            )
        )
    session.commit()


# ---------------------------------------------------------------------------
# Verification helper
# ---------------------------------------------------------------------------


def _run_search_tests() -> None:
    """Run two acceptance queries against get_retriever() to confirm ingestion worked.

    Delegates to the canonical retriever so this test exercises the same code
    path that the agent's search_knowledge tool uses at runtime.
    """
    test_cases = [
        ("lactate threshold training", "pace_zones.md"),
        ("knee pain injury prevention", "injury_prevention.md"),
    ]
    print("\n--- Verification ---")
    retriever = get_retriever()
    all_passed = True
    for query, expected_source in test_cases:
        results = retriever.search(query, top_k=3)
        sources = [r["source"] for r in results]
        passed = expected_source in sources
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] Query: '{query}'")
        for r in results:
            print(
                f"       score={r['score']:.4f}  source={r['source']}  "
                f"chunk={r['chunk_id']}"
            )
            print(f"       preview: {r['content'][:80].strip()!r}")
        if not passed:
            all_passed = False
    if all_passed:
        print("\nAll verification tests passed.")
    else:
        print("\nSome tests failed -- check embeddings and chunk content.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def ingest_knowledge_base() -> None:
    """Read all .md files, chunk, embed, and persist them in pgvector.

    This is the only public function callers outside this module need; all
    other helpers above are private implementation details.
    """
    engine = get_db_engine()
    with Session(engine) as session:
        _ensure_schema(session)

    docs = _read_markdown_files(KNOWLEDGE_DIR)
    print(f"[ingest] Found {len(docs)} documents: {sorted(docs.keys())}")

    for filename, content in docs.items():
        print(f"[ingest] Processing {filename} ...")
        records = _chunk_document(filename, content)
        with Session(get_db_engine()) as session:
            _upsert_chunks(session, records)
        print(f"[ingest] Stored {len(records)} chunks for {filename}")

    print("[ingest] Ingestion complete.")


if __name__ == "__main__":
    from dotenv import load_dotenv

    # load_dotenv is only here for direct `python rag/ingest.py` execution.
    # When run via `uv run`, the .env is loaded automatically by uv.
    load_dotenv()
    ingest_knowledge_base()
    _run_search_tests()
