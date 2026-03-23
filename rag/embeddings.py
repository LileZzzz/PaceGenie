"""Embedding utilities and pgvector database connection.

Kept separate from ingest.py so that retriever.py can import embed_query at
search time without pulling in all the file I/O and ingestion dependencies.
"""

import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import Engine, create_engine

# Module-level lazy singletons -- never initialized at import time so that
# unit tests or other modules can import this file without requiring a live
# API key or database connection.
_embeddings_client: GoogleGenerativeAIEmbeddings | None = None
_db_engine: Engine | None = None

# gemini-embedding-2-preview supports 768 / 1536 / 3072 via MRL.
# 768 is a good balance between retrieval quality and storage cost.
EMBEDDING_DIM: int = 768


def get_embeddings_client() -> GoogleGenerativeAIEmbeddings:
    """Return the shared GoogleGenerativeAIEmbeddings singleton.

    Lazy so importing this module never fails when GOOGLE_API_KEY is absent,
    e.g. in CI environments that skip RAG.
    Separate task_types (RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY) are handled
    automatically by LangChain's embed_documents / embed_query methods.
    """
    global _embeddings_client
    if _embeddings_client is None:
        _embeddings_client = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-2-preview",
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
            output_dimensionality=EMBEDDING_DIM,
        )
    return _embeddings_client


def get_db_engine() -> Engine:
    """Return the shared SQLAlchemy engine singleton.

    Lazy for the same reason as get_embeddings_client: avoids import-time
    connection failures when DATABASE_URL is not configured.
    """
    global _db_engine
    if _db_engine is None:
        db_url = os.getenv(
            "DATABASE_URL",
            "postgresql://pacegenie:password@localhost:5432/pacegenie",
        )
        _db_engine = create_engine(db_url)
    return _db_engine


def make_splitter(
    chunk_size: int = 500, overlap: int = 50
) -> RecursiveCharacterTextSplitter:
    """Build a RecursiveCharacterTextSplitter with Markdown-aware separators.

    Exposed here (not in ingest.py) so that retriever.py can reuse the same
    split parameters if it ever needs to chunk query context before re-ranking.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        # Order matters: prefer splitting on section headers before paragraphs.
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings and return their vectors.

    Used at ingest time. Centralised here so ingest.py and retriever.py
    share the same model without instantiating separate clients.
    """
    try:
        return get_embeddings_client().embed_documents(texts)
    except Exception as e:
        print(f"[embeddings] embed_texts error: {e}")
        # Return zero vectors so ingestion degrades gracefully when the API is
        # temporarily unavailable; these rows are harmless and can be re-ingested.
        return [[0.0] * EMBEDDING_DIM for _ in texts]


def embed_query(text: str) -> list[float]:
    """Embed a single query string and return its vector.

    Use for search-time embedding; embed_texts handles batch ingestion.
    """
    try:
        return get_embeddings_client().embed_query(text)
    except Exception as e:
        print(f"[embeddings] embed_query error: {e}")
        return [0.0] * EMBEDDING_DIM
