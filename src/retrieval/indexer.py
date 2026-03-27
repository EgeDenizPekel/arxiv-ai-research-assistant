import os
import pickle
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from src.pipeline.database import get_all_chunks

load_dotenv()

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
VECTOR_DIM = 1024
EMBED_BATCH_SIZE = 64
UPLOAD_BATCH_SIZE = 256

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "arxiv_papers")

BM25_PATH = Path("data/bm25_index.pkl")


# ---------------------------------------------------------------------------
# Qdrant helpers
# ---------------------------------------------------------------------------

def _get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def _ensure_collection(client: QdrantClient) -> None:
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        logger.info(f"Created Qdrant collection '{COLLECTION_NAME}'")
    else:
        logger.info(f"Qdrant collection '{COLLECTION_NAME}' already exists")


def _get_indexed_ids(client: QdrantClient) -> set[int]:
    """Return all point IDs currently in the Qdrant collection."""
    indexed: set[int] = set()
    offset = None

    while True:
        results, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        indexed.update(p.id for p in results)
        if offset is None:
            break

    return indexed


def _upload_batch(client: QdrantClient, chunks: list[dict], embeddings: np.ndarray) -> None:
    points = [
        PointStruct(
            id=chunk["id"],
            vector=embedding.tolist(),
            payload={
                "arxiv_id": chunk["arxiv_id"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "section": chunk["section"],
                "title": chunk["title"],
                "authors": chunk["authors"],
                "year": chunk["year"],
            },
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)


# ---------------------------------------------------------------------------
# BM25 helpers
# ---------------------------------------------------------------------------

def _build_bm25_index(chunks: list[dict]) -> None:
    """Build BM25 index over all chunks and serialize to disk."""
    from rank_bm25 import BM25Okapi

    logger.info(f"Building BM25 index over {len(chunks)} chunks")
    tokenized = [chunk["text"].lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized)

    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)

    logger.info(f"BM25 index saved to {BM25_PATH}")


def load_bm25_index() -> dict:
    """Load the serialized BM25 index. Returns {'bm25': BM25Okapi, 'chunks': list[dict]}."""
    if not BM25_PATH.exists():
        raise FileNotFoundError(
            f"BM25 index not found at {BM25_PATH}. Run the indexer first."
        )
    with open(BM25_PATH, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_indexer() -> dict:
    """
    Embed new chunks and upload to Qdrant (incremental).
    Rebuild BM25 index over all chunks.

    Returns:
        Summary dict with counts.
    """
    all_chunks = get_all_chunks()
    if not all_chunks:
        logger.warning("No chunks found in DB. Run the data pipeline first.")
        return {"chunks_total": 0, "chunks_embedded": 0, "chunks_skipped": 0}

    logger.info(f"Loaded {len(all_chunks)} total chunks from DB")

    # --- Qdrant: incremental embedding ---
    client = _get_qdrant_client()
    _ensure_collection(client)

    indexed_ids = _get_indexed_ids(client)
    logger.info(f"{len(indexed_ids)} chunks already indexed in Qdrant")

    new_chunks = [c for c in all_chunks if c["id"] not in indexed_ids]
    logger.info(f"{len(new_chunks)} new chunks to embed and upload")

    chunks_embedded = 0

    if new_chunks:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        model = SentenceTransformer(EMBEDDING_MODEL, device="cuda")

        texts = [chunk["text"] for chunk in new_chunks]

        logger.info(f"Embedding {len(texts)} chunks (batch_size={EMBED_BATCH_SIZE})")
        embeddings = model.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,  # required for cosine similarity with BGE
        )

        # Upload in batches to avoid large payloads
        for start in range(0, len(new_chunks), UPLOAD_BATCH_SIZE):
            batch_chunks = new_chunks[start : start + UPLOAD_BATCH_SIZE]
            batch_embeddings = embeddings[start : start + UPLOAD_BATCH_SIZE]
            _upload_batch(client, batch_chunks, batch_embeddings)
            logger.info(
                f"Uploaded {min(start + UPLOAD_BATCH_SIZE, len(new_chunks))}/{len(new_chunks)}"
            )

        chunks_embedded = len(new_chunks)

    # --- BM25: always rebuild from all chunks ---
    _build_bm25_index(all_chunks)

    summary = {
        "chunks_total": len(all_chunks),
        "chunks_embedded": chunks_embedded,
        "chunks_skipped": len(indexed_ids),
    }

    logger.info(
        f"Indexing complete: {chunks_embedded} new chunks embedded, "
        f"{len(indexed_ids)} skipped, BM25 rebuilt over {len(all_chunks)} chunks"
    )
    return summary
