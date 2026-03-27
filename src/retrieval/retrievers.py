import os
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

from .device import get_device
from .indexer import COLLECTION_NAME, EMBEDDING_MODEL, QDRANT_URL, load_bm25_index

load_dotenv()

DENSE_CANDIDATE_K = 20   # candidates fetched from each source before fusion/rerank
SPARSE_CANDIDATE_K = 20
FINAL_TOP_K = 5
RRF_K = 60               # standard RRF constant

# ---------------------------------------------------------------------------
# Shared model cache - loaded once, reused across all retriever instances
# ---------------------------------------------------------------------------

_model_cache: dict = {}


def _get_embedding_model() -> SentenceTransformer:
    if "embedder" not in _model_cache:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _model_cache["embedder"] = SentenceTransformer(EMBEDDING_MODEL, device=get_device())
    return _model_cache["embedder"]


def _get_reranker() -> CrossEncoder:
    if "reranker" not in _model_cache:
        logger.info("Loading cross-encoder reranker")
        _model_cache["reranker"] = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _model_cache["reranker"]


def _get_qdrant() -> QdrantClient:
    if "qdrant" not in _model_cache:
        _model_cache["qdrant"] = QdrantClient(url=QDRANT_URL)
    return _model_cache["qdrant"]


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _embed_query(text: str) -> list[float]:
    model = _get_embedding_model()
    return model.encode(text, normalize_embeddings=True).tolist()


def _qdrant_search(query_vector: list[float], top_k: int) -> list[dict]:
    """Dense search in Qdrant. Returns list of chunk dicts with 'score' added."""
    client = _get_qdrant()
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return [
        {**r.payload, "id": r.id, "score": r.score}
        for r in results
    ]


def _bm25_search(query: str, top_k: int) -> list[dict]:
    """Sparse BM25 search. Returns list of chunk dicts with 'score' added."""
    index = load_bm25_index()
    bm25 = index["bm25"]
    chunks = index["chunks"]

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        {**chunks[i], "score": float(scores[i])}
        for i in top_indices
    ]


def _reciprocal_rank_fusion(
    rankings: list[list[dict]],
    k: int = RRF_K,
) -> list[dict]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.
    Each list is ordered best-first. Deduplicates by chunk id.
    Returns merged list ordered by RRF score (best-first).
    """
    rrf_scores: dict[int, float] = {}
    chunk_by_id: dict[int, dict] = {}

    for ranking in rankings:
        for rank, chunk in enumerate(ranking):
            cid = chunk["id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            chunk_by_id[cid] = chunk

    merged = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
    return [
        {**chunk_by_id[cid], "score": rrf_scores[cid]}
        for cid in merged
    ]


def _cross_encoder_rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Rerank candidates with the cross-encoder and return top_k."""
    reranker = _get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [
        {**chunk, "score": float(score)}
        for score, chunk in ranked[:top_k]
    ]


def _generate_hypothetical_document(query: str) -> str:
    """Use GPT-4o-mini to generate a hypothetical answer document for HyDE."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a machine learning researcher. Write a concise paragraph "
                    "that could appear in an academic paper directly answering the question. "
                    "Be specific and technical. Do not mention that you are generating a hypothetical document."
                ),
            },
            {"role": "user", "content": query},
        ],
        max_tokens=256,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseRetriever(ABC):
    """Common interface for all retrieval configurations."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = FINAL_TOP_K) -> list[dict]:
        """
        Retrieve the top_k most relevant chunks for the query.

        Returns:
            List of chunk dicts, each containing:
            text, title, authors, year, arxiv_id, section, chunk_index, score
        """
        ...

    @property
    @abstractmethod
    def config_name(self) -> str:
        """Unique string identifier for this retrieval configuration."""
        ...


# ---------------------------------------------------------------------------
# Concrete retrievers
# ---------------------------------------------------------------------------

class NaiveRetriever(BaseRetriever):
    """Dense-only retrieval via Qdrant cosine similarity. Baseline."""

    config_name = "naive"

    def retrieve(self, query: str, top_k: int = FINAL_TOP_K) -> list[dict]:
        query_vector = _embed_query(query)
        return _qdrant_search(query_vector, top_k=top_k)


class HybridRetriever(BaseRetriever):
    """Dense + BM25 retrieval fused with Reciprocal Rank Fusion."""

    config_name = "hybrid"

    def retrieve(self, query: str, top_k: int = FINAL_TOP_K) -> list[dict]:
        query_vector = _embed_query(query)
        dense_results = _qdrant_search(query_vector, top_k=DENSE_CANDIDATE_K)
        sparse_results = _bm25_search(query, top_k=SPARSE_CANDIDATE_K)
        merged = _reciprocal_rank_fusion([dense_results, sparse_results])
        return merged[:top_k]


class RerankedRetriever(BaseRetriever):
    """Hybrid retrieval followed by cross-encoder reranking."""

    config_name = "reranked"

    def retrieve(self, query: str, top_k: int = FINAL_TOP_K) -> list[dict]:
        query_vector = _embed_query(query)
        dense_results = _qdrant_search(query_vector, top_k=DENSE_CANDIDATE_K)
        sparse_results = _bm25_search(query, top_k=SPARSE_CANDIDATE_K)
        candidates = _reciprocal_rank_fusion([dense_results, sparse_results])
        return _cross_encoder_rerank(query, candidates, top_k=top_k)


class HyDERetriever(BaseRetriever):
    """
    Hypothetical Document Embeddings (HyDE) + hybrid retrieval + reranking.

    Generates a hypothetical answer document, embeds it, and uses that
    embedding as the dense query vector instead of the raw query.
    """

    config_name = "hyde"

    def retrieve(self, query: str, top_k: int = FINAL_TOP_K) -> list[dict]:
        logger.debug(f"HyDE: generating hypothetical document for: {query[:80]}")
        hypothetical_doc = _generate_hypothetical_document(query)
        logger.debug(f"HyDE hypothesis: {hypothetical_doc[:120]}")

        # Use the hypothetical document embedding for dense search,
        # but the original query for BM25 (lexical match should stay literal)
        hyp_vector = _embed_query(hypothetical_doc)
        dense_results = _qdrant_search(hyp_vector, top_k=DENSE_CANDIDATE_K)
        sparse_results = _bm25_search(query, top_k=SPARSE_CANDIDATE_K)
        candidates = _reciprocal_rank_fusion([dense_results, sparse_results])
        return _cross_encoder_rerank(query, candidates, top_k=top_k)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_RETRIEVER_REGISTRY: dict[str, type[BaseRetriever]] = {
    "naive": NaiveRetriever,
    "hybrid": HybridRetriever,
    "reranked": RerankedRetriever,
    "hyde": HyDERetriever,
}


def get_retriever(config: str) -> BaseRetriever:
    """
    Instantiate a retriever by config name string.

    Args:
        config: One of 'naive', 'hybrid', 'reranked', 'hyde'

    Raises:
        ValueError: if config name is not recognized
    """
    if config not in _RETRIEVER_REGISTRY:
        valid = list(_RETRIEVER_REGISTRY.keys())
        raise ValueError(f"Unknown retriever config '{config}'. Valid options: {valid}")
    return _RETRIEVER_REGISTRY[config]()


def list_configs() -> list[str]:
    """Return all available retriever config names."""
    return list(_RETRIEVER_REGISTRY.keys())
