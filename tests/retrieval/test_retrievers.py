"""Unit tests for retriever implementations and RRF fusion logic."""

from unittest.mock import MagicMock, patch

import pytest

from src.retrieval.retrievers import (
    NaiveRetriever,
    HybridRetriever,
    RerankedRetriever,
    HyDERetriever,
    _reciprocal_rank_fusion,
    get_retriever,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(id_: int, text: str = "chunk text", score: float = 0.9) -> dict:
    return {
        "id": id_,
        "text": text,
        "title": "Test Paper",
        "authors": "",
        "year": 0,
        "arxiv_id": f"hf_test{id_:03d}",
        "section": "Introduction",
        "chunk_index": 0,
        "score": score,
    }


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (pure logic - no mocking needed)
# ---------------------------------------------------------------------------

def test_rrf_single_list_preserves_order():
    chunks = [_make_chunk(i) for i in range(5)]
    result = _reciprocal_rank_fusion([chunks])
    ids = [c["id"] for c in result]
    assert ids == [0, 1, 2, 3, 4]


def test_rrf_deduplicates_across_lists():
    list_a = [_make_chunk(1), _make_chunk(2)]
    list_b = [_make_chunk(2), _make_chunk(3)]
    result = _reciprocal_rank_fusion([list_a, list_b])
    ids = [c["id"] for c in result]
    assert len(ids) == len(set(ids))  # no duplicates


def test_rrf_top_ranked_in_both_lists_wins():
    """Chunk that ranks #1 in both lists should have the highest RRF score."""
    shared = _make_chunk(99)
    list_a = [shared, _make_chunk(1), _make_chunk(2)]
    list_b = [shared, _make_chunk(3), _make_chunk(4)]
    result = _reciprocal_rank_fusion([list_a, list_b])
    assert result[0]["id"] == 99


def test_rrf_empty_lists_returns_empty():
    assert _reciprocal_rank_fusion([[], []]) == []


# ---------------------------------------------------------------------------
# NaiveRetriever
# ---------------------------------------------------------------------------

def test_naive_retriever_returns_qdrant_results():
    fake_chunks = [_make_chunk(i) for i in range(5)]
    with (
        patch("src.retrieval.retrievers._embed_query", return_value=[0.1] * 1024),
        patch("src.retrieval.retrievers._qdrant_search", return_value=fake_chunks),
    ):
        retriever = NaiveRetriever()
        results = retriever.retrieve("what is attention?", top_k=5)
    assert results == fake_chunks


def test_naive_retriever_config_name():
    assert NaiveRetriever.config_name == "naive"


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

def test_hybrid_retriever_fuses_dense_and_sparse():
    dense = [_make_chunk(1), _make_chunk(2)]
    sparse = [_make_chunk(3), _make_chunk(2)]  # chunk 2 appears in both
    with (
        patch("src.retrieval.retrievers._embed_query", return_value=[0.1] * 1024),
        patch("src.retrieval.retrievers._qdrant_search", return_value=dense),
        patch("src.retrieval.retrievers._bm25_search", return_value=sparse),
    ):
        retriever = HybridRetriever()
        results = retriever.retrieve("transformer architecture", top_k=3)
    # All unique chunks should be present
    ids = {c["id"] for c in results}
    assert ids == {1, 2, 3}


# ---------------------------------------------------------------------------
# RerankedRetriever
# ---------------------------------------------------------------------------

def test_reranked_retriever_applies_cross_encoder():
    candidates = [_make_chunk(i) for i in range(5)]
    reranked_top2 = candidates[:2]
    with (
        patch("src.retrieval.retrievers._embed_query", return_value=[0.1] * 1024),
        patch("src.retrieval.retrievers._qdrant_search", return_value=candidates),
        patch("src.retrieval.retrievers._bm25_search", return_value=[]),
        patch("src.retrieval.retrievers._cross_encoder_rerank", return_value=reranked_top2) as mock_rerank,
    ):
        retriever = RerankedRetriever()
        results = retriever.retrieve("self-attention mechanism", top_k=2)
    mock_rerank.assert_called_once()
    assert len(results) == 2


# ---------------------------------------------------------------------------
# HyDERetriever
# ---------------------------------------------------------------------------

def test_hyde_retriever_generates_hypothesis_first():
    fake_chunks = [_make_chunk(i) for i in range(5)]
    with (
        patch("src.retrieval.retrievers._generate_hypothetical_document", return_value="Hypothetical doc text.") as mock_hyp,
        patch("src.retrieval.retrievers._embed_query", return_value=[0.1] * 1024),
        patch("src.retrieval.retrievers._qdrant_search", return_value=fake_chunks),
        patch("src.retrieval.retrievers._bm25_search", return_value=[]),
        patch("src.retrieval.retrievers._cross_encoder_rerank", return_value=fake_chunks[:5]),
    ):
        retriever = HyDERetriever()
        retriever.retrieve("how does LoRA work?", top_k=5)
    mock_hyp.assert_called_once_with("how does LoRA work?")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def test_get_retriever_returns_correct_types():
    for name, expected_cls in [
        ("naive", NaiveRetriever),
        ("hybrid", HybridRetriever),
        ("reranked", RerankedRetriever),
        ("hyde", HyDERetriever),
    ]:
        assert isinstance(get_retriever(name), expected_cls)


def test_get_retriever_invalid_config_raises_value_error():
    with pytest.raises(ValueError, match="Unknown retriever config"):
        get_retriever("nonexistent_config")
