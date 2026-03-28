"""FastAPI endpoint tests using TestClient with mocked dependencies."""

import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """
    TestClient fixture that patches model loading during lifespan so tests
    do not require Qdrant, BGE, or cross-encoder to be available.
    """
    with (
        patch("src.api.main.get_retriever"),
        patch("src.api.main.list_configs", return_value=["naive", "hybrid", "reranked", "hyde"]),
    ):
        from src.api.main import app
        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

def test_configs_returns_four_entries(client):
    resp = client.get("/configs")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 4
    names = {c["name"] for c in data}
    assert names == {"naive", "hybrid", "reranked", "hyde"}


def test_configs_include_descriptions(client):
    resp = client.get("/configs")
    for config in resp.json():
        assert "description" in config
        assert len(config["description"]) > 0


# ---------------------------------------------------------------------------
# Eval results
# ---------------------------------------------------------------------------

def test_eval_results_returns_valid_structure(client):
    resp = client.get("/eval-results")
    assert resp.status_code == 200
    data = resp.json()
    assert "configs" in data
    assert "dataset_size" in data
    # All four configs should be present
    for name in ["naive", "hybrid", "reranked", "hyde"]:
        assert name in data["configs"]


def test_eval_results_metrics_are_floats(client):
    resp = client.get("/eval-results")
    for config_metrics in resp.json()["configs"].values():
        for metric in ["faithfulness", "context_precision", "context_recall", "answer_relevancy"]:
            assert isinstance(config_metrics[metric], float)
            assert 0.0 <= config_metrics[metric] <= 1.0


# ---------------------------------------------------------------------------
# /query - SSE streaming with mocked stream_rag_response
# ---------------------------------------------------------------------------

async def _fake_stream(query, config, top_k):
    yield {"type": "sources", "chunks": []}
    yield {"type": "token", "content": "Test answer."}
    yield {"type": "done"}


def test_query_streams_sse_events(client):
    with patch("src.api.main.stream_rag_response", side_effect=_fake_stream):
        with patch("src.api.main.list_configs", return_value=["naive", "hybrid", "reranked", "hyde"]):
            resp = client.post(
                "/query",
                json={"query": "What is attention?", "config": "naive", "top_k": 3},
            )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]


def test_query_invalid_config_returns_400(client):
    resp = client.post(
        "/query",
        json={"query": "test question", "config": "invalid_config", "top_k": 5},
    )
    assert resp.status_code == 400
