"""
FastAPI backend for the ArXiv RAG system.

Endpoints:
    POST /query          - SSE streaming RAG response
    GET  /configs        - list retriever configurations
    GET  /eval-results   - RAGAS metrics from last evaluation run
    GET  /papers         - paginated paper list from SQLite
    GET  /health         - liveness check
"""

import json
import os
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger

from src.api.schemas import (
    ConfigInfo,
    ConfigMetrics,
    EvalResultsResponse,
    PaperItem,
    PapersResponse,
    QueryRequest,
)
from src.generation import stream_rag_response
from src.retrieval import get_retriever, list_configs

load_dotenv()

DB_PATH = Path(os.getenv("DB_PATH", "data/papers.db"))
EVAL_RESULTS_PATH = Path("eval_results.json")

CONFIG_DESCRIPTIONS = {
    "naive": "Dense-only retrieval via Qdrant cosine similarity. Fastest, no reranking.",
    "hybrid": "Dense + BM25 sparse retrieval fused with Reciprocal Rank Fusion.",
    "reranked": "Hybrid retrieval followed by cross-encoder reranking. Best quality.",
    "hyde": "Hypothetical Document Embeddings + hybrid + reranking. Experimental.",
}


# ---------------------------------------------------------------------------
# Startup: preload all models so first query is instant
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Preloading retrieval models...")
    for config in list_configs():
        get_retriever(config)  # triggers lazy model loading + caches in _model_cache
    logger.success("All models loaded. API ready.")
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ArXiv RAG API",
    description="Production RAG system over ArXiv ML/AI papers with hybrid retrieval and RAGAS evaluation.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tightened in Phase 8 for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
async def query(request: QueryRequest):
    """
    Stream a RAG response as Server-Sent Events.

    SSE event types:
        data: {"type": "sources", "chunks": [...]}   - retrieved chunks (sent first)
        data: {"type": "token",   "content": "..."}  - streamed answer token
        data: {"type": "done"}                        - stream complete
        data: {"type": "error",   "message": "..."}  - on failure
    """
    valid = list_configs()
    if request.config not in valid:
        raise HTTPException(status_code=400, detail=f"Unknown config '{request.config}'. Valid: {valid}")

    async def generate():
        try:
            async for event in stream_rag_response(
                query=request.query,
                config=request.config,
                top_k=request.top_k,
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",       # disables nginx buffering for SSE
            "Connection": "keep-alive",
        },
    )


@app.get("/configs", response_model=list[ConfigInfo])
def configs():
    """List all available retriever configurations."""
    return [
        ConfigInfo(name=name, description=CONFIG_DESCRIPTIONS.get(name, ""))
        for name in list_configs()
    ]


@app.get("/eval-results", response_model=EvalResultsResponse)
def eval_results():
    """Return RAGAS evaluation metrics from the last completed eval run."""
    if not EVAL_RESULTS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No eval results found. Run scripts/run_eval.py first.",
        )

    with open(EVAL_RESULTS_PATH) as f:
        data = json.load(f)

    return EvalResultsResponse(
        timestamp=data.get("timestamp"),
        dataset_size=data.get("dataset_size"),
        configs={
            config: ConfigMetrics(**metrics)
            for config, metrics in data.get("configs", {}).items()
        },
    )


@app.get("/papers", response_model=PapersResponse)
def papers(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: str = Query("", description="Optional title/author substring filter"),
):
    """Return a paginated list of papers in the corpus."""
    if not DB_PATH.exists():
        raise HTTPException(status_code=503, detail="Database not found. Run the pipeline first.")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    offset = (page - 1) * page_size
    if search:
        pattern = f"%{search}%"
        cur.execute(
            "SELECT COUNT(*) FROM papers WHERE title LIKE ? OR authors LIKE ?",
            (pattern, pattern),
        )
        total = cur.fetchone()[0]
        cur.execute(
            "SELECT arxiv_id, title, authors, year FROM papers WHERE title LIKE ? OR authors LIKE ? LIMIT ? OFFSET ?",
            (pattern, pattern, page_size, offset),
        )
    else:
        cur.execute("SELECT COUNT(*) FROM papers")
        total = cur.fetchone()[0]
        cur.execute(
            "SELECT arxiv_id, title, authors, year FROM papers LIMIT ? OFFSET ?",
            (page_size, offset),
        )

    rows = [PaperItem(**dict(r)) for r in cur.fetchall()]
    conn.close()

    return PapersResponse(papers=rows, total=total, page=page, page_size=page_size)
