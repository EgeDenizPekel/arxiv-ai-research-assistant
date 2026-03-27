# ArXiv AI Research Assistant

A production RAG system that answers natural-language questions over a corpus of ArXiv ML/AI papers. Goes beyond tutorial RAG by implementing hybrid retrieval, cross-encoder reranking, and HyDE query rewriting - then rigorously evaluating all four retrieval configurations head-to-head using RAGAS metrics tracked in MLflow.

## What makes this different from a basic RAG

Most RAG implementations stop at: chunk PDF -> embed -> cosine similarity -> LLM answer.

This project adds three layers on top of that baseline:

| Layer | What it does |
|-------|-------------|
| Hybrid retrieval | Dense (Qdrant) + sparse (BM25) search fused via Reciprocal Rank Fusion. BM25 catches exact keyword matches that embeddings smooth over (paper titles, author names, technical terms). |
| Cross-encoder reranking | Two-stage pipeline: fast bi-encoder retrieval for candidate selection, accurate cross-encoder for final ranking. The cross-encoder reads query + document jointly, which is much more precise but too slow to run over the full index. |
| HyDE query rewriting | Instead of embedding the raw (short, sparse) query, ask the LLM to generate a hypothetical answer document and embed that. The hypothesis lives in the same semantic space as real documents. |

All four configurations are evaluated against the same 75-question curated dataset so the improvement story has actual numbers behind it.

## Architecture

```
Offline (indexing)
------------------
ArXiv API -> PDF parser (pymupdf) -> semantic chunker
                                          |
                              +-----------+-----------+
                              |                       |
                         Qdrant (dense)         BM25 index
                    BAAI/bge-large-en-v1.5       (pickled)


Online (query)
--------------
User query
    |
    v
[optional] HyDE rewrite via GPT-4o-mini
    |
    v
Hybrid retriever
  - Qdrant top-20 (dense cosine similarity)
  - BM25 top-20 (sparse keyword match)
  - Reciprocal Rank Fusion -> merged top-20
    |
    v
[optional] Cross-encoder reranker -> top-5
    |
    v
GPT-4o-mini (streamed via SSE)
    |
    v
React dashboard
  - Streamed answer
  - Retrieved chunks with scores
  - RAGAS eval metric comparison across all 4 configs
```

## Retrieval configurations

| Config | Retrieval | Rerank | Query rewriting |
|--------|-----------|--------|-----------------|
| Naive RAG | Dense only | - | - |
| Hybrid RAG | Dense + BM25 (RRF) | - | - |
| Hybrid + Rerank | Dense + BM25 (RRF) | Cross-encoder | - |
| HyDE + Hybrid + Rerank | Dense + BM25 (RRF) | Cross-encoder | HyDE |

## Evaluation results

*(To be updated with actuals after running the full eval)*

| Config | Faithfulness | Context Precision | Context Recall | Answer Relevance |
|--------|-------------|------------------|----------------|-----------------|
| Naive RAG | - | - | - | - |
| Hybrid RAG | - | - | - | - |
| Hybrid + Rerank | - | - | - | - |
| HyDE + Hybrid + Rerank | - | - | - | - |

Evaluated against a 75-question curated dataset (factual, comparative, and multi-hop questions) using [RAGAS](https://docs.ragas.io/). Experiments tracked in MLflow.

## Stack

| Layer | Tool |
|-------|------|
| Orchestration | LangChain |
| Vector store | Qdrant |
| Embeddings | `BAAI/bge-large-en-v1.5` (sentence-transformers) |
| Sparse retrieval | rank_bm25 |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generation | GPT-4o-mini |
| Evaluation | RAGAS + MLflow |
| Backend | FastAPI + SSE streaming |
| Frontend | React |
| Deployment | Docker + AWS EC2 + Vercel |

## Setup

**Prerequisites:** Python 3.11+, Poetry, Docker

```bash
git clone https://github.com/EgeDenizPekel/arxiv-ai-research-assistant
cd ai-research-assistant

poetry install

cp .env.example .env
# Add your OPENAI_API_KEY to .env

# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant
```

## Running the pipeline

```bash
# Fetch and index 100 papers (start small, then scale)
poetry run python scripts/run_pipeline.py --max-papers 100

# Scale up to full corpus
poetry run python scripts/run_pipeline.py --max-papers 1000
```

Re-runs are safe - already-indexed papers are skipped automatically.

```bash
# Start the API
poetry run uvicorn src.api.main:app --reload

# Run tests
poetry run pytest
```

## Corpus

~1,000 ArXiv papers from `cs.LG`, `cs.AI`, and `cs.CL` (2020-2024), fetched via the `arxiv` Python library. ArXiv was chosen because the domain is immediately recognizable to AI engineer interviewers - anyone can ask the system about a paper they know and verify it works.
