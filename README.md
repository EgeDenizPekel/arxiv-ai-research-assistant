# ArXiv AI Research Assistant

A production RAG system that answers natural-language questions over 750 ArXiv ML/AI papers. Goes beyond tutorial RAG by implementing hybrid retrieval, cross-encoder reranking, and HyDE query rewriting - then rigorously evaluating all four retrieval configurations head-to-head using RAGAS metrics tracked in MLflow.

## What makes this different from basic RAG

Most RAG implementations stop at: chunk text -> embed -> cosine similarity -> LLM answer.

This project adds three layers on top of that baseline:

| Layer | What it does |
|-------|-------------|
| Hybrid retrieval | Dense (Qdrant) + sparse (BM25) fused via Reciprocal Rank Fusion. BM25 catches exact keyword matches that embeddings smooth over - paper titles, technical terms, acronyms. |
| Cross-encoder reranking | Two-stage pipeline: fast bi-encoder for candidate selection, accurate cross-encoder for final ranking. The cross-encoder reads query + document jointly, which is much more precise but too slow to run over the full index. |
| HyDE query rewriting | Instead of embedding the short, sparse query directly, GPT-4o-mini generates a hypothetical answer document and that gets embedded. The hypothesis lives in the same semantic space as real paper text. |

All four configurations are evaluated against the same 61-question curated dataset so the improvement story has real numbers behind it.

## Evaluation results

Evaluated on 61 questions using [RAGAS](https://docs.ragas.io/) with `gpt-4o-mini` as the LLM judge. Experiments tracked in MLflow.

| Config | Faithfulness | Ctx Precision | Ctx Recall | Ans Relevancy |
|--------|:-----------:|:-------------:|:----------:|:-------------:|
| Naive (dense only) | 83.4% | 80.5% | 80.9% | 81.8% |
| Hybrid (dense + BM25) | 88.9% | 84.5% | **89.6%** | **88.3%** |
| Reranked (hybrid + cross-encoder) | **90.3%** | **86.3%** | 87.2% | 84.4% |
| HyDE (hypothetical doc + reranked) | 89.1% | 86.9% | 85.5% | 81.6% |

Key findings:
- Reranked wins faithfulness (90.3%) - the cross-encoder filters out irrelevant context, keeping the answer grounded
- Hybrid wins context recall (89.6%) and answer relevancy (88.3%) - BM25+dense fusion recovers more relevant chunks overall
- Naive is last across all 4 metrics - validates the entire retrieval engineering effort

## Architecture

```
Offline (indexing)
------------------
HuggingFace scientific_papers (arxiv)
    -> ML/AI keyword filter
    -> LaTeX artifact cleaning
    -> semantic chunker (512 tok, 50 overlap)
         |
         +-- Qdrant (dense embeddings, BAAI/bge-large-en-v1.5, 1024-dim cosine)
         +-- BM25 index (rank_bm25, pickled)
         +-- SQLite (chunk text + paper metadata)


Online (query)
--------------
User query
    |
    v (HyDE only)
GPT-4o-mini generates hypothetical answer document
    |
    v
Hybrid retriever
  - Qdrant: top-20 by cosine similarity (uses hypothetical doc embedding for HyDE)
  - BM25:   top-20 by keyword score    (always uses original query)
  - Reciprocal Rank Fusion -> merged top-20
    |
    v (Reranked + HyDE only)
cross-encoder/ms-marco-MiniLM-L-6-v2 -> re-scores all 20 -> top-5
    |
    v
GPT-4o-mini (streamed via SSE)
    |
    v
React dashboard
  - Streamed answer with live cursor
  - Retrieved source chunks with relevance scores
  - RAGAS eval comparison chart across all 4 configs
```

## Retrieval configurations

| Config | Retrieval | Rerank | Query rewriting |
|--------|-----------|--------|-----------------|
| `naive` | Dense only (Qdrant cosine) | - | - |
| `hybrid` | Dense + BM25 via RRF | - | - |
| `reranked` | Dense + BM25 via RRF | Cross-encoder | - |
| `hyde` | Dense + BM25 via RRF | Cross-encoder | HyDE via GPT-4o-mini |

All four share a `BaseRetriever` interface and are hot-swappable at query time via the `?config=` parameter.

## Stack

| Layer | Tool |
|-------|------|
| Corpus | HuggingFace `scientific_papers` (arxiv config) |
| Embeddings | `BAAI/bge-large-en-v1.5` via sentence-transformers (1024-dim) |
| Vector store | Qdrant (Docker, port 6333) |
| Sparse retrieval | rank_bm25 |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Orchestration | LangChain |
| Generation | GPT-4o-mini (OpenAI) |
| Evaluation | RAGAS 0.2 + MLflow |
| Backend | FastAPI + SSE streaming (Python 3.12, Poetry) |
| Frontend | React + Vite + Tailwind + Recharts |
| Metadata store | SQLite |

## Setup

**Prerequisites:** Python 3.12+, Poetry, Docker

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
# Fetch, filter, clean and chunk 750 ML/AI papers into SQLite
poetry run python scripts/run_pipeline.py --max-papers 750

# Embed chunks into Qdrant + build BM25 index
poetry run python scripts/run_indexer.py
```

Re-runs are safe - already-indexed chunks are skipped in Qdrant (incremental upload). The BM25 index is always fully rebuilt (fast, ~seconds).

## Running the API and frontend

```bash
# Start the FastAPI backend (port 8000)
poetry run uvicorn src.api.main:app --reload

# In a separate terminal, start the React frontend (port 5173)
cd frontend
npm run dev
```

Open http://localhost:5173. The Vite dev server proxies all API calls to port 8000.

## Evaluation

```bash
# Generate Q&A candidates (review and prune the output manually)
poetry run python scripts/generate_eval_dataset.py

# Run RAGAS evaluation across all 4 configs, log to MLflow
poetry run python scripts/run_eval.py

# View MLflow experiment results
poetry run mlflow ui  # opens http://localhost:5000
```

## Project structure

```
ai-research-assistant/
├── src/
│   ├── pipeline/       # Fetch -> clean -> chunk -> store (SQLite)
│   ├── retrieval/      # Qdrant + BM25 indexer, 4 retriever implementations
│   ├── generation/     # LangChain RAG chain, SSE streaming + sync variants
│   ├── evaluation/     # RAGAS runner, MLflow logging, comparison table
│   └── api/            # FastAPI: /query (SSE), /configs, /eval-results, /papers
├── frontend/           # React dashboard (Vite + Tailwind + Recharts)
├── scripts/            # CLI entry points for pipeline, indexer, eval
├── tests/              # Unit + integration tests (pytest)
├── data/               # Runtime artifacts - gitignored (papers.db, bm25_index.pkl)
├── eval_dataset.json   # 61 curated Q&A pairs for RAGAS evaluation
├── eval_results.json   # Latest RAGAS scores (written by run_eval.py)
└── pyproject.toml      # Poetry dependencies
```

## Corpus details

750 ML/AI papers sourced from HuggingFace's `scientific_papers` (arxiv config) dataset. Papers are filtered by keyword matching on the abstract - 40+ ML/AI terms covering all major subfields (transformers, diffusion models, RL, NLP, computer vision, etc.). Full paper text is used (not just abstracts), giving 11,351 chunks at ~512 tokens each after cleaning.

The keyword filter was chosen over ArXiv API category filtering because the API only returns abstracts (~150-300 words), which is too thin for meaningful RAG chunking.
