"""
CLI entry point for the indexing step.

Embeds new chunks into Qdrant (incremental) and rebuilds the BM25 index.
Run the data pipeline first to populate the SQLite database.

Usage:
    poetry run python scripts/run_indexer.py

Prerequisites:
    docker run -p 6333:6333 qdrant/qdrant
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.indexer import run_indexer


def main() -> None:
    summary = run_indexer()

    print("\nIndexing summary:")
    print(f"  Total chunks in DB:    {summary['chunks_total']}")
    print(f"  Chunks newly embedded: {summary['chunks_embedded']}")
    print(f"  Chunks already in DB:  {summary['chunks_skipped']}")


if __name__ == "__main__":
    main()
