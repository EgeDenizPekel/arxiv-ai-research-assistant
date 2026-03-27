"""
CLI entry point for the data pipeline.

Usage:
    poetry run python scripts/run_pipeline.py
    poetry run python scripts/run_pipeline.py --max-papers 50
"""

import argparse
import sys
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and index ArXiv papers")
    parser.add_argument(
        "--max-papers",
        type=int,
        default=100,
        help="Number of new papers to fetch (default: 100)",
    )
    args = parser.parse_args()

    summary = run_pipeline(max_papers=args.max_papers)

    print("\nPipeline summary:")
    print(f"  Papers fetched:       {summary['papers_fetched']}")
    print(f"  Papers stored:        {summary['papers_stored']}")
    print(f"  Papers failed:        {summary['papers_failed']}")
    print(f"  Chunks stored:        {summary['chunks_stored']}")


if __name__ == "__main__":
    main()
