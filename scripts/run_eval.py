"""
Run RAGAS evaluation across retriever configs and log results to MLflow.

Usage:
    poetry run python scripts/run_eval.py
    poetry run python scripts/run_eval.py --configs naive hybrid reranked
    poetry run python scripts/run_eval.py --dataset eval_dataset.json --top-k 5

After running, view results with:
    poetry run mlflow ui
    # then open http://localhost:5000
"""

import argparse
import sys
from pathlib import Path

from loguru import logger

from src.evaluation import run_evaluation


def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("eval_dataset.json"),
        help="Path to eval_dataset.json (default: eval_dataset.json)",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        choices=["naive", "hybrid", "reranked", "hyde"],
        help="Retriever configs to evaluate (default: all four)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per query (default: 5)",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        logger.error(
            f"Eval dataset not found at {args.dataset}.\n"
            "Generate it first with:\n"
            "  poetry run python scripts/generate_eval_dataset.py\n"
            "Then manually review the file before running evaluation."
        )
        sys.exit(1)

    logger.info(
        f"Starting evaluation | configs={args.configs or 'all'} | "
        f"top_k={args.top_k} | dataset={args.dataset}"
    )

    results = run_evaluation(
        eval_dataset_path=args.dataset,
        configs=args.configs,
        top_k=args.top_k,
    )

    logger.success("Evaluation complete. View results with: poetry run mlflow ui")
    return results


if __name__ == "__main__":
    main()
