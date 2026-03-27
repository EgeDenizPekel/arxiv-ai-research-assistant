"""
RAGAS evaluation runner.

Runs all (or selected) retriever configs against eval_dataset.json,
evaluates with RAGAS 0.2 metrics, and logs results to MLflow.

Usage (via scripts/run_eval.py):
    poetry run python scripts/run_eval.py
    poetry run python scripts/run_eval.py --configs naive hybrid
    poetry run python scripts/run_eval.py --dataset path/to/eval_dataset.json
"""

import json
import os
from datetime import datetime
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from loguru import logger
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
from tqdm import tqdm

from src.generation import get_rag_response
from src.retrieval import list_configs

load_dotenv()

DEFAULT_EVAL_PATH = Path("eval_dataset.json")
DEFAULT_CONFIGS = ["naive", "hybrid", "reranked", "hyde"]
DEFAULT_TOP_K = 5
EXPERIMENT_NAME = "arxiv-rag-evaluation"
EVALUATOR_MODEL = "gpt-4o-mini"


def _build_ragas_evaluator():
    """Build the shared RAGAS LLM + embeddings evaluator (gpt-4o-mini to save cost)."""
    llm = LangchainLLMWrapper(
        ChatOpenAI(model=EVALUATOR_MODEL, api_key=os.getenv("OPENAI_API_KEY"))
    )
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    )
    return llm, embeddings


def _run_config(
    config: str,
    eval_pairs: list[dict],
    top_k: int,
    evaluator_llm,
    evaluator_embeddings,
) -> dict[str, float]:
    """
    Run a single retriever config against all eval pairs.
    Returns averaged RAGAS metrics.
    """
    logger.info(f"[{config}] Generating answers for {len(eval_pairs)} questions...")
    samples = []

    for pair in tqdm(eval_pairs, desc=f"{config}", unit="q"):
        question = pair["question"]
        ground_truth = pair["ground_truth"]

        result = get_rag_response(query=question, config=config, top_k=top_k)
        answer = result["answer"]
        contexts = [chunk["text"] for chunk in result["chunks"]]

        samples.append(
            SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
                reference=ground_truth,
            )
        )

    dataset = EvaluationDataset(samples=samples)

    logger.info(f"[{config}] Running RAGAS evaluation...")
    metrics = [
        Faithfulness(llm=evaluator_llm),
        ContextPrecision(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
    ]

    result = evaluate(dataset, metrics=metrics)

    scores = {
        "faithfulness": float(result["faithfulness"]),
        "context_precision": float(result["context_precision"]),
        "context_recall": float(result["context_recall"]),
        "answer_relevancy": float(result["answer_relevancy"]),
    }
    logger.success(f"[{config}] {scores}")
    return scores


def run_evaluation(
    eval_dataset_path: str | Path = DEFAULT_EVAL_PATH,
    configs: list[str] | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, dict[str, float]]:
    """
    Run RAGAS evaluation for all configs and log to MLflow.

    Args:
        eval_dataset_path: Path to eval_dataset.json.
        configs:           Retriever configs to evaluate. Defaults to all four.
        top_k:             Number of chunks to retrieve per query.

    Returns:
        Dict mapping config name -> {metric: score}.
    """
    eval_dataset_path = Path(eval_dataset_path)
    if not eval_dataset_path.exists():
        raise FileNotFoundError(
            f"Eval dataset not found at {eval_dataset_path}. "
            "Run scripts/generate_eval_dataset.py first."
        )

    with open(eval_dataset_path) as f:
        eval_pairs = json.load(f)

    logger.info(f"Loaded {len(eval_pairs)} eval pairs from {eval_dataset_path}")

    configs = configs or DEFAULT_CONFIGS
    valid_configs = list_configs()
    for c in configs:
        if c not in valid_configs:
            raise ValueError(f"Unknown config '{c}'. Valid: {valid_configs}")

    evaluator_llm, evaluator_embeddings = _build_ragas_evaluator()

    mlflow.set_experiment(EXPERIMENT_NAME)
    run_name = datetime.now().strftime("eval_%Y%m%d_%H%M%S")
    all_results: dict[str, dict[str, float]] = {}

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "configs": ",".join(configs),
                "top_k": top_k,
                "dataset_size": len(eval_pairs),
                "evaluator_model": EVALUATOR_MODEL,
            }
        )

        for config in configs:
            with mlflow.start_run(run_name=config, nested=True):
                mlflow.log_params({"config": config, "top_k": top_k})
                scores = _run_config(
                    config=config,
                    eval_pairs=eval_pairs,
                    top_k=top_k,
                    evaluator_llm=evaluator_llm,
                    evaluator_embeddings=evaluator_embeddings,
                )
                mlflow.log_metrics(scores)
                all_results[config] = scores

    _print_comparison_table(all_results)
    return all_results


def _print_comparison_table(results: dict[str, dict[str, float]]) -> None:
    """Print a formatted comparison table of all configs."""
    metrics = ["faithfulness", "context_precision", "context_recall", "answer_relevancy"]
    col_w = 20

    header = f"{'Config':<{col_w}}" + "".join(f"{m:<{col_w}}" for m in metrics)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for config, scores in results.items():
        row = f"{config:<{col_w}}" + "".join(
            f"{scores.get(m, 0.0):<{col_w}.4f}" for m in metrics
        )
        print(row)

    print("=" * len(header))
