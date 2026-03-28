"""
Regression tests asserting that more advanced retrieval configs outperform
the naive baseline on all RAGAS metrics from the last evaluation run.

These tests read eval_results.json directly - they do not re-run evaluation.
If eval_results.json does not exist, tests are skipped.
"""

import json
from pathlib import Path

import pytest

EVAL_RESULTS_PATH = Path("eval_results.json")
METRICS = ["faithfulness", "context_precision", "context_recall", "answer_relevancy"]


@pytest.fixture(scope="module")
def eval_scores():
    if not EVAL_RESULTS_PATH.exists():
        pytest.skip("eval_results.json not found - run scripts/run_eval.py first")
    with open(EVAL_RESULTS_PATH) as f:
        data = json.load(f)
    return data["configs"]


def test_hybrid_beats_naive_on_all_metrics(eval_scores):
    naive = eval_scores["naive"]
    hybrid = eval_scores["hybrid"]
    for metric in METRICS:
        assert hybrid[metric] > naive[metric], (
            f"hybrid ({hybrid[metric]:.4f}) did not beat naive ({naive[metric]:.4f}) "
            f"on {metric}"
        )


def test_reranked_beats_naive_on_faithfulness(eval_scores):
    assert eval_scores["reranked"]["faithfulness"] > eval_scores["naive"]["faithfulness"], (
        "reranked faithfulness did not improve over naive"
    )


def test_reranked_beats_naive_on_context_precision(eval_scores):
    assert eval_scores["reranked"]["context_precision"] > eval_scores["naive"]["context_precision"], (
        "reranked context_precision did not improve over naive"
    )


def test_no_config_has_zero_scores(eval_scores):
    """Guard against a silent evaluation failure producing all-zero metrics."""
    for config, metrics in eval_scores.items():
        for metric, value in metrics.items():
            assert value > 0.0, f"{config} has zero {metric} - evaluation may have failed silently"


def test_all_scores_in_valid_range(eval_scores):
    for config, metrics in eval_scores.items():
        for metric, value in metrics.items():
            assert 0.0 <= value <= 1.0, f"{config}.{metric} = {value} is outside [0, 1]"
