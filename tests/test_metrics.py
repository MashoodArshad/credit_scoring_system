"""Unit tests for evaluation metrics."""
import numpy as np

from src.evaluation.metrics import (
    business_cost,
    compute_metrics,
    find_optimal_threshold,
    kolmogorov_smirnov,
)


def test_auc_perfect_separation():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert compute_metrics(y, p, 0.5)["roc_auc"] == 1.0


def test_auc_worst_case():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.9, 0.8, 0.2, 0.1])
    assert compute_metrics(y, p, 0.5)["roc_auc"] == 0.0


def test_business_cost_known_confusion():
    # FP=1 (approve a defaulter), FN=1 (reject a good customer).
    y = np.array([0, 0, 1, 1])
    pred = np.array([1, 0, 1, 0])
    cost = business_cost(y, pred, cost_fp=5.0, cost_fn=1.0)
    assert cost["total_cost"] == 6.0
    assert cost["fp"] == 1 and cost["fn"] == 1


def test_find_optimal_threshold_in_unit_interval():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
    threshold, _ = find_optimal_threshold(y, p, cost_fp=5.0, cost_fn=1.0)
    assert 0.0 <= threshold <= 1.0


def test_kolmogorov_smirnin_range():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert 0.0 <= kolmogorov_smirnov(y, p) <= 1.0
