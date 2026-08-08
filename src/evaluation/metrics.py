"""Credit-risk evaluation metrics.

Convention: the positive class is ``creditworthy`` (1 = approve). Therefore, in a
standard confusion matrix:
    - FP = predicted good (approved) but actually a defaulter  -> the EXPENSIVE error
    - FN = predicted bad (rejected) but actually creditworthy  -> opportunity cost
This convention is stated explicitly everywhere to avoid the classic credit
metric ambiguity. Cost-based optimization favours minimizing FP.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def confusion_components(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """Return TP/TN/FP/FN for predictions (positive class = creditworthy)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}


def kolmogorov_smirnov(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """KS statistic: max separation between good/bad score CDFs (credit-industry metric)."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    pos = np.sort(y_proba[y_true == 1])
    neg = np.sort(y_proba[y_true == 0])
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    scores = np.unique(np.concatenate([pos, neg]))
    cdf_pos = np.searchsorted(pos, scores, side="right") / len(pos)
    cdf_neg = np.searchsorted(neg, scores, side="right") / len(neg)
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def compute_metrics(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Compute the full metric suite at a given decision threshold.

    Args:
        y_true: Ground-truth labels (1=creditworthy, 0=defaulter).
        y_proba: Predicted P(creditworthy).
        threshold: Decision threshold on ``y_proba`` (>= threshold -> approve).

    Returns:
        Dict of metrics (rates in [0, 1] unless noted).
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_components(y_true, y_pred)

    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
    n = tp + tn + fp + fn

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    sensitivity = recall = tp / (tp + fn) if (tp + fn) else 0.0  # P(approve|good)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0           # P(reject|bad)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    approved = y_pred == 1
    approval_rate = float(approved.mean())
    bad_rate = float((y_true[approved] == 0).mean()) if approved.sum() else 0.0

    return {
        "threshold": threshold,
        "accuracy": round((tp + tn) / n, 4),
        "precision": round(precision, 4),
        "recall_sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_proba)), 4),
        "brier": round(float(brier_score_loss(y_true, y_proba)), 4),
        "ks": round(kolmogorov_smirnov(y_true, y_proba), 4),
        "approval_rate": round(approval_rate, 4),
        "bad_rate_approved": round(bad_rate, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def business_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_fp: float = 5.0,
    cost_fn: float = 1.0,
) -> dict[str, float]:
    """Compute total business cost of a set of decisions.

    Args:
        y_true, y_pred: Labels and predicted decisions.
        cost_fp: Cost of an FP = approving a defaulter (loss given default).
        cost_fn: Cost of an FN = rejecting a good customer (lost margin).

    Returns:
        Dict with total cost, cost per applicant, and component counts.
    """
    cm = confusion_components(y_true, y_pred)
    total = cm["fp"] * cost_fp + cm["fn"] * cost_fn
    return {
        "total_cost": float(total),
        "cost_per_applicant": round(total / len(y_true), 4),
        "fp": cm["fp"],
        "fn": cm["fn"],
        "cost_fp": cost_fp,
        "cost_fn": cost_fn,
    }


def find_optimal_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, cost_fp: float = 5.0, cost_fn: float = 1.0
) -> tuple[float, float]:
    """Find the decision threshold minimizing business cost.

    Returns:
        (best_threshold, minimum_total_cost).
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = []
    for t in thresholds:
        cm = confusion_components(y_true, (y_proba >= t).astype(int))
        costs.append(cm["fp"] * cost_fp + cm["fn"] * cost_fn)
    best_idx = int(np.argmin(costs))
    return float(thresholds[best_idx]), float(costs[best_idx])
