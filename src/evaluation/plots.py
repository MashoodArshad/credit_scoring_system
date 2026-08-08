"""Evaluation visualizations: ROC, PR, confusion matrix, calibration, lift, gain, cost."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc as sk_auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from src.utils import get_logger

logger = get_logger(__name__)

sns.set_theme(style="whitegrid", font_scale=0.9)
FIG_DPI = 120
PALETTE = sns.color_palette("tab10", 12)


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure -> %s", path)


def plot_roc_curves(results: dict[str, dict], save_path: Path) -> None:
    """Overlay ROC curves for multiple models (dict: name -> {'y_true','y_proba'})."""
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, (name, data) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_proba"])
        ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)],
                label=f"{name} (AUC={sk_auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate (approve a defaulter)")
    ax.set_ylabel("True Positive Rate (approve a good)")
    ax.set_title("ROC Curves (Validation)")
    ax.legend(loc="lower right", fontsize=9)
    _save_fig(fig, save_path)


def plot_pr_curves(results: dict[str, dict], save_path: Path) -> None:
    """Overlay Precision-Recall curves for multiple models."""
    fig, ax = plt.subplots(figsize=(8, 7))
    base = None
    for i, (name, data) in enumerate(results.items()):
        prec, rec, _ = precision_recall_curve(data["y_true"], data["y_proba"])
        if base is None:
            base = (data["y_true"] == 1).mean()
        ax.plot(rec, prec, color=PALETTE[i % len(PALETTE)],
                label=f"{name} (AP={sk_auc(rec, prec):.3f})")
    if base is not None:
        ax.axhline(base, color="k", ls="--", alpha=0.5, label=f"Baseline ({base:.2f})")
    ax.set_xlabel("Recall (approve good)")
    ax.set_ylabel("Precision (of approvals)")
    ax.set_title("Precision-Recall Curves (Validation)")
    ax.legend(loc="lower left", fontsize=9)
    _save_fig(fig, save_path)


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, save_path: Path, title: str = "Confusion Matrix"
) -> None:
    """Annotated confusion matrix (positive class = creditworthy)."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Defaulter (0)", "Creditworthy (1)"],
        yticklabels=["Defaulter (0)", "Creditworthy (1)"], ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    _save_fig(fig, save_path)


def plot_calibration(
    y_true: np.ndarray, y_proba: np.ndarray, save_path: Path, name: str = "Model"
) -> None:
    """Calibration curve (reliability) + perfect-calibration diagonal."""
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(mean_pred, frac_pos, "o-", label=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted P(creditworthy)")
    ax.set_ylabel("Fraction of creditworthy (empirical)")
    ax.set_title("Calibration Curve")
    ax.legend()
    _save_fig(fig, save_path)


def _ranked_by_risk(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort by risk (ascending P(creditworthy)) and return population fraction + cumulative defaulter capture."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    order = np.argsort(y_proba)  # lowest P(good) first = highest risk first
    is_bad = (y_true[order] == 0).astype(int)
    total_bad = max(is_bad.sum(), 1)
    cum_bad = np.cumsum(is_bad)
    pop = np.arange(1, len(is_bad) + 1) / len(is_bad)
    gain = cum_bad / total_bad
    return pop, gain, is_bad


def plot_gain_chart(y_true: np.ndarray, y_proba: np.ndarray, save_path: Path) -> None:
    """Cumulative gains chart (rank by risk)."""
    pop, gain, _ = _ranked_by_risk(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(pop, gain, label="Model")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("Population reviewed (ranked by risk)")
    ax.set_ylabel("Cumulative defaulters captured")
    ax.set_title("Cumulative Gains Chart")
    ax.legend()
    _save_fig(fig, save_path)


def plot_lift_chart(y_true: np.ndarray, y_proba: np.ndarray, save_path: Path, n_bins: int = 10) -> None:
    """Cumulative lift per decile (rank by risk)."""
    pop, gain, _ = _ranked_by_risk(y_true, y_proba)
    n = len(gain)
    bin_edges = np.linspace(0, n, n_bins + 1, dtype=int)
    deciles = np.arange(1, n_bins + 1)
    lifts = []
    for k in deciles:
        idx = bin_edges[k - 1]
        frac_pop = k / n_bins
        lifts.append(gain[idx - 1] / frac_pop)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.bar(deciles * 10, lifts, width=6, color="#7c3aed", alpha=0.8)
    ax.axhline(1.0, color="k", ls="--", alpha=0.5, label="Baseline (no model)")
    ax.set_xlabel("Population reviewed by risk decile (%)")
    ax.set_ylabel("Cumulative lift")
    ax.set_title("Lift Chart")
    ax.legend()
    _save_fig(fig, save_path)


def plot_threshold_cost(
    y_true: np.ndarray, y_proba: np.ndarray, save_path: Path,
    cost_fp: float = 5.0, cost_fn: float = 1.0,
) -> tuple[float, float]:
    """Total business cost vs decision threshold; mark the cost-optimal threshold."""
    from src.evaluation.metrics import find_optimal_threshold

    thresholds = np.linspace(0.01, 0.99, 99)
    costs = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        from src.evaluation.metrics import confusion_components
        cm = confusion_components(y_true, y_pred)
        costs.append(cm["fp"] * cost_fp + cm["fn"] * cost_fn)
    best_t, best_c = find_optimal_threshold(y_true, y_proba, cost_fp, cost_fn)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(thresholds, costs, color="#b91c1c")
    ax.axvline(best_t, color="k", ls="--", label=f"Optimal threshold={best_t:.2f}")
    ax.set_xlabel("Decision threshold on P(creditworthy)")
    ax.set_ylabel(f"Total cost (FP x {cost_fp} + FN x {cost_fn})")
    ax.set_title("Business Cost vs Threshold")
    ax.legend()
    _save_fig(fig, save_path)
    return best_t, best_c
