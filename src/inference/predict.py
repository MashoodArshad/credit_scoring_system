"""Inference: prediction + per-applicant reason codes from the saved artifact.

The saved pipeline accepts RAW applicant data and returns:
    - probability of creditworthiness
    - default probability
    - approve/reject decision at the cost-optimal threshold
    - risk tier
    - top contributing reason codes (LogReg coefficient x value)
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)

# Risk tiers based on predicted probability of DEFAULT (1 - P(creditworthy)).
RISK_TIERS: list[tuple[float, str]] = [
    (0.05, "Low Risk"),
    (0.15, "Medium Risk"),
    (0.30, "High Risk"),
    (1.01, "Very High Risk"),
]


def _risk_tier(p_default: float | np.ndarray) -> Any:
    """Map a default probability (scalar or array) to a risk tier."""
    def _one(p: float) -> str:
        for cutoff, label in RISK_TIERS:
            if p < cutoff:
                return label
        return RISK_TIERS[-1][1]

    arr = np.asarray(p_default, dtype=float)
    if arr.ndim == 0:
        return _one(float(arr))
    return np.array([_one(p) for p in arr], dtype=object)


def predict_applicant(
    pipeline: Any, X: pd.DataFrame, threshold: float = 0.64,
) -> pd.DataFrame:
    """Score raw applicants and return decisions, probabilities, and risk tiers."""
    proba = pipeline.predict_proba(X)[:, 1]
    p_default = 1.0 - proba
    decision = np.where(proba >= threshold, "Approve", "Reject")
    return pd.DataFrame({
        "p_creditworthy": proba.round(4),
        "p_default": p_default.round(4),
        "decision": decision,
        "risk_tier": _risk_tier(p_default),
    }, index=X.index)


def explain_prediction(pipeline: Any, X: pd.DataFrame, top_k: int = 5) -> list[pd.DataFrame]:
    """Per-applicant top reason codes (LogReg: coefficient x feature value)."""
    model = pipeline.steps[-1][1]
    if not hasattr(model, "coef_"):
        logger.warning("Final model has no coef_; reason codes unavailable (use SHAP).")
        return [pd.DataFrame() for _ in range(len(X))]

    # NOTE: avoid `pipeline[:-1]` — in newer sklearn (1.8+) it returns an UNFITTED
    # copy (NotFittedError). Instead chain through the actual fitted step objects.
    Xt = X
    names = (
        list(X.columns)
        if hasattr(X, "columns")
        else [f"f{i}" for i in range(np.asarray(X).shape[1])]
    )
    for _, step in pipeline.steps[:-1]:
        Xt = step.transform(Xt)
        try:
            names = list(step.get_feature_names_out(names))
        except Exception:
            pass

    contributions = np.asarray(Xt) * model.coef_[0]
    explanations: list[pd.DataFrame] = []
    for row in contributions:
        df = pd.DataFrame({"feature": names, "contribution": np.asarray(row).ravel()})
        df["direction"] = np.where(df["contribution"] >= 0, "favor approval", "favor rejection")
        df = df.reindex(df["contribution"].abs().sort_values(ascending=False).index)
        explanations.append(df.head(top_k).reset_index(drop=True))
    return explanations