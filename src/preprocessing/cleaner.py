"""Row-level data cleaning: deduplication, incorrect-value correction, type safety.

This module performs *data-integrity* fixes that are deterministic and
leakage-safe (they do not depend on the training distribution). The
distribution-dependent steps (imputation, winsorization, scaling, encoding)
live in the sklearn Pipeline (see ``pipeline.py``) which is fit on TRAIN ONLY
to prevent data leakage.

WHY split cleaning from the sklearn pipeline?
    - Row-level fixes (duplicates, out-of-range values) are dataset-wide truths
      that are safe to apply to all rows regardless of split.
    - Distribution-dependent steps (medians, quantiles, scaling) MUST be learned
      from the training set only -> they belong inside the sklearn Pipeline so
      the exact same transformation is reused at inference time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.config import get_config
from src.utils import get_logger

logger = get_logger(__name__)

MONETARY_COLS: tuple[str, ...] = (
    "monthly_income", "monthly_expenses", "savings_balance",
    "total_assets", "monthly_debt_payment", "loan_amount",
)


def clean_data(
    df: pd.DataFrame,
    target: str = "creditworthy",
    id_col: str = "customer_id",
    protected: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Apply deterministic, leakage-safe row-level cleaning.

    Args:
        df: Raw dataframe.
        target: Target column (preserved, never modified).
        id_col: Identifier column used for duplicate-id detection.
        protected: Protected-attribute columns (preserved for fairness audit,
            never used as model input).

    Returns:
        A cleaned copy of the dataframe (missing values remain and are handled
        later by the imputers in the sklearn pipeline).
    """
    df = df.copy()
    n_start = len(df)
    changes: list[str] = []

    # ---- 1. Duplicates: exact rows first, then duplicate identifiers ----
    full_dups = int(df.duplicated().sum())
    if full_dups:
        df = df.drop_duplicates(keep="first")
        changes.append(f"removed {full_dups} fully duplicated rows")
    if id_col in df.columns:
        id_dups = int(df.duplicated(subset=[id_col]).sum())
        if id_dups:
            df = df.drop_duplicates(subset=[id_col], keep="first")
            changes.append(f"removed {id_dups} duplicate '{id_col}' records")

    # ---- 2. Incorrect / out-of-range values ----
    if "age" in df.columns:
        bad = df["age"] < 18
        if bad.any():
            df.loc[bad, "age"] = np.nan
            changes.append(f"set {int(bad.sum())} age<18 to NaN")

    if "credit_score" in df.columns:
        df["credit_score"] = df["credit_score"].clip(300, 850)

    if "credit_utilization_ratio" in df.columns:
        df["credit_utilization_ratio"] = df["credit_utilization_ratio"].clip(0, 1.5)

    if "interest_rate" in df.columns:
        df["interest_rate"] = df["interest_rate"].clip(0, 40)

    if "loan_term_months" in df.columns:
        bad = df["loan_term_months"] <= 0
        if bad.any():
            df.loc[bad, "loan_term_months"] = np.nan
            changes.append(f"set {int(bad.sum())} non-positive terms to NaN")

    for col in MONETARY_COLS:
        if col in df.columns:
            bad = df[col] < 0
            if bad.any():
                df.loc[bad, col] = np.nan
                changes.append(f"set {int(bad.sum())} negative {col} to NaN")

    if "dependents" in df.columns:
        df.loc[df["dependents"] < 0, "dependents"] = 0

    df = df.reset_index(drop=True)
    logger.info(
        "Cleaning complete: %d -> %d rows. Changes: %s",
        n_start, len(df), "; ".join(changes) if changes else "none",
    )
    return df


if __name__ == "__main__":
    cfg = get_config()
    raw_df = pd.read_csv(cfg["paths"]["raw_data"])
    cleaned = clean_data(
        raw_df,
        target=cfg["dataset"]["target"],
        id_col="customer_id",
        protected=cfg.get("protected_attributes"),
    )
    out_path = Path(cfg["paths"]["processed_data"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_path, index=False)
    logger.info("Saved cleaned data -> %s (%d rows)", out_path, len(cleaned))
