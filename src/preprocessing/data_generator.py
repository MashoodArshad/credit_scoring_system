"""Synthetic credit-risk dataset generator.

Generates a realistic financial dataset where the target ``creditworthy`` is
produced from a known data-generating process (DGP) over the features, plus
realistic noise and deliberately-injected data-quality issues.

WHY a generator with an explicit DGP?
    - Reproducibility: a fixed seed -> an identical dataset on every run.
    - Realism: the target is a *noisy logistic* of the features, so the
      problem is learnable (target AUC ~0.8) but not trivially separable.
    - Controlled imperfections: we inject missing values, duplicates and
      outliers so later phases (cleaning, DQ reporting) have real work.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from src.config import get_config
from src.utils import get_logger

logger = get_logger(__name__)

CATEGORICAL_COLS: Final[tuple[str, ...]] = (
    "gender", "marital_status", "education", "employment_status", "loan_purpose",
)
INT_COLS: Final[tuple[str, ...]] = (
    "age", "dependents", "num_open_accounts", "num_credit_inquiries_6m",
    "num_late_payments_12m", "num_previous_defaults", "loan_term_months",
)
# NOTE: employment_years is intentionally NOT here — it holds fractional years
# (e.g. 2.3) and must remain float64. Only whole-number integer columns that
# may contain NaN go into the nullable Int64 group.
NULLABLE_INT_COLS: Final[tuple[str, ...]] = (
    "months_since_last_delinquency", "credit_score",
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable element-wise sigmoid function."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _calibrate_to_rate(logit: np.ndarray, target_rate: float, iters: int = 60) -> np.ndarray:
    """Shift a logit array so that ``mean(sigmoid(logit)) ~= target_rate``.

    Uses bisection on an additive intercept — robust and deterministic.

    Args:
        logit: Pre-sigmoid logits (risk scores).
        target_rate: Desired mean default probability.
        iters: Bisection iterations (60 -> ~1e-18 precision).

    Returns:
        Shifted logit array whose mean sigmoid equals ``target_rate``.
    """
    lo, hi = -10.0, 10.0
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if _sigmoid(logit + mid).mean() < target_rate:
            lo = mid
        else:
            hi = mid
    return logit + (lo + hi) / 2.0


def generate_credit_dataset(
    n_samples: int = 10_000,
    seed: int = 42,
    target_default_rate: float = 0.18,
) -> pd.DataFrame:
    """Generate a realistic synthetic credit dataset.

    Args:
        n_samples: Number of applicant records to generate.
        seed: Random seed for full reproducibility.
        target_default_rate: Approx. fraction of defaulters (creditworthy == 0).

    Returns:
        DataFrame with raw features plus ``customer_id`` and ``creditworthy``.
    """
    rng = np.random.default_rng(seed)
    n = int(n_samples)
    logger.info("Generating %d synthetic credit records (seed=%s).", n, seed)

    # ---------------- Demographics ----------------
    age = np.clip(rng.normal(40, 11, n), 21, 70).round().astype(int)
    gender = rng.choice(["Male", "Female"], n, p=[0.55, 0.45])
    marital_status = rng.choice(
        ["Single", "Married", "Divorced", "Widowed"], n, p=[0.28, 0.55, 0.13, 0.04]
    )
    dependents = np.clip(rng.poisson(1.2, n), 0, 6).astype(int)
    education = rng.choice(
        ["High School", "Bachelor", "Master", "Doctorate"], n, p=[0.34, 0.41, 0.20, 0.05]
    )
    employment_status = rng.choice(
        ["Employed", "Self-Employed", "Unemployed"], n, p=[0.70, 0.22, 0.08]
    )
    employment_years = np.clip((age - 22) * rng.uniform(0.1, 0.5, n), 0, 40).round(1).astype(float)
    employment_years[employment_status == "Unemployed"] = np.nan

    # ---------------- Financial capacity ----------------
    monthly_income = np.round(rng.lognormal(mean=math.log(50_000), sigma=0.45, size=n), -2)
    unemp_mask = employment_status == "Unemployed"
    if unemp_mask.any():
        monthly_income[unemp_mask] = np.round(
            rng.lognormal(mean=math.log(15_000), sigma=0.4, size=int(unemp_mask.sum())), -2
        )
    monthly_expenses = np.clip(monthly_income * rng.uniform(0.30, 0.80, n), 1, None).round(-2)
    savings_balance = (monthly_income * rng.uniform(1, 60, n)).round(-2)
    total_assets = (monthly_income * rng.uniform(5, 120, n)).round(-2)

    # ---------------- Credit history ----------------
    num_open_accounts = np.clip(rng.poisson(4, n), 0, 15).astype(int)
    num_credit_inquiries_6m = np.clip(rng.poisson(1.0, n), 0, 12).astype(int)
    credit_utilization_ratio = np.clip(rng.beta(2, 5, n) * 1.3, 0, 1.5).round(3)

    # ---------------- Loan request ----------------
    loan_amount = np.round(rng.lognormal(mean=math.log(250_000), sigma=0.7, size=n), -3)
    loan_term_months = rng.choice(
        [12, 24, 36, 48, 60], n, p=[0.10, 0.20, 0.35, 0.25, 0.10]
    ).astype(int)
    loan_purpose = rng.choice(
        ["Debt Consolidation", "Home", "Auto", "Education", "Personal", "Business", "Medical"],
        n, p=[0.25, 0.15, 0.12, 0.08, 0.22, 0.10, 0.08],
    )

    # ---------------- Latent risk -> behavioral signals ----------------
    def z(arr: np.ndarray) -> np.ndarray:
        """Standardize an array (NaN-aware) to zero mean / unit std."""
        arr = np.asarray(arr, dtype=float)
        return (arr - np.nanmean(arr)) / (np.nanstd(arr) + 1e-9)

    latent_risk = (
        0.90 * z(credit_utilization_ratio)
        + 0.60 * z(monthly_expenses / (monthly_income + 1.0))
        - 0.50 * z(savings_balance)
        - 0.45 * z(monthly_income)
        + 0.30 * z(num_credit_inquiries_6m.astype(float))
        + 0.20 * unemp_mask.astype(float)
    )

    p_late = _sigmoid(latent_risk * 0.9 - 0.2)
    num_late_payments_12m = np.clip((rng.poisson(2, n) * p_late).astype(int), 0, 12)
    p_def = _sigmoid(latent_risk * 1.1 - 2.2)
    num_previous_defaults = np.clip((rng.poisson(1, n) * p_def).astype(int), 0, 4)

    never_delinq = (num_late_payments_12m == 0) & (num_previous_defaults == 0)
    months_since_last_delinquency = np.where(
        never_delinq, np.nan, np.clip(rng.integers(0, 70, n), 0, 84)
    ).astype(float)

    credit_score = np.clip(720 - 70 * latent_risk + rng.normal(0, 45, n), 300, 850).round().astype(int)
    interest_rate = np.clip(8 + 6 * latent_risk + rng.normal(0, 2, n), 4, 28).round(2)
    monthly_debt_payment = np.clip(
        (num_open_accounts * 2500.0) * (0.5 + credit_utilization_ratio), 0, None
    ).round(-2)

    # ---------------- Target via data-generating process ----------------
    target_logit = (
        1.10 * z(credit_utilization_ratio)
        + 1.00 * z(num_late_payments_12m.astype(float))
        + 0.90 * z(num_previous_defaults.astype(float))
        + 0.70 * z(monthly_debt_payment / (monthly_income + 1.0))
        + 0.50 * z(num_credit_inquiries_6m.astype(float))
        + 0.40 * z(loan_amount / (monthly_income + 1.0))
        - 0.80 * z(credit_score.astype(float))
        - 0.60 * z(savings_balance)
        - 0.50 * z(np.nan_to_num(employment_years, nan=0.0))
        + rng.normal(0, 0.8, n)  # irreducible noise -> realistic separability
    )
    target_logit = _calibrate_to_rate(target_logit, target_default_rate)
    is_default = rng.random(n) < _sigmoid(target_logit)
    creditworthy = (~is_default).astype(int)
    logger.info("Realized default rate: %.2f%%.", (1 - creditworthy.mean()) * 100)

    df = pd.DataFrame({
        "customer_id": [f"CS-{i + 1:05d}" for i in range(n)],
        "age": age,
        "gender": gender,
        "marital_status": marital_status,
        "dependents": dependents,
        "education": education,
        "employment_status": employment_status,
        "employment_years": employment_years,
        "monthly_income": monthly_income,
        "monthly_expenses": monthly_expenses,
        "savings_balance": savings_balance,
        "total_assets": total_assets,
        "monthly_debt_payment": monthly_debt_payment,
        "num_open_accounts": num_open_accounts,
        "num_credit_inquiries_6m": num_credit_inquiries_6m,
        "num_late_payments_12m": num_late_payments_12m,
        "num_previous_defaults": num_previous_defaults,
        "months_since_last_delinquency": months_since_last_delinquency,
        "credit_utilization_ratio": credit_utilization_ratio,
        "credit_score": credit_score,
        "interest_rate": interest_rate,
        "loan_amount": loan_amount,
        "loan_term_months": loan_term_months,
        "loan_purpose": loan_purpose,
        "creditworthy": creditworthy,
    })

    df = _enforce_dtypes(df)
    df = _inject_data_quality_issues(df, rng)

    logger.info("Dataset ready: %d rows x %d cols.", *df.shape)
    return df


def _enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to memory-efficient, semantically correct dtypes."""
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    for col in INT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("int64")
    for col in NULLABLE_INT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("Int64")
    if "customer_id" in df.columns:
        df["customer_id"] = df["customer_id"].astype("string")
    return df


def _inject_data_quality_issues(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Inject realistic missing values, outliers, and duplicate rows.

    The data is *deliberately dirty* so the cleaning and DQ phases are
    meaningful rather than ceremonial.
    """
    logger.info("Injecting realistic data-quality issues...")
    # Missing values (MCAR) on selected columns.
    for col, frac in {
        "monthly_income": 0.01,
        "savings_balance": 0.03,
        "credit_utilization_ratio": 0.02,
        "credit_score": 0.005,
    }.items():
        mask = rng.random(len(df)) < frac
        df.loc[mask, col] = np.nan

    # Outliers: inflate a small fraction of incomes (extreme earners).
    income_present = df.index[df["monthly_income"].notna()]
    out_idx = rng.choice(income_present, size=max(1, int(len(df) * 0.004)), replace=False)
    df.loc[out_idx, "monthly_income"] = df.loc[out_idx, "monthly_income"] * 8.0

    # Duplicates: copy ~0.8% of rows (same customer_id -> integrity issue).
    n_dup = max(1, int(len(df) * 0.008))
    dup_idx = rng.choice(df.index, size=n_dup, replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    logger.info("Injected ~%.1f%% missing cells across 4 cols, %d duplicate rows.", 6.5, n_dup)
    return df


if __name__ == "__main__":
    cfg = get_config()
    dataset = generate_credit_dataset(
        n_samples=cfg["dataset"]["n_samples"],
        seed=cfg["project"]["random_seed"],
        target_default_rate=cfg["dataset"]["default_rate"],
    )
    raw_path = Path(cfg["paths"]["raw_data"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(raw_path, index=False)
    logger.info("Saved %d rows -> %s", len(dataset), raw_path)
