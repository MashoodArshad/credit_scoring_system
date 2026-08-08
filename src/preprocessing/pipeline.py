"""Leakage-safe sklearn preprocessing pipeline.

Builds a ``ColumnTransformer`` that routes each feature group (raw AND the
domain-engineered features from ``feature_engineering``) to the appropriate
sub-pipeline (imputation -> transformation -> scaling / encoding). The full
pipeline is fit on the training set only and reused unchanged for validation,
test, and inference -> identical transformations, zero leakage.

Design decisions (WHY each choice):
    - Monetary features: median impute -> WINSORIZE (1-99%) -> log1p -> RobustScaler.
    - Count/behavioral features: median impute -> RobustScaler (counts have
      meaningful zeros -> NOT logged).
    - Utilization ratio: median impute -> clip[0,1.5] -> StandardScaler.
    - credit_score / interest_rate: median impute -> StandardScaler.
    - Engineered financial ratios (DTI, loan burden, savings, inquiry density):
      median impute -> RobustScaler (heavy-tailed).
    - Engineered composites (FHI, risk index, consistency, stability):
      median impute -> StandardScaler (bounded ~[0,1]).
    - Engineered binary flags: most-frequent impute -> StandardScaler.
    - months_since_last_delinquency: constant sentinel=120 + missing indicator
      (the 63% missingness is STRUCTURAL) -> StandardScaler.
    - Nominal categoricals (incl. engineered credit_tier): constant impute -> OHE.
    - education (ordinal): most-frequent impute -> OrdinalEncoder (known order).
    - Protected attributes & id dropped (never model inputs).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

from src.utils import get_logger

logger = get_logger(__name__)

EDU_ORDER: tuple[str, ...] = ("High School", "Bachelor", "Master", "Doctorate")


def _clip_utilization(x):
    """Clip utilization ratio to [0, 1.5].

    Module-level (not a lambda) so the FunctionTransformer is picklable for joblib.
    """
    return np.clip(x, 0.0, 1.5)


def get_column_groups() -> dict[str, list[str]]:
    """Return domain-driven feature group definitions (raw + engineered)."""
    return {
        "monetary": [
            "monthly_income", "monthly_expenses", "savings_balance",
            "total_assets", "monthly_debt_payment", "loan_amount",
        ],
        "count": [
            "age", "dependents", "employment_years", "num_open_accounts",
            "num_credit_inquiries_6m", "num_late_payments_12m",
            "num_previous_defaults", "loan_term_months",
        ],
        "ratio": ["credit_utilization_ratio"],
        "score_rate": ["credit_score", "interest_rate"],
        # --- Engineered feature groups (created by FeatureEngineer) ---
        "financial_ratio": [
            "dti_ratio", "loan_monthly_burden", "savings_to_income", "credit_inquiry_density",
        ],
        "composite": [
            "financial_health_index", "risk_index", "payment_consistency", "income_stability",
        ],
        "flag": ["maxed_out_flag", "has_delinquency_history"],
        # --- Special / categorical ---
        "months_since_delinq": ["months_since_last_delinquency"],
        "nominal": ["employment_status", "loan_purpose", "credit_tier"],
        "ordinal": ["education"],
    }


# --------------------------------------------------------------------------- #
# Custom transformers (sklearn-compatible, support get_feature_names_out)
# --------------------------------------------------------------------------- #
class LogTransformer(BaseEstimator, TransformerMixin):
    """Apply ``log1p`` element-wise (right-skewed -> approximately normal)."""

    def fit(self, X, y=None):  # noqa: ANN001 - sklearn signature
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):  # noqa: ANN001
        arr = np.asarray(X, dtype=float)
        return np.log1p(np.clip(arr, 0.0, None))  # clip guards against <=0

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features)


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to learned [lower, upper] quantiles (fit on train only)."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99) -> None:
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):  # noqa: ANN001
        arr = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(arr, self.lower, axis=0)
        self.hi_ = np.nanquantile(arr, self.upper, axis=0)
        self.n_features_in_ = arr.shape[1]
        return self

    def transform(self, X):  # noqa: ANN001
        arr = np.asarray(X, dtype=float)
        return np.clip(arr, self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features)


# --------------------------------------------------------------------------- #
# Pipeline builder
# --------------------------------------------------------------------------- #
def build_preprocessing_pipeline() -> ColumnTransformer:
    """Construct the full leakage-safe preprocessing ``ColumnTransformer``.

    Returns:
        An unfitted ``ColumnTransformer`` (expects engineered columns to exist
        in its input, i.e. used after ``FeatureEngineer``).
    """
    g = get_column_groups()

    monetary = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("winsorize", Winsorizer(lower=0.01, upper=0.99)),
        ("log", LogTransformer()),
        ("scale", RobustScaler()),
    ])

    count = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
    ])

    ratio = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clip", FunctionTransformer(_clip_utilization, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])

    score_rate = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    financial_ratio = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
    ])

    composite = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    flag = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("scale", StandardScaler()),
    ])

    months = Pipeline([
        ("impute", SimpleImputer(
            strategy="constant", fill_value=120, add_indicator=True,
        )),
        ("scale", StandardScaler()),
    ])

    nominal = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    ordinal = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ord", OrdinalEncoder(
            categories=[list(EDU_ORDER)],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("monetary", monetary, g["monetary"]),
            ("count", count, g["count"]),
            ("ratio", ratio, g["ratio"]),
            ("score_rate", score_rate, g["score_rate"]),
            ("financial_ratio", financial_ratio, g["financial_ratio"]),
            ("composite", composite, g["composite"]),
            ("flag", flag, g["flag"]),
            ("months", months, g["months_since_delinq"]),
            ("nominal", nominal, g["nominal"]),
            ("ordinal", ordinal, g["ordinal"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    logger.info("Preprocessing ColumnTransformer built with %d feature groups.", len(g))
    return preprocessor


def prepare_features(
    df: pd.DataFrame,
    target: str = "creditworthy",
    id_col: str = "customer_id",
    protected: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into model features X and target y.

    Drops the identifier and protected-attribute columns (never model inputs).

    Args:
        df: Cleaned dataframe.
        target: Name of the target column.
        id_col: Identifier column to drop.
        protected: Protected-attribute columns to drop.

    Returns:
        ``(X, y)`` where X is the feature matrix and y the binary target.
    """
    drop_cols = [c for c in [id_col, target, *(protected or [])] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[target]
    logger.info("Prepared features: X=%s, y=%s. Dropped: %s", X.shape, y.shape, drop_cols)
    return X, y


if __name__ == "__main__":
    cfg_groups = get_column_groups()
    total_cols = sum(len(v) for v in cfg_groups.values())
    print("ColumnTransformer feature groups:")
    for name, cols in cfg_groups.items():
        print(f"  {name:<20} ({len(cols):>2}): {cols}")
    print(f"\nTotal routed columns: {total_cols}")
    print("(Full fit/transform verification runs in src.feature_engineering)")
