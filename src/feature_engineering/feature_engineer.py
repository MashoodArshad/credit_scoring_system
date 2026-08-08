"""Domain-driven feature engineering for credit scoring.

Adds financially-meaningful, expert-knowledge features computed per-row from
raw applicant data. The transformer is *fit-less* (pure row-wise arithmetic
with fixed domain bounds) -> leakage-safe and identical at train/inference time.

WHY engineer features instead of relying on raw columns alone?
    Ratios and composite indices capture *interactions* (e.g., debt relative to
    income) that linear models cannot learn from raw columns, and they encode
    expert risk logic -> better separability, better interpretability, and a
    stronger prior for smaller-data regimes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from src.preprocessing.pipeline import build_preprocessing_pipeline
from src.utils import get_logger

logger = get_logger(__name__)

# Ordered list of added features (must match assignment order in transform()).
ENGINEERED_FEATURES: tuple[str, ...] = (
    "dti_ratio",
    "loan_monthly_burden",
    "savings_to_income",
    "payment_consistency",
    "income_stability",
    "maxed_out_flag",
    "credit_inquiry_density",
    "has_delinquency_history",
    "financial_health_index",
    "risk_index",
    "credit_tier",
)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create domain-driven financial features from raw applicant data.

    Each new feature is justified by credit-risk domain logic (see inline
    comments). The transformer adds the features listed in
    :data:`ENGINEERED_FEATURES` to the input DataFrame.
    """

    def fit(self, X, y=None):  # noqa: ANN001 - sklearn signature
        # Record input feature names so the transformer is well-behaved inside a
        # Pipeline (needed for correct get_feature_names_out composition).
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        else:
            n = np.asarray(X).shape[1]
            self.feature_names_in_ = np.asarray([f"x{i}" for i in range(n)], dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X):  # noqa: ANN001
        df = X.copy()
        income = df["monthly_income"].replace(0, np.nan)  # guard divide-by-zero
        term = df["loan_term_months"].replace(0, np.nan)
        accounts = df["num_open_accounts"]

        # 1) Debt-to-Income ratio: recurring debt service vs. income capacity.
        #    Higher DTI -> less capacity to absorb a new obligation -> more risk.
        df["dti_ratio"] = df["monthly_debt_payment"] / income

        # 2) Loan burden: approximate monthly installment vs. income.
        #    Captures the affordability of *this* requested loan specifically.
        df["loan_monthly_burden"] = (df["loan_amount"] / term) / income

        # 3) Savings ratio: savings expressed as months of income (buffer size).
        #    A larger buffer protects against income shocks -> lower risk.
        df["savings_to_income"] = df["savings_balance"] / income

        # 4) Payment consistency: fewer recent lates & no defaults -> higher.
        #    Direct proxy for repayment discipline (the strongest behavioral signal).
        df["payment_consistency"] = (
            (1 - np.clip(df["num_late_payments_12m"] / 12.0, 0, 1))
            * (1 - 0.2 * np.clip(df["num_previous_defaults"], 0, 5))
        )

        # 5) Income stability: employment status x tenure.
        #    Stable, long-tenured income reduces default probability.
        emp = df["employment_status"]
        tenure = df["employment_years"].fillna(0.0)
        df["income_stability"] = np.where(
            emp == "Employed", np.clip(tenure / 10.0, 0, 1),
            np.where(emp == "Self-Employed", 0.6 * np.clip(tenure / 10.0, 0, 1), 0.0),
        )

        # 6) Maxed-out flag: revolving utilization > 80% (credit stress indicator).
        df["maxed_out_flag"] = (df["credit_utilization_ratio"] > 0.8).astype(int)

        # 7) Credit inquiry density: hard inquiries per open account
        #    (credit-hunting behavior -> elevated risk).
        df["credit_inquiry_density"] = df["num_credit_inquiries_6m"] / (accounts + 1)

        # 8) Delinquency history flag: any late payment or prior default.
        df["has_delinquency_history"] = (
            (df["num_late_payments_12m"] > 0) | (df["num_previous_defaults"] > 0)
        ).astype(int)

        # 9) Financial Health Index: holistic normalized score in ~[0, 1].
        #    Combines buffer, burden, discipline, utilization, and stability.
        sav = np.clip(df["savings_to_income"], 0, 60) / 60.0
        dti_health = 1 - np.clip(df["dti_ratio"], 0, 2) / 2.0
        pay = np.clip(df["payment_consistency"], 0, 1)
        util_health = 1 - np.clip(df["credit_utilization_ratio"], 0, 1)
        stab = np.clip(df["income_stability"], 0, 1)
        df["financial_health_index"] = (
            0.30 * sav + 0.25 * dti_health + 0.20 * pay + 0.15 * util_health + 0.10 * stab
        )

        # 10) Risk Index: burden + behavior composite (deliberately independent of
        #     credit_score, so it adds signal beyond the bureau score).
        df["risk_index"] = (
            0.30 * np.clip(df["dti_ratio"], 0, 2) / 2.0
            + 0.25 * np.clip(df["credit_utilization_ratio"], 0, 1.5) / 1.5
            + 0.20 * np.clip(df["num_late_payments_12m"] / 6.0, 0, 1)
            + 0.15 * np.clip(df["num_previous_defaults"] / 3.0, 0, 1)
            + 0.10 * np.clip(df["loan_monthly_burden"], 0, 1)
        )

        # 11) Customer segment: credit tier bucketed from the bureau score.
        df["credit_tier"] = pd.cut(
            df["credit_score"],
            bins=[-np.inf, 550, 650, 750, np.inf],
            labels=["Deep Subprime", "Subprime", "Near-Prime", "Prime"],
        ).astype("object")

        # Replace any inf from zero-denominators with NaN (imputed downstream).
        for col in ("dti_ratio", "loan_monthly_burden", "savings_to_income", "credit_inquiry_density"):
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

        logger.info("Feature engineering complete: added %d features.", len(ENGINEERED_FEATURES))
        return df

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_
        else:
            input_features = np.asarray(input_features, dtype=object)
        return np.concatenate([input_features, np.asarray(ENGINEERED_FEATURES, dtype=object)])


def build_full_pipeline() -> Pipeline:
    """Full pipeline: feature engineering -> preprocessing (ColumnTransformer).

    Returns:
        An unfitted ``sklearn.pipeline.Pipeline`` ready to ``.fit`` on training data.
    """
    return Pipeline([
        ("feature_engineer", FeatureEngineer()),
        ("preprocess", build_preprocessing_pipeline()),
    ])


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    from src.config import get_config
    from src.preprocessing.pipeline import prepare_features

    cfg = get_config()
    df = pd.read_csv(cfg["paths"]["processed_data"])
    protected = cfg.get("protected_attributes", [])
    X, y = prepare_features(df, target=cfg["dataset"]["target"], protected=protected)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg["split"]["test_size"],
        stratify=y, random_state=cfg["project"]["random_seed"],
    )

    pipe = build_full_pipeline()
    X_tr_t = pipe.fit_transform(X_tr)  # fit on TRAIN ONLY
    X_te_t = pipe.transform(X_te)      # apply to test (no leakage)

    names = pipe.get_feature_names_out()
    print("Full pipeline verification (feature engineering + preprocessing):")
    print(f"  X_train -> {X_tr_t.shape}   X_test -> {X_te_t.shape}")
    print(f"  Output features: {len(names)}")
    print(f"  Any NaN in transformed train? {np.isnan(X_tr_t).any()}")

    fe = FeatureEngineer()
    X_fe = fe.fit_transform(X)
    print("\nEngineered feature quick stats:")
    print(X_fe[list(ENGINEERED_FEATURES)].describe().T[["mean", "std", "min", "max"]].round(3).to_string())
