"""Shared pytest fixtures for the credit scoring test suite."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.preprocessing.cleaner import clean_data
from src.preprocessing.data_generator import generate_credit_dataset
from src.preprocessing.pipeline import prepare_features


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    """A small, reproducible synthetic dataset (incl. injected quality issues)."""
    return generate_credit_dataset(n_samples=600, seed=42, target_default_rate=0.18)


@pytest.fixture(scope="session")
def cleaned_df(synthetic_df: pd.DataFrame) -> pd.DataFrame:
    """Row-level cleaned data (duplicates / out-of-range removed)."""
    return clean_data(synthetic_df, target="creditworthy", id_col="customer_id")


@pytest.fixture(scope="session")
def applicant_batch(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """A batch of raw applicants (21 model-input features, 50 rows)."""
    from src.config import get_config

    cfg = get_config()
    X, _ = prepare_features(
        cleaned_df, target="creditworthy", protected=cfg.get("protected_attributes", [])
    )
    return X.head(50).reset_index(drop=True)


@pytest.fixture(scope="session")
def service():
    """A loaded CreditScoringService (skips if the artifact is absent)."""
    from src.config import get_config
    from src.inference.service import CreditScoringService

    cfg = get_config()
    path = Path(cfg["paths"]["models_dir"]) / "credit_scoring_logreg_v1.joblib"
    if not path.exists():
        pytest.skip(f"Model artifact not found at {path}; run Phase 13 first.")
    return CreditScoringService(path)
