"""Unit tests for the synthetic data generator."""
import numpy as np
import pandas as pd

from src.preprocessing.data_generator import generate_credit_dataset


def test_shape_and_required_columns():
    # Arrange / Act
    df = generate_credit_dataset(n_samples=500, seed=42)
    # Assert
    assert len(df) >= 500  # duplicates are injected on top of the base count
    for col in ("customer_id", "creditworthy", "credit_score", "monthly_income"):
        assert col in df.columns


def test_target_is_binary():
    df = generate_credit_dataset(n_samples=500, seed=42)
    assert set(df["creditworthy"].unique()).issubset({0, 1})


def test_default_rate_near_target():
    df = generate_credit_dataset(n_samples=3000, seed=42, target_default_rate=0.18)
    rate = (df["creditworthy"] == 0).mean()
    # Allow sampling noise around the 18% target.
    assert 0.12 < rate < 0.25


def test_reproducibility_same_seed():
    a = generate_credit_dataset(n_samples=300, seed=7)
    b = generate_credit_dataset(n_samples=300, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_injected_quality_issues_present():
    # The generator deliberately injects missing values and duplicates.
    df = generate_credit_dataset(n_samples=2000, seed=42)
    assert df.isna().any().any() or df.duplicated().any()
