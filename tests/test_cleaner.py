"""Unit tests for the row-level data cleaner."""
import numpy as np
import pandas as pd

from src.preprocessing.cleaner import clean_data


def test_removes_duplicate_ids():
    df = pd.DataFrame({
        "customer_id": ["a", "b", "a"],
        "age": [30, 40, 30],
        "credit_score": [700, 650, 700],
        "creditworthy": [1, 0, 1],
    })
    out = clean_data(df, target="creditworthy", id_col="customer_id")
    assert len(out) == 2


def test_does_not_mutate_input():
    df = pd.DataFrame({
        "customer_id": ["a", "b"],
        "age": [30, 40],
        "credit_score": [900, 500],
        "creditworthy": [1, 0],
    })
    original = df.copy()
    clean_data(df, target="creditworthy")
    pd.testing.assert_frame_equal(df, original)


def test_clips_credit_score_to_valid_range():
    df = pd.DataFrame({
        "customer_id": ["a"], "age": [30], "credit_score": [1200], "creditworthy": [1],
    })
    out = clean_data(df, target="creditworthy")
    assert out["credit_score"].iloc[0] <= 850


def test_negative_monetary_becomes_nan():
    df = pd.DataFrame({
        "customer_id": ["a"], "age": [30], "credit_score": [700],
        "monthly_income": [-500.0], "creditworthy": [1],
    })
    out = clean_data(df, target="creditworthy")
    assert np.isnan(out["monthly_income"].iloc[0])
