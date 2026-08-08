"""Unit tests for the domain feature engineer."""
import numpy as np
import pandas as pd

from src.feature_engineering import ENGINEERED_FEATURES, FeatureEngineer


def test_adds_all_engineered_features(applicant_batch):
    out = FeatureEngineer().fit_transform(applicant_batch)
    for feature in ENGINEERED_FEATURES:
        assert feature in out.columns


def test_get_feature_names_out_count(applicant_batch):
    fe = FeatureEngineer()
    fe.fit(applicant_batch)
    expected = applicant_batch.shape[1] + len(ENGINEERED_FEATURES)
    assert fe.get_feature_names_out().shape[0] == expected


def test_dti_ratio_matches_definition(applicant_batch):
    out = FeatureEngineer().fit_transform(applicant_batch)
    income = applicant_batch["monthly_income"].replace(0, np.nan)
    expected = (applicant_batch["monthly_debt_payment"] / income).replace(
        [np.inf, -np.inf], np.nan
    )
    pd.testing.assert_series_equal(
        out["dti_ratio"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False, check_dtype=False,
    )


def test_no_remaining_infinities_in_ratios(applicant_batch):
    out = FeatureEngineer().fit_transform(applicant_batch)
    for col in ("dti_ratio", "loan_monthly_burden", "savings_to_income", "credit_inquiry_density"):
        assert not np.isinf(out[col]).any(), f"{col} contains infinities"
