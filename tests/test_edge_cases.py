"""Edge-case tests for the scoring service."""
import numpy as np
import pandas as pd


def test_single_row_prediction(service, applicant_batch):
    out = service.predict(applicant_batch.head(1))
    assert len(out) == 1


def test_empty_dataframe_returns_empty(service, applicant_batch):
    empty = applicant_batch.iloc[0:0]
    out = service.predict(empty)
    assert len(out) == 0


def test_all_nan_row_is_handled(service, applicant_batch):
    # All-missing row: the imputers in the pipeline must handle it gracefully.
    row = applicant_batch.head(1).copy()
    for col in row.columns:
        row[col] = np.nan
    out = service.predict(row)
    assert len(out) == 1
    assert out["decision"].iloc[0] in {"Approve", "Reject"}


def test_extra_columns_are_ignored(service, applicant_batch):
    with_extra = applicant_batch.head(2).assign(unused_column=[1, 2])
    out = service.predict(with_extra)
    assert len(out) == 2


def test_extreme_valid_values_are_scored(service, applicant_batch):
    extreme = applicant_batch.head(1).copy()
    extreme["monthly_income"] = 10_000_000
    extreme["loan_amount"] = 50_000_000
    out = service.predict(extreme)
    assert len(out) == 1
