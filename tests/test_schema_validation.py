"""Tests for input schema validation and the scoring service."""
import pandas as pd
import pytest

from src.inference.exceptions import InvalidInputError, MissingColumnsError


def test_valid_input_passes_validation(service, applicant_batch):
    out = service.validate(applicant_batch.head(5))
    assert len(out) == 5
    assert "credit_score" in out.columns


def test_missing_column_raises(service, applicant_batch):
    bad = applicant_batch.head(2).drop(columns=["credit_score"])
    with pytest.raises(MissingColumnsError):
        service.validate(bad)


def test_non_numeric_value_raises(service, applicant_batch):
    bad = applicant_batch.head(2).assign(age=["twenty", "thirty"])
    with pytest.raises(InvalidInputError):
        service.validate(bad)


def test_out_of_range_value_raises(service, applicant_batch):
    bad = applicant_batch.head(2).assign(credit_score=[999, 1234])
    with pytest.raises(InvalidInputError):
        service.validate(bad)


def test_unknown_category_is_graceful(service, applicant_batch):
    bad = applicant_batch.head(1).assign(loan_purpose=["Space Travel"])
    out = service.validate(bad)  # should NOT raise
    assert len(out) == 1


def test_predict_returns_required_outputs(service, applicant_batch):
    out = service.predict(applicant_batch.head(3))
    for col in ("p_creditworthy", "p_default", "decision", "risk_tier"):
        assert col in out.columns
    assert len(out) == 3


def test_predict_single_includes_reasons(service, applicant_batch):
    record = service.predict_single(applicant_batch.iloc[0].to_dict())
    assert record["decision"] in {"Approve", "Reject"}
    assert isinstance(record["reasons"], list) and len(record["reasons"]) > 0
