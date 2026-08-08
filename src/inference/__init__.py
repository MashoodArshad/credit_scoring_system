"""Inference subpackage: serialization, prediction, and the scoring service."""

from src.inference.exceptions import (
    CreditScoringError,
    InvalidInputError,
    MissingColumnsError,
)
from src.inference.predict import RISK_TIERS, explain_prediction, predict_applicant
from src.inference.schema import APPLICANT_SCHEMA, REQUIRED_COLUMNS, schema_dataframe
from src.inference.serialize import (
    build_metadata,
    load_artifact,
    save_artifact,
    save_pickle,
)
from src.inference.service import CreditScoringService

__all__ = [
    "save_artifact",
    "save_pickle",
    "load_artifact",
    "build_metadata",
    "predict_applicant",
    "explain_prediction",
    "RISK_TIERS",
    "CreditScoringService",
    "APPLICANT_SCHEMA",
    "REQUIRED_COLUMNS",
    "schema_dataframe",
    "CreditScoringError",
    "MissingColumnsError",
    "InvalidInputError",
]
