"""Feature engineering subpackage: domain-driven feature creation & selection."""

from src.feature_engineering.feature_engineer import (
    ENGINEERED_FEATURES,
    FeatureEngineer,
    build_full_pipeline,
)
from src.feature_engineering.feature_selection import (
    build_feature_selection_report,
    run_feature_selection,
    select_by_mutual_information,
    select_by_permutation,
    select_by_rfe,
)

__all__ = [
    "FeatureEngineer",
    "build_full_pipeline",
    "ENGINEERED_FEATURES",
    "run_feature_selection",
    "build_feature_selection_report",
    "select_by_mutual_information",
    "select_by_permutation",
    "select_by_rfe",
]
