"""Preprocessing subpackage: data generation, loading, inspection, cleaning, pipeline."""

from src.preprocessing.cleaner import clean_data
from src.preprocessing.data_generator import generate_credit_dataset
from src.preprocessing.data_inspection import (
    basic_overview,
    categorical_summary,
    column_summary,
    duplicates_report,
    generate_data_quality_report,
    missing_report,
)
from src.preprocessing.pipeline import (
    Winsorizer,
    LogTransformer,
    build_preprocessing_pipeline,
    get_column_groups,
    prepare_features,
)

__all__ = [
    "generate_credit_dataset",
    "basic_overview",
    "categorical_summary",
    "column_summary",
    "duplicates_report",
    "generate_data_quality_report",
    "missing_report",
    "clean_data",
    "build_preprocessing_pipeline",
    "get_column_groups",
    "prepare_features",
    "Winsorizer",
    "LogTransformer",
]
