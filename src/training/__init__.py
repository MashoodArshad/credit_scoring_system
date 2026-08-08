"""Training subpackage: data splitting, model building, and tuning."""

from src.training.data_splitting import (
    LEAKAGE_PROTOCOL,
    build_cv,
    build_split_report,
    class_distribution,
    split_dataframe,
)
from src.training.models import MODELS_REFERENCE, get_models
from src.training.train import (
    PositionSelector,
    build_model_pipeline,
    build_model_comparison_report,
    compare_models,
    get_selected_positions,
)
from src.training.tune import (
    build_search_spaces,
    build_tuning_report,
    refine_grid,
    tune_randomized,
    xgboost_early_stopping,
)

__all__ = [
    "split_dataframe",
    "build_cv",
    "build_split_report",
    "class_distribution",
    "LEAKAGE_PROTOCOL",
    "get_models",
    "MODELS_REFERENCE",
    "compare_models",
    "build_model_comparison_report",
    "build_model_pipeline",
    "get_selected_positions",
    "PositionSelector",
    "tune_randomized",
    "refine_grid",
    "build_search_spaces",
    "build_tuning_report",
    "xgboost_early_stopping",
]
