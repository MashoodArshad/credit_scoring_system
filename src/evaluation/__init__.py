"""Evaluation subpackage: metrics, plots, and end-to-end evaluation."""

from src.evaluation.evaluate import (
    build_evaluation_report,
    evaluate_all_models,
    recommend_model,
)
from src.evaluation.explainability import (
    build_explainability_report,
    fit_preprocessing,
    lime_local_explanations,
    load_final_models,
    logreg_reason_codes,
    plot_logreg_coefficients,
    plot_partial_dependence,
    plot_permutation_importance,
    plot_shap_summary,
)
from src.evaluation.metrics import (
    business_cost,
    compute_metrics,
    confusion_components,
    find_optimal_threshold,
    kolmogorov_smirnov,
)
from src.evaluation.plots import (
    plot_calibration,
    plot_confusion_matrix,
    plot_gain_chart,
    plot_lift_chart,
    plot_pr_curves,
    plot_roc_curves,
    plot_threshold_cost,
)

__all__ = [
    "compute_metrics",
    "confusion_components",
    "business_cost",
    "find_optimal_threshold",
    "kolmogorov_smirnov",
    "evaluate_all_models",
    "recommend_model",
    "build_evaluation_report",
    "plot_roc_curves",
    "plot_pr_curves",
    "plot_confusion_matrix",
    "plot_calibration",
    "plot_lift_chart",
    "plot_gain_chart",
    "plot_threshold_cost",
]
