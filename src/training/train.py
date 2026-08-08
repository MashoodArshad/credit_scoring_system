"""Model comparison harness (leakage-safe).

Builds a full Pipeline per candidate:
    FeatureEngineer -> preprocessing(ColumnTransformer) -> PositionSelector -> model
and scores it with stratified k-fold CV on the TRAIN set (preprocessing is refit
inside every fold -> no leakage), plus a single validation-set check. Results
are ranked by ROC-AUC and rendered into a comparison report.

WHY fit preprocessing inside each CV fold?
    Fitting imputers/scalers on the full training set before CV leaks fold
    statistics across folds. Putting preprocessing inside the Pipeline and using
    cross_val_score guarantees each fold is scored on data the preprocessor
    never saw -> an honest, generalization-aware ranking.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.feature_engineering import build_full_pipeline
from src.feature_engineering.feature_engineer import FeatureEngineer
from src.preprocessing.pipeline import build_preprocessing_pipeline
from src.training.data_splitting import build_cv
from src.training.models import MODELS_REFERENCE, get_models
from src.utils import get_logger

logger = get_logger(__name__)


class PositionSelector(BaseEstimator, TransformerMixin):
    """Select a fixed set of output columns by position (fit-less -> leakage-safe)."""

    def __init__(self, positions: list[int]) -> None:
        self.positions = positions

    def fit(self, X, y=None):  # noqa: ANN001
        return self

    def transform(self, X):  # noqa: ANN001
        return np.asarray(X)[:, self.positions]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features)[self.positions]


def get_selected_positions(
    X_train: pd.DataFrame, selected_names: list[str]
) -> list[int]:
    """Map selected (post-transform) feature names to output-column positions.

    Fits the preprocessing pipeline once on train to discover the deterministic
    output column order, then returns the indices of the selected features. The
    indices are data-independent (column order is fixed) so reusing them inside
    CV folds introduces no leakage.
    """
    preprocess = Pipeline([
        ("feature_engineer", FeatureEngineer()),
        ("preprocess", build_preprocessing_pipeline()),
    ])
    preprocess.fit(X_train)
    all_names = list(preprocess.get_feature_names_out())
    positions = [all_names.index(n) for n in selected_names if n in all_names]
    missing = [n for n in selected_names if n not in all_names]
    if missing:
        logger.warning("Selected features not found in pipeline output: %s", missing)
    logger.info("Mapped %d/%d selected features to positions.", len(positions), len(selected_names))
    return positions


def build_model_pipeline(model: Any, positions: list[int]) -> Pipeline:
    """Assemble FeatureEngineer -> preprocess -> selector -> model."""
    return Pipeline([
        ("feature_engineer", FeatureEngineer()),
        ("preprocess", build_preprocessing_pipeline()),
        ("select", PositionSelector(positions)),
        ("model", model),
    ])


def _scores(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return positive-class scores, falling back to decision_function."""
    try:
        return pipe.predict_proba(X)[:, 1]
    except (AttributeError, NotImplementedError):
        return pipe.decision_function(X)


def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    positions: list[int],
    seed: int = 42,
    cv_folds: int = 5,
) -> pd.DataFrame:
    """Run leakage-safe CV + validation evaluation for every candidate model.

    Args:
        X_train, y_train: Training features/labels (raw, pre-pipeline).
        X_val, y_val: Validation features/labels (raw, pre-pipeline).
        positions: Selected-feature output positions.
        seed: Random seed.
        cv_folds: Number of stratified CV folds.

    Returns:
        DataFrame sorted by CV ROC-AUC with timing and validation AUC.
    """
    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    spw = neg / max(pos, 1)
    models = get_models(seed=seed, n_jobs=-1, scale_pos_weight=spw)
    cv = build_cv(n_splits=cv_folds, random_state=seed)

    rows: list[dict[str, Any]] = []
    for name, model in models.items():
        logger.info("Evaluating %s ...", name)
        try:
            cv_pipe = build_model_pipeline(clone(model), positions)
            t0 = time.perf_counter()
            cv_scores = cross_val_score(
                cv_pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1,
            )
            cv_time = time.perf_counter() - t0

            val_pipe = build_model_pipeline(clone(model), positions)
            val_pipe.fit(X_train, y_train)
            val_auc = float(roc_auc_score(y_val, _scores(val_pipe, X_val)))

            rows.append({
                "model": name,
                "cv_auc_mean": round(float(cv_scores.mean()), 4),
                "cv_auc_std": round(float(cv_scores.std()), 4),
                "val_auc": round(val_auc, 4),
                "cv_time_s": round(cv_time, 2),
                "status": "ok",
            })
            logger.info("  %s: CV AUC=%.4f +/- %.4f | val AUC=%.4f (%.1fs)",
                        name, cv_scores.mean(), cv_scores.std(), val_auc, cv_time)
        except Exception as exc:  # noqa: BLE001 - keep comparing remaining models
            logger.error("  %s FAILED: %s", name, exc)
            rows.append({"model": name, "cv_auc_mean": np.nan, "cv_auc_std": np.nan,
                         "val_auc": np.nan, "cv_time_s": np.nan, "status": f"failed: {exc}"})

    results = pd.DataFrame(rows).sort_values("cv_auc_mean", ascending=False).reset_index(drop=True)
    return results


def build_model_comparison_report(results: pd.DataFrame) -> str:
    """Compile a markdown report: measured results + per-model trade-off reference."""
    results_table = results.copy()
    results_table.insert(0, "rank", results_table.index + 1)

    ref_rows = []
    for name in results["model"]:
        meta = MODELS_REFERENCE.get(name, {})
        ref_rows.append({
            "model": name,
            **{k: meta.get(k, "") for k in
               ("advantages", "disadvantages", "interpretability", "business_suitability")},
        })
    ref_table = pd.DataFrame(ref_rows)

    best = results.iloc[0]
    sections = [
        "# Model Comparison Report (Phase 9)",
        "_Protocol:_ leakage-safe Pipeline (FeatureEngineer -> preprocess -> selector -> model), "
        "stratified 5-fold CV on TRAIN + validation check.",
        "\n## 1. Measured results (ranked by CV ROC-AUC)\n",
        "```\n" + results_table.to_string(index=False) + "\n```",
        "\n## 2. Recommendation\n",
        f"- **Best by CV AUC:** **{best['model']}** "
        f"(CV AUC={best['cv_auc_mean']} +/- {best['cv_auc_std']}, val AUC={best['val_auc']}).",
        "- Boosting models (XGBoost / LightGBM / CatBoost) and Random Forest are expected to "
        "lead on tabular credit data; Logistic Regression is kept as the interpretable baseline.",
        "- Final selection considers AUC **and** interpretability/calibration/business cost "
        "(deep evaluation in Phase 10, tuning in Phase 11).",
        "\n## 3. Model trade-off reference\n",
        "```\n" + ref_table.to_string(index=False) + "\n```",
    ]
    return "\n".join(sections)


if __name__ == "__main__":
    import json

    from src.config import get_config
    from src.preprocessing.pipeline import prepare_features

    cfg = get_config()
    seed = cfg["project"]["random_seed"]
    target = cfg["dataset"]["target"]
    protected = cfg.get("protected_attributes", [])

    train_df = pd.read_csv(Path(cfg["paths"]["processed_data"]).parent / "train.csv")
    val_df = pd.read_csv(Path(cfg["paths"]["processed_data"]).parent / "validation.csv")
    X_train, y_train = prepare_features(train_df, target=target, protected=protected)
    X_val, y_val = prepare_features(val_df, target=target, protected=protected)

    selected = json.loads(Path(cfg["paths"]["artifacts_dir"]).joinpath("selected_features.json").read_text())["selected"]
    positions = get_selected_positions(X_train, selected)

    results = compare_models(
        X_train, y_train, X_val, y_val, positions,
        seed=seed, cv_folds=cfg["modeling"]["cv_folds"],
    )

    report_dir = Path(cfg["paths"]["reports_dir"])
    results.to_csv(report_dir / "model_comparison.csv", index=False)
    (report_dir / "model_comparison.md").write_text(
        build_model_comparison_report(results), encoding="utf-8"
    )
    logger.info("Model comparison saved -> %s", report_dir / "model_comparison.md")
    print("\nModel comparison (sorted by CV ROC-AUC):")
    print(results.to_string(index=False))
