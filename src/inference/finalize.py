"""Finalize, save, and verify the production model end-to-end.

Builds the tuned Logistic Regression as a single Pipeline (feature engineering ->
preprocessing -> selection -> model), saves it with joblib + metadata (+ a pickle
copy), reloads it, and runs the FIRST honest TEST-set evaluation plus an inference
demo on raw applicants.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.config import get_config
from src.evaluation.metrics import business_cost, compute_metrics, find_optimal_threshold
from src.feature_engineering.feature_engineer import FeatureEngineer
from src.inference.predict import explain_prediction, predict_applicant
from src.inference.serialize import build_metadata, load_artifact, save_artifact, save_pickle
from src.preprocessing.pipeline import build_preprocessing_pipeline, prepare_features
from src.training.train import PositionSelector, get_selected_positions
from src.utils import get_logger

logger = get_logger(__name__)


def build_final_pipeline(best_params: dict, positions: list[int], seed: int):
    """Assemble the final tuned LogReg end-to-end pipeline (unfitted)."""
    lr = best_params["Logistic Regression"]
    model = LogisticRegression(
        C=float(lr["C"]), penalty=lr["penalty"], solver=lr["solver"],
        max_iter=3000, class_weight="balanced", random_state=seed,
    )
    from sklearn.pipeline import Pipeline
    return Pipeline([
        ("feature_engineer", FeatureEngineer()),
        ("preprocess", build_preprocessing_pipeline()),
        ("select", PositionSelector(positions)),
        ("model", model),
    ])


def _eval(pipeline, X, y, label):
    """Score a dataset and return a metrics dict + cost-optimal threshold."""
    proba = pipeline.predict_proba(X)[:, 1]
    opt_t, _ = find_optimal_threshold(y, proba)
    m = compute_metrics(y, proba, opt_t)
    cost = business_cost(y, (proba >= opt_t).astype(int))
    logger.info("%s: AUC=%.4f KS=%.4f cost/app=%.4f thr=%.2f", label, m["roc_auc"], m["ks"], cost["cost_per_applicant"], opt_t)
    return {"roc_auc": m["roc_auc"], "ks": m["ks"], "brier": m["brier"],
            "opt_threshold": round(opt_t, 2), "cost_per_applicant": cost["cost_per_applicant"]}


if __name__ == "__main__":
    cfg = get_config()
    seed = cfg["project"]["random_seed"]
    target = cfg["dataset"]["target"]
    protected = cfg.get("protected_attributes", [])
    processed_dir = Path(cfg["paths"]["processed_data"]).parent
    models_dir = Path(cfg["paths"]["models_dir"])
    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    report_dir = Path(cfg["paths"]["reports_dir"])

    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df = pd.read_csv(processed_dir / "validation.csv")
    test_df = pd.read_csv(processed_dir / "test.csv")
    X_train, y_train = prepare_features(train_df, target=target, protected=protected)
    X_val, y_val = prepare_features(val_df, target=target, protected=protected)
    X_test, y_test = prepare_features(test_df, target=target, protected=protected)

    best_params = json.loads((artifacts_dir / "best_params.json").read_text())
    selected = pd.read_json(artifacts_dir / "selected_features.json")["selected"].tolist()
    positions = get_selected_positions(X_train, selected)

    # 1) Build & fit the final pipeline on TRAIN.
    final_pipe = build_final_pipeline(best_params, positions, seed)
    final_pipe.fit(X_train, y_train)
    threshold = best_params.get("threshold", 0.64)

    # 2) Evaluate on val + the untouched TEST set.
    val_metrics = _eval(final_pipe, X_val, y_val.to_numpy(), "validation")
    test_metrics = _eval(final_pipe, X_test, y_test.to_numpy(), "TEST")
    threshold = val_metrics["opt_threshold"]

    # 3) Save artifact (joblib) + metadata + a pickle copy.
    metadata = build_metadata(
        model_type="LogisticRegression (tuned)",
        threshold=threshold,
        metrics={"validation": val_metrics, "test": test_metrics},
        feature_names=selected,
        params=best_params["Logistic Regression"],
        extra={"selected_from": "consensus feature selection (Phase 7)"},
    )
    joblib_path = models_dir / "credit_scoring_logreg_v1.joblib"
    save_artifact(final_pipe, joblib_path, metadata)
    save_pickle(final_pipe, models_dir / "credit_scoring_logreg_v1.pkl")

    # 4) Reload and verify it produces identical predictions (reproducibility).
    loaded_pipe, loaded_meta = load_artifact(joblib_path)
    reloaded_proba = loaded_pipe.predict_proba(X_test)[:, 1]
    assert np.allclose(reloaded_proba, final_pipe.predict_proba(X_test)[:, 1]), "Reloaded model mismatch!"
    logger.info("Reloaded artifact reproduces predictions exactly. (v%s)", loaded_meta["artifact_version"])

    # 5) Inference demo on raw applicants from the test set.
    demo_X = X_test.iloc[[0, 1]].copy()
    preds = predict_applicant(loaded_pipe, demo_X, threshold=threshold)
    reasons = explain_prediction(loaded_pipe, demo_X, top_k=4)

    report = "\n".join([
        "# Model Finalization & Inference Report (Phase 13)",
        f"_Artifact:_ `{joblib_path.name}` (v{metadata['artifact_version']})  \n"
        f"_Created:_ {metadata['created_at_utc']}  \n"
        f"_Decision threshold:_ {threshold}",
        "\n## 1. Final performance\n",
        "```\nvalidation: " + json.dumps(val_metrics) + "\nTEST      : " + json.dumps(test_metrics) + "\n```",
        "\n## 2. Serialization\n",
        f"- Primary: joblib -> `{joblib_path.name}` (+ `.meta.json` sidecar)",
        f"- Alternative: pickle -> `credit_scoring_logreg_v1.pkl`",
        f"- Reload verified to reproduce predictions byte-for-byte.",
        "\n## 3. Inference demo (raw applicants)\n",
        "```\n" + preds.to_string() + "\n```",
        "\n### Reason codes — applicant 1\n",
        "```\n" + reasons[0].round(3).to_string(index=False) + "\n```",
        "\n### Reason codes — applicant 2\n",
        "```\n" + reasons[1].round(3).to_string(index=False) + "\n```",
    ])
    (report_dir / "model_finalization_report.md").write_text(report, encoding="utf-8")
    logger.info("Finalization report -> %s", report_dir / "model_finalization_report.md")

    print("\n=== FINAL performance ===")
    print("validation:", val_metrics)
    print("TEST      :", test_metrics)
    print("\n=== Inference demo ===")
    print(preds.to_string())
    print("\nReason codes (applicant 1):")
    print(reasons[0].round(3).to_string(index=False))
