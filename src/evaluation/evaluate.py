"""End-to-end evaluation: metrics, charts, threshold/cost analysis, recommendation.

Fits every candidate model on TRAIN (via the leakage-safe full pipeline), scores
the VALIDATION set, and produces the full credit-risk evaluation: metric table,
ROC/PR overlays, and per-model deep charts (confusion, calibration, lift, gain,
threshold-cost) for the leading models, plus a data-driven recommendation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.config import get_config
from src.evaluation.metrics import (
    business_cost,
    compute_metrics,
    find_optimal_threshold,
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
from src.preprocessing.pipeline import prepare_features
from src.training.models import get_models
from src.training.train import _scores, build_model_pipeline, get_selected_positions
from src.utils import get_logger

logger = get_logger(__name__)

COST_FP = 5.0  # cost of approving a defaulter (loss given default)
COST_FN = 1.0  # cost of rejecting a good customer (lost margin)


def evaluate_all_models(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    positions: list[int], seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    """Fit all models on train, score validation, build the metrics table.

    Returns:
        (metrics_dataframe, dict of {model_name: {y_true, y_proba}}).
    """
    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    models = get_models(seed=seed, n_jobs=-1, scale_pos_weight=neg / max(pos, 1))
    y_val_np = y_val.to_numpy()

    val_results: dict[str, dict[str, np.ndarray]] = {}
    rows: list[dict[str, Any]] = []
    for name, model in models.items():
        logger.info("Fitting %s ...", name)
        pipe = build_model_pipeline(clone(model), positions)
        pipe.fit(X_train, y_train)
        proba = _scores(pipe, X_val)
        val_results[name] = {"y_true": y_val_np, "y_proba": proba}

        m_opt_t, _ = find_optimal_threshold(y_val_np, proba, COST_FP, COST_FN)
        m_default = compute_metrics(y_val_np, proba, 0.5)
        m_opt = compute_metrics(y_val_np, proba, m_opt_t)
        cost = business_cost(y_val_np, (proba >= m_opt_t).astype(int), COST_FP, COST_FN)

        rows.append({
            "model": name,
            "roc_auc": m_default["roc_auc"],
            "pr_auc": m_default["pr_auc"],
            "ks": m_default["ks"],
            "brier": m_default["brier"],
            "opt_threshold": round(m_opt_t, 2),
            "approval_rate@opt": m_opt["approval_rate"],
            "bad_rate@opt": m_opt["bad_rate_approved"],
            "recall@opt": m_opt["recall_sensitivity"],
            "specificity@opt": m_opt["specificity"],
            "f1@opt": m_opt["f1"],
            "cost_per_applicant@opt": cost["cost_per_applicant"],
        })
        logger.info("  %s: AUC=%.4f KS=%.4f cost/app=%.3f opt_t=%.2f",
                    name, m_default["roc_auc"], m_default["ks"],
                    cost["cost_per_applicant"], m_opt_t)

    table = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    return table, val_results


def recommend_model(table: pd.DataFrame) -> str:
    """Data-driven recommendation: among AUC top-tier, pick lowest business cost."""
    best_auc = table["roc_auc"].max()
    top_tier = table[table["roc_auc"] >= best_auc - 0.005]
    return str(top_tier.sort_values("cost_per_applicant@opt").iloc[0]["model"])


def build_evaluation_report(table: pd.DataFrame, recommended: str, focus_models: list[str]) -> str:
    """Compile the markdown evaluation report."""
    ranked = table.copy()
    ranked.insert(0, "rank", ranked.index + 1)
    rec = table[table["model"] == recommended].iloc[0]

    sections = [
        "# Evaluation Report (Phase 10)",
        f"_Evaluation set:_ Validation (held-out)  \n_Cost model:_ FP(approve a defaulter)="
        f"{COST_FP} x  |  FN(reject a good customer)={COST_FN}  \n_Operating point:_ cost-optimal threshold",
        "\n## 1. Metric suite (all models, validation)\n",
        "Metrics below use each model's **cost-optimal threshold**; AUC/PR-AUC/KS/Brier are "
        "threshold-independent.",
        "```\n" + ranked.to_string(index=False) + "\n```",
        "\n## 2. Recommendation\n",
        f"- **Recommended model: {recommended}**",
        f"  - ROC-AUC={rec['roc_auc']}, KS={rec['ks']}, PR-AUC={rec['pr_auc']}, "
        f"Brier={rec['brier']}",
        f"  - At cost-optimal threshold {rec['opt_threshold']}: approval rate="
        f"{rec['approval_rate@opt']:.1%}, bad rate(among approved)={rec['bad_rate@opt']:.1%}, "
        f"cost/applicant={rec['cost_per_applicant@opt']:.3f}",
        "- Selected by: highest ROC-AUC, then lowest business cost within the AUC top-tier, "
        "with interpretability/calibration as tie-breakers (favoring regulatory-ready models).",
        "\n## 3. Charts (reports/figures/)\n",
        "- `10_roc_curves.png` / `10_pr_curves.png` — all-model overlays",
        f"- Per leading model ({', '.join(focus_models)}): `10_confusion_<model>.png`, "
        "`10_calibration_<model>.png`, `10_lift_<model>.png`, `10_gain_<model>.png`, "
        "`10_threshold_cost_<model>.png`",
    ]
    return "\n".join(sections)


if __name__ == "__main__":
    cfg = get_config()
    seed = cfg["project"]["random_seed"]
    target = cfg["dataset"]["target"]
    protected = cfg.get("protected_attributes", [])
    processed_dir = Path(cfg["paths"]["processed_data"]).parent
    figures_dir = Path(cfg["paths"]["figures_dir"])
    report_dir = Path(cfg["paths"]["reports_dir"])

    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df = pd.read_csv(processed_dir / "validation.csv")
    X_train, y_train = prepare_features(train_df, target=target, protected=protected)
    X_val, y_val = prepare_features(val_df, target=target, protected=protected)

    selected = pd.read_json(Path(cfg["paths"]["artifacts_dir"]) / "selected_features.json")["selected"].tolist()
    positions = get_selected_positions(X_train, selected)

    table, val_results = evaluate_all_models(X_train, y_train, X_val, y_val, positions, seed)
    table.to_csv(report_dir / "evaluation_metrics.csv", index=False)

    # Overlay charts (all models).
    plot_roc_curves(val_results, figures_dir / "10_roc_curves.png")
    plot_pr_curves(val_results, figures_dir / "10_pr_curves.png")

    # Deep charts for the top-2 models by AUC.
    focus = table.head(2)["model"].tolist()
    slug = lambda s: s.lower().replace(" ", "_")
    for name in focus:
        data = val_results[name]
        opt_t, _ = find_optimal_threshold(data["y_true"], data["y_proba"], COST_FP, COST_FN)
        plot_confusion_matrix(
            data["y_true"], (data["y_proba"] >= opt_t).astype(int),
            figures_dir / f"10_confusion_{slug(name)}.png",
            title=f"Confusion Matrix — {name} (thr={opt_t:.2f})",
        )
        plot_calibration(data["y_true"], data["y_proba"], figures_dir / f"10_calibration_{slug(name)}.png", name=name)
        plot_gain_chart(data["y_true"], data["y_proba"], figures_dir / f"10_gain_{slug(name)}.png")
        plot_lift_chart(data["y_true"], data["y_proba"], figures_dir / f"10_lift_{slug(name)}.png")
        plot_threshold_cost(data["y_true"], data["y_proba"], figures_dir / f"10_threshold_cost_{slug(name)}.png", cost_fp=COST_FP, cost_fn=COST_FN)

    recommended = recommend_model(table)
    report = build_evaluation_report(table, recommended, focus)
    (report_dir / "evaluation_report.md").write_text(report, encoding="utf-8")
    logger.info("Evaluation report -> %s", report_dir / "evaluation_report.md")

    print("\nValidation metric suite (cost-optimal thresholds):")
    print(table.to_string(index=False))
    print(f"\n>>> Recommended model: {recommended}")
