"""Model explainability: permutation importance, SHAP, LIME, PDP + business report.

Explains the FINAL tuned models:
    - Logistic Regression (primary): coefficients double as reason codes.
    - XGBoost (challenger): tree-based SHAP + partial dependence.

All explainers operate on the SELECTED, transformed feature matrix (the exact
representation the model sees), with feature names recovered from the pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.base import clone
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.feature_engineering.feature_engineer import FeatureEngineer
from src.preprocessing.pipeline import build_preprocessing_pipeline
from src.training.train import PositionSelector, get_selected_positions
from src.utils import get_logger

logger = get_logger(__name__)

sns.set_theme(style="whitegrid", font_scale=0.9)
FIG_DPI = 120


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure -> %s", path)


def fit_preprocessing(
    X_train: pd.DataFrame, positions: list[int]
) -> tuple[Pipeline, np.ndarray, list[str]]:
    """Fit the preprocessing+selector pipeline on train; return it + transformed train + names."""
    prep = Pipeline([
        ("feature_engineer", FeatureEngineer()),
        ("preprocess", build_preprocessing_pipeline()),
        ("select", PositionSelector(positions)),
    ])
    prep.fit(X_train)
    Xt = prep.transform(X_train)
    names = list(prep.get_feature_names_out())
    logger.info("Prepared transformed matrix %s with %d named features.", Xt.shape, len(names))
    return prep, Xt, names


def load_final_models(
    best_params: dict[str, dict], seed: int, scale_pos_weight: float,
    Xt_train: np.ndarray, y_train: np.ndarray,
) -> tuple[LogisticRegression, XGBClassifier]:
    """Instantiate & fit the tuned LogReg and XGBoost on the transformed train set."""
    lr_p = best_params["Logistic Regression"]
    logreg = LogisticRegression(
        C=float(lr_p["C"]), penalty=lr_p["penalty"], solver=lr_p["solver"],
        max_iter=3000, class_weight="balanced", random_state=seed,
    )
    logreg.fit(Xt_train, y_train)

    xgb_p = best_params["XGBoost"]
    xgboost = XGBClassifier(
        tree_method="hist", eval_metric="logloss", scale_pos_weight=scale_pos_weight,
        random_state=seed, n_jobs=-1, verbosity=0, **xgb_p,
    )
    xgboost.fit(Xt_train, y_train)
    return logreg, xgboost


def logreg_reason_codes(model: LogisticRegression, names: list[str]) -> pd.DataFrame:
    """Return LogReg coefficients as signed, ranked reason codes."""
    coef = pd.Series(model.coef_[0], index=names, name="coefficient")
    odds = np.exp(coef)
    df = pd.DataFrame({"coefficient": coef, "odds_ratio": odds.round(3)})
    df["effect"] = np.where(df["coefficient"] > 0, "increases approval odds", "decreases approval odds")
    return df.sort_values("coefficient", ascending=False)


def plot_logreg_coefficients(model: LogisticRegression, names: list[str], save_path: Path) -> pd.DataFrame:
    """Horizontal bar chart of LogReg coefficients (signed drivers)."""
    rc = logreg_reason_codes(model, names)
    ordered = rc.sort_values("coefficient")
    fig, ax = plt.subplots(figsize=(9, 8))
    colors = ["#dc2626" if c < 0 else "#16a34a" for c in ordered["coefficient"]]
    ax.barh(ordered.index, ordered["coefficient"], color=colors)
    ax.set_title("Logistic Regression Coefficients (Reason Codes)")
    ax.set_xlabel("Coefficient (effect on log-odds of approval)")
    ax.axvline(0, color="k", lw=0.8)
    _save_fig(fig, save_path)
    return rc


def plot_permutation_importance(
    model: Any, X_val: np.ndarray, y_val: np.ndarray, names: list[str],
    save_path: Path, seed: int = 42,
) -> pd.Series:
    """Permutation importance (model-agnostic) ranked by ROC-AUC drop."""
    result = permutation_importance(
        model, X_val, y_val, scoring="roc_auc", n_repeats=10,
        random_state=seed, n_jobs=-1,
    )
    imp = pd.Series(result.importances_mean, index=names).sort_values(ascending=False)
    top = imp.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top.index, top.values, color="#1d4ed8")
    ax.set_title("Permutation Importance (drop in ROC-AUC when shuffled)")
    ax.set_xlabel("Mean AUC decrease")
    _save_fig(fig, save_path)
    return imp


def plot_shap_summary(
    model: XGBClassifier, X_sample: np.ndarray, names: list[str], save_path: Path
) -> pd.Series:
    """SHAP TreeExplainer beeswarm + return mean |SHAP| per feature."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=names).sort_values(ascending=False)

    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=names, show=False)
    fig = plt.gcf()
    _save_fig(fig, save_path)
    return mean_abs


def plot_partial_dependence(
    model: Any, X: np.ndarray, names: list[str], features: list[str], save_path: Path
) -> None:
    """Partial dependence plots for the most impactful features."""
    indices = [names.index(f) for f in features if f in names]
    if not indices:
        return
    display = PartialDependenceDisplay.from_estimator(
        model, X, features=indices, feature_names=names,
        grid_resolution=30, kind="average",
    )
    fig = display.figure_
    fig.set_size_inches(12, 8)
    _save_fig(fig, save_path)


def lime_local_explanations(
    model: Any, Xt_train: np.ndarray, Xt_val: np.ndarray, y_val: np.ndarray,
    proba_val: np.ndarray, names: list[str], save_dir: Path, seed: int = 42,
) -> list[str]:
    """Explain one approved-good and one rejected-bad applicant with LIME."""
    save_dir.mkdir(parents=True, exist_ok=True)
    explainer = LimeTabularExplainer(
        Xt_train, feature_names=names, class_names=["Defaulter", "Creditworthy"],
        mode="classification", discretize_continuous=True, random_state=seed,
    )
    summaries: list[str] = []
    scenarios = [
        ("approved_good", np.where((proba_val >= 0.64) & (y_val == 1))[0]),
        ("rejected_bad", np.where((proba_val < 0.64) & (y_val == 0))[0]),
    ]
    for label, candidates in scenarios:
        if len(candidates) == 0:
            continue
        idx = int(candidates[0])
        exp = explainer.explain_instance(Xt_val[idx], model.predict_proba, num_features=8)
        fig = exp.as_pyplot_figure()
        _save_fig(fig, save_dir / f"12_lime_{label}.png")
        summaries.append(f"[{label}] P(creditworthy)={proba_val[idx]:.2f} | "
                         + "; ".join(f"{w} {v}" for v, w in exp.as_list()[:5]))
    return summaries


def build_explainability_report(
    reason_codes: pd.DataFrame, perm_imp: pd.Series, shap_imp: pd.Series,
    lime_summaries: list[str], feature_names: list[str],
) -> str:
    """Compile a business-facing explainability report."""
    top_pos = reason_codes.head(8)
    top_neg = reason_codes.tail(8).iloc[::-1]
    top_perm = perm_imp.head(10)
    top_shap = shap_imp.head(10)

    def _business_narrative():
        lines = [
            "- The strongest APPROVAL drivers (raise log-odds): "
            + ", ".join(top_pos.index[:5]),
            "- The strongest REJECTION drivers (lower log-odds): "
            + ", ".join(top_neg.index[:5]),
            "- These map directly to underwriting logic: capacity (income/savings), "
            "discipline (payment consistency, no delinquency), and burden (DTI, utilization).",
        ]
        return "\n".join(lines)

    sections = [
        "# Model Explainability Report (Phase 12)",
        "_Models explained:_ tuned Logistic Regression (primary) + tuned XGBoost (challenger). "
        "Reason codes come from LogReg; SHAP/PDP from the tree model.",
        "\n## 1. Reason codes — Logistic Regression coefficients\n",
        "Coefficients are the change in log-odds of approval per +1 (scaled) unit. "
        "`odds_ratio = exp(coef)`. Green = increases approval odds; red = decreases.",
        "```\n" + reason_codes.round(3).to_string() + "\n```",
        "\n## 2. Top approval drivers\n",
        "```\n" + top_pos[["coefficient", "odds_ratio", "effect"]].round(3).to_string() + "\n```",
        "\n## 3. Top rejection drivers\n",
        "```\n" + top_neg[["coefficient", "odds_ratio", "effect"]].round(3).to_string() + "\n```",
        "\n## 4. Permutation importance (model-agnostic)\n",
        "```\n" + top_perm.round(4).to_string() + "\n```",
        "\n## 5. SHAP mean |contribution| (XGBoost)\n",
        "```\n" + top_shap.round(4).to_string() + "\n```",
        "\n## 6. Business narrative\n",
        _business_narrative(),
        "\n## 7. Local explanations (LIME)\n",
        "\n".join(f"- {s}" for s in lime_summaries) if lime_summaries else "- (not generated)",
        "\n## 8. Figures (reports/figures/)\n",
        "- `12_logreg_coefficients.png` — reason codes (signed)",
        "- `12_permutation_importance.png` — model-agnostic ranking",
        "- `12_shap_summary.png` — SHAP beeswarm (XGBoost)",
        "- `12_partial_dependence.png` — PDPs for top features",
        "- `12_lime_approved_good.png` / `12_lime_rejected_bad.png` — per-applicant explanations",
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
    processed_dir = Path(cfg["paths"]["processed_data"]).parent
    figures_dir = Path(cfg["paths"]["figures_dir"])
    report_dir = Path(cfg["paths"]["reports_dir"])
    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])

    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df = pd.read_csv(processed_dir / "validation.csv")
    X_train, y_train = prepare_features(train_df, target=target, protected=protected)
    X_val, y_val = prepare_features(val_df, target=target, protected=protected)

    selected = pd.read_json(artifacts_dir / "selected_features.json")["selected"].tolist()
    positions = get_selected_positions(X_train, selected)
    prep, Xt_train, names = fit_preprocessing(X_train, positions)
    Xt_val = prep.transform(X_val)
    y_train_np, y_val_np = y_train.to_numpy(), y_val.to_numpy()

    best_params = json.loads((artifacts_dir / "best_params.json").read_text())
    neg, pos = int((y_train_np == 0).sum()), int((y_train_np == 1).sum())
    spw = neg / max(pos, 1)
    logreg, xgboost = load_final_models(best_params, seed, spw, Xt_train, y_train_np)

    logger.info("LogReg val AUC=%.4f | XGBoost val AUC=%.4f",
                roc_auc_score(y_val_np, logreg.predict_proba(Xt_val)[:, 1]),
                roc_auc_score(y_val_np, xgboost.predict_proba(Xt_val)[:, 1]))

    rc = plot_logreg_coefficients(logreg, names, figures_dir / "12_logreg_coefficients.png")
    perm = plot_permutation_importance(logreg, Xt_val, y_val_np, names,
                                       figures_dir / "12_permutation_importance.png", seed)
    sample_idx = np.random.RandomState(seed).choice(len(Xt_val), size=min(800, len(Xt_val)), replace=False)
    shap_imp = plot_shap_summary(xgboost, Xt_val[sample_idx], names, figures_dir / "12_shap_summary.png")
    plot_partial_dependence(xgboost, Xt_val, names, perm.head(4).index.tolist(),
                            figures_dir / "12_partial_dependence.png")

    proba_val = logreg.predict_proba(Xt_val)[:, 1]
    lime_summaries = lime_local_explanations(
        logreg, Xt_train, Xt_val, y_val_np, proba_val, names, figures_dir, seed,
    )

    report = build_explainability_report(rc, perm, shap_imp, lime_summaries, names)
    (report_dir / "explainability_report.md").write_text(report, encoding="utf-8")
    logger.info("Explainability report -> %s", report_dir / "explainability_report.md")

    print("\nTop reason codes (LogReg):")
    print(rc.head(6).round(3).to_string())
    print("\nBottom reason codes (LogReg):")
    print(rc.tail(6).round(3).to_string())
    print("\nSHAP top (XGBoost):")
    print(shap_imp.head(6).round(4).to_string())
