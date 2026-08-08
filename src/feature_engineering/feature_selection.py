"""Feature selection: compare multiple methods and build a consensus subset.

Five complementary techniques are applied to the *transformed* training matrix
(the output of the full preprocessing pipeline) so we evaluate exactly what the
model sees. All selection is fit on TRAIN ONLY to prevent leakage.

WHY compare many methods instead of trusting one?
    - Filters (variance, correlation/VIF) remove useless/redundant features
      but ignore the target.
    - Wrappers (RFE) optimize for a specific model but are slow & can overfit.
    - Embedded/agnostic (mutual information, permutation importance) capture
      non-linear + target-aware signal.
    A *consensus* across independent methods is far more robust than any single
    ranking, especially for a regulated, auditable domain like credit risk.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.utils import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# 1. Correlation / redundancy
# --------------------------------------------------------------------------- #
def select_by_correlation(
    X: np.ndarray, names: list[str], threshold: float = 0.9
) -> tuple[list[tuple[str, str, float]], set[str]]:
    """Find highly-correlated feature pairs and flag one per pair for removal.

    Args:
        X: Transformed feature matrix.
        names: Feature names aligned with ``X`` columns.
        threshold: Absolute Pearson correlation above which two features are
            considered redundant.

    Returns:
        (list of (feature_a, feature_b, |corr|) pairs, set of drop candidates).
    """
    corr = pd.DataFrame(X, columns=names).corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs: list[tuple[str, str, float]] = []
    drop_candidates: set[str] = set()
    for col in upper.columns:
        highly = upper[col][upper[col] > threshold]
        for other, value in highly.items():
            pairs.append((col, other, round(float(value), 3)))
            drop_candidates.add(other)  # drop the second of the pair by default
    return pairs, drop_candidates


# --------------------------------------------------------------------------- #
# 2. Variance threshold (near-constant features)
# --------------------------------------------------------------------------- #
def select_by_variance(
    X: np.ndarray, names: list[str], threshold: float = 0.01
) -> tuple[set[str], pd.Series]:
    """Flag features whose variance is below ``threshold`` (near-constant)."""
    variances = pd.Series(X.var(axis=0), index=names)
    dropped = set(variances[variances < threshold].index)
    return dropped, variances.sort_values()


# --------------------------------------------------------------------------- #
# 3. Variance Inflation Factor (multicollinearity for linear models)
# --------------------------------------------------------------------------- #
def variance_inflation_factors(X: np.ndarray, names: list[str]) -> pd.Series:
    """Compute VIF per feature (multicollinearity diagnostic, no statsmodels)."""
    Xc = np.asarray(X, dtype=float)
    Xc = Xc - Xc.mean(axis=0)
    n, k = Xc.shape
    vifs = np.zeros(k)
    for i in range(k):
        others = np.delete(Xc, i, axis=1)
        design = np.column_stack([others, np.ones(n)])
        coef, *_ = np.linalg.lstsq(design, Xc[:, i], rcond=None)
        residual = Xc[:, i] - design @ coef
        ss_res = float(residual @ residual)
        ss_tot = float(((Xc[:, i] - Xc[:, i].mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vifs[i] = 1.0 / (1.0 - r2) if r2 < 0.9999 else np.inf
    return pd.Series(vifs, index=names).sort_values(ascending=False)


# --------------------------------------------------------------------------- #
# 4. Mutual information (target-aware, model-free, non-linear)
# --------------------------------------------------------------------------- #
def select_by_mutual_information(
    X: np.ndarray, y: np.ndarray, names: list[str], seed: int = 42
) -> pd.Series:
    """Return mutual-information scores between each feature and the target."""
    scores = mutual_info_classif(X, y, random_state=seed)
    return pd.Series(scores, index=names).sort_values(ascending=False)


# --------------------------------------------------------------------------- #
# 5. Recursive Feature Elimination (wrapper, model-specific)
# --------------------------------------------------------------------------- #
def select_by_rfe(
    X: np.ndarray, y: np.ndarray, names: list[str], top_k: int = 25, seed: int = 42
) -> set[str]:
    """RFE with balanced logistic regression -> set of selected feature names."""
    base = LogisticRegression(
        max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=seed
    )
    selector = RFE(base, n_features_to_select=top_k)
    selector.fit(X, y)
    return set(np.array(names)[selector.get_support()])


# --------------------------------------------------------------------------- #
# 6. Permutation importance (model-agnostic, non-linear)
# --------------------------------------------------------------------------- #
def select_by_permutation(
    X: np.ndarray,
    y: np.ndarray,
    names: list[str],
    top_k: int = 25,
    seed: int = 42,
) -> pd.Series:
    """Train a RandomForest, score permutation importance on a held-out fold."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    model = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=-1
    )
    model.fit(X_tr, y_tr)
    result = permutation_importance(
        model, X_val, y_val, scoring="roc_auc", n_repeats=5, random_state=seed, n_jobs=-1
    )
    return pd.Series(result.importances_mean, index=names).sort_values(ascending=False)


# --------------------------------------------------------------------------- #
# 7. Consensus + orchestrator
# --------------------------------------------------------------------------- #
def run_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    names: list[str],
    top_k: int = 25,
    corr_threshold: float = 0.9,
    var_threshold: float = 0.01,
    seed: int = 42,
) -> dict[str, Any]:
    """Run all methods and derive a consensus feature subset.

    Args:
        X: Transformed TRAIN-ONLY feature matrix.
        y: Training target.
        names: Feature names aligned with ``X``.
        top_k: Number of features each ranking method contributes.
        corr_threshold, var_threshold: filter thresholds.
        seed: Random seed.

    Returns:
        Dict with per-method results, a vote table, and the final selected set.
    """
    logger.info("Running feature selection on %d features (train only).", len(names))

    corr_pairs, corr_drop = select_by_correlation(X, names, corr_threshold)
    var_drop, variances = select_by_variance(X, names, var_threshold)
    vifs = variance_inflation_factors(X, names)
    mi = select_by_mutual_information(X, y, names, seed)
    rfe_set = select_by_rfe(X, y, names, top_k, seed)
    perm = select_by_permutation(X, y, names, top_k, seed)

    mi_top = set(mi.head(top_k).index)
    perm_top = set(perm.head(top_k).index)
    method_tops = {"mutual_info": mi_top, "rfe": rfe_set, "permutation": perm_top}

    votes = {f: int(sum(f in s for s in method_tops.values())) for f in names}
    consensus = {f for f, v in votes.items() if v >= 2}

    # Redundancy pruning: among consensus features, for each high-corr pair drop
    # the one with lower mutual information.
    removed_for_redundancy: set[str] = set()
    for a, b, _ in corr_pairs:
        if a in consensus and b in consensus:
            loser = a if mi.get(a, 0.0) < mi.get(b, 0.0) else b
            removed_for_redundancy.add(loser)

    final_selected = sorted(
        consensus - removed_for_redundancy - var_drop
    )

    return {
        "names": names,
        "n_total": len(names),
        "corr_pairs": corr_pairs,
        "variance_dropped": sorted(var_drop),
        "vifs": vifs,
        "mutual_info": mi,
        "rfe_set": sorted(rfe_set),
        "permutation": perm,
        "votes": votes,
        "consensus": sorted(consensus),
        "removed_for_redundancy": sorted(removed_for_redundancy),
        "final_selected": final_selected,
        "n_selected": len(final_selected),
    }


# --------------------------------------------------------------------------- #
# 8. Report
# --------------------------------------------------------------------------- #
def _md_block(df: pd.DataFrame, title: str | None = None) -> str:
    header = f"**{title}**\n\n" if title else ""
    return f"{header}```\n{df.to_string()}\n```"


def build_feature_selection_report(results: dict[str, Any]) -> str:
    """Compile a markdown report comparing methods and justifying the choice."""
    names = results["names"]
    mi = results["mutual_info"]
    perm = results["permutation"]
    rfe_set = set(results["rfe_set"])
    votes = results["votes"]
    vifs = results["vifs"]
    final = set(results["final_selected"])

    table = pd.DataFrame({
        "feature": names,
        "mutual_info": [round(float(mi.get(f, 0.0)), 4) for f in names],
        "perm_imp": [round(float(perm.get(f, 0.0)), 4) for f in names],
        "rfe_kept": [f in rfe_set for f in names],
        "votes": [votes[f] for f in names],
        "vif": [round(float(vifs.get(f, 0.0)), 1) for f in names],
        "final_keep": [f in final for f in names],
    })
    table = table.sort_values(["votes", "mutual_info"], ascending=[False, False])

    high_vif = vifs[vifs > 10]

    sections = [
        "# Feature Selection Report",
        f"_Features evaluated:_ {results['n_total']}  \n_Final selected:_ "
        f"**{results['n_selected']}** (consensus: selected by >=2 of MI/RFE/Permutation, "
        "minus near-constant & redundant)",
        "\n## 1. Method comparison\n",
        "| Method | Type | Keeps | Note |",
        "|---|---|---|---|",
        f"| Variance threshold (>={0.01}) | Filter (unsupervised) | {results['n_total'] - len(results['variance_dropped'])} | "
        f"dropped {len(results['variance_dropped'])} near-constant |",
        f"| Correlation (>0.{int(results.get('_corr', 0.9)*10)}) | Filter (redundancy) | - | "
        f"{len(results['corr_pairs'])} redundant pairs found |",
        f"| Mutual Information | Filter (target-aware) | top 25 | non-linear relevance |",
        f"| RFE (LogReg) | Wrapper | top 25 | model-specific ranking |",
        f"| Permutation (RF) | Agnostic | top 25 | non-linear, held-out scored |",
        "\n## 2. Consensus ranking\n",
        _md_block(table, "All features ranked by votes then MI"),
        "\n## 3. Multicollinearity (VIF > 10, caution for linear models)\n",
        _md_block(high_vif.to_frame("vif")) if not high_vif.empty else "_No feature with VIF > 10._",
        "\n## 4. Decision & rationale\n",
        f"- **Final feature set ({results['n_selected']}):** kept features appearing in "
        ">=2 of {MI, RFE, Permutation} top-25, with near-constant and lower-MI redundant "
        "partners removed.",
        f"- Removed for redundancy: {results['removed_for_redundancy'] or 'none'}.",
        f"- Tree/boosting models tolerate multicollinearity, so high-VIF composites "
        "(e.g. risk_index, FHI) are retained for them; for logistic regression we monitor VIF.",
        "- This consensus is more robust than any single method and is auditable.",
    ]
    return "\n".join(sections)


if __name__ == "__main__":
    from src.config import get_config
    from src.feature_engineering import build_full_pipeline
    from src.preprocessing.pipeline import prepare_features

    cfg = get_config()
    seed = cfg["project"]["random_seed"]
    df = pd.read_csv(cfg["paths"]["processed_data"])
    X, y = prepare_features(df, target=cfg["dataset"]["target"], protected=cfg.get("protected_attributes", []))

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg["split"]["test_size"],
        stratify=y, random_state=seed,
    )

    pipe = build_full_pipeline()
    X_tr_t = pipe.fit_transform(X_tr)  # fit on TRAIN ONLY
    names = list(pipe.get_feature_names_out())

    results = run_feature_selection(X_tr_t, y_tr.to_numpy(), names, seed=seed)
    results["_corr"] = 0.9  # for report rendering

    report = build_feature_selection_report(results)
    report_path = Path(cfg["paths"]["reports_dir"]) / "feature_selection_report.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("Feature selection report -> %s", report_path)

    artifacts = Path(cfg["paths"]["artifacts_dir"])
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "selected_features.json").write_text(
        json.dumps({"selected": results["final_selected"], "n": results["n_selected"]},
                   indent=2),
        encoding="utf-8",
    )
    logger.info("Saved selected features -> %s", artifacts / "selected_features.json")

    print(f"Total features: {results['n_total']}  ->  Selected: {results['n_selected']}")
    print(f"Variance-dropped: {results['variance_dropped']}")
    print(f"Redundant pairs: {len(results['corr_pairs'])}  -> removed: {results['removed_for_redundancy']}")
    print("\nTop 10 by mutual information:")
    print(results["mutual_info"].head(10).round(4).to_string())
