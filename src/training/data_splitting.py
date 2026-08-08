"""Data splitting & cross-validation strategy (leakage-safe).

Establishes the single, reproducible train/validation/test split used by every
downstream phase, plus the stratified k-fold CV strategy for model selection
and hyperparameter tuning. Splitting is performed on *rows* of the cleaned
dataframe so all columns (features, target, id, protected attrs) stay aligned.

WHY stratified splitting?
    The target is imbalanced (~18% defaulters). Plain random splits could leave
    a fold/split with too few minority examples -> unstable metrics. Stratified
    splits preserve the class proportion in every partition.

LEAKAGE-PREVENTION PROTOCOL (enforced project-wide):
    1. Split FIRST, before any fitting of imputers/scalers/selectors.
    2. Preprocessing pipeline is fit on TRAIN ONLY, then transformed onto val/test.
    3. Feature selection is fit on TRAIN ONLY.
    4. Hyperparameter tuning uses CV folds drawn from TRAIN ONLY (test untouched).
    5. The TEST set is touched exactly ONCE, for final evaluation.
    6. No row-level information (e.g., target-derived, post-outcome) is used as a feature.
    7. Point-in-time design: features measured as-of application (no future leakage).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.utils import get_logger

logger = get_logger(__name__)

LEAKAGE_PROTOCOL: tuple[str, ...] = (
    "Split FIRST, before fitting any imputer/scaler/selector.",
    "Preprocessing pipeline fit on TRAIN ONLY, then applied to val/test.",
    "Feature selection fit on TRAIN ONLY.",
    "Hyperparameter tuning uses CV folds from TRAIN ONLY; test set untouched.",
    "TEST set evaluated exactly ONCE, for final reporting.",
    "No target-derived or post-outcome information used as a feature.",
    "Point-in-time features only (measured as-of application).",
)


def split_dataframe(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.20,
    val_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/validation/test split on dataframe rows.

    Args:
        df: Cleaned dataframe (any column layout).
        target: Stratification column.
        test_size: Fraction of the full data reserved for the test set.
        val_size: Fraction of the *remaining* (post-test) data reserved for
            validation. E.g. test_size=0.2, val_size=0.2 -> 64/16/20 split.
        random_state: Seed for full reproducibility.

    Returns:
        ``(train_df, val_df, test_df)``.
    """
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[target], random_state=random_state
    )
    train, val = train_test_split(
        train_val, test_size=val_size, stratify=train_val[target], random_state=random_state
    )
    logger.info(
        "Split complete (stratified): train=%d, val=%d, test=%d (total=%d).",
        len(train), len(val), len(test), len(df),
    )
    return train, val, test


def build_cv(n_splits: int = 5, random_state: int = 42) -> StratifiedKFold:
    """Return a stratified k-fold CV strategy (shuffle + seed-pinned)."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def class_distribution(df: pd.DataFrame, target: str) -> dict[str, float]:
    """Return class proportions (fractions) for a dataframe."""
    counts = df[target].value_counts(normalize=True).sort_index()
    return {str(int(k)): round(float(v), 4) for k, v in counts.items()}


def build_split_report(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    cv_folds: int = 5,
) -> str:
    """Compile a markdown report of the split + CV + leakage protocol."""
    splits = {"train": train, "validation": val, "test": test}
    rows = []
    for name, frame in splits.items():
        dist = class_distribution(frame, target)
        n_bad = int((frame[target] == 0).sum())
        rows.append({
            "split": name,
            "n_rows": len(frame),
            "pct_of_total": round(len(frame) / (len(train) + len(val) + len(test)) * 100, 1),
            "creditworthy(1)": dist.get("1", 0.0),
            "defaulter(0)": dist.get("0", 0.0),
            "default_rate": round(n_bad / len(frame), 4),
        })
    table = pd.DataFrame(rows)

    sections = [
        "# Data Split & Cross-Validation Report",
        f"_Stratification column:_ `{target}`  \n_Seed:_ pinned (random_state=42)  \n"
        f"_CV:_ Stratified {cv_folds}-fold (on train only)",
        "\n## 1. Partition sizes & class balance\n",
        "```\n" + table.to_string(index=False) + "\n```",
        "\n**Check:** default rate is preserved across all splits (stratification working).",
        "\n## 2. Cross-validation strategy\n",
        f"- **Stratified {cv_folds}-fold** CV is used for model comparison and tuning.",
        "- Folds are drawn from the TRAIN set only; validation/test are never in CV.",
        "- A single validation set additionally supports early-stopping for boosting models.",
        "\n## 3. Leakage-prevention protocol (enforced project-wide)\n",
        "\n".join(f"- {step}" for step in LEAKAGE_PROTOCOL),
    ]
    return "\n".join(sections)


if __name__ == "__main__":
    from src.config import get_config

    cfg = get_config()
    df = pd.read_csv(cfg["paths"]["processed_data"])
    target = cfg["dataset"]["target"]
    seed = cfg["project"]["random_seed"]

    train, val, test = split_dataframe(
        df, target=target,
        test_size=cfg["split"]["test_size"],
        val_size=cfg["split"]["validation_size"],
        random_state=seed,
    )

    # Persist reproducible split artifacts for downstream phases.
    processed_dir = Path(cfg["paths"]["processed_data"]).parent
    for name, frame in (("train", train), ("validation", val), ("test", test)):
        out = processed_dir / f"{name}.csv"
        frame.to_csv(out, index=False)
        logger.info("Saved %s -> %s (%d rows)", name, out, len(frame))

    report = build_split_report(train, val, test, target, cv_folds=cfg["modeling"]["cv_folds"])
    report_path = Path(cfg["paths"]["reports_dir"]) / "data_split_report.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("Split report -> %s", report_path)

    print("\nClass balance per split (default rate):")
    for name, frame in (("train", train), ("validation", val), ("test", test)):
        print(f"  {name:<10} n={len(frame):<5} default_rate="
              f"{(frame[target]==0).mean():.4f}")
