"""Exploratory Data Analysis (EDA): styled plotting + programmatic profiling.

WHY a dedicated EDA module (instead of ad-hoc notebook plotting)?
    - Reproducible, consistently-styled, portfolio-ready figures saved as artifacts.
    - Business insights are derived *programmatically* (data-driven), not eyeballed.
    - Figures go to ``reports/figures/`` so they can be embedded in the README/report.

All heavy matplotlib/seaborn calls use the headless ``Agg`` backend so the module
runs in any environment (notebook, CI, server) without a display.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive backend -> safe in headless/CI environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kurtosis, skew

from src.utils import get_logger

logger = get_logger(__name__)

# ---------- Global visual style (applied once at import) ----------
sns.set_theme(style="whitegrid", palette="deep", font_scale=0.9)
PALETTE: dict[int, str] = {0: "#dc2626", 1: "#16a34a"}  # red=defaulter, green=creditworthy
TARGET_LABELS: dict[int, str] = {0: "Defaulter", 1: "Creditworthy"}
FIG_DPI = 120


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _save_fig(fig: plt.Figure, path: Path) -> None:
    """Save a figure to ``path`` with tight layout and close it (free memory)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure -> %s", path)


def _flatten_axes(n: int, n_cols: int, fig_size: tuple[float, float]) -> tuple[plt.Figure, list]:
    """Create a subplot grid sized for ``n`` panels and return flattened axes."""
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
    axes = np.array(axes).flatten().tolist()
    return fig, axes


# --------------------------------------------------------------------------- #
# 1. Target distribution
# --------------------------------------------------------------------------- #
def plot_target_distribution(df: pd.DataFrame, target: str, save_path: Path) -> pd.Series:
    """Bar plot of the target class balance; returns the value-count series."""
    counts = df[target].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        [TARGET_LABELS.get(int(i), str(i)) for i in counts.index],
        counts.values,
        color=[PALETTE.get(int(i), "#999") for i in counts.index],
    )
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,}\n({value / len(df) * 100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_title("Target Distribution — Creditworthiness", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of applicants")
    ax.set_ylim(0, counts.max() * 1.15)
    _save_fig(fig, save_path)
    return counts


# --------------------------------------------------------------------------- #
# 2. Numeric histograms (class-colored)
# --------------------------------------------------------------------------- #
def plot_numeric_histograms(
    df: pd.DataFrame, cols: list[str], target: str, save_path: Path
) -> None:
    """Histograms of numeric features, overlaid by target class."""
    fig, axes = _flatten_axes(len(cols), 3, (16, 4 * math.ceil(len(cols) / 3)))
    for ax, col in zip(axes, cols):
        for val in sorted(df[target].dropna().unique()):
            data = df[df[target] == val][col].dropna()
            ax.hist(
                data, bins=30, alpha=0.55,
                label=TARGET_LABELS.get(int(val), str(val)),
                color=PALETTE.get(int(val), "#999"),
            )
        ax.set_title(col, fontsize=11)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.suptitle("Numeric Feature Distributions by Class", fontsize=15, y=1.0)
    _save_fig(fig, save_path)


# --------------------------------------------------------------------------- #
# 3 & 4. Boxplots & Violin plots (class-separated) — matplotlib-native for control
# --------------------------------------------------------------------------- #
def _grouped_box_or_violin(
    df: pd.DataFrame, cols: list[str], target: str, save_path: Path, kind: str
) -> None:
    fig, axes = _flatten_axes(len(cols), 3, (16, 4 * math.ceil(len(cols) / 3)))
    vals = sorted(df[target].dropna().unique())
    labels = [TARGET_LABELS.get(int(v), str(v)) for v in vals]
    for ax, col in zip(axes, cols):
        data = [df[df[target] == v][col].dropna().values for v in vals]
        if kind == "box":
            bp = ax.boxplot(
                data, labels=labels, patch_artist=True, showfliers=True,
                flierprops=dict(marker=".", markersize=3, alpha=0.4),
            )
            for patch, v in zip(bp["boxes"], vals):
                patch.set_facecolor(PALETTE.get(int(v), "#999"))
                patch.set_alpha(0.6)
        else:  # violin
            parts = ax.violinplot(data, showmedians=True)
            for body, v in zip(parts["bodies"], vals):
                body.set_facecolor(PALETTE.get(int(v), "#999"))
                body.set_alpha(0.6)
            ax.set_xticks(range(1, len(vals) + 1))
            ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(col, fontsize=11)
        ax.tick_params(labelsize=8)
    for ax in axes[len(cols):]:
        ax.axis("off")
    title = "Boxplots by Class (outliers visible)" if kind == "box" else "Violin Plots by Class (distribution shape)"
    fig.suptitle(title, fontsize=15, y=1.0)
    _save_fig(fig, save_path)


def plot_boxplots_by_target(df: pd.DataFrame, cols: list[str], target: str, save_path: Path) -> None:
    """Class-separated boxplots to compare medians, spread, and outliers."""
    _grouped_box_or_violin(df, cols, target, save_path, kind="box")


def plot_violins_by_target(df: pd.DataFrame, cols: list[str], target: str, save_path: Path) -> None:
    """Class-separated violins to compare full distribution shapes & multimodality."""
    _grouped_box_or_violin(df, cols, target, save_path, kind="violin")


# --------------------------------------------------------------------------- #
# 5. Correlation heatmap
# --------------------------------------------------------------------------- #
def plot_correlation_heatmap(
    df: pd.DataFrame, cols: list[str], target: str, save_path: Path
) -> pd.DataFrame:
    """Pearson correlation heatmap of numeric features + target; returns the matrix."""
    corr = df[cols + [target]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True,
        linewidths=0.5, cbar_kws={"shrink": 0.8}, annot_kws={"size": 7}, ax=ax,
    )
    ax.set_title("Feature Correlation Matrix", fontsize=15, fontweight="bold")
    _save_fig(fig, save_path)
    return corr


# --------------------------------------------------------------------------- #
# 6. Pairplot of key features
# --------------------------------------------------------------------------- #
def plot_pairplot(
    df: pd.DataFrame, cols: list[str], target: str, save_path: Path, sample_n: int = 1500
) -> None:
    """Seaborn pairplot of selected key features, colored by class (subsampled)."""
    sample = df.sample(min(len(df), sample_n), random_state=42)
    grid = sns.pairplot(
        sample[cols + [target]], hue=target, palette=PALETTE, diag_kind="kde",
        plot_kws={"alpha": 0.4, "s": 14}, height=2.2,
    )
    grid.fig.suptitle("Pairplot of Key Credit Features", y=1.02, fontsize=14)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    grid.fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(grid.fig)
    logger.info("Saved figure -> %s", save_path)


# --------------------------------------------------------------------------- #
# 7. Categorical analysis: counts + default rate by category
# --------------------------------------------------------------------------- #
def plot_categorical_counts(
    df: pd.DataFrame, cols: list[str], target: str, save_path: Path
) -> None:
    """Stacked count bars (by class) for each categorical feature."""
    fig, axes = _flatten_axes(len(cols), 3, (16, 4 * math.ceil(len(cols) / 3)))
    vals = sorted(df[target].dropna().unique())
    for ax, col in zip(axes, cols):
        ct = pd.crosstab(df[col], df[target])
        bottom = np.zeros(len(ct))
        for v in vals:
            if v in ct.columns:
                ax.bar(
                    ct.index.astype(str), ct[v], bottom=bottom,
                    label=TARGET_LABELS.get(int(v), str(v)),
                    color=PALETTE.get(int(v), "#999"),
                )
                bottom += ct[v].values
        ax.set_title(col, fontsize=11)
        ax.tick_params(labelsize=8, axis="x", rotation=30)
        ax.legend(fontsize=8)
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.suptitle("Categorical Feature Counts by Class", fontsize=15, y=1.0)
    _save_fig(fig, save_path)


def plot_default_rate_by_category(
    df: pd.DataFrame, cols: list[str], target: str, save_path: Path
) -> dict[str, pd.Series]:
    """Default rate (1 - mean creditworthy) per category; returns a dict of series."""
    fig, axes = _flatten_axes(len(cols), 3, (16, 4 * math.ceil(len(cols) / 3)))
    result: dict[str, pd.Series] = {}
    for ax, col in zip(axes, cols):
        rate = (1 - df.groupby(col, observed=True)[target].mean()).sort_values(ascending=False)
        result[col] = rate
        bars = ax.bar(rate.index.astype(str), rate.values, color="#b91c1c")
        overall = 1 - df[target].mean()
        ax.axhline(overall, color="#1f2937", ls="--", lw=1, label=f"Overall {overall:.1%}")
        ax.set_title(f"Default rate by {col}", fontsize=11)
        ax.set_ylabel("Default rate")
        ax.tick_params(labelsize=8, axis="x", rotation=30)
        ax.legend(fontsize=8)
        for bar, value in zip(bars, rate.values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.0%}",
                    ha="center", va="bottom", fontsize=8)
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.suptitle("Default Rate by Categorical Feature", fontsize=15, y=1.0)
    _save_fig(fig, save_path)
    return result


# --------------------------------------------------------------------------- #
# 8. Skewness / Kurtosis + Outlier (IQR) profiling
# --------------------------------------------------------------------------- #
def skew_kurtosis_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Fisher skewness and excess kurtosis per numeric feature."""
    rows = []
    for col in cols:
        series = df[col].dropna()
        rows.append({
            "feature": col,
            "skewness": round(float(skew(series)), 3),
            "excess_kurtosis": round(float(kurtosis(series)), 3),
            "interpretation": _skew_label(float(skew(series))),
        })
    return pd.DataFrame(rows).sort_values("skewness", key=lambda s: s.abs(), ascending=False)


def _skew_label(sk: float) -> str:
    a = abs(sk)
    if a < 0.5:
        return "approx. symmetric"
    if a < 1.0:
        return "moderately skewed"
    return "highly skewed"


def outlier_summary_iqr(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Tukey IQR outlier counts/bounds per numeric feature."""
    rows = []
    for col in cols:
        series = df[col].dropna()
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((series < low) | (series > high)).sum())
        rows.append({
            "feature": col,
            "lower_bound": round(float(low), 2),
            "upper_bound": round(float(high), 2),
            "n_outliers": n_out,
            "pct_outliers": round(n_out / len(series) * 100, 2),
        })
    return pd.DataFrame(rows).sort_values("pct_outliers", ascending=False)


# --------------------------------------------------------------------------- #
# 9. Orchestrator + data-driven insights report
# --------------------------------------------------------------------------- #
def _md_block(df: pd.DataFrame, title: str | None = None) -> str:
    header = f"**{title}**\n\n" if title else ""
    return f"{header}```\n{df.to_string()}\n```"


def run_full_eda(
    df: pd.DataFrame,
    target: str,
    figures_dir: Path,
    pairplot_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Run the complete EDA suite: generate all figures + profiling tables.

    Args:
        df: Raw dataframe.
        target: Name of the binary target column.
        figures_dir: Directory to save figures into.
        pairplot_cols: Subset of features for the pairplot (defaults to key features).

    Returns:
        Dict of saved figure paths and computed profiling tables.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c != target]
    cat_cols = list(df.select_dtypes(include=["object", "category"]).columns)
    cat_cols = [c for c in cat_cols if c != "customer_id"]
    if pairplot_cols is None:
        pairplot_cols = [
            c for c in ["credit_score", "credit_utilization_ratio",
                        "num_late_payments_12m", "monthly_income", "interest_rate"]
            if c in df.columns
        ]

    logger.info("Running full EDA on %d numeric + %d categorical features.", len(numeric_cols), len(cat_cols))

    counts = plot_target_distribution(df, target, figures_dir / "01_target_distribution.png")
    plot_numeric_histograms(df, numeric_cols, target, figures_dir / "02_numeric_histograms.png")
    plot_boxplots_by_target(df, numeric_cols, target, figures_dir / "03_boxplots_by_class.png")
    plot_violins_by_target(df, numeric_cols, target, figures_dir / "04_violins_by_class.png")
    corr = plot_correlation_heatmap(df, numeric_cols, target, figures_dir / "05_correlation_heatmap.png")
    plot_pairplot(df, pairplot_cols, target, figures_dir / "06_pairplot.png")
    plot_categorical_counts(df, cat_cols, target, figures_dir / "07_categorical_counts.png")
    cat_rates = plot_default_rate_by_category(df, cat_cols, target, figures_dir / "08_default_rate_by_category.png")

    skew_df = skew_kurtosis_table(df, numeric_cols)
    outlier_df = outlier_summary_iqr(df, numeric_cols)

    corr_with_target = corr[target].drop(labels=target).sort_values(key=lambda s: s.abs(), ascending=False)

    return {
        "figures_dir": figures_dir,
        "counts": counts,
        "corr_with_target": corr_with_target,
        "skew_kurtosis": skew_df,
        "outliers": outlier_df,
        "categorical_default_rates": cat_rates,
    }


def build_eda_report(df: pd.DataFrame, target: str, results: dict[str, Any]) -> str:
    """Compile a markdown EDA report with data-driven business insights."""
    counts = results["counts"]
    corr_t = results["corr_with_target"]
    skew_df = results["skew_kurtosis"]
    out_df = results["outliers"]
    cat_rates = results["categorical_default_rates"]
    overall_default = 1 - df[target].mean()

    top_corr = corr_t.head(5)
    top_skew = skew_df.head(5)
    top_out = out_df.head(5)

    # Categorical highlights: most & least risky categories.
    cat_highlights = []
    for col, rate in cat_rates.items():
        if not rate.empty:
            cat_highlights.append(
                f"- **{col}** — highest default: *{rate.index[0]}* ({rate.iloc[0]:.1%}); "
                f"lowest: *{rate.index[-1]}* ({rate.iloc[-1]:.1%})"
            )

    sections = [
        "# Exploratory Data Analysis (EDA) Report",
        f"_Records:_ {len(df):,}  \n_Target:_ `{target}`  \n_Overall default rate:_ **{overall_default:.1%}**",
        "\n## 1. Target balance\n",
        f"- Creditworthy (1): **{counts.get(1, 0):,}** ({counts.get(1, 0) / len(df):.1%})",
        f"- Defaulter (0): **{counts.get(0, 0):,}** ({counts.get(0, 0) / len(df):.1%})",
        f"- **Insight:** The dataset is **imbalanced** ({overall_default:.0%} minority). "
        "We will use stratified splits and class weighting; accuracy is misleading here.",
        "\n## 2. Features most correlated with the target\n",
        _md_block(top_corr.to_frame("pearson_corr"), "Top |correlation| with target"),
        "\n## 3. Skewness / Kurtosis (top |skew|)\n",
        _md_block(top_skew, "Most skewed features"),
        "- **Insight:** Highly skewed monetary features (income, assets, loan amount) "
        "will benefit from log/quantile transformation in Phase 5 to stabilize model learning.",
        "\n## 4. Outlier profile (IQR, top by %)\n",
        _md_block(top_out, "Features with most outliers"),
        "- **Insight:** Many 'outliers' in credit features are **legitimate extreme risk** "
        "(high utilization, many late payments) — we will *capping (winsorize)* monetary "
        "fields rather than dropping behavioral extremes.",
        "\n## 5. Default rate by categorical feature\n",
        "\n".join(cat_highlights) if cat_highlights else "_No categorical features._",
        "\n## 6. Figures (saved to reports/figures/)\n",
        "- `01_target_distribution.png` — class balance",
        "- `02_numeric_histograms.png` — distributions by class",
        "- `03_boxplots_by_class.png` — spread & outliers by class",
        "- `04_violins_by_class.png` — distribution shape by class",
        "- `05_correlation_heatmap.png` — feature/target correlations",
        "- `06_pairplot.png` — pairwise relationships (key features)",
        "- `07_categorical_counts.png` — category counts by class",
        "- `08_default_rate_by_category.png` — risk ranking per category",
    ]
    return "\n".join(sections)


if __name__ == "__main__":
    from src.config import get_config

    cfg = get_config()
    raw_df = pd.read_csv(cfg["paths"]["raw_data"])
    figures = Path(cfg["paths"]["figures_dir"])
    eda_results = run_full_eda(raw_df, cfg["dataset"]["target"], figures)
    report_md = build_eda_report(raw_df, cfg["dataset"]["target"], eda_results)
    out = Path(cfg["paths"]["reports_dir"]) / "eda_report.md"
    out.write_text(report_md, encoding="utf-8")
    logger.info("EDA report written -> %s", out)
    print("Top correlations with target:")
    print(eda_results["corr_with_target"].head(8).to_string())
