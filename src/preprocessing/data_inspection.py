"""Data inspection utilities and Data Quality (DQ) report generation.

These functions are *pure reporting* tools — they never mutate the input
DataFrame, so they are safe to call at any pipeline stage for diagnostics.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import get_config
from src.utils import get_logger

logger = get_logger(__name__)


def basic_overview(df: pd.DataFrame) -> dict[str, Any]:
    """Return high-level shape and memory metadata.

    Args:
        df: Input dataframe.

    Returns:
        Dict with row count, column count, memory (MB), and duplicate-row count.
    """
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "memory_mb": round(float(df.memory_usage(deep=True).sum()) / 1024**2, 3),
        "n_duplicate_rows": int(df.duplicated().sum()),
    }


def column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-column profile: dtype, nulls, missing %, cardinality.

    Args:
        df: Input dataframe.

    Returns:
        One row per column with profile metrics.
    """
    rows: list[dict[str, Any]] = []
    for col in df.columns:
        missing = int(df[col].isna().sum())
        rows.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "non_null": int(df[col].notna().sum()),
            "missing": missing,
            "missing_pct": round(missing / len(df) * 100, 2),
            "n_unique": int(df[col].nunique(dropna=True)),
        })
    return pd.DataFrame(rows)


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Columns containing missing values, sorted by missing % (desc)."""
    summary = column_summary(df)
    miss = summary[summary["missing"] > 0].sort_values("missing_pct", ascending=False)
    return miss[["column", "dtype", "missing", "missing_pct"]].reset_index(drop=True)


def duplicates_report(df: pd.DataFrame, id_col: str = "customer_id") -> dict[str, Any]:
    """Report full-row duplicates and duplicate identifier values.

    Args:
        df: Input dataframe.
        id_col: Identifier column used for the duplicate-id check.

    Returns:
        Dict of duplicate counts.
    """
    result: dict[str, Any] = {"full_duplicate_rows": int(df.duplicated().sum())}
    if id_col in df.columns:
        result[f"duplicate_{id_col}"] = int(df.duplicated(subset=[id_col]).sum())
    return result


def categorical_summary(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Long-format summary of top values per categorical column."""
    cat = df.select_dtypes(include=["object", "category"])
    rows: list[dict[str, Any]] = []
    for col in cat.columns:
        for value, count in df[col].value_counts().head(top_n).items():
            rows.append({"column": col, "value": value, "count": int(count)})
    return pd.DataFrame(rows)


def _md_block(df: pd.DataFrame, title: str | None = None) -> str:
    """Render a DataFrame as a fenced markdown code block."""
    header = f"**{title}**\n\n" if title else ""
    return f"{header}```\n{df.to_string()}\n```"


def generate_data_quality_report(
    df: pd.DataFrame,
    id_col: str = "customer_id",
    source: str = "",
) -> str:
    """Compile a complete markdown Data Quality report for a DataFrame.

    Args:
        df: Input dataframe to audit.
        id_col: Identifier column for duplicate checks.
        source: Optional source path string for the report header.

    Returns:
        A markdown string describing data quality.
    """
    overview = basic_overview(df)
    col_summ = column_summary(df)
    miss = missing_report(df)
    dups = duplicates_report(df, id_col=id_col)

    numeric = df.select_dtypes(include="number")
    numeric_desc = numeric.describe().T.round(2) if not numeric.empty else numeric
    cat_summ = categorical_summary(df)

    total_missing_pct = round(float(df.isna().mean().mean()) * 100, 2)
    flags: list[str] = []
    if miss.shape[0]:
        flags.append(f"{miss.shape[0]} columns contain missing values "
                     f"(overall cell-level missing: {total_missing_pct}%).")
    if dups["full_duplicate_rows"]:
        flags.append(f"{dups['full_duplicate_rows']} fully duplicated rows detected.")
    if dups.get(f"duplicate_{id_col}"):
        flags.append(f"{dups[f'duplicate_{id_col}']} duplicate `{id_col}` values (integrity issue).")
    if not flags:
        flags.append("No major data-quality issues detected.")

    sections = [
        "# Data Quality Report",
        f"_Generated:_ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n"
        f"_Source:_ `{source or 'in-memory'}`  \n"
        f"_Records:_ {overview['n_rows']:,} × _Columns:_ {overview['n_cols']}",
        "\n## 1. Overview\n",
        f"- Rows: **{overview['n_rows']:,}**",
        f"- Columns: **{overview['n_cols']}**",
        f"- Memory: **{overview['memory_mb']} MB**",
        f"- Fully duplicated rows: **{overview['n_duplicate_rows']:,}**",
        "\n## 2. Column Summary\n",
        _md_block(col_summ, "Per-column profile"),
        "\n## 3. Missing Values\n",
        _md_block(miss, "Columns with missing values") if not miss.empty else "_No missing values._",
        "\n## 4. Duplicates\n",
        "\n".join(f"- {k}: {v:,}" for k, v in dups.items()),
        "\n## 5. Numeric Summary Statistics\n",
        _md_block(numeric_desc) if not numeric_desc.empty else "_No numeric columns._",
        "\n## 6. Categorical Summary (top 5 per column)\n",
        _md_block(cat_summ) if not cat_summ.empty else "_No categorical columns._",
        "\n## 7. Data Quality Flags\n",
        "\n".join(
            f"- ✅ {f}" if "No major" in f else f"- ⚠️ {f}" for f in flags
        ),
    ]
    return "\n".join(sections)


if __name__ == "__main__":
    cfg = get_config()
    raw_path = Path(cfg["paths"]["raw_data"])
    dataframe = pd.read_csv(raw_path)
    report = generate_data_quality_report(dataframe, id_col="customer_id", source=str(raw_path))
    out_path = Path(cfg["paths"]["reports_dir"]) / "data_quality_report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Data Quality Report written -> %s", out_path)
    print("Overview:", basic_overview(dataframe))
