"""Input schema for the credit scoring service.

Defines the contract for RAW applicant data (the 21 model-input features), used
to validate requests before scoring. Keeping the schema explicit and centralized
is what makes the service safe and self-documenting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    """Specification of a single raw input feature."""

    name: str
    dtype: str  # "numeric" or "categorical"
    nullable: bool = True
    minimum: float | None = None
    maximum: float | None = None
    allowed: tuple[str, ...] | None = None
    description: str = ""


def _num(name: str, lo: float | None, hi: float | None, desc: str = "") -> FeatureSpec:
    return FeatureSpec(name=name, dtype="numeric", minimum=lo, maximum=hi, description=desc)


def _cat(name: str, allowed: tuple[str, ...], desc: str = "") -> FeatureSpec:
    return FeatureSpec(name=name, dtype="categorical", allowed=allowed, description=desc)


# The 21 raw model-input features (id / target / protected attributes excluded).
APPLICANT_SCHEMA: tuple[FeatureSpec, ...] = (
    _num("age", 18, 100, "Applicant age in years"),
    _num("dependents", 0, 20, "Number of dependents"),
    _cat("education", ("High School", "Bachelor", "Master", "Doctorate")),
    _cat("employment_status", ("Employed", "Self-Employed", "Unemployed")),
    _num("employment_years", 0, 50, "Years in current employment"),
    _num("monthly_income", 0, None, "Gross monthly income"),
    _num("monthly_expenses", 0, None, "Monthly living expenses"),
    _num("savings_balance", 0, None, "Total savings balance"),
    _num("total_assets", 0, None, "Total declared assets"),
    _num("monthly_debt_payment", 0, None, "Monthly debt service"),
    _num("num_open_accounts", 0, 50, "Open credit accounts"),
    _num("num_credit_inquiries_6m", 0, 30, "Hard inquiries last 6m"),
    _num("num_late_payments_12m", 0, 12, "Late payments last 12m"),
    _num("num_previous_defaults", 0, 10, "Prior defaults"),
    _num("months_since_last_delinquency", 0, 120, "Months since last delinquency"),
    _num("credit_utilization_ratio", 0, 1.5, "Revolving utilization"),
    _num("credit_score", 300, 850, "Bureau credit score"),
    _num("interest_rate", 0, 40, "Offered annual interest rate %"),
    _num("loan_amount", 0, None, "Requested loan amount"),
    _num("loan_term_months", 6, 120, "Loan term in months"),
    _cat("loan_purpose", ("Debt Consolidation", "Home", "Auto", "Education",
                          "Personal", "Business", "Medical")),
)

REQUIRED_COLUMNS: tuple[str, ...] = tuple(spec.name for spec in APPLICANT_SCHEMA)


def schema_dataframe() -> pd.DataFrame:
    """Return the schema as a human-readable DataFrame."""
    rows: list[dict[str, Any]] = []
    for spec in APPLICANT_SCHEMA:
        rows.append({
            "feature": spec.name,
            "type": spec.dtype,
            "nullable": spec.nullable,
            "min": spec.minimum,
            "max": spec.maximum,
            "allowed": ", ".join(spec.allowed) if spec.allowed else "-",
        })
    return pd.DataFrame(rows)
