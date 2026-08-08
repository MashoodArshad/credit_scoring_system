"""Production credit scoring service: validation + scoring + logging.

Wraps the saved end-to-end artifact in a robust service that:
    - validates raw applicant data against an explicit schema,
    - handles bad inputs with clear exceptions (no silent failures),
    - logs every request (volume, decision distribution, latency),
    - scores single applicants or batches, returning probability, decision,
      risk tier, and optional per-applicant reason codes.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.inference.exceptions import InvalidInputError, MissingColumnsError
from src.inference.predict import explain_prediction, predict_applicant
from src.inference.schema import APPLICANT_SCHEMA, REQUIRED_COLUMNS, schema_dataframe
from src.inference.serialize import load_artifact
from src.utils import get_logger

logger = get_logger(__name__)


class CreditScoringService:
    """Load once, score many: the production scoring facade over the artifact."""

    def __init__(self, artifact_path: str | Path, threshold: float | None = None) -> None:
        """Load the model artifact and configuration.

        Args:
            artifact_path: Path to the .joblib artifact.
            threshold: Decision threshold on P(creditworthy); defaults to the
                value stored in the artifact metadata.
        """
        self.pipeline, self.metadata = load_artifact(artifact_path)
        self.threshold = float(threshold) if threshold is not None \
            else float(self.metadata.get("decision_threshold", 0.5))
        self.model_type = self.metadata.get("model_type", "unknown")
        logger.info("CreditScoringService ready (model=%s, threshold=%.2f).",
                    self.model_type, self.threshold)

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate a batch of applicants against the schema.

        Args:
            df: Raw applicant features.

        Returns:
            A clean DataFrame restricted to required columns (in order).

        Raises:
            MissingColumnsError: If any required column is absent.
            InvalidInputError: If numeric values are non-numeric or out of range.
        """
        if not isinstance(df, pd.DataFrame):
            raise InvalidInputError(f"Expected a DataFrame, got {type(df).__name__}.")

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise MissingColumnsError(
                f"Missing required columns: {missing}. Required: {list(REQUIRED_COLUMNS)}"
            )

        extra = [c for c in df.columns if c not in REQUIRED_COLUMNS]
        if extra:
            logger.warning("Ignoring extra columns: %s", extra)

        errors: list[str] = []
        clean = pd.DataFrame(index=df.index)
        for spec in APPLICANT_SCHEMA:
            col = df[spec.name]
            if spec.dtype == "numeric":
                coerced = pd.to_numeric(col, errors="coerce")
                bad_coerce = col.notna() & coerced.isna()
                if bad_coerce.any():
                    sample = col[bad_coerce].head(3).tolist()
                    errors.append(f"'{spec.name}' has non-numeric values: {sample}")
                if spec.minimum is not None:
                    below = coerced.notna() & (coerced < spec.minimum)
                    if below.any():
                        errors.append(f"'{spec.name}' has values < {spec.minimum}: {coerced[below].head(3).tolist()}")
                if spec.maximum is not None:
                    above = coerced.notna() & (coerced > spec.maximum)
                    if above.any():
                        errors.append(f"'{spec.name}' has values > {spec.maximum}: {coerced[above].head(3).tolist()}")
                clean[spec.name] = coerced
            else:  # categorical
                unknown = col.notna() & ~col.isin(spec.allowed or ())
                if unknown.any():
                    # OHE handles unknowns gracefully -> warn, do not fail.
                    logger.warning("'%s' has unknown categories (allowed=%s): %s",
                                   spec.name, spec.allowed, col[unknown].unique()[:5])
                clean[spec.name] = col

        if errors:
            raise InvalidInputError("Input validation failed: " + " | ".join(errors))
        return clean[list(REQUIRED_COLUMNS)]

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    def predict(
        self, df: pd.DataFrame, return_reasons: bool = False
    ) -> pd.DataFrame | tuple[pd.DataFrame, list[pd.DataFrame]]:
        """Validate and score a batch of applicants.

        Args:
            df: Raw applicant features.
            return_reasons: If True, also return per-applicant reason codes.

        Returns:
            Results DataFrame, or (results, reasons) if return_reasons.
        """
        start = time.perf_counter()
        validated = self.validate(df)
        if len(validated) == 0:
            # Graceful no-op for empty requests (pipeline imputers need >=1 row).
            logger.info("Scored 0 applicants (empty input).")
            empty = pd.DataFrame(columns=["p_creditworthy", "p_default", "decision", "risk_tier"])
            return (empty, []) if return_reasons else empty
        results = predict_applicant(self.pipeline, validated, self.threshold)
        reasons = explain_prediction(self.pipeline, validated) if return_reasons else None
        latency = time.perf_counter() - start
        decision_counts = results["decision"].value_counts().to_dict()
        logger.info("Scored %d applicants -> %s in %.3fs",
                    len(df), decision_counts, latency)
        return (results, reasons) if return_reasons else results

    def predict_single(self, applicant: dict[str, Any]) -> dict[str, Any]:
        """Score a single applicant from a dict; returns results + reason codes."""
        df = pd.DataFrame([applicant])
        results, reasons = self.predict(df, return_reasons=True)
        record = results.iloc[0].to_dict()
        record["reasons"] = reasons[0].to_dict("records")
        return record

    def health_check(self) -> dict[str, Any]:
        """Return service status & artifact metadata (for ops/monitoring)."""
        return {
            "status": "ready",
            "model_type": self.model_type,
            "threshold": self.threshold,
            "artifact_version": self.metadata.get("artifact_version"),
            "created_at": self.metadata.get("created_at_utc"),
            "test_metrics": self.metadata.get("metrics", {}).get("test"),
            "n_features": self.metadata.get("n_features"),
        }


if __name__ == "__main__":
    from src.config import get_config
    from src.preprocessing.pipeline import prepare_features

    cfg = get_config()
    artifact = Path(cfg["paths"]["models_dir"]) / "credit_scoring_logreg_v1.joblib"
    service = CreditScoringService(artifact)

    print("=== Health check ===")
    print(service.health_check())

    # Batch scoring on the held-out test set.
    test_df = pd.read_csv(Path(cfg["paths"]["processed_data"]).parent / "test.csv")
    X_test, _ = prepare_features(test_df, target=cfg["dataset"]["target"],
                                 protected=cfg.get("protected_attributes", []))
    print("\n=== Batch scoring (5 applicants) ===")
    print(service.predict(X_test.head(5)).to_string())

    print("\n=== Single applicant (with reasons) ===")
    single = service.predict_single(X_test.iloc[0].to_dict())
    print({k: v for k, v in single.items() if k != "reasons"})
    print("Reasons:", single["reasons"][:3])

    # Robustness: error handling demonstrations.
    print("\n=== Validation / exception handling ===")
    demos = {
        "missing column": X_test.head(2).drop(columns=["credit_score"]),
        "non-numeric age": X_test.head(2).assign(age=["twenty", "thirty"]),
        "out-of-range score": X_test.head(2).assign(credit_score=[999, 1234]),
    }
    for label, bad in demos.items():
        try:
            service.predict(bad)
            print(f"  [{label}] -> no error (unexpected!)")
        except (MissingColumnsError, InvalidInputError) as exc:
            print(f"  [{label}] -> {type(exc).__name__}: {str(exc)[:90]}...")

    # Unknown category -> graceful warning, still scores.
    print("\n=== Unknown category (graceful) ===")
    weird = X_test.head(1).assign(loan_purpose=["Space Travel"])
    print(service.predict(weird).to_string())

    print("\nSchema:")
    print(schema_dataframe().head(6).to_string(index=False))
