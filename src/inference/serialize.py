"""Model serialization & versioning.

Saves the trained end-to-end pipeline with joblib (preferred for sklearn/numpy
objects) and a JSON metadata sidecar capturing version, environment, threshold,
metrics, feature names, and hyperparameters -> fully reproducible, auditable
artifacts. A pickle alternative is provided for completeness.
"""
from __future__ import annotations

import json
import pickle
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import sklearn

from src.utils import get_logger

logger = get_logger(__name__)

ARTIFACT_VERSION = "1.0.0"


def build_metadata(
    model_type: str,
    threshold: float,
    metrics: dict[str, float],
    feature_names: list[str],
    params: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a reproducibility/audit metadata dictionary."""
    return {
        "artifact_version": ARTIFACT_VERSION,
        "model_type": model_type,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "python_version": sys.version.split()[0],
            "sklearn_version": sklearn.__version__,
            "platform": platform.platform(),
        },
        "decision_threshold": threshold,
        "metrics": metrics,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "params": {k: v for k, v in params.items()},
        "extra": extra or {},
    }


def save_artifact(pipeline: Any, path: str | Path, metadata: dict[str, Any]) -> tuple[Path, Path]:
    """Persist a fitted pipeline (joblib) + JSON metadata sidecar.

    Args:
        pipeline: Fitted sklearn Pipeline (the end-to-end artifact).
        path: Destination path for the .joblib file.
        metadata: Metadata dict from :func:`build_metadata`.

    Returns:
        (joblib_path, metadata_path).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    meta_path = path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    logger.info("Saved artifact -> %s (+ %s)", path, meta_path)
    return path, meta_path


def save_pickle(obj: Any, path: str | Path) -> Path:
    """Alternative serialization via pickle (less efficient for numpy/sklearn)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(obj, fh)
    logger.info("Saved pickle -> %s", path)
    return path


def load_artifact(path: str | Path) -> tuple[Any, dict[str, Any]]:
    """Load a joblib artifact and its metadata sidecar.

    Args:
        path: Path to the .joblib file.

    Returns:
        (pipeline, metadata). metadata is {} if no sidecar is found.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    pipeline = joblib.load(path)
    meta_path = path.with_suffix(".meta.json")
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    logger.info("Loaded artifact -> %s (v%s)", path, metadata.get("artifact_version", "?"))
    return pipeline, metadata
