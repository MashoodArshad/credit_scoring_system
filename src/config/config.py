"""Centralized configuration management.

Loads parameters from YAML files so that *no* path, seed, or hyperparameter
is hard-coded in source code. This is a core industry practice: it makes the
project reproducible, easy to tune, and portable across environments.

WHY a config file instead of constants in code?
    - Reproducibility: the same config + code => the same model.
    - Separation of concerns: code describes *how*, config describes *what*.
    - Portability: switch datasets/paths/seeds without touching logic.
    - Auditing: a single file documents every modeling decision.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Project root = two levels up from this file (src/config/config.py -> root).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load a YAML configuration file into a nested dictionary.

    Args:
        config_path: Path to the YAML config file. Defaults to
            ``config/config.yaml`` at the project root.

    Returns:
        Parsed configuration as a nested dictionary.

    Raises:
        FileNotFoundError: If ``config_path`` does not exist.
        yaml.YAMLError: If the file exists but is not valid YAML.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    logger.info("Loading configuration from %s", config_path)
    with config_path.open("r", encoding="utf-8") as file_handle:
        config: dict[str, Any] = yaml.safe_load(file_handle)

    logger.info("Configuration loaded successfully (%d top-level keys).", len(config))
    return config


def get_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Public accessor used across modules to obtain the configuration dict.

    Thin wrapper around :func:`load_config` kept for a stable, descriptive API
    at call sites (``config = get_config()``).
    """
    return load_config(config_path)
