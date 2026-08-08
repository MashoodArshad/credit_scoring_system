"""Logging configuration for the credit scoring system.

Provides a single ``get_logger`` factory that writes to both the console and a
timestamped log file under ``logs/``. Consistent, structured logging is
essential for debugging, auditing, and reproducibility.

WHY a dedicated logger factory?
    - Every module logs with a consistent format and destination.
    - File logs persist per run for post-hoc debugging and audit trails.
    - Avoids the common "duplicate handler" bug on re-import.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
LOG_DIR: Path = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str = "credit_scoring", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger with console and file handlers.

    Args:
        name: Logger name; pass ``__name__`` from the calling module so log
            entries are traceable to their source file.
        level: Minimum severity level to emit (e.g., ``logging.DEBUG``).

    Returns:
        A configured :class:`logging.Logger` with two handlers attached.

    Note:
        If the logger already has handlers (e.g., on re-import), it is returned
        as-is to prevent duplicated log lines.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:  # idempotent: avoid stacking handlers on re-import
        return logger

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Console handler -> stdout (visible in the running process / notebook).
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler -> one timestamped file per run (audit trail).
    log_file = LOG_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Logging initialized -> %s", log_file)
    return logger
