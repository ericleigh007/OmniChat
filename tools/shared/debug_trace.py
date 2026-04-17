"""Helpers for lightweight OmniChat debug tracing."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


_TRACE_LOGGER_NAME = "omnichat.trace"
_TRACE_LOG_PATH = Path(__file__).resolve().parents[2] / "outputs" / "debug" / "omnichat_trace.log"


def get_trace_log_path() -> Path:
    """Return the canonical trace log path for OmniChat runs."""
    return _TRACE_LOG_PATH


def get_trace_logger() -> logging.Logger:
    """Return a process-wide trace logger that writes to outputs/debug."""
    logger = logging.getLogger(_TRACE_LOGGER_NAME)
    if logger.handlers:
        return logger

    log_dir = _TRACE_LOG_PATH.parent
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(_TRACE_LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
