"""Project-wide logging configuration."""

from __future__ import annotations

import logging
import os


def configure_logging(default_level: str = "INFO") -> None:
    """Configure root logging once for CLI, pipeline, and dashboard usage."""
    level_name = os.getenv("LOG_LEVEL", default_level).upper()
    level = getattr(logging, level_name, logging.INFO)

    if logging.getLogger().handlers:
        logging.getLogger().setLevel(level)
        return

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

