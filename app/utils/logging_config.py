"""Logging helpers for deployment services."""

from __future__ import annotations

import logging


def configure_app_logging(level: int = logging.INFO) -> None:
    """Configure standard application logging.

    Args:
        level: Logging level to use for the root configuration.
    """
    logging.basicConfig(level=level, format="%(levelname)s: %(name)s: %(message)s")


def get_logger(name: str) -> logging.Logger:
    """Return a logger for an application module.

    Args:
        name: Logger name.

    Returns:
        logging.Logger: Configured logger instance.
    """
    return logging.getLogger(name)
