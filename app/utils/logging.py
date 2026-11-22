"""Minimal, consistent logging setup shared across modules.

Modules call ``get_logger(__name__)`` at import time. Entry points (CLI, UI) call
``configure_logging(cfg.logging.level)`` once the config is loaded, so the level
in config.yaml is what actually takes effect.
"""
from __future__ import annotations

import logging

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """(Re)apply the root log level; installs the handler on first call."""
    global _CONFIGURED
    numeric = getattr(logging, str(level).upper(), logging.INFO)
    if not _CONFIGURED:
        logging.basicConfig(level=numeric, format=_FORMAT, datefmt="%H:%M:%S")
        _CONFIGURED = True
    logging.getLogger().setLevel(numeric)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, installing a default (INFO) handler once."""
    if not _CONFIGURED:
        configure_logging("INFO")
    return logging.getLogger(name)
