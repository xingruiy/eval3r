"""Rich-based logger used across eval3r."""

from __future__ import annotations

import logging

from rich.logging import RichHandler

_CONFIGURED = False


def get_logger(name: str = "eval3r") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        handler = RichHandler(rich_tracebacks=True, show_time=False, show_path=False)
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[handler],
        )
        _CONFIGURED = True
    return logging.getLogger(name)
