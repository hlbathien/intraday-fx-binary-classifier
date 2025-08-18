"""Project logging initializer.

Provides init_logging(level) to standardize formatting.
"""
from __future__ import annotations

import logging
from typing import Optional

_FMT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def init_logging(level: int | str = logging.INFO) -> None:
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    if root.handlers:
        for h in root.handlers:
            root.removeHandler(h)
    logging.basicConfig(level=level, format=_FMT)

__all__ = ["init_logging"]
