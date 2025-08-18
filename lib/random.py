"""Deterministic seeding utility for Python, NumPy, and torch.

Call seed_all() once at process start for reproducibility.
"""
from __future__ import annotations

import os, random
from typing import Optional

import numpy as np

try:  # optional torch
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

_DEF_SEED = 1337


def seed_all(seed: Optional[int] = None) -> int:
    """Seed Python, NumPy and torch (if available). Returns the seed used."""
    s = int(seed if seed is not None else os.environ.get("GLOBAL_SEED", _DEF_SEED))
    random.seed(s)
    np.random.seed(s)
    if torch is not None:
        try:
            torch.manual_seed(s)
            torch.cuda.manual_seed_all(s)
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover
            pass
    return s

__all__ = ["seed_all"]
