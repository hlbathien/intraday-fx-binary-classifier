"""Configuration loader utility.

Resolves '@/relative' paths relative to project root. Provides cached
access to key directories. Ensures output dirs exist.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = PROJECT_ROOT / ".config"


@dataclass(frozen=True)
class Config:
    raw_data_path: Path
    cleaned_data_path: Path
    featured_data_path: Path
    sequenced_data_path: Path
    model_output: Path


def _resolve(root: Path, value: str) -> Path:
    if value.startswith("@/"):
        return (root / value[2:]).resolve()
    return Path(value).resolve()


@lru_cache(maxsize=1)
def load_config() -> Config:
    if not CONFIG_FILE.exists():  # pragma: no cover
        raise FileNotFoundError(f"Missing config file: {CONFIG_FILE}")
    kv: Dict[str, str] = {}
    root = PROJECT_ROOT
    for line in CONFIG_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        kv[k.strip()] = v.strip()
    raw = _resolve(root, kv["RAW_DATA_PATH"])
    cleaned = _resolve(root, kv["CLEANED_DATA_PATH"])
    featured = _resolve(root, kv["FEATURED_DATA_PATH"])
    sequenced = _resolve(root, kv.get("SEQUENCED_DATA_PATH", "@/data/sequenced"))
    model_out = _resolve(root, kv["MODEL_OUTPUT"])
    for p in (sequenced, model_out):
        p.mkdir(parents=True, exist_ok=True)
    return Config(raw, cleaned, featured, sequenced, model_out)


__all__ = ["Config", "load_config"]
