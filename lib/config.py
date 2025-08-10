"""
Config loader for project-wide settings from .config and .env.
Reads RAW_DATA_PATH and CLEANED_DATA_PATH for data preprocessing steps.
"""
import os
from pathlib import Path
from typing import Dict

CONFIG_PATH = Path(__file__).parent.parent / ".config"


def _parse_config(path: Path) -> Dict[str, str]:
    """Parse .config file as key=value pairs, ignoring comments and blanks."""
    config = {}
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            config[k.strip()] = v.strip()
    return config


def get_config() -> Dict[str, str]:
    return _parse_config(CONFIG_PATH)


def get_raw_data_path() -> Path:
    cfg = get_config()
    return Path(cfg["RAW_DATA_PATH"])


def get_cleaned_data_path() -> Path:
    cfg = get_config()
    return Path(cfg["CLEANED_DATA_PATH"])
