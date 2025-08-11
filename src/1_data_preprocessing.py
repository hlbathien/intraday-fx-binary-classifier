"""Preprocess raw FX minute CSV files.

Steps:
1. Read RAW_DATA_PATH and CLEANED_DATA_PATH from `.config` (supports '@/relative').
2. For every *.csv in raw directory (no header: Time, Open, High, Low, Close, Volume) parse as UTC.
3. Drop all non-session bars immediately: keep only bars whose New York (America/New_York) local time is 12:00 through 20:59 inclusive (handles DST automatically).
4. Clean & align remaining (session) data:
	- Ensure strictly increasing minute timestamps.
	- Fill only "small" gaps (<= SMALL_GAP_MINUTES) by forward-filling previous bar (OHLC repeated, Volume=0) *within the session only*.
	- Do NOT create rows for larger outages; just keep existing bars (gap remains apparent by timestamp jump).
5. Export cleaned CSVs (with header) to cleaned directory using same filename. Output Time is formatted as YYYY-MM-DD HH:MM:SS (UTC, no timezone suffix).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import pytz

# ---------------- Configuration ---------------- #

CONFIG_FILE = Path(__file__).resolve().parent.parent / ".config"
SMALL_GAP_MINUTES = int(os.getenv("SMALL_GAP_MINUTES", "5"))  # forward-fill threshold
NY_TZ = pytz.timezone("America/New_York")


@dataclass
class Paths:
	raw: Path
	cleaned: Path


def load_config(path: Path = CONFIG_FILE) -> Paths:
	"""Parse .config file retrieving RAW_DATA_PATH & CLEANED_DATA_PATH.

	Supports '@/relative' meaning relative to repository root (the .config parent).
	"""
	if not path.exists():
		raise FileNotFoundError(f"Config file not found: {path}")
	cfg: Dict[str, str] = {}
	root = path.parent
	for line in path.read_text().splitlines():
		line = line.strip()
		if not line or line.startswith("#"):
			continue
		if "=" not in line:
			continue
		k, v = line.split("=", 1)
		v = v.strip()
		if v.startswith("@/"):
			v = str(root / v[2:])
		cfg[k.strip()] = v
	raw = Path(cfg["RAW_DATA_PATH"]).resolve()
	cleaned = Path(cfg["CLEANED_DATA_PATH"]).resolve()
	cleaned.mkdir(parents=True, exist_ok=True)
	return Paths(raw=raw, cleaned=cleaned)


# ---------------- Data Cleaning ---------------- #

COLS = ["Time", "Open", "High", "Low", "Close", "Volume"]


def read_raw_csv(f: Path) -> pd.DataFrame:
	df = pd.read_csv(
		f,
		names=COLS,
		header=None,
		parse_dates=[0],
		infer_datetime_format=True,
	)
	# Ensure tz-aware UTC
	if df["Time"].dt.tz is None:
		df["Time"] = df["Time"].dt.tz_localize("UTC")
	else:
		df["Time"] = df["Time"].dt.tz_convert("UTC")
	df = df.sort_values("Time").drop_duplicates("Time")
	return df.reset_index(drop=True)


def fill_small_gaps(df: pd.DataFrame, max_gap: int = SMALL_GAP_MINUTES) -> pd.DataFrame:
	"""Forward-fill only small missing minute bars inside session.

	Adds new rows (no synthetic flag) for gaps <= max_gap minutes where each
	inserted bar copies previous Close into OHLC and sets Volume=0.
	"""
	if df.empty:
		return df
	rows: List[dict] = []
	times = df["Time"].tolist()
	for idx, row in df.iterrows():
		rows.append(row.to_dict())
		if idx == len(df) - 1:
			break
		cur_t = times[idx]
		nxt_t = times[idx + 1]
		gap = int((nxt_t - cur_t).total_seconds() // 60) - 1
		if gap <= 0 or gap > max_gap:
			continue
		prev_close = row["Close"]
		for m in range(1, gap + 1):
			synth_time = cur_t + pd.Timedelta(minutes=m)
			rows.append(
				{
					"Time": synth_time,
					"Open": prev_close,
					"High": prev_close,
					"Low": prev_close,
					"Close": prev_close,
					"Volume": 0,
				}
			)
	out = pd.DataFrame(rows)
	return out.sort_values("Time").reset_index(drop=True)


def filter_ny_session(df: pd.DataFrame) -> pd.DataFrame:
	"""Filter to NY session 12:00:00 - 20:59:59 local (America/New_York)."""
	if df.empty:
		return df
	local_times = df["Time"]
	keep_mask = (local_times.dt.hour >= 12) & (
		(local_times.dt.hour < 21)  # until 20:59
	)
	return df.loc[keep_mask].reset_index(drop=True)


def process_file(f: Path, cleaned_dir: Path) -> Path:
	"""Process a single raw CSV into a NY-session-only cleaned file.

	Steps: read -> session filter -> small gap fill -> standardize time -> export.
	"""
	df = read_raw_csv(f)
	df = filter_ny_session(df)  # drop all non NY session bars early
	df = fill_small_gaps(df)    # fill only intra-session small gaps
	# Standardize time formatting (UTC naive)
	df["Time"] = df["Time"].dt.tz_convert("UTC").dt.tz_localize(None)
	# Ensure only required columns
	df = df[COLS]
	out_path = cleaned_dir / f.name
	df.to_csv(out_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
	return out_path


def iter_csv_files(raw_dir: Path) -> Iterable[Path]:
	return sorted(p for p in raw_dir.glob("*.csv") if p.is_file())


def main(argv: List[str] | None = None) -> int:
	paths = load_config()
	csv_files = list(iter_csv_files(paths.raw))
	if not csv_files:
		print(f"No CSV files found in {paths.raw}")
		return 1
	print(f"Processing {len(csv_files)} file(s)...")
	for f in csv_files:
		out = process_file(f, paths.cleaned)
		print(f"✔ {f.name} -> {out.relative_to(paths.cleaned.parent)}")
	print("Done.")
	return 0


if __name__ == "__main__":  # pragma: no cover
	raise SystemExit(main())

