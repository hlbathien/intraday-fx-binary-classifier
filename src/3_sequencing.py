"""Build fixed-length (60-minute) per-symbol sequence tensors and aligned multi-horizon
targets from previously generated featured parquet files.

Spec (per user pseudocode):
1. Read FEATURED_DATA_PATH & SEQUENCED_DATA_PATH from .config (supports '@/relative').
   If SEQUENCED_DATA_PATH absent, default to '@/data/sequenced'.
2. Featured directory structure: <FEATURED_DATA_PATH>/<horizon>m/<SYMBOL>.parquet
   with horizons in HORIZONS.
3. Candidate prediction timestamps t are ONLY top-of-hour (minute == 0). Each sequence
   covers strictly the preceding hour window [t-60m, t-1m] inclusive consisting of 60
   contiguous minute rows whose start minute must also be 0 (strict hour block).
4. Core per-minute features inside each sequence (order preserved oldest->newest):
	  SEQ_CORE_FEATS = ["r1", "rv_5m", "rv_30m", "vol_z_5", "vol_z_30"].
   NOTE: The featured parquet files do not contain raw 'r1'. We reconstruct r1 via:
		 r1_at_time_t = r_1m shifted -1 (since 'r_1m' is the previous minute's r1).
5. Export for each symbol an .npz file stored at SEQUENCED_DATA_PATH/<symbol>_seq.npz
   containing:
	  - seq: float32 array shape (N, 60, F) where F=len(SEQ_CORE_FEATS)
	  - times: ISO-8601 UTC strings for prediction timestamps t (length N)
	  - For each horizon k: y_sign_{k}, y_mag_{k} arrays length N (float32/int8)
	  - meta JSON (serialized str) with feature list, horizons, sequence_length.
6. Walk-forward split metadata (date spans) is also produced (JSON list) using
   configured block sizes & embargo per pseudocode and saved once to
   SEQUENCED_DATA_PATH/splits.json (only if not already present or overwrite=True).

Determinism: purely deterministic transforms (no RNG). Any future stochastic
components should seed explicitly.

Assumptions / Notes:
* If any minute inside the required hour window is missing, that prediction time is skipped.
* Rows with any NaN among required core features inside the window are skipped.
* A prediction time is also skipped if ANY horizon target (sign/mag) is NaN.
* The feature parquet files (one per horizon) contain identical feature columns; we
  load only the 1m horizon file for per-minute features and then merge target columns
  from all horizons to avoid redundant memory usage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence
import json
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------- Configuration Constants ---------------- #

CONFIG_FILE = Path(__file__).resolve().parent.parent / ".config"
HORIZONS: List[int] = [1, 5, 10, 15]
SEQ_LEN = 60
SEQ_CORE_FEATS = ["r1", "rv_5m", "rv_30m", "vol_z_5", "vol_z_30"]

# Walk-forward split parameters (days based)
EMBARGO_MIN = 60
TRAIN_BLOCK_DAYS = 60
VAL_BLOCK_DAYS = 7
TEST_BLOCK_DAYS = 21


@dataclass
class Paths:
	featured: Path
	sequenced: Path


def load_config(path: Path = CONFIG_FILE) -> Paths:
	if not path.exists():  # pragma: no cover
		raise FileNotFoundError(f"Config file not found: {path}")
	cfg: Dict[str, str] = {}
	root = path.parent
	for line in path.read_text().splitlines():
		line = line.strip()
		if not line or line.startswith("#") or "=" not in line:
			continue
		k, v = line.split("=", 1)
		v = v.strip()
		if v.startswith("@/"):
			v = str(root / v[2:])
		cfg[k.strip()] = v
	featured = Path(cfg["FEATURED_DATA_PATH"]).resolve()
	# Optional sequenced path with fallback
	seq_path_str = cfg.get("SEQUENCED_DATA_PATH", str(root / "data" / "sequenced"))
	sequenced = Path(seq_path_str).resolve()
	sequenced.mkdir(parents=True, exist_ok=True)
	return Paths(featured=featured, sequenced=sequenced)


# ---------------- Splits (walk-forward with embargo) ---------------- #

def make_walkforward_splits(
	index: pd.DatetimeIndex,
	train_days: int,
	val_days: int,
	test_days: int,
	embargo_minutes: int,
) -> List[Dict[str, str]]:
	if index.empty:
		return []
	index = index.sort_values()
	splits: List[Dict[str, str]] = []
	start = index.min().normalize()  # align to day start
	last_time = index.max()
	while True:
		train_start = start
		train_end = train_start + pd.Timedelta(days=train_days) - pd.Timedelta(minutes=1)
		val_start = train_end + pd.Timedelta(minutes=1 + embargo_minutes)
		val_end = val_start + pd.Timedelta(days=val_days) - pd.Timedelta(minutes=1)
		test_start = val_end + pd.Timedelta(minutes=1 + embargo_minutes)
		test_end = test_start + pd.Timedelta(days=test_days) - pd.Timedelta(minutes=1)
		if test_end > last_time:
			break
		splits.append(
			{
				"train": f"{train_start.isoformat()}|{train_end.isoformat()}",
				"val": f"{val_start.isoformat()}|{val_end.isoformat()}",
				"test": f"{test_start.isoformat()}|{test_end.isoformat()}",
				"embargo_minutes": embargo_minutes,
			}
		)
		# Slide by test block
		start = test_start
	return splits


# ---------------- Core Logic ---------------- #

def load_symbol_feature_frames(symbol: str, featured_root: Path) -> Dict[int, pd.DataFrame]:
	frames: Dict[int, pd.DataFrame] = {}
	for k in HORIZONS:
		fp = featured_root / f"{k}m" / f"{symbol}.parquet"
		if not fp.exists():
			raise FileNotFoundError(f"Missing featured file for {symbol} horizon {k}: {fp}")
		df = pd.read_parquet(fp)
		if not isinstance(df.index, pd.DatetimeIndex):
			# Attempt to restore index if stored as column
			if "Time" in df.columns:
				df = df.set_index(pd.to_datetime(df["Time"]))
				df = df.drop(columns=["Time"])  # drop duplicate
			else:
				raise ValueError(f"DataFrame lacks DatetimeIndex and 'Time' column: {fp}")
		df = df.sort_index()
		frames[k] = df
	return frames


def build_base_feature_view(frames: Dict[int, pd.DataFrame]) -> pd.DataFrame:
	# Use 1-minute horizon frame as canonical; they share feature columns.
	base = frames[min(frames.keys())].copy()
	# Reconstruct r1 from shifted r_1m
	if "r_1m" not in base.columns:
		raise ValueError("Expected column 'r_1m' in featured dataset to reconstruct r1")
	base["r1"] = base["r_1m"].shift(-1)  # r_1m at t+1 corresponds to r1 at t
	return base


def attach_targets(base_index: pd.DatetimeIndex, frames: Dict[int, pd.DataFrame]) -> pd.DataFrame:
	tgt_cols = {}
	for k, df in frames.items():
		for col in [f"y_sign_{k}", f"y_mag_{k}"]:
			if col not in df.columns:
				raise ValueError(f"Missing target column {col} in horizon {k} frame")
			tgt_cols[col] = df[col]
	# Align on base index (outer join then subset to base)
	targets = pd.DataFrame(tgt_cols)
	targets = targets.reindex(base_index)
	return targets


def contiguous_hour_window_ok(index: pd.DatetimeIndex, t: pd.Timestamp) -> bool:
	if t.minute != 0:
		return False
	start = t - pd.Timedelta(minutes=SEQ_LEN)
	if start.minute != 0:
		return False  # enforce whole prior hour
	# Slice expected window
	window = index[(index >= start) & (index < t)]
	if len(window) != SEQ_LEN:
		return False
	# Check minute spacing
	diffs = window.to_series().diff().dropna().dt.total_seconds() / 60.0
	return bool((diffs == 1).all())


def build_sequences_for_symbol(symbol: str, featured_root: Path) -> Path:
	frames = load_symbol_feature_frames(symbol, featured_root)
	base = build_base_feature_view(frames)
	targets = attach_targets(base.index, frames)
	# Candidate prediction times
	candidate_times = base.index[base.index.minute == 0]
	seq_list: List[np.ndarray] = []
	keep_times: List[pd.Timestamp] = []
	tgt_records: Dict[str, List] = {f"y_sign_{k}": [] for k in HORIZONS}
	tgt_records.update({f"y_mag_{k}": [] for k in HORIZONS})

	core_available = all(f in base.columns for f in SEQ_CORE_FEATS)
	if not core_available:
		missing = [f for f in SEQ_CORE_FEATS if f not in base.columns]
		raise ValueError(f"Missing required core features for sequences: {missing}")

	for t in candidate_times:
		if not contiguous_hour_window_ok(base.index, t):
			continue
		start = t - pd.Timedelta(minutes=SEQ_LEN)
		window = base.loc[start : t - pd.Timedelta(minutes=1), SEQ_CORE_FEATS]
		if window.isna().any().any():
			continue
		# Targets at prediction time t must all be present
		row_targets = targets.loc[t]
		if row_targets.isna().any():
			continue
		seq_list.append(window.to_numpy(dtype=np.float32))
		keep_times.append(t)
		for col in row_targets.index:
			tgt_records[col].append(row_targets[col])

	if not seq_list:
		raise RuntimeError(f"No valid sequences generated for symbol {symbol}")

	seq_arr = np.stack(seq_list)  # (N, 60, F)
	times_arr = np.array([ts.isoformat() for ts in keep_times])
	out_dir = load_config().sequenced  # ensure directory exists
	out_path = out_dir / f"{symbol}_seq.npz"
	meta = {
		"symbol": symbol,
		"sequence_length": SEQ_LEN,
		"features": SEQ_CORE_FEATS,
		"horizons": HORIZONS,
		"num_sequences": int(seq_arr.shape[0]),
	}
	np.savez_compressed(
		out_path,
		seq=seq_arr,
		times=times_arr,
		meta=json.dumps(meta),
		**{k: np.array(v, dtype=np.float32) for k, v in tgt_records.items()},
	)
	return out_path


def discover_symbols(featured_root: Path) -> List[str]:
	# Inspect 1m horizon folder for parquet files
	folder = featured_root / "1m"
	symbols = [p.stem for p in folder.glob("*.parquet") if p.is_file()]
	return sorted(symbols)


def write_splits(index: pd.DatetimeIndex, seq_root: Path, overwrite: bool = True) -> Path:
	splits = make_walkforward_splits(
		index,
		TRAIN_BLOCK_DAYS,
		VAL_BLOCK_DAYS,
		TEST_BLOCK_DAYS,
		EMBARGO_MIN,
	)
	out_path = seq_root / "splits.json"
	if out_path.exists() and not overwrite:
		return out_path
	out_path.write_text(json.dumps(splits, indent=2))
	return out_path


def main():  # pragma: no cover
	logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
	logger = logging.getLogger("sequencing")
	paths = load_config()
	symbols = discover_symbols(paths.featured)
	if not symbols:
		logger.error("No symbols discovered under %s", paths.featured / "1m")
		return 1
	logger.info("Building sequences for %d symbols (features root=%s, output=%s)", len(symbols), paths.featured, paths.sequenced)
	all_indices: List[pd.DatetimeIndex] = []
	for sym in tqdm(symbols, desc="Symbols"):
		try:
			out = build_sequences_for_symbol(sym, paths.featured)
			logger.info("%s -> %s", sym, out.name)
			# Collect index for splits (top-of-hour prediction times)
			data = np.load(out, allow_pickle=True)
			times = pd.to_datetime(data["times"])  # prediction times
			all_indices.append(pd.DatetimeIndex(times))
		except Exception as e:  # pragma: no cover - robust batch
			logger.exception("Failed symbol %s: %s", sym, e)
	# Build unified index for splits (union)
	if all_indices:
		master_index = all_indices[0]
		for idx in all_indices[1:]:
			master_index = master_index.union(idx)
		write_splits(master_index, paths.sequenced, overwrite=True)
		logger.info("Wrote walk-forward splits (count=%d)", len(json.loads((paths.sequenced / "splits.json").read_text())))
	return 0


if __name__ == "__main__":  # pragma: no cover
	raise SystemExit(main())

