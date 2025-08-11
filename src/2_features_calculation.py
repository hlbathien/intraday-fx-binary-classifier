"""Generate per-pair feature matrices and classification targets for multiple horizons.

Reads CLEANED_DATA_PATH & FEATURED_DATA_PATH from .config, iterates all cleaned CSVs
and produces one parquet per symbol per horizon inside FEATURED_DATA_PATH/<horizon>m/.

Horizons implemented: 1,5,10,15 minutes ahead.

Feature design derives strictly from past data (no leakage) using a 60-minute lookback.
Targets: sign (binary) of future k-minute log return (winsorized) and magnitude scaled by
realized volatility.

NOTE: This script keeps implementation self-contained; if library utilities are later
added they should be factored into lib/ and imported (reuse-first principle).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

# Parallel utility
try:  # pragma: no cover - import resolution
	from lib.parallel import parallel_map, determine_workers
except Exception:  # fallback sequential
	parallel_map = None  # type: ignore
	determine_workers = lambda: 1  # type: ignore

# ---------------- Configuration ---------------- #

CONFIG_FILE = Path(__file__).resolve().parent.parent / ".config"
HORIZONS = [1, 5, 10, 15]
SEQ_LEN = 60  # minutes lookback window
VOL_WIN = 120
WINSOR_Q = (0.001, 0.999)


@dataclass
class Paths:
	cleaned: Path
	featured: Path


def load_config(path: Path = CONFIG_FILE) -> Paths:
	if not path.exists():
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
	cleaned = Path(cfg["CLEANED_DATA_PATH"]).resolve()
	featured = Path(cfg["FEATURED_DATA_PATH"]).resolve()
	featured.mkdir(parents=True, exist_ok=True)
	return Paths(cleaned=cleaned, featured=featured)


# ---------------- Utility Functions ---------------- #

def typical_price(df: pd.DataFrame) -> pd.Series:
	return (df["High"] + df["Low"] + 2 * df["Close"]) / 4.0


def log_return(series: pd.Series) -> pd.Series:
	return np.log(series).diff()


def rolling_std(series: pd.Series, win: int) -> pd.Series:
	return series.rolling(win, min_periods=win).std()


def winsorize(series: pd.Series, q_low: float, q_high: float) -> pd.Series:
	if series.empty:
		return series
	lo = series.quantile(q_low)
	hi = series.quantile(q_high)
	return series.clip(lo, hi)


def zscore(series: pd.Series, win: int) -> pd.Series:
	roll = series.rolling(win, min_periods=win)
	mu = roll.mean()
	sig = roll.std()
	return (series - mu) / sig


def parkinson_vol(high: pd.Series, low: pd.Series, win: int) -> pd.Series:
	r2 = (np.log(high / low)) ** 2
	return (r2.rolling(win, min_periods=win).mean() / (4 * math.log(2))) ** 0.5


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
	prev_close = close.shift(1)
	tr = pd.concat([
		high - low,
		(high - prev_close).abs(),
		(low - prev_close).abs(),
	], axis=1).max(axis=1)
	return tr.rolling(n, min_periods=n).mean()


def iter_cleaned_csv(cleaned_dir: Path) -> Iterable[Path]:
	return sorted(p for p in cleaned_dir.glob("*.csv") if p.is_file())


def read_cleaned(f: Path) -> pd.DataFrame:
	df = pd.read_csv(f, parse_dates=["Time"])  # Time is UTC naive
	df = df.sort_values("Time").drop_duplicates("Time")
	df = df.set_index("Time")
	return df


def one_hour_window_ok(df: pd.DataFrame, t: pd.Timestamp) -> bool:
	start = t - pd.Timedelta(minutes=SEQ_LEN)
	if start < df.index[0]:
		return False
	# Must have continuous minute coverage: compare expected size
	window_index = df.loc[start : t - pd.Timedelta(minutes=1)].index
	if len(window_index) != SEQ_LEN:
		return False
	# ensure last point exists just before t
	return True


def build_base_series(df: pd.DataFrame) -> pd.DataFrame:
	out = df.copy()
	out["mid"] = typical_price(out)
	out["r1"] = log_return(out["mid"])  # 1m log returns
	out["sigma_vol"] = rolling_std(out["r1"], VOL_WIN)
	out["parkinson_5"] = parkinson_vol(out["High"], out["Low"], 5)
	out["parkinson_15"] = parkinson_vol(out["High"], out["Low"], 15)
	out["atr_5"] = atr(out["High"], out["Low"], out["Close"], 5)
	out["atr_14"] = atr(out["High"], out["Low"], out["Close"], 14)
	out["r1_w"] = winsorize(out["r1"], *WINSOR_Q)
	return out


def build_targets(df: pd.DataFrame, k: int) -> pd.DataFrame:
	eps = 1e-8
	col_rt = f"r_tk_{k}"
	col_rtw = f"r_tk_w_{k}"
	col_mag = f"y_mag_{k}"
	col_sign = f"y_sign_{k}"
	df[col_rt] = np.log(df["mid"].shift(-k)) - np.log(df["mid"])
	df[col_rtw] = winsorize(df[col_rt], *WINSOR_Q)
	df[col_mag] = (df[col_rtw].abs()) / (df["sigma_vol"].replace(0, np.nan))
	df[col_mag] = df[col_mag].fillna(0.0)
	df[col_sign] = (df[col_rtw] > 0).astype(int)
	return df


def _compute_contiguous_window_mask(index: pd.DatetimeIndex, window: int) -> np.ndarray:
	"""Return boolean mask (aligned to index) where a full prior `window` minutes exist.

	For prediction time t (index position i) we require the previous `window` minutes
	[t-window, t-1] be present contiguously. Implementation derives run-lengths of
	contiguous 1-minute spacing and shifts by one.
	"""
	if len(index) == 0:
		return np.array([], dtype=bool)
	diff = index.to_series().diff().dt.total_seconds().fillna(60) / 60.0
	arr = diff.to_numpy()
	run = np.empty_like(arr, dtype=int)
	run[0] = 1
	for i in range(1, len(arr)):
		if arr[i] == 1.0:
			run[i] = run[i - 1] + 1
		else:
			run[i] = 1
	# Need run-length at t-1 (previous row) >= window
	has_window = np.zeros(len(arr), dtype=bool)
	has_window[1:] = run[:-1] >= window
	return has_window


def build_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
	"""Vectorized construction of all feature columns (excluding targets).

	Returns DataFrame of features aligned to df.index (prediction times) and a boolean
	mask indicating which rows have a full prior SEQ_LEN-minute history.
	"""
	r1 = df["r1"]
	features = pd.DataFrame(index=df.index)
	# Return aggregations (past-only, ending at t-1)
	features["r_1m"] = r1.shift(1)
	for n in [2, 5, 10, 15, 60]:
		col = f"r_{n}m" if n != 2 else "r_2m"
		features[col] = r1.rolling(n).sum().shift(1)
	# Z-scores of returns
	for win in [5, 15, 30]:
		features[f"rz_{win}m"] = zscore(r1, win).shift(1)
	# Realized volatility and range measures (shifted to be past-only)
	features["rv_5m"] = rolling_std(r1, 5).shift(1)
	features["rv_10m"] = rolling_std(r1, 10).shift(1)
	features["rv_30m"] = rolling_std(r1, 30).shift(1)
	for col in ["parkinson_5", "parkinson_15", "atr_5", "atr_14"]:
		features[col.replace("atr_", "atr").replace("parkinson_", "pkn_") if col.startswith("parkinson_") else col] = df[col].shift(1)
	# Rename corrections for pkn & atr
	features.rename(columns={
		"parkinson_5": "pkn_5",
		"parkinson_15": "pkn_15",
		"atr_5": "atr5",
		"atr_14": "atr14",
	}, inplace=True)
	# Time-of-day (computed at prediction timestamp t)
	idx = features.index
	hours = idx.hour.to_numpy()
	minutes = idx.minute.to_numpy()
	features["tod_sin"] = np.sin(2 * np.pi * hours / 24.0)
	features["tod_cos"] = np.cos(2 * np.pi * hours / 24.0)
	features["min_sin"] = np.sin(2 * np.pi * minutes / 60.0)
	features["min_cos"] = np.cos(2 * np.pi * minutes / 60.0)
	features["min_to_hour_end"] = 60 - minutes
	# Volume features
	vol = df["Volume"]
	for win in [5, 30, 120]:
		features[f"vol_z_{win}"] = zscore(vol, win).shift(1)
	features["vol_spike"] = (features["vol_z_30"] >= 2.0).astype(float)
	# vol_to_volratio
	denom = features["rv_30m"].replace(0, np.nan)
	features["vol_to_volratio"] = features["vol_z_30"] / denom
	features["vol_to_volratio"] = features["vol_to_volratio"].replace([np.inf, -np.inf], np.nan)
	# Contiguous window mask
	has_window = _compute_contiguous_window_mask(df.index, SEQ_LEN)
	return features, has_window


def build_dataset(
	df: pd.DataFrame,
	k: int,
	features: pd.DataFrame,
	has_window: np.ndarray,
) -> pd.DataFrame:
	"""Assemble dataset for horizon k using precomputed feature matrix.

	Parameters
	----------
	df : DataFrame with base & target series
	k : horizon minutes ahead
	features : DataFrame of feature columns (past-only aligned to df.index)
	has_window : bool array indicating full prior SEQ_LEN coverage
	"""
	# Targets
	df = build_targets(df, k)
	sign_col = f"y_sign_{k}"
	mag_col = f"y_mag_{k}"
	# Future existence mask
	future_idx = df.index + pd.Timedelta(minutes=k)
	future_exists = future_idx.isin(df.index)
	# Valid mask: has full history and future exists and target not NaN
	valid_mask = has_window & future_exists & (~df[sign_col].isna())
	# Subset
	X = features.loc[valid_mask].copy()
	X[sign_col] = df.loc[valid_mask, sign_col].values
	X[mag_col] = df.loc[valid_mask, mag_col].values
	return X


def _process_symbol(args: Tuple[str, str]):
	"""Worker: compute all horizon datasets for one symbol.

	Parameters
	----------
	args : tuple[str, str]
		(csv_path_str, featured_dir_str)
	"""
	csv_path_str, featured_dir_str = args
	csv_path = Path(csv_path_str)
	featured_root = Path(featured_dir_str)
	symbol = csv_path.stem
	df = read_cleaned(csv_path)
	df = build_base_series(df)
	# Precompute vectorized features once per symbol
	features, has_window = build_feature_matrix(df)
	for k in HORIZONS:
		# Note: build_dataset adds target columns based on df copy to avoid cross-horizon overwrite
		ds = build_dataset(df.copy(), k, features, has_window)
		out_path = featured_root / f"{k}m" / f"{symbol}.parquet"
		ds.to_parquet(out_path)
	return symbol


def main():  # pragma: no cover
	logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
	logger = logging.getLogger("features")
	paths = load_config()
	csv_files = list(iter_cleaned_csv(paths.cleaned))
	if not csv_files:
		logger.warning("No cleaned CSV files in %s", paths.cleaned)
		return 1
	for horizon in HORIZONS:
		(paths.featured / f"{horizon}m").mkdir(parents=True, exist_ok=True)
	logger.info("Generating features for %d symbols across horizons %s", len(csv_files), HORIZONS)
	work_args = [(str(p), str(paths.featured)) for p in csv_files]
	if parallel_map is not None and len(work_args) > 1:
		workers = determine_workers()
		logger.info("Using %d worker processes", workers)
		parallel_map(_process_symbol, work_args, workers=workers, desc="Symbols")
	else:
		for a in tqdm(work_args, desc="Symbols"):
			_process_symbol(a)
	logger.info("Completed feature generation.")
	return 0


if __name__ == "__main__":  # pragma: no cover
	raise SystemExit(main())

