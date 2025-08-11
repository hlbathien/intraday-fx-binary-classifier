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
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

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


def build_feature_row(df: pd.DataFrame, t: pd.Timestamp) -> Dict[str, float]:
	# slice strictly past window
	win = df.loc[t - pd.Timedelta(minutes=SEQ_LEN) : t - pd.Timedelta(minutes=1)]
	r1 = df["r1"]
	def sum_last(n: int) -> float:
		return float(r1.loc[t - pd.Timedelta(minutes=n) : t - pd.Timedelta(minutes=1)].sum())

	feats: Dict[str, float] = {
		"r_1m": float(r1.loc[t - pd.Timedelta(minutes=1)]),
		"r_2m": sum_last(2),
		"r_5m": sum_last(5),
		"r_10m": sum_last(10),
		"r_15m": sum_last(15),
		"r_60m": sum_last(60),
	}
	# Z scores of returns
	r1_roll = r1.copy()
	feats.update(
		{
			"rz_5m": float(zscore(r1_roll, 5).loc[t - pd.Timedelta(minutes=1)]),
			"rz_15m": float(zscore(r1_roll, 15).loc[t - pd.Timedelta(minutes=1)]),
			"rz_30m": float(zscore(r1_roll, 30).loc[t - pd.Timedelta(minutes=1)]),
		}
	)
	# Realized vol & ranges (past value at t-1)
	feats.update(
		{
			"rv_5m": float(rolling_std(r1, 5).loc[t - pd.Timedelta(minutes=1)]),
			"rv_10m": float(rolling_std(r1, 10).loc[t - pd.Timedelta(minutes=1)]),
			"rv_30m": float(rolling_std(r1, 30).loc[t - pd.Timedelta(minutes=1)]),
			"pkn_5": float(df["parkinson_5"].loc[t - pd.Timedelta(minutes=1)]),
			"pkn_15": float(df["parkinson_15"].loc[t - pd.Timedelta(minutes=1)]),
			"atr5": float(df["atr_5"].loc[t - pd.Timedelta(minutes=1)]),
			"atr14": float(df["atr_14"].loc[t - pd.Timedelta(minutes=1)]),
		}
	)
	# Time of day features (NY session already 12:00-20:59). Use UTC naive -> treat as UTC.
	hour = t.hour
	minute = t.minute
	feats.update(
		{
			"tod_sin": math.sin(2 * math.pi * hour / 24.0),
			"tod_cos": math.cos(2 * math.pi * hour / 24.0),
			"min_sin": math.sin(2 * math.pi * minute / 60.0),
			"min_cos": math.cos(2 * math.pi * minute / 60.0),
			"min_to_hour_end": float(60 - minute),
		}
	)
	# Volume features
	vol = df["Volume"]
	feats.update(
		{
			"vol_z_5": float(zscore(vol, 5).loc[t - pd.Timedelta(minutes=1)]),
			"vol_z_30": float(zscore(vol, 30).loc[t - pd.Timedelta(minutes=1)]),
			"vol_z_120": float(zscore(vol, 120).loc[t - pd.Timedelta(minutes=1)]),
		}
	)
	feats["vol_spike"] = 1.0 if feats["vol_z_30"] >= 2.0 else 0.0
	# vol_to_volratio - price rv_30m in denominator
	denom = feats["rv_30m"] if feats["rv_30m"] not in (0.0, np.nan) else 1e-8
	feats["vol_to_volratio"] = feats["vol_z_30"] / denom
	return feats


def build_dataset(df: pd.DataFrame, k: int) -> pd.DataFrame:
	# Build targets columns if not already
	df = build_targets(df, k)
	sign_col = f"y_sign_{k}"
	mag_col = f"y_mag_{k}"
	# Candidate prediction times: choose all timestamps (could filter to top-of-hour if desired)
	# For now every minute within data range where targets valid
	valid_times: List[pd.Timestamp] = []
	for t in df.index:
		if pd.isna(df.loc[t, sign_col]):
			continue
		future_t = t + pd.Timedelta(minutes=k)
		if future_t not in df.index:
			continue
		if one_hour_window_ok(df, t):
			valid_times.append(t)
	rows: List[Dict[str, float]] = []
	for t in valid_times:
		rows.append(build_feature_row(df, t))
	X = pd.DataFrame(rows, index=valid_times)
	X[sign_col] = df.loc[valid_times, sign_col].values
	X[mag_col] = df.loc[valid_times, mag_col].values
	return X


def main():  # pragma: no cover
	paths = load_config()
	csv_files = list(iter_cleaned_csv(paths.cleaned))
	if not csv_files:
		print(f"No cleaned CSV files in {paths.cleaned}")
		return 1
	for horizon in HORIZONS:
		(paths.featured / f"{horizon}m").mkdir(parents=True, exist_ok=True)
	for f in tqdm(csv_files, desc="Symbols"):
		symbol = f.stem
		df = read_cleaned(f)
		df = build_base_series(df)
		for k in HORIZONS:
			ds = build_dataset(df.copy(), k)
			out_path = paths.featured / f"{k}m" / f"{symbol}.parquet"
			ds.to_parquet(out_path)
	return 0


if __name__ == "__main__":  # pragma: no cover
	raise SystemExit(main())

