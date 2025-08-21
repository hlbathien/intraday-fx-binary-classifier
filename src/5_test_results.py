"""Step 5 main: Aggregate trained model results & rank best performers.

Collects each symbol/horizon directory under the configured model output path,
reads its ``agg_metrics.json`` (produced in step 4), and builds a consolidated
DataFrame. It then computes composite ranking scores and exports summary
artifacts (CSV/Parquet/JSON/Markdown) for downstream review.

Ranking methodology (deterministic, no randomness):
 1. Primary objective: maximize ev_per_trade (expected value per trade).
 2. Secondary tie-breakers: total_ev (higher better), roc_auc (higher),
	brier (lower), ece (lower).
 3. A composite score is computed via min-max scaling of each metric and a
	weighted sum:  w = {ev_per_trade:0.35, total_ev:0.25, roc_auc:0.2,
						brier:0.1 (negative), ece:0.1 (negative)}.

All outputs are written under ``<MODEL_OUTPUT>/rankings`` so re-runs overwrite
the same deterministic artifacts.

Outputs:
  * all_models_metrics.parquet : full row-wise metrics table
  * rankings_sorted.csv        : all models sorted by composite score
  * top_per_horizon.csv        : best model per symbol+horizon
  * best_overall.json          : top N overall entries (N=20)
  * summary.md                 : human-readable summary table
"""
from __future__ import annotations

import json, logging, sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- Ensure project root on path (mirror pattern from prior steps) ---
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
	sys.path.insert(0, str(_ROOT))

from lib.config import load_config
from lib.random import seed_all
from lib.logging import init_logging


RANK_DIR_NAME = "rankings"
TOP_N_EXPORT = 20

PRIMARY_METRICS = [
	"ev_per_trade", "total_ev", "gross_return", "gross_return_per_trade", "roc_auc", "brier", "ece", "n_trades", "acc@0.5",
]

WEIGHTS = {
	"ev_per_trade": 0.30,
	"total_ev": 0.20,
	"gross_return_per_trade": 0.15,
	"gross_return": 0.10,
	"roc_auc": 0.15,
	"brier": -0.05,  # negative = lower is better
	"ece": -0.05,    # negative = lower is better
}


def _safe_read_json(fp: Path) -> Dict[str, Any]:
	try:
		return json.loads(fp.read_text())
	except Exception:  # pragma: no cover
		logging.warning("Failed reading %s", fp)
		return {}


def _collect(cfg) -> pd.DataFrame:
	rows: List[Dict[str, Any]] = []
	model_root = cfg.model_output
	symbols = [p for p in model_root.iterdir() if p.is_dir() and (p / 'config_snapshot.json').exists()]
	for sym_dir in tqdm(symbols, desc="Symbols"):
		sym = sym_dir.name
		for hz_dir in sorted(sym_dir.glob('horizon_*')):
			if not hz_dir.is_dir():
				continue
			agg_fp = hz_dir / 'agg_metrics.json'
			if not agg_fp.exists():
				continue
			data = _safe_read_json(agg_fp)
			# Horizon number extraction
			try:
				horizon = int(hz_dir.name.split('_')[-1])
			except Exception:
				horizon = None
			row = {**data, 'symbol': sym, 'horizon': horizon}
			rows.append(row)
	if not rows:
		return pd.DataFrame(columns=["symbol","horizon", *PRIMARY_METRICS])
	df = pd.DataFrame(rows)
	# Standardize column order (metrics first for readability)
	# Some metrics may be missing if folds empty; keep robust.
	return df


def _min_max(series: pd.Series) -> pd.Series:
	s = series.astype(float)
	lo, hi = s.min(), s.max()
	if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
		return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
	return (s - lo) / (hi - lo)


def _compute_ranking(df: pd.DataFrame) -> pd.DataFrame:
	if df.empty:
		df['composite_score'] = []  # type: ignore[index]
		return df
	score = np.zeros(len(df), dtype=float)
	for k, w in WEIGHTS.items():
		if k not in df.columns:
			logging.warning("Metric %s not present; skipping in ranking", k)
			continue
		mm = _min_max(df[k])
		score += w * mm.to_numpy()
		df[f'z_{k}'] = mm
	df['composite_score'] = score
	df.sort_values('composite_score', ascending=False, inplace=True)
	df.reset_index(drop=True, inplace=True)
	# Rank columns
	df['rank'] = np.arange(1, len(df) + 1)
	return df


def _best_per_symbol_horizon(df: pd.DataFrame) -> pd.DataFrame:
	if df.empty:
		return df
	key_cols = ['symbol', 'horizon']
	idx = df.groupby(key_cols)['composite_score'].idxmax()
	return df.loc[idx].sort_values(['symbol', 'horizon']).reset_index(drop=True)


def _write_outputs(df_all: pd.DataFrame, df_top: pd.DataFrame, out_dir: Path):
	out_dir.mkdir(parents=True, exist_ok=True)
	# Persist full parquet & csv ranking
	if not df_all.empty:
		(out_dir / 'all_models_metrics.parquet').unlink(missing_ok=True)  # type: ignore[arg-type]
		df_all.to_parquet(out_dir / 'all_models_metrics.parquet', index=False)
		df_all.to_csv(out_dir / 'rankings_sorted.csv', index=False)
		df_top.to_csv(out_dir / 'top_per_horizon.csv', index=False)
		best_overall = df_all.head(TOP_N_EXPORT).to_dict(orient='records')
		(out_dir / 'best_overall.json').write_text(json.dumps({'top_n': TOP_N_EXPORT, 'entries': best_overall}, indent=2))
		# Produce a concise markdown summary
		# Dynamically keep only columns that exist (older metrics may miss new fields)
		base_cols_common = ['rank','symbol','horizon','ev_per_trade','gross_return_per_trade','total_ev','gross_return','roc_auc','brier','ece','n_trades','composite_score']
		existing_common = [c for c in base_cols_common if c in df_all.columns]
		base_per_cols = ['symbol','horizon','ev_per_trade','gross_return_per_trade','total_ev','gross_return','roc_auc','brier','ece','n_trades','composite_score']
		existing_per = [c for c in base_per_cols if c in df_top.columns]
		try:
			top10_md = df_all.head(10)[existing_common].to_markdown(index=False)
			per_md = df_top[existing_per].to_markdown(index=False)
		except Exception:  # pragma: no cover
			top10_md = df_all.head(10)[existing_common].to_csv(index=False)
			per_md = df_top[existing_per].to_csv(index=False)
		md_lines = [
			'# Model Ranking Summary',
			'',
			f'Total evaluated entries: {len(df_all)}',
			'',
			'## Top 10 Overall',
			'',
			top10_md,
			'',
			'## Best Per Symbol + Horizon',
			'',
			per_md
		]
		(out_dir / 'summary.md').write_text('\n'.join(md_lines))
	else:
		(out_dir / 'summary.md').write_text('No model metrics found to rank.\n')


def main():  # pragma: no cover
	seed_all()
	init_logging()
	cfg = load_config()
	logging.info('Collecting aggregated metrics from %s', cfg.model_output)
	df = _collect(cfg)
	logging.info('Loaded %d symbol-horizon aggregate rows', len(df))
	df_ranked = _compute_ranking(df.copy())
	df_top = _best_per_symbol_horizon(df_ranked)
	out_dir = cfg.model_output / RANK_DIR_NAME
	_write_outputs(df_ranked, df_top, out_dir)
	logging.info('Ranking artifacts written to %s', out_dir)
	return 0


if __name__ == '__main__':  # pragma: no cover
	raise SystemExit(main())

