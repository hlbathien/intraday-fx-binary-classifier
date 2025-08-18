"""Step 4 main: Train & evaluate models per symbol & horizon.

Ensures project root is on sys.path so that 'lib' imports resolve regardless of CWD.
"""
from __future__ import annotations

import json, logging, pickle, sys, hashlib
from pathlib import Path
from typing import Dict, List, Any
import numpy as np, pandas as pd
from tqdm import tqdm

# --- Ensure root path for 'lib' imports when running from src/ directory ---
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.config import load_config, CONFIG_FILE
from lib.random import seed_all
from lib.logging import init_logging
from lib.modeling import (
    build_models,
    train_model,
    calibrate_sign_prob,
    predict_proba_sign,
    predict_mag,
    optimize_thresholds,
    evaluate_on_set,
)

# Horizons will be inferred per symbol from available y_sign_<k> keys unless overridden here
EXPLICIT_HORIZONS: list[int] | None = None  # e.g. [1,5]
F_MAX = 0.02


def _load_npz(fp: Path) -> Dict[str, Any]:
    d = np.load(fp, allow_pickle=True)
    meta = json.loads(str(d['meta']))
    out = {k: d[k] for k in d.files if k != 'meta'}; out['meta']=meta
    return out

def _tabular(seq: np.ndarray) -> np.ndarray:
    mean=seq.mean(axis=1); std=seq.std(axis=1); last=seq[:,-1,:]; first=seq[:,0,:]; diff=last-first
    return np.concatenate([mean,std,last,diff], axis=1).astype(np.float32)

def _load_splits(root: Path):
    f=root/'splits.json'
    if not f.exists(): raise FileNotFoundError('splits.json missing (run sequencing).')
    return json.loads(f.read_text())

def _span(span: str): s,e=span.split('|'); return pd.Timestamp(s), pd.Timestamp(e)

def _mask(times: pd.DatetimeIndex, s, e):
    m=(times>=s)&(times<=e); return np.where(m)[0]

def _agg(folds: List[Dict[str, Any]]):
    if not folds: return {}
    keys=[k for k in folds[0] if k not in {'model','features'}]
    return {**{k: float(np.nanmean([f[k] for f in folds])) for k in keys if isinstance(folds[0][k], (int,float))}, "folds": len(folds)}

def _available_horizons(data: Dict[str, Any]) -> list[int]:
    hs = sorted({int(k.split('_')[-1]) for k in data if k.startswith('y_sign_')})
    return [h for h in hs if f'y_mag_{h}' in data]

def _config_snapshot(cfg) -> Dict[str, Any]:
    txt = CONFIG_FILE.read_text() if CONFIG_FILE.exists() else ''
    return {
        'config_file_sha256': hashlib.sha256(txt.encode()).hexdigest(),
        'raw_config': txt.splitlines(),
    }

def _run_symbol(sym: str, cfg):
    seq_fp = cfg.sequenced_data_path / f"{sym}_seq.npz"
    if not seq_fp.exists():
        logging.warning("Skip %s (no sequence file)", sym); return
    data = _load_npz(seq_fp); seq=data['seq']; times=pd.to_datetime(data['times'])
    tab=_tabular(seq); splits=_load_splits(cfg.sequenced_data_path)
    out_root = cfg.model_output / sym; out_root.mkdir(parents=True, exist_ok=True)
    horizons = EXPLICIT_HORIZONS if EXPLICIT_HORIZONS else _available_horizons(data)
    if not horizons:
        logging.warning("No horizons for %s", sym); return
    for k in horizons:
        yS=data[f'y_sign_{k}']; yM=data[f'y_mag_{k}']
        hz_dir=out_root / f'horizon_{k}'; hz_dir.mkdir(exist_ok=True)
        fold_metrics=[]; fold_meta=[]
        for i, sp in enumerate(splits):
            trS,trE=_span(sp['train']); vS,vE=_span(sp['val']); teS,teE=_span(sp['test'])
            idx_tr=_mask(times,trS,trE); idx_v=_mask(times,vS,vE); idx_te=_mask(times,teS,teE)
            if min(len(idx_tr), len(idx_v), len(idx_te))==0: continue
            Xtr_tab, Xv_tab, Xte_tab = tab[idx_tr], tab[idx_v], tab[idx_te]
            Xtr_seq, Xv_seq, Xte_seq = seq[idx_tr], seq[idx_v], seq[idx_te]
            yS_tr, yS_v, yS_te = yS[idx_tr], yS[idx_v], yS[idx_te]
            yM_tr, yM_v, yM_te = yM[idx_tr], yM[idx_v], yM[idx_te]
            models = build_models(seq.shape[-1])
            best=None; best_name=None; best_ev=-1e9; best_cal=None; best_thresh=(0.6, float(np.median(np.abs(yM_v))))
            for name, spec in models.items():
                bundle = train_model(spec, Xtr_tab, Xtr_seq, yS_tr, yM_tr, Xv_tab, Xv_seq, yS_v, yM_v)
                calib = calibrate_sign_prob(bundle, Xv_tab, Xv_seq, yS_v)
                p_val = predict_proba_sign(bundle, Xv_tab, Xv_seq); p_val = calib.predict(p_val) if hasattr(calib,'predict') else p_val
                m_val = predict_mag(bundle, Xv_tab, Xv_seq)
                ev_val,(p_star,theta)=optimize_thresholds(p_val, m_val, yS_v)
                if ev_val>best_ev:
                    best_ev=ev_val; best=bundle; best_name=name; best_cal=calib; best_thresh=(p_star,theta)
            if best is None: continue
            metrics = evaluate_on_set(best, best_cal, best_thresh[0], best_thresh[1], Xte_tab, Xte_seq, yS_te, f_max=F_MAX)
            metrics['model']=best_name
            (hz_dir / f'metrics_fold_{i}.json').write_text(json.dumps(metrics, indent=2))
            fold_metrics.append(metrics)
            # Persist model (per best fold) & artifacts with fold id
            if 'joint' in best:
                import torch; torch.save(best['joint'].state_dict(), hz_dir / f'best_model_{best_name}_fold_{i}.pt')
            else:
                with open(hz_dir / f'best_model_{best_name}_fold_{i}.pkl','wb') as f: pickle.dump(best, f)
            with open(hz_dir / f'calibrator_fold_{i}.pkl','wb') as f: pickle.dump(best_cal, f)
            (hz_dir/ f'thresholds_fold_{i}.json').write_text(json.dumps({'p_star': best_thresh[0], 'theta': best_thresh[1]}, indent=2))
            fold_meta.append({'fold': i, 'train': sp['train'], 'val': sp['val'], 'test': sp['test'], 'n_train': len(idx_tr), 'n_val': len(idx_v), 'n_test': len(idx_te)})
        (hz_dir / 'agg_metrics.json').write_text(json.dumps({**_agg(fold_metrics), 'horizons_used': horizons}, indent=2))
        (hz_dir / 'fold_meta.json').write_text(json.dumps({'folds': fold_meta}, indent=2))
    # Config snapshot at symbol root
    (out_root / 'config_snapshot.json').write_text(json.dumps(_config_snapshot(cfg), indent=2))

def main():  # pragma: no cover
    seed_all()
    init_logging()
    cfg=load_config()
    symbols=sorted([p.stem.replace('_seq','') for p in cfg.sequenced_data_path.glob('*_seq.npz')])
    logging.info('Training symbols=%d', len(symbols))
    for s in tqdm(symbols, desc='Symbols'): _run_symbol(s, cfg)
    logging.info('Training complete')
    return 0

if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
