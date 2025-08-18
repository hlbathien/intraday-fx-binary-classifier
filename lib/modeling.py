"""Model building & training utilities (tabular + sequence).

Enhancements (Aug 2025):
 - Added structured configuration for sequence models (LSTM/GRU/Transformer)
 - Support variable hidden sizes, num layers, dropout, feedforward dims
 - Added accuracy-focused threshold search (0.45–0.65) per new pipeline spec
 - Integrated optional class imbalance weighting & label smoothing
 - Training hyperparameters (batch size, lr, epochs, patience, weight decay) configurable via spec dict
"""
from __future__ import annotations

from typing import Dict, Tuple, Any, List, Optional
import math
import logging
from contextlib import nullcontext
import numpy as np
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

try:  # optional
    import xgboost as xgb  # type: ignore
except Exception:  # pragma: no cover
    xgb = None  # type: ignore

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
try:  # Prefer new unified torch.amp API (PyTorch >=2.0+ with unified interface)
    from torch.amp import autocast as _autocast_amp, GradScaler as _GradScaler_amp  # type: ignore
    _HAS_UNIFIED_AMP = True
except Exception:  # Fallback to legacy CUDA AMP
    from torch.cuda.amp import autocast as _autocast_cuda, GradScaler as _GradScaler_cuda  # type: ignore
    _HAS_UNIFIED_AMP = False


def _make_scaler(enabled: bool):
    """Return a GradScaler honoring deprecation differences between APIs."""
    if not enabled:
        # Dummy object with required interface
        class _NoScaler:
            def is_enabled(self): return False
            def scale(self, loss): return loss
            def step(self, opt): opt.step()
            def update(self): pass
            def unscale_(self, opt): pass
        return _NoScaler()
    if _HAS_UNIFIED_AMP:
        return _GradScaler_amp('cuda', enabled=True)
    return _GradScaler_cuda(enabled=True)  # legacy


def _autocast_ctx(device_type: str, dtype, enabled: bool):
    if device_type != 'cuda' or not enabled:
        return nullcontext()
    if _HAS_UNIFIED_AMP:
        return _autocast_amp('cuda', dtype=dtype, enabled=True)
    # legacy signature: autocast(enabled=True, dtype=None, cache_enabled=True)
    return _autocast_cuda(enabled=True, dtype=dtype)


def build_models(input_dim_seq: int, horizon: int) -> Dict[str, Any]:
    """Return dictionary of model specs keyed by short name.

    The returned values can be:
      * sklearn/xgboost estimators (tabular)
      * dict specs for sequence models consumed by ``train_model``
    Horizon is used to pick recommended defaults as per provided configuration sets.
    """
    models: Dict[str, Any] = {}

    # ---- Logistic L1 (sparse) ----
    # Horizon-dependent C (smaller for k=1 noisy)
    C_map = {1: 0.05, 5: 0.1, 10: 0.1, 15: 0.1}
    models["log_l1"] = LogisticRegression(
        penalty="l1",
        C=C_map.get(horizon, 0.1),
        solver="liblinear",
        max_iter=2000,
    )

    # ---- XGBoost configs (Small & Medium) ----
    if xgb is not None:
        xgb_common = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", verbosity=0)
        models["xgb_s"] = xgb.XGBClassifier(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=12,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.0,
            reg_lambda=1.0,
            **xgb_common,
        )
        models["xgb_m"] = xgb.XGBClassifier(
            n_estimators=900,
            learning_rate=0.05,
            max_depth=4 if horizon <= 5 else 5,
            min_child_weight=8 if horizon >= 10 else 8,
            subsample=0.7,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            **xgb_common,
        )

    # ---- Sequence model spec dictionaries ----
    # Provide minimal feature size; training loop uses these.
    seq_base = {
        "batch_size": 256,
        "val_batch_size": 512,
        "epochs": 50,
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "patience": 8,
        "optimizer": "adamw",
        "scheduler": "cosine",  # or onecycle
        "grad_clip": 1.0,
        # PyTorch 2.8+ execution controls (may be overridden per spec)
        "device": "auto",          # "auto" | "cuda" | "cpu" | "mps"
        "precision": "auto",       # "auto" | "fp16" | "bf16" | "fp32"
        "compile": False,           # leverage torch.compile if True (PyTorch >=2.0)
    }
    # Horizon guided shapes
    if horizon in (1,):
        # GRU small, LSTM small, Transformer small
        models["gru_s"] = {"type": "gru", "input_dim": input_dim_seq, "hidden_size": 48, "num_layers": 1, "dropout": 0.3, **seq_base}
        models["lstm_s"] = {"type": "lstm", "input_dim": input_dim_seq, "hidden_size": 48, "num_layers": 1, "dropout": 0.3, "label_smoothing": 0.05, **seq_base}
        models["transf_s"] = {"type": "transf", "input_dim": input_dim_seq, "d_model": 64, "num_layers": 1, "nhead": 2, "dim_ff": 128, "dropout": 0.2, **seq_base}
    elif horizon in (5,):
        models["gru"] = {"type": "gru", "input_dim": input_dim_seq, "hidden_size": 64, "num_layers": 2, "dropout": 0.2, **seq_base}
        models["lstm_m"] = {"type": "lstm", "input_dim": input_dim_seq, "hidden_size": 64, "num_layers": 2, "dropout": 0.2, **seq_base}
        models["transf"] = {"type": "transf", "input_dim": input_dim_seq, "d_model": 64, "num_layers": 2, "nhead": 2, "dim_ff": 128, "dropout": 0.1, **seq_base}
    elif horizon in (10,):
        models["lstm_m"] = {"type": "lstm", "input_dim": input_dim_seq, "hidden_size": 64, "num_layers": 2, "dropout": 0.2, **seq_base}
        models["transf_m"] = {"type": "transf", "input_dim": input_dim_seq, "d_model": 64, "num_layers": 2, "nhead": 2, "dim_ff": 128, "dropout": 0.1, **seq_base}
        models["gru_m"] = {"type": "gru", "input_dim": input_dim_seq, "hidden_size": 64, "num_layers": 2, "dropout": 0.2, **seq_base}
    else:  # horizon 15 or others (cleaner)
        models["lstm_l"] = {"type": "lstm", "input_dim": input_dim_seq, "hidden_size": 96, "num_layers": 2, "dropout": 0.15, **seq_base}
        models["transf_l"] = {"type": "transf", "input_dim": input_dim_seq, "d_model": 96, "num_layers": 3, "nhead": 2, "dim_ff": 192, "dropout": 0.1, **seq_base}
        models["gru_l"] = {"type": "gru", "input_dim": input_dim_seq, "hidden_size": 96, "num_layers": 2, "dropout": 0.15, **seq_base}

    return models


class IdentityCalibrator:
    """Picklable identity calibrator (used when not enough samples)."""
    def predict(self, x):  # type: ignore[override]
        return x


class JointSeqNet(nn.Module):
    def __init__(self, backbone: nn.Module, hidden: int, head_dropout: float = 0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(hidden)
        self.head_sign = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.head_mag = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(x)
        h = self.norm(h)
        return {"sign_logit": self.head_sign(h).squeeze(-1), "mag_pred": torch.relu(self.head_mag(h).squeeze(-1))}


class RNNBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden: int, kind: str, num_layers: int = 1, dropout: float = 0.0, bidirectional: bool = False):
        super().__init__()
        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM}[kind]
        self.rnn = rnn_cls(input_dim, hidden, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0, bidirectional=bidirectional)
        self.hidden = hidden * (2 if bidirectional else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, h = self.rnn(x)
        if isinstance(h, tuple):
            h = h[0]
        return h[-1]


class TransformerBackbone(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 64, heads: int = 2, num_layers: int = 1, dim_ff: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        ff = dim_ff if dim_ff is not None else d_model * 2
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=ff, batch_first=True, dropout=dropout)
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        z = self.enc(z)
        return self.pool(z.transpose(1, 2)).squeeze(-1)


def _build_net(spec: Dict[str, Any]) -> JointSeqNet:
    kind = spec["type"]
    if kind in {"gru", "lstm"}:
        backbone = RNNBackbone(spec["input_dim"], spec.get("hidden_size", 64), kind, num_layers=spec.get("num_layers", 1), dropout=spec.get("dropout", 0.0), bidirectional=spec.get("bidirectional", False))
        hidden = backbone.hidden
    elif kind == "transf":
        hidden = spec.get("d_model", 64)
        backbone = TransformerBackbone(spec["input_dim"], d_model=hidden, heads=spec.get("nhead", 2), num_layers=spec.get("num_layers", 1), dim_ff=spec.get("dim_ff"), dropout=spec.get("dropout", 0.1))
    else:  # pragma: no cover
        raise ValueError(kind)
    return JointSeqNet(backbone, hidden, head_dropout=spec.get("dropout", 0.2))


def train_model(spec: Any,
                X_tr_tab: np.ndarray, X_tr_seq: np.ndarray, yS_tr: np.ndarray, yM_tr: np.ndarray,
                X_v_tab: np.ndarray, X_v_seq: np.ndarray, yS_v: np.ndarray, yM_v: np.ndarray) -> Dict[str, Any]:
    """Train a model (tabular or sequence).

    Sequence specs are dictionaries produced by ``build_models``; classical estimators are fit directly.
    """
    # Classical tabular classifiers path
    if isinstance(spec, (RidgeClassifier, LogisticRegression)) or (xgb is not None and isinstance(spec, xgb.XGBClassifier)):
        model_sign = spec
        model_sign.fit(X_tr_tab, yS_tr)
        try:
            mean_mag = float(np.abs(yM_tr).mean())
            mag_reg = MeanMagPredictor(mean_mag)
        except Exception:  # pragma: no cover
            mag_reg = MeanMagPredictor(0.0)
        return {"sign": model_sign, "mag": mag_reg}

    # Sequence joint network path (spec dict)
    if not isinstance(spec, dict):  # pragma: no cover
        raise ValueError("Sequence spec must be dict")
    # ------------------- Device & precision selection (PyTorch 2.8 compat) -------------------
    req_device = spec.get("device", "auto")
    if req_device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():  # type: ignore[attr-defined]
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        # Allow explicit device specification; fallback gracefully if unavailable
        try:
            device = torch.device(req_device)
            if device.type == 'cuda' and not torch.cuda.is_available():  # pragma: no cover
                logging.warning("Requested CUDA but not available. Falling back to CPU.")
                device = torch.device('cpu')
        except Exception:  # pragma: no cover
            logging.warning("Invalid device '%s'; falling back to CPU", req_device)
            device = torch.device('cpu')

    # Precision logic
    req_prec = spec.get("precision", "auto")
    if req_prec == "auto":
        # Prefer bf16 if CUDA with native support, else fp16 if CUDA, else fp32
        if device.type == 'cuda' and torch.cuda.is_bf16_supported():  # type: ignore[attr-defined]
            precision = 'bf16'
        elif device.type == 'cuda':
            precision = 'fp16'
        else:
            precision = 'fp32'
    else:
        precision = req_prec
    if precision not in {"fp16","bf16","fp32"}:  # pragma: no cover
        logging.warning("Unknown precision '%s'; defaulting to fp32", precision)
        precision = 'fp32'

    # Map to dtype for autocast
    if precision == 'fp16':
        amp_dtype = torch.float16
    elif precision == 'bf16':
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float32

    net = _build_net(spec)
    net.to(device)

    # Optional torch.compile (available PyTorch >=2.0) — safe fallback if unsupported
    if spec.get("compile", False):
        try:  # pragma: no cover (compile not needed for simple tests)
            net = torch.compile(net)  # type: ignore[attr-defined]
        except Exception as e:  # graceful fallback
            logging.warning("torch.compile failed (%s); continuing without compilation", e)
    batch_size = spec.get("batch_size", 256)
    val_batch = spec.get("val_batch_size", 512)
    epochs = spec.get("epochs", 50)
    patience = spec.get("patience", 8)
    lr = spec.get("lr", 1e-3)
    weight_decay = spec.get("weight_decay", 1e-2)
    optimizer_name = spec.get("optimizer", "adamw")
    label_smoothing = spec.get("label_smoothing", 0.0)
    grad_clip = spec.get("grad_clip", 1.0)

    # Class imbalance weighting (pos_weight = #neg/#pos) if imbalance > 55/45
    pos_frac = yS_tr.mean()
    pos_weight = None
    if pos_frac > 0 and (pos_frac < 0.45 or pos_frac > 0.55):
        pos_weight = float((1 - pos_frac) / pos_frac) if pos_frac < 0.5 else float(pos_frac / (1 - pos_frac))

    if optimizer_name == "adamw":
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    else:  # pragma: no cover
        opt = torch.optim.Adam(net.parameters(), lr=lr)
    use_amp = device.type in {"cuda"} and precision in {"fp16","bf16"}
    # Gradient scaling only needed for fp16 (not bf16)
    scaler = _make_scaler(use_amp and precision == 'fp16')
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device) if pos_weight and pos_weight > 1.0 else None)
    l1 = nn.L1Loss()

    def _smooth(y: torch.Tensor) -> torch.Tensor:
        if label_smoothing and label_smoothing > 0:
            return y * (1 - label_smoothing) + 0.5 * label_smoothing
        return y

    tr_ds = TensorDataset(torch.tensor(X_tr_seq, dtype=torch.float32), torch.tensor(yS_tr, dtype=torch.float32), torch.tensor(np.abs(yM_tr), dtype=torch.float32))
    v_ds = TensorDataset(torch.tensor(X_v_seq, dtype=torch.float32), torch.tensor(yS_v, dtype=torch.float32), torch.tensor(np.abs(yM_v), dtype=torch.float32))
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    v_loader = DataLoader(v_ds, batch_size=val_batch, shuffle=False, pin_memory=True)

    best_val = math.inf; bad = 0; best_state = None
    for _ in range(epochs):
        net.train()
        for xb, ybS, ybM in tr_loader:
            xb = xb.to(device, non_blocking=True); ybS = ybS.to(device, non_blocking=True); ybM = ybM.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            # Autocast context (disabled for fp32)
            ac = _autocast_ctx(device.type, amp_dtype, use_amp)
            with ac:
                out = net(xb)
                loss = 0.6 * bce(out['sign_logit'], _smooth(ybS)) + 0.4 * l1(out['mag_pred'], ybM)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
                opt.step()
        # Validation
        net.eval(); v_loss = 0.0
        with torch.no_grad():
            for xb, ybS, ybM in v_loader:
                xb = xb.to(device, non_blocking=True); ybS = ybS.to(device, non_blocking=True); ybM = ybM.to(device, non_blocking=True)
                ac = _autocast_ctx(device.type, amp_dtype, use_amp)
                with ac:
                    out = net(xb)
                    v_loss += (0.6 * bce(out['sign_logit'], _smooth(ybS)) + 0.4 * l1(out['mag_pred'], ybM)).item()
        if v_loss < best_val:
            best_val = v_loss; best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state:
        net.load_state_dict(best_state)
    return {"joint": net, "device": device, "kind": spec["type"], "spec": {**spec, "_resolved_device": str(device), "_resolved_precision": precision}}


def predict_proba_sign(bundle: Dict[str, Any], X_tab: np.ndarray, X_seq: np.ndarray) -> np.ndarray:
    if 'sign' in bundle:
        model = bundle['sign']
        # Preferred: native predict_proba
        if hasattr(model, 'predict_proba'):
            try:
                return model.predict_proba(X_tab)[:, 1]
            except Exception:  # pragma: no cover
                pass
        # Some linear models expose _predict_proba_lr
        if hasattr(model, '_predict_proba_lr'):
            try:
                return model._predict_proba_lr(X_tab)[:, 1]  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover
                pass
        # Fallback: decision_function -> logistic transform
        if hasattr(model, 'decision_function'):
            logits = model.decision_function(X_tab)
            return 1.0 / (1.0 + np.exp(-logits))
        # Last resort: use discrete predictions as hard probs
        preds = model.predict(X_tab)
        return preds.astype(float)
    net=bundle['joint']; device=bundle['device']; spec_meta=bundle.get('spec', {})
    precision = spec_meta.get('_resolved_precision', 'fp32')
    use_amp = device.type=='cuda' and precision in {'fp16','bf16'}
    amp_dtype = torch.float16 if precision=='fp16' else (torch.bfloat16 if precision=='bf16' else torch.float32)
    with torch.no_grad():
        try:
            ac = _autocast_ctx(device.type, amp_dtype, use_amp)
            with ac:
                out_dict = net(torch.tensor(X_seq, dtype=torch.float32, device=device))
                logits = out_dict['sign_logit']
                # Cast to float32 before moving to CPU to avoid bf16 -> numpy incompat
                return logits.float().sigmoid().cpu().numpy()
        except RuntimeError as e:  # numpy interop failure
            return np.full(len(X_seq), 0.5, dtype=np.float32)


def predict_mag(bundle: Dict[str, Any], X_tab: np.ndarray, X_seq: np.ndarray) -> np.ndarray:
    if 'mag' in bundle:
        return bundle['mag'].predict(X_tab)
    net=bundle['joint']; device=bundle['device']; spec_meta=bundle.get('spec', {})
    precision = spec_meta.get('_resolved_precision', 'fp32')
    use_amp = device.type=='cuda' and precision in {'fp16','bf16'}
    amp_dtype = torch.float16 if precision=='fp16' else (torch.bfloat16 if precision=='bf16' else torch.float32)
    with torch.no_grad():
        try:
            ac = _autocast_ctx(device.type, amp_dtype, use_amp)
            with ac:
                out_dict = net(torch.tensor(X_seq, dtype=torch.float32, device=device))
                mag = out_dict['mag_pred']
                return mag.float().cpu().numpy()
        except RuntimeError:
            return np.full(len(X_seq), float(np.abs(X_tab).mean()) if len(X_tab) else 0.0, dtype=np.float32)


def calibrate_sign_prob(bundle: Dict[str, Any], X_v_tab: np.ndarray, X_v_seq: np.ndarray, yS_v: np.ndarray):
    p_raw = predict_proba_sign(bundle, X_v_tab, X_v_seq)
    if len(p_raw) < 50:
        return IdentityCalibrator()
    try:
        return IsotonicRegression(out_of_bounds='clip').fit(p_raw, yS_v)
    except Exception:  # pragma: no cover
        return IdentityCalibrator()


def optimize_thresholds(p_val: np.ndarray, m_val: np.ndarray, y_sign: np.ndarray):
    """Original EV-driven joint probability & magnitude threshold search (retained)."""
    grid_p = np.linspace(0.56, 0.70, 15)
    grid_m = np.unique(np.quantile(m_val, [0, .2, .4, .6, .8, .9, .95]))
    best_ev = -1e9; best = (0.6, float(np.median(m_val)))
    for p_star in grid_p:
        for theta in grid_m:
            sel = (p_val >= p_star) & (m_val >= theta)
            if sel.sum() < 20:
                continue
            wins = (y_sign == 1)[sel]
            ev = (0.8 * wins - (1 - wins)).mean()
            if ev > best_ev:
                best_ev = float(ev); best = (float(p_star), float(theta))
    return best_ev, best


def optimize_accuracy_threshold(p_val: np.ndarray, y_sign: np.ndarray, lo: float = 0.45, hi: float = 0.65, step: float = 0.01) -> Tuple[float, float]:
    """Search probability threshold maximizing accuracy.

    Returns (best_accuracy, threshold).
    """
    best_acc = -1.0; best_t = lo
    t = lo
    while t <= hi + 1e-9:
        pred = (p_val >= t).astype(int)
        acc = float((pred == y_sign).mean())
        if acc > best_acc:
            best_acc = acc; best_t = t
        t += step
    return best_acc, best_t


def size_capped_kelly(p_hat: np.ndarray, f_max: float) -> np.ndarray:
    return np.clip(((0.8*p_hat)-(1-p_hat))/0.8, 0.0, f_max)


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins:int=10)->float:
    cuts=np.linspace(0,1,bins+1); ece=0.0
    for i in range(bins):
        m=(p>=cuts[i]) & (p<cuts[i+1]) if i<bins-1 else (p>=cuts[i]) & (p<=cuts[i+1])
        if m.any(): e=p[m].mean()-y[m].mean(); ece += m.mean()*abs(e)
    return float(ece)


def evaluate_on_set(bundle: Dict[str, Any], calib: Any, p_star: float, theta: float,
                     X_tab: np.ndarray, X_seq: np.ndarray, y_sign: np.ndarray, f_max: float=0.02) -> Dict[str, Any]:
    p_raw = predict_proba_sign(bundle, X_tab, X_seq)
    p_hat = calib.predict(p_raw) if hasattr(calib,'predict') else p_raw
    m_hat = predict_mag(bundle, X_tab, X_seq)
    trade_mask = (p_hat>=p_star) & (m_hat>=theta)
    wins=(y_sign==1)
    # Position sizing using capped Kelly (expected edge vs 0.8 payout structure) if trades selected
    if trade_mask.any():
        sizes = size_capped_kelly(p_hat[trade_mask], f_max)
    else:
        sizes = np.array([], dtype=np.float32)
    raw_payoffs = 0.8*wins[trade_mask] - (1-wins[trade_mask])
    payoffs = raw_payoffs * (sizes if len(sizes)==len(raw_payoffs) else 1.0)
    ev_per_trade = float(payoffs.mean()) if trade_mask.any() else 0.0
    total_ev = float(payoffs.sum())
    brier=float(brier_score_loss(y_sign, p_hat))
    ece=expected_calibration_error(p_hat, y_sign)
    try: roc=float(roc_auc_score(y_sign, p_hat))
    except Exception: roc=float('nan')
    try: pr=float(average_precision_score(y_sign, p_hat))
    except Exception: pr=float('nan')
    acc_default = float(((p_hat >= 0.5).astype(int) == y_sign).mean())
    return {"n_trades": int(trade_mask.sum()), "ev_per_trade": ev_per_trade, "total_ev": total_ev, "brier": brier, "ece": ece, "pr_auc": pr, "roc_auc": roc, "p_star": p_star, "theta": theta, "acc@0.5": acc_default}


__all__ = [
    "build_models",
    "train_model",
    "predict_proba_sign",
    "predict_mag",
    "calibrate_sign_prob",
    "optimize_thresholds",
    "optimize_accuracy_threshold",
    "evaluate_on_set",
    "size_capped_kelly",
    "IdentityCalibrator",
]


class MeanMagPredictor:
    """Predictor returning constant mean absolute magnitude (picklable)."""
    def __init__(self, v: float):
        self.v = float(v)
    def predict(self, X):  # type: ignore[override]
        return np.full(len(X), self.v, dtype=np.float32)

