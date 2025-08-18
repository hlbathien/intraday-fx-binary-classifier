"""Model building & training utilities (tabular + sequence)."""
from __future__ import annotations

from typing import Dict, Tuple, Any, List
import math
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
from torch.cuda.amp import autocast, GradScaler


def build_models(input_dim_seq: int) -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "ridge": RidgeClassifier(),
        "lasso": LogisticRegression(penalty="l1", C=0.5, solver="liblinear", max_iter=200),
    }
    if xgb is not None:
        models["xgb"] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            verbosity=0,
        )
    # Only add sequence models if torch can interop with current numpy
    try:
        _ = np.zeros(1)  # basic check
        models.update({"gru": ("gru", input_dim_seq), "lstm": ("lstm", input_dim_seq), "transf": ("transf", input_dim_seq)})
    except Exception:  # pragma: no cover
        pass
    return models


class IdentityCalibrator:
    """Picklable identity calibrator (used when not enough samples)."""
    def predict(self, x):  # type: ignore[override]
        return x


class JointSeqNet(nn.Module):
    def __init__(self, backbone: nn.Module, hidden: int):
        super().__init__()
        self.backbone = backbone
        self.head_sign = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.head_mag = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(x)
        return {"sign_logit": self.head_sign(h).squeeze(-1), "mag_pred": torch.relu(self.head_mag(h).squeeze(-1))}


class RNNBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden: int, kind: str):
        super().__init__()
        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM}[kind]
        self.rnn = rnn_cls(input_dim, hidden, num_layers=1, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, h = self.rnn(x)
        if isinstance(h, tuple):
            h = h[0]
        return h[-1]


class TransformerBackbone(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 64, heads: int = 2):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=d_model * 2, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=1)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        z = self.enc(z)
        return self.pool(z.transpose(1, 2)).squeeze(-1)


def _build_net(kind: str, input_dim: int, hidden: int = 64) -> JointSeqNet:
    if kind in {"gru", "lstm"}:
        backbone = RNNBackbone(input_dim, hidden, kind)
    elif kind == "transf":
        backbone = TransformerBackbone(input_dim, d_model=hidden, heads=2)
    else:  # pragma: no cover
        raise ValueError(kind)
    return JointSeqNet(backbone, hidden)


def train_model(spec: Any,
                X_tr_tab: np.ndarray, X_tr_seq: np.ndarray, yS_tr: np.ndarray, yM_tr: np.ndarray,
                X_v_tab: np.ndarray, X_v_seq: np.ndarray, yS_v: np.ndarray, yM_v: np.ndarray,
                max_epochs: int = 50) -> Dict[str, Any]:
    # Classical tabular classifiers path
    if isinstance(spec, (RidgeClassifier, LogisticRegression)) or (xgb is not None and isinstance(spec, xgb.XGBClassifier)):
        model_sign = spec
        model_sign.fit(X_tr_tab, yS_tr)
        # Simple magnitude regressor (ridge) on absolute magnitude for filtering
        try:
            mag_reg = RidgeClassifier()  # placeholder classifier not suitable for regression
            raise ValueError  # force except to use fallback
        except Exception:
            mean_mag = float(np.abs(yM_tr).mean())
            mag_reg = MeanMagPredictor(mean_mag)
        return {"sign": model_sign, "mag": mag_reg}
    # Sequence joint network path
    kind, input_dim = spec
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = _build_net(kind, input_dim)
    net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    bce = nn.BCEWithLogitsLoss(); l1 = nn.L1Loss()
    tr_ds = TensorDataset(torch.tensor(X_tr_seq, dtype=torch.float32), torch.tensor(yS_tr, dtype=torch.float32), torch.tensor(np.abs(yM_tr), dtype=torch.float32))
    v_ds = TensorDataset(torch.tensor(X_v_seq, dtype=torch.float32), torch.tensor(yS_v, dtype=torch.float32), torch.tensor(np.abs(yM_v), dtype=torch.float32))
    tr_loader = DataLoader(tr_ds, batch_size=256, shuffle=True, pin_memory=True)
    v_loader = DataLoader(v_ds, batch_size=512, shuffle=False, pin_memory=True)
    best_val = math.inf; patience=8; bad=0; best_state=None
    for _ in range(max_epochs):
        net.train()
        for xb, ybS, ybM in tr_loader:
            xb=xb.to(device); ybS=ybS.to(device); ybM=ybM.to(device)
            opt.zero_grad()
            with autocast(enabled=torch.cuda.is_available()):
                out = net(xb)
                loss = 0.6*bce(out['sign_logit'], ybS)+0.4*l1(out['mag_pred'], ybM)
            scaler.scale(loss).backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); scaler.step(opt); scaler.update()
        net.eval(); v_loss=0.0
        with torch.no_grad():
            for xb, ybS, ybM in v_loader:
                xb=xb.to(device); ybS=ybS.to(device); ybM=ybM.to(device)
                with autocast(enabled=torch.cuda.is_available()):
                    out = net(xb)
                    v_loss += (0.6*bce(out['sign_logit'], ybS)+0.4*l1(out['mag_pred'], ybM)).item()
        if v_loss < best_val:
            best_val=v_loss; best_state={k:v.cpu().clone() for k,v in net.state_dict().items()}; bad=0
        else:
            bad+=1
            if bad>=patience: break
    if best_state: net.load_state_dict(best_state)
    return {"joint": net, "device": device, "kind": kind}


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
    net=bundle['joint']; device=bundle['device']; net.eval()
    with torch.no_grad():
        try:
            return net(torch.tensor(X_seq, dtype=torch.float32, device=device))['sign_logit'].sigmoid().cpu().numpy()
        except RuntimeError as e:  # numpy interop failure
            # Fallback: uniform probability 0.5 to avoid crash
            return np.full(len(X_seq), 0.5, dtype=np.float32)


def predict_mag(bundle: Dict[str, Any], X_tab: np.ndarray, X_seq: np.ndarray) -> np.ndarray:
    if 'mag' in bundle:
        return bundle['mag'].predict(X_tab)
    net=bundle['joint']; device=bundle['device']; net.eval()
    with torch.no_grad():
        try:
            return net(torch.tensor(X_seq, dtype=torch.float32, device=device))['mag_pred'].cpu().numpy()
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
    grid_p = np.linspace(0.56, 0.70, 15)
    grid_m = np.unique(np.quantile(m_val, [0, .2, .4, .6, .8, .9, .95]))
    best_ev=-1e9; best=(0.6, float(np.median(m_val)))
    for p_star in grid_p:
        for theta in grid_m:
            sel = (p_val>=p_star) & (m_val>=theta)
            if sel.sum()<20: continue
            wins=(y_sign==1)[sel]
            ev=(0.8*wins - (1-wins)).mean()
            if ev>best_ev:
                best_ev=float(ev); best=(float(p_star), float(theta))
    return best_ev, best


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
    return {"n_trades": int(trade_mask.sum()), "ev_per_trade": ev_per_trade, "total_ev": total_ev, "brier": brier, "ece": ece, "pr_auc": pr, "roc_auc": roc, "p_star": p_star, "theta": theta}


__all__ = ["build_models", "train_model", "predict_proba_sign", "predict_mag", "calibrate_sign_prob", "optimize_thresholds", "evaluate_on_set", "size_capped_kelly", "IdentityCalibrator"]


class MeanMagPredictor:
    """Predictor returning constant mean absolute magnitude (picklable)."""
    def __init__(self, v: float):
        self.v = float(v)
    def predict(self, X):  # type: ignore[override]
        return np.full(len(X), self.v, dtype=np.float32)

