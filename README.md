# Intraday FX Binary Classifier

Deterministic, reproducible intraday foreign exchange (FX) modeling pipeline for predicting short‑horizon (1,5,10,15 minute) directional moves during the New York session and estimating move magnitude. The project produces both machine‑learning metrics (ROC AUC, Brier, calibration) and trading KPIs (expected value per trade, gross return) under disciplined time‑series hygiene (walk‑forward splits with embargo, no leakage, strictly past‑only features, top‑of‑hour sequence predictions).

---
## ✨ Key Features
- Full end‑to‑end 5‑step pipeline (raw → cleaned → features → sequences → model training → ranking)
- NY session filtering (12:00–20:59 America/New_York) with controlled small‑gap forward fill
- Multi‑horizon targets: future k‑minute log return sign & volatility‑scaled magnitude
- Vectorized feature engineering (returns, volatility, range, volume, time‑of‑day, z‑scores)
- Hourly sequence extraction: 60 contiguous 1‑minute windows ending at top‑of‑hour
- Dual modality modeling:
  - Tabular summarizations of sequences (statistical aggregates)
  - Joint sequence deep nets (GRU/LSTM/Transformer backbones) predicting sign (classification) + absolute magnitude (regression) simultaneously
- Probability calibration (Isotonic) with accuracy‑oriented threshold search plus EV‑driven secondary thresholds
- Walk‑forward evaluation blocks (train / val / test) with embargo to reduce leakage
- Deterministic seeding across Python / NumPy / Torch
- Ranking engine computing composite model score & exports consolidated artifacts
- All paths centralized in `.config` (supports `@/` repo‑relative resolution)

---
## 🗂 Directory Structure (core)
```
lib/                Reusable utilities (config, logging, randomness, modeling)
src/                Step scripts (1–5)
data/raw            Raw minute CSVs (Time, Open, High, Low, Close, Volume)
data/cleaned        NY‑session filtered + small gap filled
/data/featured/<k>m Parquet feature + target tables per horizon
/data/sequenced     Sequence tensors (.npz) + walk‑forward splits.json
/models/<SYMBOL>/   Trained artifacts per horizon + metrics
/models/rankings/   Aggregated ranking outputs
```

---
## ⚙️ Configuration
Edit `.config` at project root (example):
```
RAW_DATA_PATH=@/data/raw
CLEANED_DATA_PATH=@/data/cleaned
FEATURED_DATA_PATH=@/data/featured
SEQUENCED_DATA_PATH=@/data/sequenced
MODEL_OUTPUT=@/models
```
All scripts resolve these via `lib.config.load_config()`. Paths are created if missing.

---
## 📦 Dependencies
Pinned in `requirements.txt` (Python ≥3.12 recommended):
- pandas, numpy, pyarrow / fastparquet
- scikit-learn, xgboost (optional if installed), torch
- tqdm, tabulate
Install:
```bash
pip install -r requirements.txt
```

---
## 🚀 Pipeline Steps
Run from project root (or `src/`). Each step is idempotent.

1. Data Preprocessing (`src/1_data_preprocessing.py`)
   - Reads raw CSVs → filters NY session → fills small (≤5m) intra‑session gaps → writes cleaned CSV.

2. Feature Calculation (`src/2_features_calculation.py`)
   - Builds horizon‑specific parquet datasets with past‑only engineered features and future targets for horizons {1,5,10,15}.

3. Sequencing (`src/3_sequencing.py`)
   - Extracts strictly contiguous 60‑minute windows ending at each top‑of‑hour; reconstructs per‑minute return `r1`; packs sequences + multi‑horizon targets into compressed `.npz` per symbol; generates walk‑forward `splits.json` (train/val/test blocks with embargo).

4. Model Training (`src/4_model_training.py`)
   - For each symbol & horizon: creates tabular summary features, builds candidate models (logistic L1, XGBoost variants, GRU/LSTM/Transformer specs), trains with early stopping, calibrates probabilities, optimizes accuracy threshold (0.45–0.65 search), derives EV trade filter theta, evaluates on test fold(s), saves metrics + model weights/state.

5. Results Aggregation / Ranking (`src/5_test_results.py`)
   - Aggregates `agg_metrics.json` across symbols/horizons; computes composite min‑max weighted score; exports parquet, CSV rankings, best per horizon, top N JSON, and Markdown summary.

---
## 🧪 Walk‑Forward Splits
Defined in sequencing step:
- Train: 60 days
- Validation: 7 days (after 60‑minute embargo)
- Test: 21 days (after embargo)
- Slide forward by test block length; all folds processed independently.

This preserves chronological integrity and reduces leakage.

---
## 🧠 Modeling Details
Sequence models: joint network produces sign logit + non‑negative magnitude head. Training loss = 0.6 * BCE(logits, sign) + 0.4 * L1(magnitude, |target|) with optional label smoothing & class imbalance weighting. Optional mixed precision (fp16/bf16) and `torch.compile` support.

Tabular models: logistic regression (L1) and XGBoost (small/medium) when XGBoost is installed.

Probability calibration: Isotonic (fallback identity if insufficient validation points).

Threshold selection:
- Accuracy threshold search (probability) across [0.45, 0.65] increments 0.01.
- EV threshold search (p* & magnitude theta) retained for trade filtering and reporting.

Trading sizing: capped Kelly fraction (max fraction f_max=0.02) applied to selected trades.

---
## 📊 Metrics
Per fold (test set):
- Classification: roc_auc, average precision (if available), Brier score, expected calibration error (ECE), accuracy @0.5
- Trading: n_trades, ev_per_trade, total_ev, gross_return, gross_return_per_trade
- Thresholds: probability p_star (accuracy driven), magnitude theta (from EV search)

Aggregated (`agg_metrics.json`): mean across folds + fold count.

Ranking composite weights (see `5_test_results.py`):
```
ev_per_trade: 0.30
total_ev: 0.20
gross_return_per_trade: 0.15
gross_return: 0.10
roc_auc: 0.15
brier: -0.05 (lower better)
ece: -0.05 (lower better)
```

---
## 🛠 Usage Quickstart
```bash
# 1. Prepare raw CSVs in data/raw (Time,Open,High,Low,Close,Volume)
python src/1_data_preprocessing.py

# 2. Generate features
python src/2_features_calculation.py

# 3. Build sequences & splits
python src/3_sequencing.py

# 4. Train models
python src/4_model_training.py

# 5. Aggregate & rank results
python src/5_test_results.py
```
Outputs land in `models/` and `models/rankings/`.

---
## 🧩 Extending
- Add new engineered features: modify `2_features_calculation.py` ensuring past‑only semantics.
- Add sequence backbone: extend `_build_net` in `lib/modeling.py` (reuse pattern) and update `build_models` spec mapping.
- Adjust walk‑forward regime: edit constants in `3_sequencing.py`.
- Change thresholds search ranges: modify `optimize_accuracy_threshold` / `optimize_thresholds` logic in `lib/modeling.py`.

---
## 🔒 Reproducibility
`lib/random.seed_all()` seeds Python, NumPy, Torch (deterministic flags) once per run. Config snapshots stored with each symbol under `models/<SYMBOL>/config_snapshot.json` including SHA‑256 of `.config` content.

---
## ⚠️ Notes & Assumptions
- No data look‑ahead: all features shifted to exclude information from or after prediction timestamp.
- Sequence sampling restricted to top‑of‑hour to reduce temporal correlation.
- Magnitude model predicts absolute (vol‑scaled) move size; trade decision combines probability & magnitude thresholds.
- XGBoost models skipped gracefully if library not installed.

---
## 📜 License
(Insert project license here if applicable.)

---
## 🤝 Contributions
Issues & pull requests welcome. Please keep additions deterministic, time‑aware, and reuse utilities in `lib/` (avoid duplicating logic).

---
## 🧾 Citation
If you use this project in research or a report, consider citing with the repository URL and commit hash.

---
Happy modeling.
