# 💧 Liquidity Stress Early Warning System

### Production-Grade Machine Learning Pipeline | Financial Risk Analytics

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-02569B?style=flat-square&logo=lightgbm)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-FF6600?style=flat-square&logo=xgboost)
![CatBoost](https://img.shields.io/badge/CatBoost-Yandex-FFD700?style=flat-square)
![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Tuning-6DB3F2?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Pipelines-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?style=flat-square&logo=numpy&logoColor=white)

![FastAPI](https://img.shields.io/badge/FastAPI-Deployment-009688?style=flat-square&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerisation-2496ED?style=flat-square&logo=docker&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2?style=flat-square&logo=mlflow&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-FF6B6B?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-TabNet-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![YAML](https://img.shields.io/badge/YAML-Config%20Management-CB171E?style=flat-square&logo=yaml&logoColor=white)

![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen?style=flat-square)
![Competition](https://img.shields.io/badge/Zindi-AI4EAC%20Challenge-6236FF?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

</div>

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement & Business Impact](#problem-statement--business-impact)
- [Competition Details](#competition-details)
- [Dataset](#dataset)
- [Architecture & Design Decisions](#architecture--design-decisions)
- [Pipeline Stages](#pipeline-stages)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Reproducing Results](#reproducing-results)
- [Roadmap](#roadmap)
- [Author](#author)

---

## Project Overview

This project implements a **production-grade machine learning early warning system** to predict which mobile money customers are likely to experience **liquidity stress within the next 30 days**, using six months of transactional behavioral data.

Built for the **AI for Economic Activity Challenge (AI4EAC)** on Zindi Africa, this system goes beyond competition-level modeling to demonstrate **real-world MLOps practices**: modular pipelines, reproducible experiments, calibrated probability outputs, a three-layer ensemble, and a deployable REST API.

The competition metric is a **weighted composite score**:

```
Score = 0.6 × LogLoss + 0.4 × (1 − AUC)     [lower is better]
```

---

## Problem Statement & Business Impact

Mobile money is the primary financial infrastructure for millions of households and small businesses across Africa. When liquidity pressure builds, customers show early behavioral signals in their transaction patterns — often 2–4 weeks before formal financial distress becomes visible.

**This project addresses:** Can we detect those behavioral signals from mobile money data and provide a 30-day early warning?

Early warning enables:
- Proactive financial support before customers reach crisis
- Targeted credit or savings products at the right moment
- Reduced default risk for financial service providers
- Measurable social good — protecting vulnerable populations from financial hardship

**Key behavioral insight discovered:** Transaction *frequency* decline precedes transaction *value* decline by 1–2 months. A customer who stops depositing regularly before their balance drops is showing a stress signal the value-only view misses entirely. This finding drove the v2.2 feature expansion.

---

## Competition Details

| Item | Detail |
|---|---|
| Competition | AI for Economic Activity Challenge (AI4EAC) — Zindi Africa |
| Task | Binary classification — predict P(liquidity stress in next 30 days) |
| Primary metric | Log Loss (60%) + ROC-AUC (40%) composite |
| Output format | Predicted probabilities (not labels) |
| Training set | 40,000 customer snapshots × 184 raw columns |
| Test set | 30,000 customer snapshots |
| Class balance | ~15% positive (6,000 stressed / 34,000 non-stressed) |

---

## Dataset

The dataset contains a **6-month mobile money panel** in wide format. Each row is a unique customer snapshot. There are no time-series rows — this is a supervised classification problem with temporal structure encoded in column naming.

### Feature Groups

| Group | Description | Columns |
|---|---|---|
| Customer profile | Age, gender, region, segment, earning pattern, smartphone | 8 |
| Balance | Monthly average daily balance (M1–M6) | 6 |
| Deposits | Volume, value, highest amount, unique agents (M1–M6) | 24 |
| Withdrawals | Volume, value, highest amount, unique agents (M1–M6) | 24 |
| Paybill | Volume, value, highest amount, unique companies (M1–M6) | 24 |
| Merchant payments | Volume, value, highest amount, unique merchants (M1–M6) | 24 |
| MM sends (P2P out) | Volume, value, highest amount, unique recipients (M1–M6) | 24 |
| MM received (P2P in) | Volume, value, highest amount, unique senders (M1–M6) | 24 |
| Bank transfers in | Volume, value, highest amount, unique banks (M1–M6) | 24 |
| Activity | 90-day activity rate, ARPU | 2 |
| **Target** | `liquidity_stress_next_30d` (binary) | 1 |

**Convention:** M1 = most recent month, M6 = oldest month. Null values mean zero activity per competition specification.

---

## Architecture & Design Decisions

### Why gradient boosted trees as the base?
LightGBM, XGBoost, and CatBoost are consistently the strongest performers on tabular financial data. They handle sparsity natively (60–98% zero rates in this dataset), require no standardisation, and produce interpretable feature importances via SHAP.

### Why add Logistic Regression and TabNet?
Post-tuning diversity analysis showed inter-model correlation of **0.978** between all three GBM models. Stacking adds no value when base models are this correlated — the meta-model has nothing structurally different to learn. Logistic Regression (linear boundary) and TabNet (instance-wise attention) introduce genuinely orthogonal decision boundaries, reducing expected correlation to 0.72–0.90.

### Why Platt calibration?
The competition weights Log Loss at 60%. Log Loss penalises overconfident wrong predictions severely. GBMs without calibration tend to produce miscalibrated probabilities (too confident in edge cases). Platt scaling on OOF predictions improves calibration curves and directly reduces Log Loss without affecting AUC.

### Why OOF stacking over simple averaging?
Simple averaging of highly correlated models adds noise, not signal. The ensemble meta-model (LogisticRegression on OOF predictions) learns *when* to trust each base model conditionally — particularly useful in boundary cases where the models disagree. Nested cross-validation prevents meta-model overfitting.

### Why a global ratio explosion guard in feature engineering?
Transaction data has 60–98% zero rates. With EPS=1e-6 in ratio denominators, monetary values in the millions (KES) produce ratios in the 10¹¹–10¹² range. A global clip at the 99th percentile in `_clean_features()` covers all current and future ratio features automatically.

---

## Pipeline Stages

### Stage 1 — Data Validation ✅
**Module:** `src/data/load_data.py`

- Schema validation against expected column set
- Duplicate ID detection
- Target distribution check (verified: 15% positive, 34k/6k split)
- Train/test column alignment with explicit mismatch reporting
- Memory optimisation (downcasting to uint8/float32 where possible)
- High-missing and high-zero feature flagging

---

### Stage 2 — Feature Engineering v2.2.1 ✅
**Module:** `src/features/feature_engineering.py`

**226 raw columns → 825 engineered features**

23 feature blocks across 6 layers:

| Layer | Blocks | Description |
|---|---|---|
| Raw signal | 01–09 | Temporal aggregations, trends, momentum, acceleration, volatility, recency — applied to value, volume, and highest_amount caches |
| Balance intelligence | 10–12 | Balance trend, drawdown, pressure vs obligations |
| Cashflow intelligence | 13–15, 23 | Net flow, slope, volatility, volume-based frequency cashflow |
| P2P & banking | 16–17 | Peer-to-peer ratios, bank transfer signals |
| Interaction | 18 | SHAP-evidenced cross-feature products |
| Encoding & indicators | 19–20 | Categorical encoding, zero-sparsity binary flags |
| Value-volume intelligence | 21–22 | Avg transaction size trends, unique entity network contraction |
| Log-transformed | 01,02,05,03,08 | Same blocks applied to log1p(value) and log1p(volume) |

**Key design decisions:**
- Three-cache architecture (`{group}`, `{group}_vol`, `{group}_hi`) — adding a new cache type auto-propagates through all 9 existing feature blocks
- Global ratio explosion guard clips all ratio columns at 99th percentile in `_clean_features()`
- CV-safe: fully stateless, no fitting step
- Deterministic: same input always produces identical output

**v2.2.1 validation results:**

| Check | Result |
|---|---|
| Features built | 643 (+389 vs v2.1) |
| Final processed shape | 40,000 × 825 |
| Duplicate columns | 0 |
| NaN / Inf values | 0 |
| Constant features | 0 |
| LightGBM AUC (300 est. quickcheck) | 0.9017 |
| `inflow_volume_recency_ratio` importance rank | 5th overall |

---

### Stage 3 — Preprocessing ✅
**Module:** `src/preprocessing/preprocessing.py`

- Quantile clipping at [0.001, 0.999] — 749 columns clipped
- Constant column removal (fit-time)
- Nullable dtype → numpy-native conversion (fixes pandas ambiguous truth-value errors)
- Feature contract enforcement: exact column set locked at `fit()`, asserted at `transform()`
- Memory optimisation: float64 → float32, int64 → smallest signed int
- `scale_features` flag for downstream LogReg/TabNet (StandardScaler applied here, not in feature engineering)
- Full persistence via `joblib` with `save()` / `load()` API

---

### Stage 4 — CV Training ✅
**Module:** `src/training/cv.py` | `src/orchestration/run_all_models.py`

- 5-fold stratified K-Fold (seed=42, shuffle=True)
- OOF prediction arrays saved per model (ensemble input)
- Per-fold and global metrics: LogLoss, ROC-AUC, composite score
- Model-agnostic factory: LightGBM / XGBoost / CatBoost / LogReg (planned: TabNet)
- CatBoost native categorical handling (`cat_features` pass-through)
- Windows-safe UTF-8 logging with ASCII fallback for console
- Per-model failure isolation — one model failure never aborts siblings
- Full artifact contract per run:

```
outputs/experiments/{stage}/{model}/run_YYYYMMDD_HHMMSS/
    oof_preds.npy           fold_scores.json
    y_true.npy              fold_indices.pkl
    feature_importance.csv  fold_predictions.pkl
    feature_list.json       preprocessor.pkl
    metadata.json           config_used.yaml
    models/{model}/         {model}_fold_{i}.pkl
```

---

### Stage 5 — Calibration ✅
**Module:** `notebooks/07_calibration_analysis.ipynb`

- Isotonic regression and Platt scaling evaluated per model
- Cross-validated Platt chosen for test-set generalisation safety
- Calibrated OOF predictions saved to `outputs/calibration/{model}/`
- Calibration curves (reliability diagrams) and ECE computed for all models

**Baseline calibrated OOF results:**

| Model | LogLoss | AUC | Composite Score |
|---|---|---|---|
| LightGBM (tuned) | — | — | **0.1941** |
| XGBoost (tuned) | — | — | **0.1921** |
| CatBoost (baseline) | — | — | ~0.195 |

---

### Stage 6 — Hyperparameter Tuning ✅
**Module:** `src/tuning/tune.py`

- Optuna TPE Bayesian search (100 trials per model)
- MedianPruner eliminates unpromising trials after fold 2 (~40% compute saved)
- Warm start from baseline params (trial 0) — tuned model guaranteed ≥ baseline
- Calibrated composite score as objective (Platt on OOF inside each trial)
- Separate SQLite study per model for independent resumption
- Post-tuning diversity check: inter-model correlation = **0.978** (unchanged)
- Result: tuning found better params within the GBM family; structural diversity requires new model families

---

### Stage 7 — Ensemble v4 ✅
**Module:** `src/ensemble/ensemble.py`

Three-layer ensemble architecture:

| Layer | Strategy | Method |
|---|---|---|
| Layer 1 | Simple average | Equal-weight baseline |
| Layer 2 | Optimised weighted average | Scipy Nelder-Mead on OOF composite score, 16 initialisations |
| Layer 3 | Stacking | LogisticRegression(C=0.1) meta-model on OOF + disagreement feature |
| Layer 4 | Calibrated stacking | Platt calibration applied to stacking OOF output |

**Key design decisions:**
- Platt over isotonic for test-set generalisation (isotonic used for OOF analysis only)
- `use_disagreement=True`: inter-model std as 4th meta-feature captures prediction uncertainty
- Nested CV prevents meta-model overfitting
- All artifacts saved to timestamped run directory

**Current limitation:** Inter-model correlation 0.978 means stacking gains are marginal. Adding LogReg and TabNet base models (Phase 2–3) is the correct next step.

---

### Stage 8 — Inference Pipeline 🔄 In Progress
**Module:** `src/inference/predict.py`

5-step inference contract:
1. Load and validate raw test data
2. Per-model prediction: feature engineering → preprocessing → fold-average
3. Platt calibration per model
4. Ensemble meta-model application
5. OOF vs test distribution drift check + Zindi-format submission CSV

**Submission format:**
```csv
ID,Target,TargetLogLoss,TargetRAUC
ID_XYZ,0.45,0.45,0.45
```

---

### Stages 9–12 — Planned 🔴
See [Roadmap](#roadmap) section.

---

## Key Results

| Milestone | Metric | Value |
|---|---|---|
| LightGBM baseline (v2.1 features) | Composite score | ~0.200 |
| LightGBM tuned (v2.1 features) | Composite score | 0.1941 |
| XGBoost tuned (v2.1 features) | Composite score | 0.1921 |
| LightGBM quickcheck (v2.2 features, 300 est.) | AUC | 0.9017 |
| Feature count v2.1 → v2.2 | Features | 254 → 643 (+389) |
| Inter-model correlation (GBMs) | Pearson r | 0.978 |
| New feature signal: inflow_volume_recency_ratio | Importance rank | 5th overall |

---

## Project Structure

```
liquidity-stress-early-warning/
│
├── .vscode/                          # Editor settings (gitignored)
│
├── configs/                          # All YAML configuration files
│   ├── baseline.yaml                 # Shared baseline config
│   ├── lgbm_v2.yaml                  # LightGBM baseline config
│   ├── xgb_v2.yaml                   # XGBoost baseline config
│   ├── catboost_v2.yaml              # CatBoost config (native cat handling)
│   ├── lightgbm_tuned.yaml           # LightGBM post-Optuna config
│   ├── xgboost_tuned.yaml            # XGBoost post-Optuna config
│   ├── logreg_v1.yaml                # [PLANNED] LogReg config
│   └── tabnet_v1.yaml                # [PLANNED] TabNet config
│
├── data/
│   ├── raw/
│   │   ├── Train.csv                 # 40,000 × 184 (immutable)
│   │   ├── Test.csv                  # 30,000 × 183 (immutable)
│   │   ├── data_dictionary.csv       # Column definitions
│   │   └── SampleSubmission.csv      # Zindi submission format
│   └── processed/                    # Intermediate feature matrices
│
├── notebooks/
│   ├── 01_data_validation.ipynb      # ✅ Schema, quality, class balance
│   ├── 02_EDA.ipynb                  # ✅ Distributions, correlations
│   ├── 03_feature_engineering_       # ✅ v2.2.1 validation + new feature
│   │   validation.ipynb               #    signal confirmation
│   ├── 04_preprocessing_             # ✅ Pipeline contract verification
│   │   validation.ipynb
│   ├── 05_full_pipeline_             # ✅ End-to-end smoke test
│   │   validation.ipynb
│   ├── 06_baseline_analysis.ipynb    # ✅ SHAP, errors, calibration diagnosis
│   ├── 07_calibration_analysis.ipynb # ✅ Platt vs isotonic per model
│   ├── 08_multi_model_analysis.ipynb # ✅ 3-model comparison  corr matrix
│   ├── 09_ensemble.ipynb             # ✅ 3-model ensemble 
│   ├── 10_feature_engineering_       # ✅ Feature engineering pipleine validation
│   │  validation_v2_2# 
│   └── 11_    
│
├── outputs/
│   ├── experiments/
│   │   ├── baseline/
│   │   │   ├── lightgbm/run_*/       # Fold models, OOF, metrics, artifacts
│   │   │   ├── xgboost/run_*/
│   │   │   └── catboost/run_*/
│   │   ├── v2_feature_expansion/
│   │   │   ├── lightgbm/run_*/       # Tuned LightGBM artifacts
│   │   │   └── xgboost/run_*/        # Tuned XGBoost artifacts
│   │   └── v4_ensemble/
│   │       └── run_YYYYMMDD_HHMMSS/  # Ensemble artifacts (meta-model, weights)
│   ├── calibration/
│   │   ├── lgb/                      # calibrator_platt.pkl, calibrator_isotonic.pkl
│   │   ├── xgb/
│   │   └── cat/
│   ├── tuning/
│   │   ├── lightgbm_best_params.yaml
│   │   ├── lightgbm_study_summary.json
│   │   ├── xgboost_best_params.yaml
│   │   └── xgboost_study_summary.json
│   ├── multi_model/                  # Calibrated OOF for ensemble input
│   │   ├── oof_calibrated_lightgbm.npy
│   │   ├── oof_calibrated_xgboost.npy
│   │   ├── oof_calibrated_catboost.npy
│   │   └── y_true.npy
│   ├── logs/                         # Multi-model run logs + YAML summaries
│   └── submissions/                  # Generated submission CSV files
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   └── load_data.py              # Data ingestion + schema validation
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_engineering.py   # 23-block feature pipeline v2.2.1
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── preprocessing.py         # CV-safe PreprocessingPipeline
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── cv.py                    # 5-fold CV engine + artifact saver
│   │   └── train_baseline.py        # Single-model training script
│   │
│   ├── ensemble/
│   │   ├── __init__.py
│   │   └── ensemble.py              # 4-strategy ensemble pipeline v4
│   │
│   ├── orchestration/
│   │   ├── __init__.py
│   │   └── run_all_models.py        # Multi-model training orchestrator
│   │
│   ├── tuning/
│   │   ├── __init__.py
│   │   └── tune.py                  # Optuna tuning for LGBM + XGB
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   └── predict.py               # End-to-end inference + submission
│   │
│   └── utils/
│       ├── __init__.py
│       └── paths.py                  # Project root resolution utilities
│
├── api/                              # [PLANNED] FastAPI deployment
│   ├── app.py                        # REST API with /predict + /health
│   ├── schemas.py                    # Pydantic request/response models
│   ├── model_loader.py               # Startup artifact loading (lru_cache)
│   └── predictor.py                  # Single-customer inference wrapper
│
├── tests/                            # [PLANNED] Unit + integration tests
│
├── artifacts/                        # Preprocessor PKLs for quick access
│   ├── preprocessing_pipeline.pkl
│   └── catboost_info/
│
├── Dockerfile                        # [PLANNED] API containerisation
├── docker-compose.yml                # [PLANNED]
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

---

## Tech Stack

### Core ML & Data
| Library | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Runtime |
| pandas | 2.x | Data manipulation |
| NumPy | 1.x | Vectorised computation |
| scikit-learn | 1.x | CV, preprocessing, LogReg |
| LightGBM | 4.x | Primary GBM (best calibration) |
| XGBoost | 2.x | Secondary GBM (best AUC) |
| CatBoost | 1.x | Native categorical handling |
| pytorch-tabnet | 4.x | [PLANNED] Attention-based tabular NN |

### Tuning & Tracking
| Library | Purpose |
|---|---|
| Optuna | Bayesian HPO with TPE sampler + MedianPruner |
| MLflow | [PLANNED] Experiment tracking + model registry |

### Interpretation
| Library | Purpose |
|---|---|
| SHAP | Global + local feature importance |
| matplotlib / seaborn | Visualisation |

### Deployment
| Tool | Purpose |
|---|---|
| FastAPI | [PLANNED] REST API endpoint |
| Pydantic | [PLANNED] Request/response validation |
| uvicorn | [PLANNED] ASGI server |
| Docker | [PLANNED] Containerisation |

### Configuration & Serialisation
| Tool | Purpose |
|---|---|
| YAML | All hyperparameters, paths, seeds |
| joblib | Model and pipeline serialisation |
| SQLite (via Optuna) | Study storage for tuning resumption |

---

## Reproducing Results

### 1. Clone and install

```bash
git clone https://github.com/henryotsyula/liquidity-stress-early-warning.git
cd liquidity-stress-early-warning
pip install -r requirements.txt
```

### 2. Place data

```bash
# Place competition files in:
data/raw/Train.csv
data/raw/Test.csv
```

### 3. Run baseline training (all 3 GBM models)

```bash
python -m src.orchestration.run_all_models \
  --configs configs/lgbm_v2.yaml configs/xgb_v2.yaml configs/catboost_v2.yaml
```

### 4. Run Optuna tuning

```bash
python -m src.tuning.tune --model all --n-trials 100
```

### 5. Retrain with tuned configs

```bash
python -m src.orchestration.run_all_models \
  --configs configs/lightgbm_tuned.yaml configs/xgboost_tuned.yaml
```

### 6. Run ensemble

```bash
python -m src.ensemble.ensemble
```

### 7. Generate submission

```bash
python -m src.inference.predict \
  --ensemble-run outputs/experiments/v4_ensemble/run_YYYYMMDD_HHMMSS \
  --stage v2_feature_expansion \
  --output submissions/submission_v5.csv
```

### Reproducibility guarantees

All random seeds are set globally at pipeline entry via `seed: 42` in every config file. The feature engineering module is fully stateless and deterministic. The preprocessing pipeline locks the feature contract at `fit()` time and enforces it at `transform()` time with a hard assertion.

---

## Roadmap

| Phase | Description | Status |
|---|---|---|
| Phase 1 | Feature engineering v2.2.1 (volume + entity + ratio guard) | ✅ Complete |
| Phase 2 | Logistic Regression base model (ElasticNet, saga, class_weight=balanced) | 🔴 Next |
| Phase 3 | TabNet base model (pytorch-tabnet, instance-wise attention) | 🔴 Planned |
| Phase 4 | 5-model analysis notebook (correlation matrix, calibration curves) | 🔴 Planned |
| Phase 5 | Ensemble v5 (5-model stacking + raw SHAP features in meta-model) | 🔴 Planned |
| Phase 6 | SHAP interpretability notebook + business narrative | 🔴 Planned |
| Phase 7 | MLflow experiment tracking integration | 🔴 Planned |
| Phase 8 | Final inference pipeline + submission notebook | 🔄 Drafted |
| Phase 9 | FastAPI deployment + Docker containerisation | 🔴 Planned |

**Strategic context:** The diversity check confirmed that all three GBM models correlate at 0.978 — too high for stacking to add value. Phases 2–3 introduce structurally different models (linear boundary + attention-based neural network) to reduce inter-model correlation to the 0.72–0.90 range where stacking becomes the correct ensemble strategy.

---

## What Makes This Project Senior-Level

**Pipeline design:** Every component is modular, config-driven, and reproducible. Adding a new model requires one config file and one line in the orchestrator — no changes to the CV engine, calibration, or ensemble.

**Data integrity:** The feature engineering module has a leakage validation guard, a global ratio explosion guard, a duplicate column safety net, and stateless deterministic computation. Five critical bugs found and documented in v2.1 — v2.2.1 continues that discipline.

**Calibration-first thinking:** The competition weights Log Loss at 60%. Every modeling decision accounts for calibration quality, not just ranking accuracy. Platt calibration is applied at the base model level and again at the ensemble output level.

**Honest evaluation:** All ensemble inputs are OOF predictions — the meta-model never sees predictions made on data it was trained on. Nested CV is used for stacking. The diversity check explicitly measures and documents when adding a new model *won't* help.

**Domain translation:** Feature engineering is grounded in mobile money domain logic. `inflow_volume_recency_ratio` ranking 5th in feature importance validates the hypothesis that frequency decline precedes value decline. The SHAP interpretability notebook will translate findings into plain-English risk narratives.

---

## Author

**Henry Otsyula**
Machine Learning Engineer | Financial Risk Analytics

---

*Built as part of the AI4EAC Liquidity Stress Early Warning Challenge on Zindi Africa.*
*All pipeline code is original and production-ready. Competition data is not included in this repository.*