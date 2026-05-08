"""
Ensemble Pipeline — v5
======================
Project : AI4EAC Liquidity Stress Early Warning (Zindi Africa)
Module  : src/ensemble/ensemble.py
Author  : Henry Otsyula
Updated : 2026-05-08  (v4 → v5: 3-model → 5-model; meta_C tightened)

Architecture
------------
Three-layer ensemble designed around the specific properties of this problem:

  Layer 1 — Base models (already trained, artefacts on disk)
    LightGBM  : strong calibration, best Log Loss baseline
    XGBoost   : best single-model composite (0.19350), most stable folds
    CatBoost  : not Optuna-tuned, but fat-tailed disagreements add value
    LogReg    : ElasticNet saga — LINEAR boundary; structurally orthogonal
                to all three GBMs.  Composite 0.26003 post-Platt.
    TabNet    : instance-wise sequential attention — per-customer feature
                selection.  Composite 0.25296 post-Platt.

  Layer 2 — Calibration (Platt scaling per model, already applied)
    OOF Platt calibrators are fitted in notebooks 07 / 07b and saved to
    outputs/calibration/{model}/calibrator_platt.pkl.
    Calibrated OOF arrays are saved to outputs/multi_model/.
    This module consumes those artefacts — it does NOT re-calibrate base
    model outputs.

  Layer 3 — Ensemble strategies (evaluated in order of complexity)
    3a. Simple average           — equal-weight sanity check
    3b. Optimised weighted avg   — Scipy Nelder-Mead on composite score
    3c. Stacking                 — LogisticRegression meta-model on OOF
    3d. Calibrated stacking      — Platt pass on stacking OOF output

Why v5 changes are needed
--------------------------
v4 was built for three GBMs that shared inductive bias and correlated at
0.962–0.978 — effectively one signal bloc.  Stacking gave minimal lift
over simple averaging in that regime.

v5 adds LogReg and TabNet, which reduce cross-cluster correlation to
0.708–0.753.  Three genuinely independent signal clusters now exist:

    GBM bloc    (LGBM / XGB / CAT)  : within-cluster r = 0.962–0.978
    TabNet                           : vs GBMs r = 0.745–0.753
    LogReg                           : vs GBMs r = 0.748–0.750
                                       vs TabNet r = 0.752

In this regime the meta-model learns CONDITIONAL trust — when TabNet
disagrees with the GBM bloc, whose signal is more likely correct —
producing genuine discriminative lift beyond simple averaging.

Key design decisions
---------------------
1. META_C TIGHTENED: 0.1 → 0.05
   Five meta-features instead of three; stronger regularisation prevents
   the meta-model from over-fitting to the OOF signal.

2. DISAGREEMENT FEATURE (std across ALL 5 models)
   High disagreement cases are exactly where conditional trust matters.
   Using all 5 models' std captures the full signal cluster separation.

3. LEAKAGE PREVENTION — OOF-only stacking
   The meta-model is NEVER trained on in-sample base-model predictions.
   Nested 5-fold CV produces honest OOF meta-predictions for evaluation.
   The final meta-model (for test inference) is fitted on full OOF data.

4. CALIBRATION LAYER ON ENSEMBLE OUTPUT
   LogisticRegression meta-model produces well-scaled probabilities, but
   a final Platt pass corrects residual miscalibration from input scale
   differences between GBM and non-GBM predictions.

5. COMPOSITE SCORE OBJECTIVE throughout
   All weight optimisation and evaluation uses: 0.6 * LogLoss + 0.4 * (1 - AUC)
   This is the exact competition metric — no proxy objectives.

6. SCALE AWARENESS
   LogReg and TabNet OOF arrays have different probability ranges from GBMs
   post-Platt (GBMs centre at 0.15; LogReg/TabNet shift from class weights).
   The meta-model receives all 5 calibrated OOF arrays — each already
   mapped to the correct probability scale by the per-model Platt calibrators
   fitted in notebooks 07 / 07b.  No additional scaling is applied here.

7. MODEL NAME → FILE NAME MAPPING
   lightgbm  →  oof_calibrated_lightgbm.npy  /  calibration/lgb/
   xgboost   →  oof_calibrated_xgboost.npy   /  calibration/xgb/
   catboost  →  oof_calibrated_catboost.npy  /  calibration/cat/
   logreg    →  oof_calibrated_logreg.npy    /  calibration/logreg/
   tabnet    →  oof_calibrated_tabnet.npy    /  calibration/tabnet/
   NOTE: GBMs use abbreviated subfolder names (lgb/xgb/cat) — preserved
   from v4 to avoid breaking existing artefact paths.

Output contract
---------------
outputs/experiments/v5_ensemble/run_YYYYMMDD_HHMMSS/
    ensemble_oof.npy           final calibrated stacking OOF predictions
    stacking_oof.npy           raw stacking output (pre-final-calibration)
    ensemble_weights.json      optimised weights per strategy
    meta_model.pkl             fitted stacking LogisticRegression
    meta_calibrator.pkl        Platt calibrator fitted on ensemble OOF
    stacking_results.json      full metrics for all strategies
    feature_importance.csv     meta-model coefficient table
    metadata.json              run record with all hyperparameters
    correlation_matrix.csv     post-calibration OOF Pearson correlation
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold


# =============================================================================
# CONSTANTS
# =============================================================================

EPS             : float      = 1e-15
CLIP_LOW        : float      = 1e-6
CLIP_HIGH       : float      = 1.0 - 1e-6
LOG_LOSS_WEIGHT : float      = 0.6
AUC_WEIGHT      : float      = 0.4
SEED            : int        = 42

# v5: Five models spanning three independent signal clusters.
# ORDER MATTERS — must match the order used in _build_meta_features().
MODEL_NAMES: List[str] = [
    "lightgbm",   # GBM cluster
    "xgboost",    # GBM cluster
    "catboost",   # GBM cluster
    "logreg",     # Linear cluster
    "tabnet",     # Attention cluster
]

# GBM models use abbreviated folder names for calibration artefact paths.
# Non-GBMs use their full model name as the folder name.
_CALIBRATION_FOLDER: Dict[str, str] = {
    "lightgbm" : "lgb",
    "xgboost"  : "xgb",
    "catboost" : "cat",
    "logreg"   : "logreg",
    "tabnet"   : "tabnet",
}

# Reference composite scores from individual model evaluation (post-Platt).
# Used in the results table for easy comparison.
_KNOWN_INDIVIDUAL_SCORES: Dict[str, float] = {
    "lightgbm" : 0.19557,
    "xgboost"  : 0.19350,
    "catboost" : 0.19430,
    "logreg"   : 0.26003,
    "tabnet"   : 0.25296,
}


# =============================================================================
# METRICS
# =============================================================================

def composite_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Competition metric: 0.6 * LogLoss + 0.4 * (1 - AUC).
    Lower is better.  Clips predictions before LogLoss computation.
    """
    y_clipped = np.clip(y_pred, EPS, 1.0 - EPS)
    ll  = log_loss(y_true, y_clipped)
    auc = roc_auc_score(y_true, y_pred)
    return LOG_LOSS_WEIGHT * ll + AUC_WEIGHT * (1.0 - auc)


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Full metric suite for a set of predictions.
    Returns logloss, auc, brier score, composite score,
    and prediction distribution statistics.
    """
    y_clipped = np.clip(y_pred, EPS, 1.0 - EPS)
    ll        = float(log_loss(y_true, y_clipped))
    auc       = float(roc_auc_score(y_true, y_pred))
    return {
        "name"      : name,
        "logloss"   : ll,
        "auc"       : auc,
        "brier"     : float(brier_score_loss(y_true, y_pred)),
        "score"     : float(LOG_LOSS_WEIGHT * ll + AUC_WEIGHT * (1.0 - auc)),
        "mean_pred" : float(y_pred.mean()),
        "std_pred"  : float(y_pred.std()),
        "min_pred"  : float(y_pred.min()),
        "max_pred"  : float(y_pred.max()),
    }


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class EnsembleConfig:
    """
    Full configuration for the ensemble pipeline.
    All hyperparameters in one place — no magic numbers in code.

    v5 changes vs v4
    ----------------
    meta_C          : 0.1 → 0.05   (tighter regularisation for 5 meta-inputs)
    output_dir      : v4_ensemble → v5_ensemble
    use_raw_features: now properly wired (activated post-SHAP in Step 3)
    """
    # Paths
    project_root    : str   = ""
    oof_dir         : str   = "outputs/multi_model"
    output_dir      : str   = "outputs/experiments/v5_ensemble"
    calibration_dir : str   = "outputs/calibration"

    # Cross-validation
    n_splits        : int   = 5
    seed            : int   = SEED

    # Weight optimisation
    optimise_weights  : bool  = True
    optimiser_method  : str   = "Nelder-Mead"
    optimiser_maxiter : int   = 5000

    # Stacking meta-model
    # v5: C tightened from 0.1 → 0.05 because we now have 5 meta-features
    # (vs 3 in v4).  With 5 inputs, a looser C risks the meta-model learning
    # spurious correlations in the 40 k OOF sample.
    meta_C          : float = 0.05
    meta_max_iter   : int   = 1000
    meta_solver     : str   = "lbfgs"

    # Stacking feature augmentation
    # use_disagreement: std across ALL 5 models' predictions per row.
    # This captures cross-cluster uncertainty — exactly where stacking
    # adds value over averaging.
    use_disagreement    : bool  = True
    # use_raw_features: activated in Step 3 after SHAP identifies top-5.
    use_raw_features    : bool  = False
    top_k_raw_features  : int   = 5

    # Final calibration of ensemble output
    calibrate_ensemble  : bool  = True

    # Artifact saving
    save_models : bool  = True
    save_oof    : bool  = True
    save_metrics: bool  = True


@dataclass
class EnsembleResults:
    """Collects all results across ensemble strategies for comparison."""
    strategy_metrics  : List[Dict]  = field(default_factory=list)
    best_strategy     : str         = ""
    best_score        : float       = 999.0
    optimised_weights : Dict        = field(default_factory=dict)
    stacking_oof      : np.ndarray  = field(default_factory=lambda: np.array([]))
    ensemble_oof      : np.ndarray  = field(default_factory=lambda: np.array([]))
    run_dir           : str         = ""
    runtime_sec       : float       = 0.0


# =============================================================================
# CALIBRATION UTILITIES
# =============================================================================

class PlattCalibrator:
    """
    Platt scaling calibrator (logistic regression on raw predictions).

    Wraps sklearn LogisticRegression(C=1e10) — effectively unconstrained —
    fitting the sigmoid: calibrated = sigmoid(a * raw + b).

    This class exists to keep the interface consistent with the per-model
    calibrators saved in notebooks 07 / 07b.  The ensemble module uses it
    only for the FINAL calibration of the stacking output, not for
    re-calibrating base model predictions (those are already calibrated).
    """

    def __init__(self, C: float = 1e10, max_iter: int = 1000) -> None:
        self.C        = C
        self.max_iter = max_iter
        self._model   = LogisticRegression(
            C=C, solver="lbfgs", max_iter=max_iter
        )
        self._fitted  = False

    # Slope and intercept expose the sigmoid parameters for inspection.
    @property
    def slope(self) -> float:
        return float(self._model.coef_[0][0]) if self._fitted else float("nan")

    @property
    def intercept(self) -> float:
        return float(self._model.intercept_[0]) if self._fitted else float("nan")

    def fit(self, raw_preds: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        self._model.fit(raw_preds.reshape(-1, 1), y)
        self._fitted = True
        return self

    def predict(self, raw_preds: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("PlattCalibrator must be fitted before predict().")
        return self._model.predict_proba(raw_preds.reshape(-1, 1))[:, 1]

    def fit_predict(self, raw_preds: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(raw_preds, y).predict(raw_preds)

    def summary(self) -> Dict[str, float]:
        return {"slope": self.slope, "intercept": self.intercept}


def cv_platt_calibrate(
    raw_preds : np.ndarray,
    y         : np.ndarray,
    n_splits  : int  = 5,
    seed      : int  = SEED,
    C         : float = 1e10,
) -> Tuple[np.ndarray, "PlattCalibrator"]:
    """
    Cross-validated Platt calibration.

    Fits the calibrator on held-out splits to prevent in-sample optimism.
    Returns both calibrated OOF predictions AND a final calibrator fitted
    on all data (for applying to the test set).

    Returns
    -------
    calibrated_oof   : np.ndarray — CV-calibrated predictions (honest)
    final_calibrator : PlattCalibrator — fitted on full data, for test set
    """
    calibrated = np.zeros_like(raw_preds, dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for train_idx, val_idx in skf.split(raw_preds, y):
        cal = PlattCalibrator(C=C)
        cal.fit(raw_preds[train_idx], y[train_idx])
        calibrated[val_idx] = cal.predict(raw_preds[val_idx])

    final_calibrator = PlattCalibrator(C=C).fit(raw_preds, y)

    return calibrated, final_calibrator


# =============================================================================
# ARTIFACT LOADING
# =============================================================================

def load_oof_artifacts(config: EnsembleConfig) -> Dict[str, np.ndarray]:
    """
    Load Platt-calibrated OOF predictions and ground truth.

    Expects (all produced by notebooks 07 / 07b / 08):
        outputs/multi_model/y_true.npy
        outputs/multi_model/oof_calibrated_{model}.npy   for all 5 models

    Validates:
        - All arrays have the same length as y_true
        - No NaN or Inf values in any array
        - Positive rate matches expected 15% (±1%)
    """
    root    = Path(config.project_root)
    oof_dir = root / config.oof_dir

    artifacts: Dict[str, np.ndarray] = {}

    # Ground truth
    y_path = oof_dir / "y_true.npy"
    if not y_path.exists():
        raise FileNotFoundError(
            f"y_true.npy not found at {y_path}.\n"
            "Run notebooks 07 / 07b / 08 to produce OOF artefacts first."
        )
    artifacts["y_true"] = np.load(y_path)
    n = len(artifacts["y_true"])
    pos_rate = artifacts["y_true"].mean()
    print(f"  y_true       : shape={artifacts['y_true'].shape}  "
          f"positive_rate={pos_rate:.4f}")
    if not 0.13 <= pos_rate <= 0.17:
        raise ValueError(
            f"Unexpected positive rate {pos_rate:.4f}. "
            "Expected ~0.15 (6,000 / 40,000). Check y_true.npy."
        )

    # Per-model calibrated OOF
    for m in MODEL_NAMES:
        path = oof_dir / f"oof_calibrated_{m}.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"Calibrated OOF not found for {m}: {path}.\n"
                f"  • GBMs: run notebook 07_calibration_analysis.ipynb\n"
                f"  • LogReg/TabNet: run notebook 07b_logreg_tabnet_calibration.ipynb"
            )
        arr = np.load(path)

        # Integrity checks — explicit ValueError, not assert (assert is
        # disabled under python -O and must never guard production paths).
        if len(arr) != n:
            raise ValueError(
                f"Shape mismatch: y_true has {n} rows, {m} OOF has {len(arr)}. "
                f"Re-run the calibration notebook to regenerate this artefact."
            )
        if np.isnan(arr).any():
            raise ValueError(f"NaN values detected in {m} OOF predictions.")
        if np.isinf(arr).any():
            raise ValueError(f"Inf values detected in {m} OOF predictions.")
        if not (0.0 <= arr.min() and arr.max() <= 1.0):
            raise ValueError(
                f"{m} OOF predictions out of [0, 1]: "
                f"min={arr.min():.6f}, max={arr.max():.6f}"
            )

        artifacts[m] = arr
        known = _KNOWN_INDIVIDUAL_SCORES.get(m)
        known_str = f"  (ref composite={known:.5f})" if known else ""
        print(f"  {m:12s}: shape={arr.shape}  "
              f"range=[{arr.min():.4f}, {arr.max():.4f}]  "
              f"mean={arr.mean():.4f}{known_str}")

    print(f"\n  All 5 models loaded and validated ✓")
    return artifacts


def compute_oof_correlations(
    oof_preds: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Compute Pearson correlation matrix across all 5 models' OOF predictions.
    Used both for reporting and to verify the three-cluster structure.
    """
    df = pd.DataFrame({m: oof_preds[m] for m in MODEL_NAMES})
    return df.corr(method="pearson")


def build_run_dir(config: EnsembleConfig) -> Path:
    """Create timestamped run directory aligned to output spec."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        Path(config.project_root)
        / config.output_dir
        / f"run_{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# =============================================================================
# ENSEMBLE STRATEGY 1 — SIMPLE AVERAGE
# =============================================================================

def simple_average(
    oof_preds : Dict[str, np.ndarray],
    y_true    : np.ndarray,
) -> Tuple[np.ndarray, Dict]:
    """
    Equal-weight average across all 5 base model predictions.

    With GBM intra-cluster correlations of 0.962–0.978, this effectively
    gives ~75% weight to the GBM signal bloc.  Serves as the lower-bound
    baseline — stacking should clearly beat this with r=0.745–0.753
    cross-cluster diversity.
    """
    active_models = [m for m in MODEL_NAMES if m in oof_preds]
    pred_arr = np.stack([oof_preds[m] for m in active_models], axis=1)
    avg_pred = pred_arr.mean(axis=1)
    avg_pred = np.clip(avg_pred, CLIP_LOW, CLIP_HIGH)
    metrics  = evaluate("simple_average", y_true, avg_pred)
    return avg_pred, metrics


# =============================================================================
# ENSEMBLE STRATEGY 2 — OPTIMISED WEIGHTED AVERAGE
# =============================================================================

def _weight_objective(
    weights  : np.ndarray,
    pred_arr : np.ndarray,
    y_true   : np.ndarray,
) -> float:
    """
    Composite score objective for Nelder-Mead weight optimisation.
    Projects weights onto the probability simplex (sum=1, all ≥ 0).
    """
    w = np.clip(weights, 0.0, 1.0)

    if w.shape[0] != pred_arr.shape[1]:
        raise ValueError(
            f"Weight mismatch: got {w.shape[0]} weights for "
            f"{pred_arr.shape[1]} models"
        )

    w_sum = w.sum()
    if w_sum < 1e-10:
        return 999.0
    w       = w / w_sum
    blended = (pred_arr * w).sum(axis=1)
    blended = np.clip(blended, EPS, 1.0 - EPS)
    return composite_score(y_true, blended)


def optimised_weighted_average(
    oof_preds : Dict[str, np.ndarray],
    y_true    : np.ndarray,
    config    : EnsembleConfig,
) -> Tuple[np.ndarray, Dict, np.ndarray]:
    """
    Scipy Nelder-Mead optimisation of per-model ensemble weights.

    Minimises the exact competition composite score on OOF predictions.
    Uses 16+ initialisations (deterministic + random simplex points) to
    escape local minima.  Weights are projected onto the simplex after
    optimisation.

    WARNING: These weights are OOF-optimised — the reported score is
    slightly optimistic.  The stacking meta-model (Strategy 3) is the
    deployment-safe alternative with honest nested-CV evaluation.
    """
    model_list = [m for m in MODEL_NAMES if m in oof_preds]

    missing_models = [m for m in MODEL_NAMES if m not in oof_preds]
    if missing_models:
        raise ValueError(
            f"Missing OOF predictions for models: {missing_models}"
        )

    pred_arr = np.stack([oof_preds[m] for m in model_list], axis=1)
    n_models = pred_arr.shape[1]

    print("pred_arr shape:", pred_arr.shape)
    print("n_models:", n_models)
    print("model_list:", model_list)

    best_score   = 999.0
    best_weights = np.full(n_models, 1.0 / n_models)

    # Deterministic initialisations — encode domain knowledge
    initialisations = [
        np.full(n_models, 1.0 / n_models),                    # equal
        np.array([0.30, 0.35, 0.20, 0.10, 0.05]),             # XGB-heavy GBM
        np.array([0.35, 0.30, 0.20, 0.10, 0.05]),             # LGB-heavy GBM
        np.array([0.25, 0.25, 0.20, 0.15, 0.15]),             # GBM-balanced
        np.array([0.25, 0.30, 0.15, 0.15, 0.15]),             # diversified
        np.array([0.20, 0.25, 0.15, 0.20, 0.20]),             # cluster-balanced
        np.array([0.33, 0.33, 0.17, 0.10, 0.07]),             # top-2 GBM + others
        np.array([0.28, 0.32, 0.15, 0.13, 0.12]),             # XGB-dominant
        np.array([0.50, 0.30, 0.10, 0.05, 0.05]),             # LGB dominant
        np.array([0.30, 0.40, 0.10, 0.10, 0.10]),             # XGB dominant
    ]

    # Random Dirichlet initialisations for broad exploration
    rng = np.random.default_rng(config.seed)
    for _ in range(12):
        initialisations.append(rng.dirichlet(np.ones(n_models)))

    print(f"  Running weight optimisation ({len(initialisations)} initialisations)...")

    for i, w0 in enumerate(initialisations):
        result = minimize(
            _weight_objective,
            x0      = w0,
            args    = (pred_arr, y_true),
            method  = config.optimiser_method,
            options = {
                "maxiter" : config.optimiser_maxiter,
                "xatol"   : 1e-8,
                "fatol"   : 1e-8,
            },
        )
        if result.fun < best_score:
            best_score   = result.fun
            best_weights = result.x

    # Project onto probability simplex
    best_weights = np.clip(best_weights, 0.0, 1.0)
    best_weights = best_weights / best_weights.sum()

    best_pred = (pred_arr * best_weights).sum(axis=1)
    best_pred = np.clip(best_pred, CLIP_LOW, CLIP_HIGH)
    metrics   = evaluate("optimised_weighted_average", y_true, best_pred)

    weight_dict = {m: float(w) for m, w in zip(model_list, best_weights)}
    print(f"  Optimised weights : {weight_dict}")
    print(f"  Best score        : {best_score:.6f}")

    return best_pred, metrics, best_weights


# =============================================================================
# ENSEMBLE STRATEGY 3 — STACKING (PRIMARY STRATEGY)
# =============================================================================

def _build_meta_features(
    oof_preds      : Dict[str, np.ndarray],
    config         : EnsembleConfig,
    extra_features : Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build meta-feature matrix and corresponding feature names.

    Base features (always included, 5 columns):
        Platt-calibrated OOF from each of the 5 base models.
        Order: lightgbm, xgboost, catboost, logreg, tabnet.

    Optional — inter-model disagreement (1 column):
        Standard deviation across ALL 5 models' predictions per row.
        High-disagreement rows are exactly where the meta-model adds value:
        the GBM bloc and LogReg/TabNet clusters disagree, and the meta-model
        has learned which cluster to trust conditionally.

    Optional — raw features from SHAP analysis (k columns):
        Top-k raw features identified in notebook 10_shap_interpretability.
        Activated post-SHAP by setting use_raw_features=True in config.
        extra_features array must be shaped (n, k).

    Returns
    -------
    meta_X         : np.ndarray of shape (n, n_meta_features)
    feature_names  : List[str] for coefficient reporting
    """
    meta_X         = np.stack([oof_preds[m] for m in MODEL_NAMES], axis=1)
    feature_names  = list(MODEL_NAMES)

    if config.use_disagreement:
        # Std across all 5 models per customer — disagreement signal
        disagreement  = meta_X.std(axis=1, keepdims=True)
        meta_X        = np.hstack([meta_X, disagreement])
        feature_names.append("inter_model_disagreement")

    if config.use_raw_features and extra_features is not None:
        if extra_features.shape[0] != meta_X.shape[0]:
            raise ValueError(
                f"extra_features row count ({extra_features.shape[0]}) "
                f"does not match OOF row count ({meta_X.shape[0]}). "
                "Ensure extra_features is aligned to the training set."
            )
        meta_X = np.hstack([meta_X, extra_features])
        for i in range(extra_features.shape[1]):
            feature_names.append(f"shap_raw_feature_{i}")

    print(f"  Meta-feature matrix : shape={meta_X.shape}  "
          f"features={feature_names}")

    return meta_X, feature_names


def stacking_ensemble(
    oof_preds      : Dict[str, np.ndarray],
    y_true         : np.ndarray,
    config         : EnsembleConfig,
    extra_features : Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict, LogisticRegression, pd.DataFrame]:
    """
    Stacking ensemble with LogisticRegression(C=0.05) meta-model.

    Nested 5-fold CV protocol:
    ─────────────────────────
    For each outer fold (val_idx):
      - Meta-model trained on train_idx OOF predictions
      - Meta-model predicts val_idx → honest OOF meta-prediction
    After CV: final meta-model fitted on ALL OOF data (for test inference).

    This is a strict OOF stacking implementation — the meta-model is NEVER
    evaluated on the same rows it was trained on.  The CV composite score
    is the honest estimate of deployment performance.

    Why LogisticRegression (not GBM) as meta-model?
    ────────────────────────────────────────────────
    With 5–7 meta-features and 40,000 samples, a tree-based meta-model
    would memorise rather than generalise.  LogisticRegression with
    C=0.05 regularisation produces stable, interpretable weights and
    respects the probability output contract (values in [0, 1]).

    Parameters
    ----------
    oof_preds      : calibrated OOF predictions for all 5 models
    y_true         : ground truth labels
    config         : EnsembleConfig (meta_C=0.05 for v5)
    extra_features : optional SHAP raw features (n, k) — post-SHAP step

    Returns
    -------
    stacking_oof     : np.ndarray — CV stacking OOF predictions
    metrics          : dict
    final_meta_model : LogisticRegression fitted on full OOF data
    coefficients     : pd.DataFrame — coefficient table for interpretability
    """
    meta_X, feature_names = _build_meta_features(oof_preds, config, extra_features)
    n_meta_features       = meta_X.shape[1]

    stacking_oof = np.zeros(len(y_true), dtype=float)
    skf          = StratifiedKFold(
        n_splits=config.n_splits, shuffle=True, random_state=config.seed
    )

    fold_scores  = []
    fold_lls     = []
    fold_aucs    = []

    print(f"  Nested {config.n_splits}-fold CV for honest OOF stacking...")
    print(f"  {'Fold':>4}  {'Score':>8}  {'LogLoss':>8}  {'AUC':>8}")
    print(f"  {'-'*40}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(meta_X, y_true)):
        X_tr, y_tr = meta_X[train_idx], y_true[train_idx]
        X_val, y_val = meta_X[val_idx], y_true[val_idx]

        meta_model = LogisticRegression(
            C            = config.meta_C,
            solver       = config.meta_solver,
            max_iter     = config.meta_max_iter,
            random_state = config.seed,
        )
        meta_model.fit(X_tr, y_tr)

        fold_pred             = meta_model.predict_proba(X_val)[:, 1]
        stacking_oof[val_idx] = fold_pred

        fs  = composite_score(y_val, fold_pred)
        fll = float(log_loss(y_val, np.clip(fold_pred, EPS, 1.0 - EPS)))
        fau = float(roc_auc_score(y_val, fold_pred))
        fold_scores.append(fs)
        fold_lls.append(fll)
        fold_aucs.append(fau)

        print(f"  {fold:>4}  {fs:>8.5f}  {fll:>8.5f}  {fau:>8.5f}")

    mean_score = float(np.mean(fold_scores))
    std_score  = float(np.std(fold_scores))
    mean_ll    = float(np.mean(fold_lls))
    mean_auc   = float(np.mean(fold_aucs))

    print(f"  {'-'*40}")
    print(f"  {'MEAN':>4}  {mean_score:>8.5f}  {mean_ll:>8.5f}  {mean_auc:>8.5f}")
    print(f"  {'STD':>4}  {std_score:>8.5f}  {float(np.std(fold_lls)):>8.5f}  "
          f"{float(np.std(fold_aucs)):>8.5f}")

    # Final meta-model fitted on ALL OOF data — used for test set inference
    final_meta_model = LogisticRegression(
        C            = config.meta_C,
        solver       = config.meta_solver,
        max_iter     = config.meta_max_iter,
        random_state = config.seed,
    )
    final_meta_model.fit(meta_X, y_true)

    # Coefficient table — meta-model interpretability
    coefficients = pd.DataFrame({
        "feature"     : feature_names[:n_meta_features],
        "coefficient" : final_meta_model.coef_[0][:n_meta_features],
    }).sort_values("coefficient", ascending=False).reset_index(drop=True)

    print(f"\n  Meta-model coefficients (C={config.meta_C}):")
    print(coefficients.to_string(index=False))

    # Interpretation guidance: high coefficient = meta-model trusts this model more
    best_meta_feature = coefficients.iloc[0]["feature"]
    print(f"\n  → Meta-model trusts '{best_meta_feature}' most strongly.")
    print(f"  → Compare to individual composite scores for context.")

    metrics = evaluate("stacking", y_true, stacking_oof)

    return stacking_oof, metrics, final_meta_model, coefficients


# =============================================================================
# ENSEMBLE STRATEGY 4 — STACKING WITH FINAL CALIBRATION
# =============================================================================

def calibrated_stacking(
    stacking_oof : np.ndarray,
    y_true       : np.ndarray,
    config       : EnsembleConfig,
) -> Tuple[np.ndarray, Dict, "PlattCalibrator"]:
    """
    Apply cross-validated Platt calibration to the stacking OOF predictions.

    Although LogisticRegression is calibrated by design, the input feature
    scale (mixing GBM predictions near 0.15 mean with LogReg/TabNet at
    higher post-Platt means) can introduce mild residual miscalibration.
    A final Platt pass corrects this at a cost of ~2 parameters.

    Uses nested CV Platt (5 folds) for an honest calibrated OOF estimate.
    The final_calibrator (fitted on all data) is saved for test inference.

    Known limitation — fold misalignment:
        The Platt CV folds here are NOT aligned to the stacking CV folds
        that produced stacking_oof.  Each calibration fold's held-out rows
        were predicted by meta-models trained on overlapping (but not
        identical) samples from the stacking phase.  The resulting optimism
        is negligible in practice (the Platt calibrator has only 2 free
        parameters across 40k rows), but treat calibrated_stacking scores
        as a lower-bound estimate rather than a perfectly honest OOF score.
        The rigorous fix — calibrating within each stacking outer fold —
        requires restructuring stacking_ensemble() to expose per-fold
        predictions before aggregation.  Deferred post-competition.

    Returns
    -------
    calibrated_oof   : np.ndarray — CV-calibrated stacking OOF predictions
    metrics          : dict
    final_calibrator : PlattCalibrator — apply to test set stacking output
    """
    calibrated_oof, final_calibrator = cv_platt_calibrate(
        stacking_oof, y_true,
        n_splits = config.n_splits,
        seed     = config.seed,
        C        = 1e10,                    # unconstrained — just fit the sigmoid
    )
    calibrated_oof = np.clip(calibrated_oof, CLIP_LOW, CLIP_HIGH)
    metrics        = evaluate("stacking_calibrated", y_true, calibrated_oof)

    print(f"  Final calibrator: slope={final_calibrator.slope:.4f}  "
          f"intercept={final_calibrator.intercept:.4f}")
    print(f"  Slope ≈ 1.0 → stacking output already well-calibrated ✓")
    print(f"  Slope >> 1.0 → residual compression was present (expected if large)")

    return calibrated_oof, metrics, final_calibrator


# =============================================================================
# REPORTING UTILITIES
# =============================================================================

def print_correlation_report(corr_matrix: pd.DataFrame) -> None:
    """Print the OOF Pearson correlation matrix with cluster analysis."""
    print("\n  OOF Pearson Correlation Matrix (post-Platt):")
    print(corr_matrix.round(3).to_string())

    # Extract cross-cluster correlations for interpretation
    gbm_models    = ["lightgbm", "xgboost", "catboost"]
    non_gbm       = ["logreg", "tabnet"]

    gbm_intra     = [corr_matrix.loc[a, b]
                     for i, a in enumerate(gbm_models)
                     for j, b in enumerate(gbm_models) if i < j]
    cross_cluster = [corr_matrix.loc[a, b]
                     for a in gbm_models for b in non_gbm]
    non_gbm_cross = [corr_matrix.loc["logreg", "tabnet"]]

    print(f"\n  Cluster structure:")
    print(f"  GBM intra-cluster         : {min(gbm_intra):.3f} – {max(gbm_intra):.3f}")
    print(f"  GBM vs LogReg/TabNet      : {min(cross_cluster):.3f} – {max(cross_cluster):.3f}")
    print(f"  LogReg vs TabNet          : {non_gbm_cross[0]:.3f}")

    gbm_cross_max = max(cross_cluster)
    if gbm_cross_max < 0.82:
        print(f"\n  ✓ Cross-cluster diversity target MET (<0.82). Stacking justified.")
    else:
        print(f"\n  ⚠ Cross-cluster correlation {gbm_cross_max:.3f} > 0.82 target.")
        print(f"    Stacking may not outperform optimised weighting.")


def print_results_table(results: EnsembleResults) -> None:
    """Print a formatted comparison table of all ensemble strategies."""
    sep = "=" * 90
    print(f"\n{sep}")
    print("ENSEMBLE v5 — RESULTS SUMMARY")
    print(sep)
    print(f"{'Strategy':40s}  {'LogLoss':>8}  {'AUC':>8}  "
          f"{'Score':>8}  {'Brier':>8}")
    print("-" * 90)

    sorted_metrics = sorted(results.strategy_metrics, key=lambda x: x["score"])

    for m in sorted_metrics:
        marker = "  ← BEST" if m["name"] == results.best_strategy else ""
        # Add improvement vs best single model for ensemble strategies
        ref = _KNOWN_INDIVIDUAL_SCORES.get("xgboost", 0.19350)
        improvement = ""
        if m["name"] not in _KNOWN_INDIVIDUAL_SCORES:
            delta = m["score"] - ref
            sign  = "+" if delta > 0 else ""
            improvement = f"  ({sign}{delta:.5f} vs XGB baseline)"

        print(f"{m['name']:40s}  {m['logloss']:>8.5f}  {m['auc']:>8.5f}  "
              f"{m['score']:>8.5f}  {m['brier']:>8.5f}"
              f"{marker}{improvement}")

    print(sep)
    print(f"\nBest strategy : {results.best_strategy}")
    print(f"Best score    : {results.best_score:.6f}")
    print(f"XGB baseline  : {_KNOWN_INDIVIDUAL_SCORES['xgboost']:.6f}")
    delta = results.best_score - _KNOWN_INDIVIDUAL_SCORES["xgboost"]
    sign  = "+" if delta >= 0 else ""
    print(f"Improvement   : {sign}{delta:.6f}")
    print(f"Runtime       : {results.runtime_sec:.1f} seconds")


# =============================================================================
# ARTIFACT SAVING
# =============================================================================

def save_ensemble_artifacts(
    results         : EnsembleResults,
    meta_model      : LogisticRegression,
    meta_calibrator : Optional["PlattCalibrator"],
    coefficients    : pd.DataFrame,
    corr_matrix     : pd.DataFrame,
    config          : EnsembleConfig,
    run_dir         : Path,
) -> None:
    """
    Save all ensemble artefacts to the run directory.

    Output layout (v5):
    outputs/experiments/v5_ensemble/run_YYYYMMDD_HHMMSS/
        ensemble_oof.npy           final calibrated stacking OOF
        stacking_oof.npy           raw stacking output (pre-calibration)
        ensemble_weights.json      optimised weights per strategy
        meta_model.pkl             fitted stacking LogisticRegression
        meta_calibrator.pkl        Platt calibrator on ensemble OOF
        stacking_results.json      full metrics for all strategies
        feature_importance.csv     meta-model coefficient table
        correlation_matrix.csv     OOF Pearson correlation matrix
        metadata.json              run record with all hyperparameters
    """
    # OOF predictions
    if results.ensemble_oof is not None and len(results.ensemble_oof) > 0:
        np.save(run_dir / "ensemble_oof.npy", results.ensemble_oof)
    if results.stacking_oof is not None and len(results.stacking_oof) > 0:
        np.save(run_dir / "stacking_oof.npy", results.stacking_oof)

    # Ensemble weights
    weights_data = {
        "optimised"  : results.optimised_weights,
        "model_order": MODEL_NAMES,
        "note"       : (
            "Weights are OOF-optimised and slightly optimistic. "
            "Use stacking (meta_model.pkl) for deployment."
        ),
    }
    with open(run_dir / "ensemble_weights.json", "w") as f:
        json.dump(weights_data, f, indent=4)

    # Meta-model and calibrator
    joblib.dump(meta_model, run_dir / "meta_model.pkl")
    if meta_calibrator is not None:
        joblib.dump(meta_calibrator, run_dir / "meta_calibrator.pkl")

    # Strategy results
    with open(run_dir / "stacking_results.json", "w") as f:
        json.dump(results.strategy_metrics, f, indent=4)

    # Meta-model coefficients
    coefficients.to_csv(run_dir / "feature_importance.csv", index=False)

    # OOF correlation matrix
    corr_matrix.to_csv(run_dir / "correlation_matrix.csv")

    # Metadata
    best_single_score = _KNOWN_INDIVIDUAL_SCORES.get("xgboost", None)
    metadata = {
        "created_at"       : datetime.now().isoformat(),
        "version"          : "v5",
        "model_names"      : MODEL_NAMES,
        "best_strategy"    : results.best_strategy,
        "best_score"       : results.best_score,
        "xgb_baseline"     : best_single_score,
        "improvement_vs_xgb": (
            round(results.best_score - best_single_score, 6)
            if best_single_score is not None else None
        ),
        "runtime_sec"      : results.runtime_sec,
        "config"           : asdict(config),
        "strategy_summary" : [
            {k: v for k, v in m.items() if k != "name"}
            | {"strategy": m["name"]}
            for m in results.strategy_metrics
        ],
        "meta_model_summary": {
            "C"                 : config.meta_C,
            "solver"            : config.meta_solver,
            "use_disagreement"  : config.use_disagreement,
            "use_raw_features"  : config.use_raw_features,
            # Explicit feature order so any future reload is self-describing.
            # The meta-model's coef_[0] indices map to this list in order.
            "meta_feature_order": (
                list(MODEL_NAMES)
                + (["inter_model_disagreement"] if config.use_disagreement else [])
                + ([f"shap_raw_feature_{i}" for i in range(config.top_k_raw_features)]
                   if config.use_raw_features else [])
            ),
        },
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4, default=str)

    print(f"\n  Artefacts saved to: {run_dir}")
    for fname in sorted(run_dir.iterdir()):
        print(f"    {fname.name}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_ensemble_pipeline(
    config         : EnsembleConfig,
    extra_features : Optional[np.ndarray] = None,
) -> EnsembleResults:
    """
    Full 5-model ensemble pipeline — v5.

    Executes all four ensemble strategies in order:
      1. Simple average (equal-weight baseline)
      2. Optimised weighted average (Scipy Nelder-Mead)
      3. Stacking (LogisticRegression meta-model, C=0.05)
      4. Calibrated stacking (Platt pass on stacking OOF output)

    Parameters
    ----------
    config         : EnsembleConfig
    extra_features : optional np.ndarray of shape (n, k)
                     Top-k raw features from SHAP analysis.
                     Only used when config.use_raw_features=True.
                     If None and use_raw_features=True, raises ValueError.

    Returns
    -------
    EnsembleResults with all metrics, predictions, and artefacts saved.
    """
    start_time = time.perf_counter()
    results    = EnsembleResults()

    if config.use_raw_features and extra_features is None:
        raise ValueError(
            "config.use_raw_features=True but extra_features is None.\n"
            "Run SHAP analysis (notebook 10) first and pass the top-k "
            "feature matrix here."
        )

    print("=" * 70)
    print("ENSEMBLE PIPELINE — v5 (5-Model Stacking)")
    print("=" * 70)
    print(f"Models     : {MODEL_NAMES}")
    print(f"meta_C     : {config.meta_C}")
    print(f"Disagreement feature: {config.use_disagreement}")
    print(f"Raw features (SHAP) : {config.use_raw_features}")

    # ── 0. Setup ──────────────────────────────────────────────────────────────
    run_dir         = build_run_dir(config)
    results.run_dir = str(run_dir)
    print(f"\nRun directory: {run_dir}")

    # ── 1. Load and validate OOF artefacts ───────────────────────────────────
    print("\n[1/5] Loading OOF artefacts...")
    oof_artifacts = load_oof_artifacts(config)
    y_true    = oof_artifacts["y_true"]
    oof_preds = {m: oof_artifacts[m] for m in MODEL_NAMES}

    # Correlation matrix — confirm three-cluster structure
    print("\n  Computing OOF correlation matrix...")
    corr_matrix = compute_oof_correlations(oof_preds)
    print_correlation_report(corr_matrix)

    # Individual model baselines
    print("\n  Individual model metrics (Platt-calibrated OOF):")
    print(f"  {'Model':12s}  {'LogLoss':>8}  {'AUC':>8}  {'Score':>8}")
    print(f"  {'-'*46}")
    for m in MODEL_NAMES:
        m_metrics = evaluate(m, y_true, oof_preds[m])
        results.strategy_metrics.append(m_metrics)
        print(f"  {m:12s}  {m_metrics['logloss']:>8.5f}  "
              f"{m_metrics['auc']:>8.5f}  {m_metrics['score']:>8.5f}")

    # ── 2. Simple average ─────────────────────────────────────────────────────
    print("\n[2/5] Strategy 1 — Simple average (equal-weight baseline)...")
    avg_pred, avg_metrics = simple_average(oof_preds, y_true)
    results.strategy_metrics.append(avg_metrics)
    print(f"  LL={avg_metrics['logloss']:.5f}  "
          f"AUC={avg_metrics['auc']:.5f}  "
          f"Score={avg_metrics['score']:.5f}")

    # ── 3. Optimised weighted average ─────────────────────────────────────────
    print("\n[3/5] Strategy 2 — Optimised weighted average...")
    opt_pred, opt_metrics, opt_weights = optimised_weighted_average(
        oof_preds, y_true, config
    )
    results.strategy_metrics.append(opt_metrics)
    results.optimised_weights = {
        m: float(w) for m, w in zip(MODEL_NAMES, opt_weights)
    }
    print(f"  LL={opt_metrics['logloss']:.5f}  "
          f"AUC={opt_metrics['auc']:.5f}  "
          f"Score={opt_metrics['score']:.5f}")

    # ── 4. Stacking ───────────────────────────────────────────────────────────
    print("\n[4/5] Strategy 3 — Stacking meta-model (LogReg, C=0.05)...")
    stacking_oof, stacking_metrics, meta_model, coefficients = stacking_ensemble(
        oof_preds, y_true, config, extra_features
    )
    results.strategy_metrics.append(stacking_metrics)
    results.stacking_oof = stacking_oof
    print(f"  LL={stacking_metrics['logloss']:.5f}  "
          f"AUC={stacking_metrics['auc']:.5f}  "
          f"Score={stacking_metrics['score']:.5f}")

    # ── 5. Calibrated stacking ────────────────────────────────────────────────
    meta_calibrator = None
    if config.calibrate_ensemble:
        print("\n[5/5] Strategy 4 — Calibrated stacking (Platt on ensemble OOF)...")
        cal_oof, cal_metrics, meta_calibrator = calibrated_stacking(
            stacking_oof, y_true, config
        )
        results.strategy_metrics.append(cal_metrics)
        results.ensemble_oof = cal_oof
        print(f"  LL={cal_metrics['logloss']:.5f}  "
              f"AUC={cal_metrics['auc']:.5f}  "
              f"Score={cal_metrics['score']:.5f}")
    else:
        results.ensemble_oof = stacking_oof

    # ── 6. Identify best strategy ─────────────────────────────────────────────
    for m in results.strategy_metrics:
        if m["score"] < results.best_score:
            results.best_score    = m["score"]
            results.best_strategy = m["name"]

    # ── 7. Record runtime BEFORE saving — metadata.json must capture it ───────
    # (Bug fix: previously set after save, causing metadata to always record 0.0)
    results.runtime_sec = time.perf_counter() - start_time

    # ── 8. Save all artefacts ─────────────────────────────────────────────────
    save_ensemble_artifacts(
        results, meta_model, meta_calibrator,
        coefficients, corr_matrix, config, run_dir
    )

    print_results_table(results)

    return results


# =============================================================================
# PRODUCTION INFERENCE API
# =============================================================================

class EnsembleInference:
    """
    Production inference wrapper for the trained 5-model ensemble.

    Loads all artefacts from a completed ensemble run directory and
    exposes a single predict() method for use in the inference pipeline
    (notebooks 11_final_submission.ipynb and src/inference/predict.py).

    Usage
    -----
    inference = EnsembleInference.from_run_dir(run_dir, config)
    probabilities = inference.predict(raw_preds_per_model)

    TabNet note
    -----------
    raw_preds_per_model["tabnet"] should be the RAW (un-Platt-calibrated)
    fold-averaged predictions from TabNet.  The per-model Platt calibrators
    loaded here will calibrate them before the meta-model.
    """

    def __init__(
        self,
        meta_model       : LogisticRegression,
        meta_calibrator  : Optional["PlattCalibrator"],
        base_calibrators : Dict[str, "PlattCalibrator"],
        config           : EnsembleConfig,
    ) -> None:
        self.meta_model       = meta_model
        self.meta_calibrator  = meta_calibrator
        self.base_calibrators = base_calibrators
        self.config           = config

    @classmethod
    def from_run_dir(
        cls,
        run_dir : str,
        config  : EnsembleConfig,
    ) -> "EnsembleInference":
        """
        Load inference artefacts from a completed ensemble run directory.

        Loads:
          run_dir/meta_model.pkl              — stacking meta-model
          run_dir/meta_calibrator.pkl         — final Platt calibrator
          calibration_dir/{abbrev}/calibrator_platt.pkl  — per-model Platt
        """
        run_path = Path(run_dir)
        root     = Path(config.project_root)

        meta_model      = joblib.load(run_path / "meta_model.pkl")

        meta_calibrator = None
        cal_path        = run_path / "meta_calibrator.pkl"
        if cal_path.exists():
            meta_calibrator = joblib.load(cal_path)
        else:
            print("  Warning: meta_calibrator.pkl not found — "
                  "final Platt step will be skipped.")

        # Per-model Platt calibrators (use _CALIBRATION_FOLDER for abbreviated paths)
        base_calibrators: Dict[str, "PlattCalibrator"] = {}
        cal_dir = root / config.calibration_dir
        for m in MODEL_NAMES:
            folder     = _CALIBRATION_FOLDER[m]
            platt_path = cal_dir / folder / "calibrator_platt.pkl"
            if platt_path.exists():
                base_calibrators[m] = joblib.load(platt_path)
                print(f"  Loaded Platt calibrator for {m} from {folder}/")
            else:
                print(f"  Warning: Platt calibrator not found for {m}: {platt_path}")

        return cls(meta_model, meta_calibrator, base_calibrators, config)

    def predict(
        self,
        raw_preds_per_model : Dict[str, np.ndarray],
        extra_features      : Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Produce final ensemble predictions for test data.

        Pipeline
        --------
        1. Platt-calibrate each model's raw predictions (or pass through
           if calibrator not loaded — emit warning).
        2. Build meta-feature matrix (5 OOF cols + optional disagreement
           + optional raw SHAP features).
        3. Meta-model (LogisticRegression) → ensemble probability.
        4. Final Platt calibration (if available).
        5. Clip to [1e-6, 1−1e-6].

        Parameters
        ----------
        raw_preds_per_model : dict model_name → np.ndarray
            Raw (un-calibrated) predictions from each base model,
            averaged across CV folds.
        extra_features      : optional np.ndarray (n, k)
            Top-k SHAP raw features (post-Step 3 only).

        Returns
        -------
        np.ndarray of shape (n,) — final ensemble probabilities
        """
        # Step 1: per-model Platt calibration
        # Validate all input arrays have equal length before any computation.
        # Silent misalignment here would produce a corrupt submission with no error.
        lengths = {m: len(raw_preds_per_model[m]) for m in MODEL_NAMES}
        unique_lengths = set(lengths.values())
        if len(unique_lengths) != 1:
            raise ValueError(
                f"Inconsistent prediction array lengths across models: {lengths}. "
                "All models must produce predictions for the same number of rows."
            )

        calibrated: Dict[str, np.ndarray] = {}
        for m in MODEL_NAMES:
            if m not in raw_preds_per_model:
                raise ValueError(
                    f"Missing predictions for model '{m}' in raw_preds_per_model."
                )
            if m in self.base_calibrators:
                calibrated[m] = self.base_calibrators[m].predict(
                    raw_preds_per_model[m]
                )
            else:
                calibrated[m] = raw_preds_per_model[m]
                print(f"  Warning: Using un-calibrated predictions for {m}.")

        # Step 2: meta-feature matrix
        meta_X = np.stack([calibrated[m] for m in MODEL_NAMES], axis=1)
        if self.config.use_disagreement:
            meta_X = np.hstack([meta_X, meta_X.std(axis=1, keepdims=True)])

        if self.config.use_raw_features and extra_features is not None:
            meta_X = np.hstack([meta_X, extra_features])

        # Step 3: meta-model
        ensemble_pred = self.meta_model.predict_proba(meta_X)[:, 1]

        # Step 4: final Platt calibration
        if self.meta_calibrator is not None:
            ensemble_pred = self.meta_calibrator.predict(ensemble_pred)

        # Step 5: clip
        return np.clip(ensemble_pred, CLIP_LOW, CLIP_HIGH)


# =============================================================================
# ENTRY POINT (direct execution)
# =============================================================================

if __name__ == "__main__":
    """
    Run the ensemble pipeline from project root:
        python -m src.ensemble.ensemble

    Expects:
        outputs/multi_model/y_true.npy
        outputs/multi_model/oof_calibrated_{model}.npy  (all 5 models)
    """
    import sys

    project_root = str(Path(__file__).resolve().parents[2])

    config = EnsembleConfig(
        project_root       = project_root,
        n_splits           = 5,
        seed               = 42,
        optimise_weights   = True,
        use_disagreement   = True,
        use_raw_features   = False,   # activate post-SHAP (Step 3)
        calibrate_ensemble = True,
        meta_C             = 0.05,    # v5: tightened from 0.1 for 5 meta-inputs
    )

    print(f"Project root : {project_root}")
    print(f"Model names  : {MODEL_NAMES}")
    print(f"meta_C       : {config.meta_C}")
    print()

    results = run_ensemble_pipeline(config)

    sys.exit(0 if results.best_score < 999.0 else 1)