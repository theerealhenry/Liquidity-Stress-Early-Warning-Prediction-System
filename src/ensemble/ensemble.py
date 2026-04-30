"""
Ensemble Pipeline
=================
Project : AI4EAC Liquidity Stress Early Warning (Zindi Africa)
Module  : src/ensemble/ensemble.py

Architecture
------------
Three-layer ensemble designed around the specific properties of this problem:

  Layer 1 — Base models (already trained)
    LightGBM  : best Log Loss (0.29269), strong calibration
    XGBoost   : best AUC (0.90378), most stable folds (±0.00465)
    CatBoost  : post-calibration equivalent; fat-tailed disagreements add value

  Layer 2 — Calibration (Platt scaling per model)
    Applied before all ensemble operations.
    Platt chosen over isotonic for test-set generalisation safety.
    Isotonic used for OOF analysis only — never for inference.

  Layer 3 — Ensemble strategies (evaluated in order of complexity)
    3a. Simple average       — sanity check baseline
    3b. Weighted average     — scipy-optimised weights on OOF composite score
    3c. Stacking             — LogisticRegression meta-model on OOF inputs
    3d. Stacking + features  — meta-model augmented with key raw features

Key design decisions
--------------------
1. HIGH CORRELATION PROBLEM (0.962–0.978)
   Simple averaging on highly correlated models adds noise rather than
   signal. Stacking is the correct strategy: the meta-model learns *when*
   to trust each base model conditionally, not just averaging their outputs.

2. META-MODEL CHOICE: LogisticRegression
   With only 3–7 inputs (OOF preds ± disagreement ± raw features),
   a complex meta-model would overfit aggressively. LogisticRegression
   with C=0.1 (strong regularisation) is the principled choice.

3. LEAKAGE PREVENTION
   The meta-model is trained exclusively on OOF predictions from the
   base models — never on predictions made on the training data itself.
   Cross-validation inside stacking (nested CV) is used to produce
   meta-model OOF predictions for honest evaluation.

4. COMPOSITE SCORE OBJECTIVE
   All weight optimisation uses the competition's exact scoring formula:
   score = 0.6 * LogLoss + 0.4 * (1 - AUC)
   This ensures optimisation targets the actual leaderboard metric.

5. CALIBRATION LAYER
   The ensemble output is passed through a final Platt calibration step
   fitted on the stacking OOF predictions. This corrects any residual
   miscalibration introduced by the meta-model.

6. INFERENCE CONTRACT
   The public API is a single function: predict(X_raw_test) → np.ndarray
   Internally: load base models → feature engineer → preprocess →
   predict per fold → average → Platt calibrate → ensemble → clip

Output contract
---------------
outputs/experiments/v4_ensemble/run_YYYYMMDD_HHMMSS/
    ensemble_oof.npy          stacking OOF predictions (meta-model output)
    ensemble_weights.json     optimised weights per strategy
    meta_model.pkl            fitted stacking meta-model
    meta_calibrator.pkl       Platt calibrator fitted on ensemble OOF
    stacking_results.json     full metrics for all ensemble strategies
    feature_importance.csv    meta-model coefficient analysis
    metadata.json             run record with all hyperparameters
"""

from __future__ import annotations

import json
import os
import pickle
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold


# =============================================================================
# CONSTANTS
# =============================================================================

EPS                 : float      = 1e-15
CLIP_LOW            : float      = 1e-6
CLIP_HIGH           : float      = 1.0 - 1e-6
LOG_LOSS_WEIGHT     : float      = 0.6
AUC_WEIGHT          : float      = 0.4
SEED                : int        = 42
MODEL_NAMES         : List[str]  = ["lightgbm", "xgboost", "catboost"]


# =============================================================================
# METRICS
# =============================================================================

def composite_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Competition metric: 0.6 * LogLoss + 0.4 * (1 - AUC).
    Lower is better. Clips predictions before Log Loss computation.
    """
    y_clipped = np.clip(y_pred, EPS, 1 - EPS)
    ll  = log_loss(y_true, y_clipped)
    auc = roc_auc_score(y_true, y_pred)
    return LOG_LOSS_WEIGHT * ll + AUC_WEIGHT * (1.0 - auc)


def evaluate(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """Full metric suite for a set of predictions."""
    y_clipped = np.clip(y_pred, EPS, 1 - EPS)
    return {
        "name"      : name,
        "logloss"   : float(log_loss(y_true, y_clipped)),
        "auc"       : float(roc_auc_score(y_true, y_pred)),
        "brier"     : float(brier_score_loss(y_true, y_pred)),
        "score"     : float(composite_score(y_true, y_pred)),
        "mean_pred" : float(y_pred.mean()),
        "std_pred"  : float(y_pred.std()),
    }


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class EnsembleConfig:
    """
    Full configuration for the ensemble pipeline.
    All hyperparameters in one place — no magic numbers in code.
    """
    # Paths
    project_root        : str   = ""
    oof_dir             : str   = "outputs/multi_model"
    output_dir          : str   = "outputs/experiments/v4_ensemble"
    calibration_dir     : str   = "outputs/calibration"

    # Cross-validation
    n_splits            : int   = 5
    seed                : int   = SEED

    # Weight optimisation
    optimise_weights    : bool  = True
    weight_bounds       : Tuple = (0.0, 1.0)
    optimiser_method    : str   = "Nelder-Mead"
    optimiser_maxiter   : int   = 5000

    # Stacking meta-model
    meta_C              : float = 0.1        # strong regularisation — 3 features only
    meta_max_iter       : int   = 1000
    meta_solver         : str   = "lbfgs"

    # Stacking feature augmentation
    use_disagreement    : bool  = True       # add inter-model std as meta-feature
    use_raw_features    : bool  = False      # add top raw features (Phase C option)
    top_k_raw_features  : int   = 5

    # Final calibration of ensemble output
    calibrate_ensemble  : bool  = True

    # Artifact saving
    save_models         : bool  = True
    save_oof            : bool  = True
    save_metrics        : bool  = True


@dataclass
class EnsembleResults:
    """Collects all results across ensemble strategies for comparison."""
    strategy_metrics    : List[Dict]   = field(default_factory=list)
    best_strategy       : str          = ""
    best_score          : float        = 999.0
    optimised_weights   : Dict         = field(default_factory=dict)
    stacking_oof        : np.ndarray   = field(default_factory=lambda: np.array([]))
    ensemble_oof        : np.ndarray   = field(default_factory=lambda: np.array([]))
    run_dir             : str          = ""
    runtime_sec         : float        = 0.0


# =============================================================================
# CALIBRATION UTILITIES
# =============================================================================

class PlattCalibrator:
    """
    Platt scaling calibrator (logistic regression on raw predictions).

    Wraps sklearn LogisticRegression with the 1D interface expected
    by the ensemble pipeline. Chosen over isotonic for test-set
    generalisation safety (2 parameters vs non-parametric staircase).
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000) -> None:
        self.C        = C
        self.max_iter = max_iter
        self._model   = LogisticRegression(
            C=C, solver="lbfgs", max_iter=max_iter
        )
        self._fitted  = False

    def fit(self, raw_preds: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        self._model.fit(raw_preds.reshape(-1, 1), y)
        self._fitted = True
        return self

    def predict(self, raw_preds: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("PlattCalibrator must be fitted before predict()")
        return self._model.predict_proba(
            raw_preds.reshape(-1, 1)
        )[:, 1]

    def fit_predict(
        self, raw_preds: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        return self.fit(raw_preds, y).predict(raw_preds)


def cv_platt_calibrate(
    raw_preds: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = SEED,
    C: float = 1.0,
) -> Tuple[np.ndarray, PlattCalibrator]:
    """
    Cross-validated Platt calibration.

    Fits calibrator on held-out splits to prevent in-sample optimism.
    Returns both calibrated predictions AND a final calibrator fitted
    on the full data (for applying to test set).

    Returns
    -------
    calibrated_oof : np.ndarray
        CV-calibrated predictions (honest OOF estimate)
    final_calibrator : PlattCalibrator
        Fitted on full data — use for test set inference
    """
    calibrated = np.zeros_like(raw_preds, dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for train_idx, val_idx in skf.split(raw_preds, y):
        cal = PlattCalibrator(C=C)
        cal.fit(raw_preds[train_idx], y[train_idx])
        calibrated[val_idx] = cal.predict(raw_preds[val_idx])

    # Final calibrator on full data for test set use
    final_calibrator = PlattCalibrator(C=C).fit(raw_preds, y)

    return calibrated, final_calibrator


# =============================================================================
# ARTIFACT LOADING
# =============================================================================

def load_oof_artifacts(config: EnsembleConfig) -> Dict[str, np.ndarray]:
    """
    Load Platt-calibrated OOF predictions and ground truth from
    the multi-model analysis outputs.

    Expects:
        outputs/multi_model/oof_calibrated_{model}.npy
        outputs/multi_model/y_true.npy
    """
    root    = Path(config.project_root)
    oof_dir = root / config.oof_dir

    artifacts = {}

    # Ground truth
    y_path = oof_dir / "y_true.npy"
    if not y_path.exists():
        raise FileNotFoundError(
            f"y_true.npy not found at {y_path}. "
            "Run 08_multi_model_analysis.ipynb first."
        )
    artifacts["y_true"] = np.load(y_path)

    # Per-model Platt-calibrated OOF predictions
    for m in MODEL_NAMES:
        path = oof_dir / f"oof_calibrated_{m}.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"Calibrated OOF not found for {m}: {path}. "
                "Run 08_multi_model_analysis.ipynb first."
            )
        artifacts[m] = np.load(path)
        print(f"  Loaded {m}: shape={artifacts[m].shape}  "
              f"range=[{artifacts[m].min():.4f}, {artifacts[m].max():.4f}]")

    # Validate alignment
    n = len(artifacts["y_true"])
    for m in MODEL_NAMES:
        assert len(artifacts[m]) == n, \
            f"Shape mismatch: y_true has {n} rows, {m} has {len(artifacts[m])}"
    assert not any(np.isnan(artifacts[m]).any() for m in MODEL_NAMES), \
        "NaN values detected in OOF predictions"

    print(f"\n  y_true shape     : {artifacts['y_true'].shape}")
    print(f"  Positive rate    : {artifacts['y_true'].mean():.4f}")
    print(f"  All shapes match : True")

    return artifacts


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
    oof_preds: Dict[str, np.ndarray],
    y_true: np.ndarray,
) -> Tuple[np.ndarray, Dict]:
    """
    Equal-weight average of all base model predictions.
    Sanity check baseline — expected to underperform stacking
    given correlations of 0.962–0.978.
    """
    pred_arr = np.stack([oof_preds[m] for m in MODEL_NAMES], axis=1)
    avg_pred = pred_arr.mean(axis=1)
    metrics  = evaluate("simple_average", y_true, avg_pred)
    return avg_pred, metrics


# =============================================================================
# ENSEMBLE STRATEGY 2 — OPTIMISED WEIGHTED AVERAGE
# =============================================================================

def _weight_objective(
    weights: np.ndarray,
    pred_arr: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """
    Objective function for weight optimisation.
    Minimises the competition composite score.
    Weights are projected onto the simplex (sum to 1, all non-negative).
    """
    w = np.clip(weights, 0.0, 1.0)
    w_sum = w.sum()
    if w_sum < 1e-10:
        return 999.0
    w = w / w_sum
    blended = (pred_arr * w).sum(axis=1)
    blended = np.clip(blended, EPS, 1 - EPS)
    return composite_score(y_true, blended)


def optimised_weighted_average(
    oof_preds: Dict[str, np.ndarray],
    y_true: np.ndarray,
    config: EnsembleConfig,
) -> Tuple[np.ndarray, Dict, np.ndarray]:
    """
    Scipy Nelder-Mead optimisation of ensemble weights.

    Minimises the composite competition score (0.6*LL + 0.4*(1-AUC)).
    Uses multiple random initialisations to avoid local minima.
    Projects weights onto the probability simplex after optimisation.

    Returns
    -------
    best_pred    : np.ndarray of shape (n,)
    metrics      : dict
    best_weights : np.ndarray of shape (3,)

    WARNING: These weights are OOF-optimised and will be slightly
    optimistic. The meta-model (stacking) is the deployment-safe
    alternative.
    """
    pred_arr = np.stack([oof_preds[m] for m in MODEL_NAMES], axis=1)

    best_score   = 999.0
    best_weights = np.array([1/3, 1/3, 1/3])

    # Multiple initialisations to escape local minima
    initialisations = [
        np.array([1/3, 1/3, 1/3]),        # equal weights
        np.array([0.50, 0.30, 0.20]),      # LightGBM heavy
        np.array([0.55, 0.35, 0.10]),      # LightGBM + XGB
        np.array([0.40, 0.40, 0.20]),      # LGB + XGB equal
        np.array([0.60, 0.40, 0.00]),      # Top 2 only
        np.array([0.45, 0.35, 0.20]),      # Balanced with CatBoost
    ]

    rng = np.random.default_rng(config.seed)
    for _ in range(10):
        # Random initialisations on simplex
        r = rng.dirichlet(np.ones(len(MODEL_NAMES)))
        initialisations.append(r)

    print(f"  Running weight optimisation ({len(initialisations)} initialisations)...")

    for i, w0 in enumerate(initialisations):
        result = minimize(
            _weight_objective,
            x0=w0,
            args=(pred_arr, y_true),
            method=config.optimiser_method,
            options={
                "maxiter" : config.optimiser_maxiter,
                "xatol"   : 1e-8,
                "fatol"   : 1e-8,
            },
        )
        if result.fun < best_score:
            best_score   = result.fun
            best_weights = result.x

    # Project onto simplex
    best_weights = np.clip(best_weights, 0.0, 1.0)
    best_weights = best_weights / best_weights.sum()

    best_pred = (pred_arr * best_weights).sum(axis=1)
    best_pred = np.clip(best_pred, CLIP_LOW, CLIP_HIGH)
    metrics   = evaluate("optimised_weighted_average", y_true, best_pred)

    weight_dict = {m: float(w) for m, w in zip(MODEL_NAMES, best_weights)}
    print(f"  Optimised weights: {weight_dict}")
    print(f"  Score: {best_score:.6f}")

    return best_pred, metrics, best_weights


# =============================================================================
# ENSEMBLE STRATEGY 3 — STACKING (PRIMARY STRATEGY)
# =============================================================================

def _build_meta_features(
    oof_preds: Dict[str, np.ndarray],
    config: EnsembleConfig,
    extra_features: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Build meta-feature matrix for the stacking meta-model.

    Base features (always included):
        - Platt-calibrated OOF from each base model (3 columns)

    Optional features:
        - Inter-model standard deviation (disagreement signal)
        - Top-k raw features from feature_importance_combined.csv

    The disagreement feature (std across models) captures the meta-model's
    uncertainty: high disagreement cases are where stacking adds most value
    relative to simple averaging.
    """
    # Base: stacked OOF predictions
    meta_X = np.stack([oof_preds[m] for m in MODEL_NAMES], axis=1)   # (n, 3)

    if config.use_disagreement:
        # Inter-model std — fat-tailed disagreements are the ensemble's opportunity
        disagreement = meta_X.std(axis=1, keepdims=True)              # (n, 1)
        meta_X = np.hstack([meta_X, disagreement])
        print(f"  Meta-features: 3 OOF + 1 disagreement = {meta_X.shape[1]} features")
    else:
        print(f"  Meta-features: 3 OOF = {meta_X.shape[1]} features")

    if config.use_raw_features and extra_features is not None:
        meta_X = np.hstack([meta_X, extra_features])
        print(f"  Meta-features: +{extra_features.shape[1]} raw = {meta_X.shape[1]} total")

    return meta_X


def stacking_ensemble(
    oof_preds: Dict[str, np.ndarray],
    y_true: np.ndarray,
    config: EnsembleConfig,
    extra_features: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict, LogisticRegression]:
    """
    Stacking ensemble with LogisticRegression meta-model.

    Uses nested cross-validation to produce honest OOF meta-predictions:
    - Outer CV: 5 folds on the full dataset
    - In each fold: meta-model trained on train-fold OOF, predicts val-fold

    This prevents leakage: the meta-model never sees the same predictions
    it is evaluated on during its own training.

    Parameters
    ----------
    oof_preds       : dict of model_name → np.ndarray (Platt-calibrated)
    y_true          : np.ndarray
    config          : EnsembleConfig
    extra_features  : optional np.ndarray of additional meta-features

    Returns
    -------
    stacking_oof    : np.ndarray — CV stacking OOF predictions
    metrics         : dict — full metric suite
    final_meta_model: LogisticRegression fitted on full data
    """
    meta_X = _build_meta_features(oof_preds, config, extra_features)
    n_meta_features = meta_X.shape[1]

    stacking_oof = np.zeros(len(y_true), dtype=float)
    skf = StratifiedKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.seed,
    )

    fold_scores = []
    print(f"  Running stacking CV ({config.n_splits} folds)...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(meta_X, y_true)):
        X_train_meta = meta_X[train_idx]
        y_train_meta = y_true[train_idx]
        X_val_meta   = meta_X[val_idx]
        y_val_meta   = y_true[val_idx]

        meta_model = LogisticRegression(
            C         = config.meta_C,
            solver    = config.meta_solver,
            max_iter  = config.meta_max_iter,
            random_state = config.seed,
        )
        meta_model.fit(X_train_meta, y_train_meta)

        fold_pred = meta_model.predict_proba(X_val_meta)[:, 1]
        stacking_oof[val_idx] = fold_pred

        fold_score = composite_score(y_val_meta, fold_pred)
        fold_ll    = log_loss(y_val_meta, np.clip(fold_pred, EPS, 1 - EPS))
        fold_auc   = roc_auc_score(y_val_meta, fold_pred)
        fold_scores.append(fold_score)

        print(f"  [Fold {fold}] Score={fold_score:.5f}  "
              f"LL={fold_ll:.5f}  AUC={fold_auc:.5f}")

    mean_score = np.mean(fold_scores)
    std_score  = np.std(fold_scores)
    print(f"\n  Stacking CV score: {mean_score:.5f} ± {std_score:.5f}")

    # Final meta-model fitted on all data (for test inference)
    final_meta_model = LogisticRegression(
        C            = config.meta_C,
        solver       = config.meta_solver,
        max_iter     = config.meta_max_iter,
        random_state = config.seed,
    )
    final_meta_model.fit(meta_X, y_true)

    # Log meta-model coefficients for interpretability
    feature_names = MODEL_NAMES.copy()
    if config.use_disagreement:
        feature_names.append("inter_model_disagreement")
    coeffs = pd.DataFrame({
        "feature"     : feature_names[:n_meta_features],
        "coefficient" : final_meta_model.coef_[0][:n_meta_features],
    }).sort_values("coefficient", ascending=False)
    print(f"\n  Meta-model coefficients:")
    print(coeffs.to_string(index=False))

    metrics = evaluate("stacking", y_true, stacking_oof)

    return stacking_oof, metrics, final_meta_model, coeffs


# =============================================================================
# ENSEMBLE STRATEGY 4 — STACKING WITH FINAL CALIBRATION
# =============================================================================

def calibrated_stacking(
    stacking_oof    : np.ndarray,
    y_true          : np.ndarray,
    config          : EnsembleConfig,
) -> Tuple[np.ndarray, Dict, PlattCalibrator]:
    """
    Apply Platt calibration to the stacking OOF predictions.

    The stacking meta-model (LogisticRegression) produces well-scaled
    probabilities but may have residual miscalibration from the input
    feature scale. A final calibration pass corrects this.

    Uses cross-validated Platt to produce honest OOF estimates.

    Returns
    -------
    calibrated_oof     : np.ndarray — CV-calibrated stacking predictions
    metrics            : dict
    final_calibrator   : PlattCalibrator — for test set inference
    """
    calibrated_oof, final_calibrator = cv_platt_calibrate(
        stacking_oof, y_true,
        n_splits=config.n_splits,
        seed=config.seed,
        C=1.0,
    )
    calibrated_oof = np.clip(calibrated_oof, CLIP_LOW, CLIP_HIGH)
    metrics = evaluate("stacking_calibrated", y_true, calibrated_oof)

    return calibrated_oof, metrics, final_calibrator


# =============================================================================
# ARTIFACT SAVING
# =============================================================================

def save_ensemble_artifacts(
    results         : EnsembleResults,
    meta_model      : LogisticRegression,
    meta_calibrator : Optional[PlattCalibrator],
    coefficients    : pd.DataFrame,
    config          : EnsembleConfig,
    run_dir         : Path,
) -> None:
    """
    Save all ensemble artifacts to the run directory.

    Aligned to output spec:
    outputs/experiments/v4_ensemble/run_YYYYMMDD_HHMMSS/
        ensemble_oof.npy
        ensemble_weights.json
        meta_model.pkl
        meta_calibrator.pkl
        stacking_results.json
        feature_importance.csv
        metadata.json
    """
    # ── OOF predictions ───────────────────────────────────────────────
    if len(results.ensemble_oof) > 0:
        np.save(run_dir / "ensemble_oof.npy", results.ensemble_oof)

    if len(results.stacking_oof) > 0:
        np.save(run_dir / "stacking_oof.npy", results.stacking_oof)

    # ── Ensemble weights ──────────────────────────────────────────────
    weights_data = {
        "optimised"  : results.optimised_weights,
        "model_order": MODEL_NAMES,
        "note"       : "Weights are OOF-optimised. Use stacking for deployment.",
    }
    with open(run_dir / "ensemble_weights.json", "w") as f:
        json.dump(weights_data, f, indent=4)

    # ── Meta-model ────────────────────────────────────────────────────
    joblib.dump(meta_model, run_dir / "meta_model.pkl")

    # ── Meta-calibrator ───────────────────────────────────────────────
    if meta_calibrator is not None:
        joblib.dump(meta_calibrator, run_dir / "meta_calibrator.pkl")

    # ── Strategy results ──────────────────────────────────────────────
    with open(run_dir / "stacking_results.json", "w") as f:
        json.dump(results.strategy_metrics, f, indent=4)

    # ── Meta-model coefficients ───────────────────────────────────────
    coefficients.to_csv(run_dir / "feature_importance.csv", index=False)

    # ── Metadata ──────────────────────────────────────────────────────
    metadata = {
        "created_at"        : datetime.now().isoformat(),
        "best_strategy"     : results.best_strategy,
        "best_score"        : results.best_score,
        "runtime_sec"       : results.runtime_sec,
        "config"            : asdict(config),
        "strategy_summary"  : [
            {k: v for k, v in m.items() if k != "name"}
            | {"strategy": m["name"]}
            for m in results.strategy_metrics
        ],
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4, default=str)

    print(f"\nAll artifacts saved to: {run_dir}")


# =============================================================================
# RESULTS REPORTING
# =============================================================================

def print_results_table(results: EnsembleResults) -> None:
    """Print a formatted comparison table of all ensemble strategies."""
    print("\n" + "=" * 80)
    print("ENSEMBLE RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Strategy':35s}  {'LogLoss':>8}  {'AUC':>8}  "
          f"{'Score':>8}  {'Brier':>8}")
    print("-" * 80)

    sorted_metrics = sorted(results.strategy_metrics, key=lambda x: x["score"])

    for m in sorted_metrics:
        marker = " ← BEST" if m["name"] == results.best_strategy else ""
        print(f"{m['name']:35s}  {m['logloss']:>8.5f}  {m['auc']:>8.5f}  "
              f"{m['score']:>8.5f}  {m['brier']:>8.5f}{marker}")

    print("=" * 80)
    print(f"\nBest strategy : {results.best_strategy}")
    print(f"Best score    : {results.best_score:.6f}")
    print(f"Runtime       : {results.runtime_sec:.1f} seconds")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_ensemble_pipeline(
    config              : EnsembleConfig,
    extra_features      : Optional[np.ndarray] = None,
) -> EnsembleResults:
    """
    Full ensemble pipeline.

    Executes all four ensemble strategies in order:
    1. Simple average (baseline)
    2. Optimised weighted average
    3. Stacking (LogisticRegression meta-model)
    4. Calibrated stacking (final Platt pass on stacking output)

    Parameters
    ----------
    config          : EnsembleConfig
    extra_features  : optional np.ndarray of shape (n, k) — additional
                      meta-features beyond OOF predictions (e.g. top-k
                      raw features identified by SHAP analysis)

    Returns
    -------
    EnsembleResults with all metrics, predictions, and artifacts saved
    """
    start_time = time.perf_counter()
    results    = EnsembleResults()

    print("=" * 70)
    print("ENSEMBLE PIPELINE")
    print("=" * 70)

    # ── 0. Setup ──────────────────────────────────────────────────────
    run_dir = build_run_dir(config)
    results.run_dir = str(run_dir)
    print(f"Run directory: {run_dir}")

    # ── 1. Load artifacts ─────────────────────────────────────────────
    print("\n[1/5] Loading OOF artifacts...")
    oof_artifacts = load_oof_artifacts(config)
    y_true = oof_artifacts["y_true"]
    oof_preds = {m: oof_artifacts[m] for m in MODEL_NAMES}

    # Per-model baseline metrics
    print("\n  Individual model metrics (Platt-calibrated OOF):")
    for m in MODEL_NAMES:
        m_metrics = evaluate(m, y_true, oof_preds[m])
        results.strategy_metrics.append(m_metrics)
        print(f"  {m:12s}: LL={m_metrics['logloss']:.5f}  "
              f"AUC={m_metrics['auc']:.5f}  Score={m_metrics['score']:.5f}")

    # ── 2. Simple average ─────────────────────────────────────────────
    print("\n[2/5] Strategy 1 — Simple average...")
    avg_pred, avg_metrics = simple_average(oof_preds, y_true)
    results.strategy_metrics.append(avg_metrics)
    print(f"  LL={avg_metrics['logloss']:.5f}  "
          f"AUC={avg_metrics['auc']:.5f}  "
          f"Score={avg_metrics['score']:.5f}")

    # ── 3. Optimised weighted average ─────────────────────────────────
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

    # ── 4. Stacking ───────────────────────────────────────────────────
    print("\n[4/5] Strategy 3 — Stacking meta-model...")
    stacking_oof, stacking_metrics, meta_model, coefficients = stacking_ensemble(
        oof_preds, y_true, config, extra_features
    )
    results.strategy_metrics.append(stacking_metrics)
    results.stacking_oof = stacking_oof
    print(f"  LL={stacking_metrics['logloss']:.5f}  "
          f"AUC={stacking_metrics['auc']:.5f}  "
          f"Score={stacking_metrics['score']:.5f}")

    # ── 5. Calibrated stacking ────────────────────────────────────────
    meta_calibrator = None
    if config.calibrate_ensemble:
        print("\n[5/5] Strategy 4 — Calibrated stacking...")
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

    # ── 6. Identify best strategy ─────────────────────────────────────
    for m in results.strategy_metrics:
        if m["score"] < results.best_score:
            results.best_score    = m["score"]
            results.best_strategy = m["name"]

    # ── 7. Save artifacts ─────────────────────────────────────────────
    save_ensemble_artifacts(
        results, meta_model, meta_calibrator, coefficients, config, run_dir
    )

    results.runtime_sec = time.perf_counter() - start_time
    print_results_table(results)

    return results


# =============================================================================
# INFERENCE API
# =============================================================================

class EnsembleInference:
    """
    Production inference wrapper for the trained ensemble.

    Loads all artifacts from a completed ensemble run and exposes
    a single predict() method for use in the inference pipeline.

    Usage
    -----
    inference = EnsembleInference.from_run_dir(run_dir, config)
    probabilities = inference.predict(X_test_processed_per_model)
    """

    def __init__(
        self,
        meta_model      : LogisticRegression,
        meta_calibrator : Optional[PlattCalibrator],
        base_calibrators: Dict[str, PlattCalibrator],
        config          : EnsembleConfig,
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
        Load inference artifacts from a completed ensemble run directory.
        """
        run_path = Path(run_dir)

        meta_model = joblib.load(run_path / "meta_model.pkl")

        meta_calibrator = None
        cal_path = run_path / "meta_calibrator.pkl"
        if cal_path.exists():
            meta_calibrator = joblib.load(cal_path)

        # Load per-model Platt calibrators from calibration directory
        base_calibrators = {}
        cal_dir = Path(config.project_root) / config.calibration_dir
        for m in MODEL_NAMES:
            platt_path = cal_dir / m / "calibrator_platt.pkl"
            if platt_path.exists():
                base_calibrators[m] = joblib.load(platt_path)
            else:
                print(f"  Warning: Platt calibrator not found for {m}: {platt_path}")

        return cls(meta_model, meta_calibrator, base_calibrators, config)

    def predict(
        self,
        raw_preds_per_model: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Produce final ensemble predictions for test data.

        Parameters
        ----------
        raw_preds_per_model : dict of model_name → np.ndarray
            Raw (uncalibrated) predictions from each base model,
            averaged across CV folds.

        Returns
        -------
        np.ndarray of shape (n,)
            Final ensemble probabilities, clipped to [1e-6, 1−1e-6]
        """
        # Step 1: Platt-calibrate each base model's predictions
        calibrated = {}
        for m in MODEL_NAMES:
            if m in self.base_calibrators:
                calibrated[m] = self.base_calibrators[m].predict(
                    raw_preds_per_model[m]
                )
            else:
                calibrated[m] = raw_preds_per_model[m]
                print(f"  Warning: Using uncalibrated predictions for {m}")

        # Step 2: Build meta-features
        meta_X = np.stack([calibrated[m] for m in MODEL_NAMES], axis=1)
        if self.config.use_disagreement:
            disagreement = meta_X.std(axis=1, keepdims=True)
            meta_X = np.hstack([meta_X, disagreement])

        # Step 3: Meta-model prediction
        ensemble_pred = self.meta_model.predict_proba(meta_X)[:, 1]

        # Step 4: Final calibration
        if self.meta_calibrator is not None:
            ensemble_pred = self.meta_calibrator.predict(ensemble_pred)

        # Step 5: Clip
        return np.clip(ensemble_pred, CLIP_LOW, CLIP_HIGH)


# =============================================================================
# ENTRY POINT (direct execution)
# =============================================================================

if __name__ == "__main__":
    """
    Run the ensemble pipeline from project root:
        python -m src.ensemble.ensemble

    Expects:
        outputs/multi_model/oof_calibrated_{model}.npy   (from notebook 08)
        outputs/multi_model/y_true.npy
    """
    import sys

    project_root = str(Path(__file__).resolve().parents[2])

    config = EnsembleConfig(
        project_root     = project_root,
        n_splits         = 5,
        seed             = 42,
        optimise_weights = True,
        use_disagreement = True,
        use_raw_features = False,
        calibrate_ensemble = True,
        meta_C           = 0.1,
    )

    print(f"Project root : {project_root}")
    print(f"Config       : {asdict(config)}\n")

    results = run_ensemble_pipeline(config)

    sys.exit(0 if results.best_score < 999.0 else 1)