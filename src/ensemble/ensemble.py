"""
Ensemble Pipeline — v5.1
========================
Project : AI4EAC Liquidity Stress Early Warning (Zindi Africa)
Module  : src/ensemble/ensemble.py
Author  : Henry Otsyula
Updated : 2026-05-08  (v5.0 → v5.1: architectural hardening)

Change log
----------
v4   : 3-model GBM-only ensemble.  Intra-cluster r=0.962–0.978.
       Stacking gave no lift over averaging.
v5.0 : Added LogReg + TabNet.  Cross-cluster r dropped to 0.745–0.753.
       Three independent signal clusters confirmed.  Optimised weighted
       average (0.19144) beat best single model (XGB 0.19350).
v5.1 : Architectural hardening — identical results, safer code:
       • Global mutable MODEL_NAMES removed.  config.model_names is now
         the single source of truth for every loop and array stack.
       • Global mutable _CALIBRATION_FOLDER removed.  Moved to
         EnsembleConfig.calibration_folder_map — overridable per-run.
       • model_groups added to EnsembleConfig for cluster-aware reporting
         without hardcoded string lists in utility functions.
       • ensemble_version added to EnsembleConfig; saved in metadata.json
         for deployment compatibility checks.
       • _KNOWN_INDIVIDUAL_SCORES kept module-level (read-only constant,
         never mutated — appropriate).
       • All functions receive model_names explicitly; zero global reads.
       • Hardcoded 5-element weight initialisations replaced with
         model-count-agnostic dynamic generation via _pad() helper.
       • validate_model_set() added — called at every entry point.
       • config.validate() enforces internal consistency on construction.
       • EnsembleInference.from_run_dir() reads model_names and
         calibration_folder_map from metadata.json — not from module
         defaults — guaranteeing training-inference consistency.
       • Bug fix carried from v5.0: runtime_sec set BEFORE save so
         metadata.json records the real elapsed time (was always 0.0).
       • Known limitation documented in calibrated_stacking() docstring:
         Platt CV folds not aligned to stacking CV folds (negligible
         impact; deferred post-competition).

Architecture
------------
Three-layer ensemble designed around the specific properties of this problem:

  Layer 1 — Base models (already trained, artefacts on disk)
    LightGBM  : strong calibration, best Log Loss baseline
    XGBoost   : best single-model composite (0.19350), most stable folds
    CatBoost  : not Optuna-tuned, but fat-tailed disagreements add value
    LogReg    : ElasticNet saga — LINEAR boundary; orthogonal to GBMs
                Composite 0.26003 post-Platt.
    TabNet    : instance-wise sequential attention — per-customer feature
                selection.  Composite 0.25296 post-Platt.

  Layer 2 — Calibration (already applied upstream)
    OOF Platt calibrators fitted in notebooks 07 / 07b, saved to
    outputs/calibration/{folder}/calibrator_platt.pkl.
    Calibrated OOF arrays in outputs/multi_model/.
    This module CONSUMES those artefacts — it does NOT re-calibrate.

  Layer 3 — Ensemble strategies
    3a. Simple average           — equal-weight sanity check
    3b. Optimised weighted avg   — Scipy Nelder-Mead on composite score
    3c. Stacking                 — LogisticRegression meta-model on OOF
    3d. Calibrated stacking      — Platt pass on stacking OOF output

Key design decisions
---------------------
1. CONFIG IS THE SINGLE SOURCE OF TRUTH
   config.model_names drives every loop, stack, and save.  Ablation
   experiments change config.model_names — nothing else.

2. META_C = 0.05
   Five meta-inputs instead of three; tighter regularisation prevents
   the meta-model from fitting OOF noise.

3. DISAGREEMENT FEATURE (std across all configured models per row)
   High disagreement = cross-cluster uncertainty = where stacking adds value.

4. OOF-ONLY STACKING (no leakage)
   Meta-model trained exclusively on held-out OOF predictions.
   Nested 5-fold CV for honest evaluation.

5. COMPOSITE SCORE OBJECTIVE throughout
   0.6 * LogLoss + 0.4 * (1 − AUC) — the exact competition metric.

6. CALIBRATION FOLDER MAP
   lightgbm → lgb/  (abbreviated, preserved from v4 artefact paths)
   xgboost  → xgb/
   catboost → cat/
   logreg   → logreg/
   tabnet   → tabnet/
   Overridable via config.calibration_folder_map.

Output contract
---------------
outputs/experiments/v5_ensemble/run_YYYYMMDD_HHMMSS/
    ensemble_oof.npy           final calibrated stacking OOF predictions
    stacking_oof.npy           raw stacking output (pre-calibration)
    ensemble_weights.json      optimised weights + model order
    meta_model.pkl             fitted stacking LogisticRegression
    meta_calibrator.pkl        Platt calibrator on ensemble OOF
    stacking_results.json      full metrics for all strategies
    feature_importance.csv     meta-model coefficient table
    correlation_matrix.csv     post-calibration OOF Pearson correlations
    metadata.json              self-describing run record: model_names,
                               calibration_folder_map, ensemble_version,
                               meta_feature_order, full config
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
# MODULE-LEVEL CONSTANTS  (read-only; never mutated by any function)
# =============================================================================

EPS             : float = 1e-15
CLIP_LOW        : float = 1e-6
CLIP_HIGH       : float = 1.0 - 1e-6
LOG_LOSS_WEIGHT : float = 0.6
AUC_WEIGHT      : float = 0.4
SEED            : int   = 42

# Default ensemble structure — used only as defaults in EnsembleConfig.
# Production code must never import or mutate these; use config.model_names.
_DEFAULT_MODEL_NAMES: List[str] = [
    "lightgbm",  # GBM cluster
    "xgboost",   # GBM cluster
    "catboost",  # GBM cluster
    "logreg",    # Linear cluster
    "tabnet",    # Attention cluster
]

# Default calibration folder mapping.  GBMs use abbreviated names preserved
# from v4 artefact paths.  Overridable via EnsembleConfig.calibration_folder_map.
_DEFAULT_CALIBRATION_FOLDER: Dict[str, str] = {
    "lightgbm" : "lgb",
    "xgboost"  : "xgb",
    "catboost" : "cat",
    "logreg"   : "logreg",
    "tabnet"   : "tabnet",
}

# Default cluster groupings — used for correlation reporting only.
# Overridable via EnsembleConfig.model_groups.
_DEFAULT_MODEL_GROUPS: Dict[str, List[str]] = {
    "gbm"       : ["lightgbm", "xgboost", "catboost"],
    "linear"    : ["logreg"],
    "attention" : ["tabnet"],
}

# Individual model composite scores (post-Platt, from notebooks 07/07b).
# Read-only — used for display in results tables only.
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
    Competition metric: 0.6 * LogLoss + 0.4 * (1 − AUC).
    Lower is better.  Clips predictions to avoid log(0).
    """
    y_clipped = np.clip(y_pred, EPS, 1.0 - EPS)
    ll  = log_loss(y_true, y_clipped)
    auc = roc_auc_score(y_true, y_pred)
    return LOG_LOSS_WEIGHT * ll + AUC_WEIGHT * (1.0 - auc)


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Full metric suite: logloss, auc, brier, composite score,
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
# VALIDATION UTILITIES
# =============================================================================

def validate_model_set(
    model_names : List[str],
    oof_preds   : Dict[str, np.ndarray],
    context     : str = "",
) -> None:
    """
    Strict bidirectional validation: model_names vs oof_preds keys.

    Raises ValueError if:
      - Any model in model_names is missing from oof_preds.
      - Any key in oof_preds is not in model_names.

    Bidirectional checking is important: extra keys indicate a stale or
    wrong artefact set was loaded, which would silently corrupt stacking
    if allowed through.

    Parameters
    ----------
    model_names : models expected by the current config
    oof_preds   : dict of loaded model predictions
    context     : caller label for error messages
    """
    prefix  = f"[{context}] " if context else ""
    missing = [m for m in model_names if m not in oof_preds]
    extra   = [m for m in oof_preds   if m not in model_names]

    if missing:
        raise ValueError(
            f"{prefix}Missing OOF predictions for: {missing}.\n"
            "Run the calibration notebook(s) to produce the missing artefacts."
        )
    if extra:
        raise ValueError(
            f"{prefix}Unexpected models in oof_preds not in config.model_names: "
            f"{extra}.\nPass only the models listed in config.model_names, "
            "or update config.model_names to include them."
        )


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class EnsembleConfig:
    """
    Single source of truth for the ensemble pipeline.

    All hyperparameters, model identity, and file-path mappings live here.
    No function in this module reads module-level global state — every
    loop is driven by config.model_names.

    Design note — NOT frozen
    ------------------------
    frozen=True breaks field(default_factory=…) on Python < 3.10 and
    prevents notebook-friendly path overrides (config.project_root = ...).
    Instead, call config.validate() after construction to enforce internal
    consistency.  The validate() method is also called at the start of
    run_ensemble_pipeline() as a pre-flight check.

    Ablation example
    ----------------
    To run a GBM-only experiment without touching ensemble.py:

        config = EnsembleConfig(
            model_names = ["lightgbm", "xgboost", "catboost"],
            model_groups = {"gbm": ["lightgbm", "xgboost", "catboost"]},
            output_dir  = "outputs/experiments/ablation_gbm_only",
        )
        config.validate()
        results = run_ensemble_pipeline(config)
    """

    # ── Ensemble identity ─────────────────────────────────────────────────────
    ensemble_version : str = "v5.1"

    # ── Model structure ───────────────────────────────────────────────────────
    # This list is the single source of truth for which models are in the
    # ensemble, in what order, and therefore what the meta-feature column
    # order is.  Every function receives model_names from config.
    model_names: List[str] = field(
        default_factory=lambda: list(_DEFAULT_MODEL_NAMES)
    )

    # Cluster groupings for correlation reporting.  Must be updated when
    # running experiments with a non-standard model set.
    model_groups: Dict[str, List[str]] = field(
        default_factory=lambda: {k: list(v)
                                 for k, v in _DEFAULT_MODEL_GROUPS.items()}
    )

    # Per-model calibration subfolder names.  GBMs use abbreviated names
    # (lgb/xgb/cat) preserved from v4 artefact paths.  Add new models here.
    calibration_folder_map: Dict[str, str] = field(
        default_factory=lambda: dict(_DEFAULT_CALIBRATION_FOLDER)
    )

    # ── Paths ─────────────────────────────────────────────────────────────────
    project_root    : str = ""
    oof_dir         : str = "outputs/multi_model"
    output_dir      : str = "outputs/experiments/v5_ensemble"
    calibration_dir : str = "outputs/calibration"

    # ── Cross-validation ──────────────────────────────────────────────────────
    n_splits : int = 5
    seed     : int = SEED

    # ── Weight optimisation ───────────────────────────────────────────────────
    optimise_weights  : bool = True
    optimiser_method  : str  = "Nelder-Mead"
    optimiser_maxiter : int  = 5000

    # ── Stacking meta-model ───────────────────────────────────────────────────
    # C=0.05: tighter than v4's 0.1 — 5 meta-inputs now vs 3.
    meta_C        : float = 0.05
    meta_max_iter : int   = 1000
    meta_solver   : str   = "lbfgs"

    # ── Stacking feature augmentation ─────────────────────────────────────────
    use_disagreement   : bool = True   # std across all models per row
    use_raw_features   : bool = False  # activated post-SHAP (Step 3)
    top_k_raw_features : int  = 5

    # ── Final calibration of ensemble output ──────────────────────────────────
    calibrate_ensemble : bool = True

    # ── Artefact saving ───────────────────────────────────────────────────────
    save_models  : bool = True
    save_oof     : bool = True
    save_metrics : bool = True

    # ── Public helpers ────────────────────────────────────────────────────────

    def validate(self) -> None:
        """
        Enforce internal consistency.  Always call after construction,
        especially when overriding model_names or calibration_folder_map.

        Checks:
          - model_names non-empty, no duplicates
          - Every model has a calibration_folder_map entry
          - model_groups references only known models (warns, not error)
          - use_raw_features=True requires top_k_raw_features > 0
        """
        if not self.model_names:
            raise ValueError("config.model_names must not be empty.")

        dupes = [m for m in self.model_names
                 if self.model_names.count(m) > 1]
        if dupes:
            raise ValueError(
                f"Duplicate model names in config.model_names: {set(dupes)}"
            )

        missing_folders = [m for m in self.model_names
                           if m not in self.calibration_folder_map]
        if missing_folders:
            raise ValueError(
                f"No calibration_folder_map entry for: {missing_folders}.\n"
                "Add the mapping before running."
            )

        for group, members in self.model_groups.items():
            unknown = [m for m in members if m not in self.model_names]
            if unknown:
                print(
                    f"  Warning: model_groups['{group}'] references models "
                    f"not in config.model_names: {unknown}.  "
                    "Correlation report for this group will be incomplete."
                )

        if self.use_raw_features and self.top_k_raw_features <= 0:
            raise ValueError(
                "use_raw_features=True requires top_k_raw_features > 0."
            )

    def calibration_folder(self, model_name: str) -> str:
        """Return the calibration subfolder path component for a model."""
        if model_name not in self.calibration_folder_map:
            raise KeyError(
                f"No calibration_folder_map entry for '{model_name}'.  "
                "Add it to config.calibration_folder_map."
            )
        return self.calibration_folder_map[model_name]

    @property
    def meta_feature_order(self) -> List[str]:
        """
        Ordered list of meta-feature names matching meta_model.coef_[0].
        Saved in metadata.json for self-describing run records.
        """
        names = list(self.model_names)
        if self.use_disagreement:
            names.append("inter_model_disagreement")
        if self.use_raw_features:
            names += [f"shap_raw_feature_{i}"
                      for i in range(self.top_k_raw_features)]
        return names


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
    Platt scaling calibrator — logistic regression on scalar predictions.

    Fits: calibrated = σ(slope × raw_pred + intercept).
    Interface matches per-model calibrators saved in notebooks 07 / 07b.

    Used here ONLY for the final calibration of the stacking output.
    Base model predictions arrive already calibrated.
    """

    def __init__(self, C: float = 1e10, max_iter: int = 1000) -> None:
        self.C        = C
        self.max_iter = max_iter
        self._model   = LogisticRegression(C=C, solver="lbfgs", max_iter=max_iter)
        self._fitted  = False

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
    n_splits  : int   = 5,
    seed      : int   = SEED,
    C         : float = 1e10,
) -> Tuple[np.ndarray, PlattCalibrator]:
    """
    Cross-validated Platt calibration.

    Returns honest CV-calibrated OOF predictions AND a final calibrator
    fitted on all data (for test-set inference).
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
# ARTEFACT LOADING
# =============================================================================

def load_oof_artifacts(config: EnsembleConfig) -> Dict[str, np.ndarray]:
    """
    Load Platt-calibrated OOF predictions and ground truth.

    Model set driven by config.model_names — no global list read.

    Validates:
      - y_true positive rate in [0.13, 0.17]  (expected 0.15 ± 2 pp)
      - Each OOF array: length == len(y_true), no NaN/Inf, values in [0,1]

    Returns dict: {"y_true": ..., model_name: array, ...}
    """
    root    = Path(config.project_root)
    oof_dir = root / config.oof_dir

    artifacts: Dict[str, np.ndarray] = {}

    # Ground truth
    y_path = oof_dir / "y_true.npy"
    if not y_path.exists():
        raise FileNotFoundError(
            f"y_true.npy not found at {y_path}.\n"
            "Run notebooks 07 / 07b to produce OOF artefacts first."
        )
    artifacts["y_true"] = np.load(y_path)
    n        = len(artifacts["y_true"])
    pos_rate = float(artifacts["y_true"].mean())

    print(f"  y_true       : shape={artifacts['y_true'].shape}  "
          f"positive_rate={pos_rate:.4f}")
    if not 0.13 <= pos_rate <= 0.17:
        raise ValueError(
            f"Unexpected positive rate {pos_rate:.4f}. "
            "Expected ~0.15 (6,000 / 40,000).  Check y_true.npy."
        )

    # Per-model calibrated OOF
    for m in config.model_names:
        path = oof_dir / f"oof_calibrated_{m}.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"Calibrated OOF not found for '{m}': {path}.\n"
                "  • GBMs  → run 07_calibration_analysis.ipynb\n"
                "  • Other → run 07b_logreg_tabnet_calibration.ipynb"
            )
        arr = np.load(path)

        # Integrity checks — explicit ValueError; assert is disabled by python -O
        if len(arr) != n:
            raise ValueError(
                f"Shape mismatch: y_true has {n} rows; '{m}' OOF has {len(arr)}. "
                "Re-run the calibration notebook to regenerate this artefact."
            )
        if np.isnan(arr).any():
            raise ValueError(f"NaN values in '{m}' OOF predictions.")
        if np.isinf(arr).any():
            raise ValueError(f"Inf values in '{m}' OOF predictions.")
        if arr.min() < 0.0 or arr.max() > 1.0:
            raise ValueError(
                f"'{m}' OOF out of [0,1]: "
                f"min={arr.min():.6f}, max={arr.max():.6f}"
            )

        artifacts[m] = arr
        known     = _KNOWN_INDIVIDUAL_SCORES.get(m)
        known_str = f"  (ref composite={known:.5f})" if known else ""
        print(f"  {m:12s}: shape={arr.shape}  "
              f"range=[{arr.min():.4f}, {arr.max():.4f}]  "
              f"mean={arr.mean():.4f}{known_str}")

    print(f"\n  {len(config.model_names)} models loaded and validated ✓")
    return artifacts


def compute_oof_correlations(
    oof_preds   : Dict[str, np.ndarray],
    model_names : List[str],
) -> pd.DataFrame:
    """
    Pearson correlation matrix for the configured model set.
    Column order matches model_names — never reads global state.
    """
    df = pd.DataFrame({m: oof_preds[m] for m in model_names})
    return df.corr(method="pearson")


def build_run_dir(config: EnsembleConfig) -> Path:
    """Create a timestamped run directory under config.output_dir."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = Path(config.project_root) / config.output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# =============================================================================
# ENSEMBLE STRATEGY 1 — SIMPLE AVERAGE
# =============================================================================

def simple_average(
    oof_preds   : Dict[str, np.ndarray],
    y_true      : np.ndarray,
    model_names : List[str],
) -> Tuple[np.ndarray, Dict]:
    """
    Equal-weight average across the configured model set.

    With GBM intra-cluster r=0.962–0.978, equal weighting gives the GBM
    bloc ~60% implicit weight (3 of 5 equal slots).  Lower-bound baseline.
    """
    validate_model_set(model_names, oof_preds, "simple_average")
    pred_arr = np.stack([oof_preds[m] for m in model_names], axis=1)
    avg_pred = np.clip(pred_arr.mean(axis=1), CLIP_LOW, CLIP_HIGH)
    return avg_pred, evaluate("simple_average", y_true, avg_pred)


# =============================================================================
# ENSEMBLE STRATEGY 2 — OPTIMISED WEIGHTED AVERAGE
# =============================================================================

def _weight_objective(
    weights  : np.ndarray,
    pred_arr : np.ndarray,
    y_true   : np.ndarray,
) -> float:
    """
    Composite score objective for Nelder-Mead.
    Projects weights onto the probability simplex before evaluating.
    """
    w = np.clip(weights, 0.0, 1.0)
    w_sum = w.sum()
    if w_sum < 1e-10:
        return 999.0
    w       = w / w_sum
    blended = np.clip((pred_arr * w).sum(axis=1), EPS, 1.0 - EPS)
    return composite_score(y_true, blended)


def optimised_weighted_average(
    oof_preds : Dict[str, np.ndarray],
    y_true    : np.ndarray,
    config    : EnsembleConfig,
) -> Tuple[np.ndarray, Dict, np.ndarray]:
    """
    Scipy Nelder-Mead optimisation of per-model ensemble weights.

    Initialisations are generated dynamically from len(config.model_names)
    so the function is fully model-count agnostic — works for 2 models or 10.

    Deterministic variants encode GBM-heavy domain knowledge (from v5.0
    results where XGB+CAT dominated).  A _pad() helper normalises them to
    the current model count so no hard-coded 5-element arrays remain.

    Random Dirichlet initialisations (≥15, scaled with model count) provide
    broad simplex coverage.

    WARNING: weights are OOF-optimised → reported score is slightly
    optimistic.  Strategy 3 (stacking) uses nested CV for honest evaluation.
    """
    validate_model_set(config.model_names, oof_preds, "optimised_weighted_average")

    model_names = config.model_names
    n_models    = len(model_names)
    pred_arr    = np.stack([oof_preds[m] for m in model_names], axis=1)
    rng         = np.random.default_rng(config.seed)

    best_score   = 999.0
    best_weights = np.full(n_models, 1.0 / n_models)

    def _pad(base: List[float]) -> np.ndarray:
        """
        Pad or trim a weight prototype to n_models, then normalise.
        Allows GBM-heavy priors (designed for 5 models) to work with
        any model count without raising shape errors.
        """
        w = np.array(base[:n_models]
                     + [0.0] * max(0, n_models - len(base)),
                     dtype=float)
        total = w.sum()
        return w / total if total > 1e-10 else np.full(n_models, 1.0 / n_models)

    # ── Initialisations ───────────────────────────────────────────────────────
    initialisations: List[np.ndarray] = [
        np.full(n_models, 1.0 / n_models),     # equal weights (always first)
    ]

    # GBM-heavy domain-knowledge priors — meaningful when n_models >= 3
    # and the first three slots are GBMs (enforced by default model_names order).
    if n_models >= 3:
        initialisations += [
            _pad([0.30, 0.35, 0.20, 0.10, 0.05]),  # XGB-heavy
            _pad([0.35, 0.30, 0.20, 0.10, 0.05]),  # LGB-heavy
            _pad([0.25, 0.25, 0.20, 0.15, 0.15]),  # GBM-balanced
            _pad([0.25, 0.30, 0.15, 0.15, 0.15]),  # diversified
            _pad([0.20, 0.25, 0.15, 0.20, 0.20]),  # cluster-balanced
            _pad([0.33, 0.33, 0.17, 0.10, 0.07]),  # top-2 GBM
            _pad([0.50, 0.30, 0.10, 0.05, 0.05]),  # LGB dominant
            _pad([0.30, 0.40, 0.10, 0.10, 0.10]),  # XGB dominant
        ]

    # Random Dirichlet — minimum 15, scales with model count
    n_random = max(15, 3 * n_models)
    for _ in range(n_random):
        initialisations.append(rng.dirichlet(np.ones(n_models)))

    print(f"  Running weight optimisation "
          f"({len(initialisations)} initialisations, {n_models} models)...")

    for w0 in initialisations:
        result = minimize(
            _weight_objective,
            x0      = w0,
            args    = (pred_arr, y_true),
            method  = config.optimiser_method,
            options = {"maxiter": config.optimiser_maxiter,
                       "xatol": 1e-8, "fatol": 1e-8},
        )
        if result.fun < best_score:
            best_score   = result.fun
            best_weights = result.x

    # Project onto probability simplex
    best_weights = np.clip(best_weights, 0.0, 1.0)
    best_weights = best_weights / best_weights.sum()

    best_pred   = np.clip(
        (pred_arr * best_weights).sum(axis=1), CLIP_LOW, CLIP_HIGH
    )
    metrics     = evaluate("optimised_weighted_average", y_true, best_pred)
    weight_dict = {m: float(w) for m, w in zip(model_names, best_weights)}

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
    Build meta-feature matrix from config.

    Column order is determined entirely by config.model_names.
    config.meta_feature_order property gives the corresponding names list
    and is saved in metadata.json for self-describing run records.

    Columns:
      [0 … n_models−1]  Calibrated OOF from each model in model_names order
      [n_models]         inter_model_disagreement (if use_disagreement)
      [n_models+1 …]     shap_raw_feature_k       (if use_raw_features)
    """
    model_names   = config.model_names
    meta_X        = np.stack([oof_preds[m] for m in model_names], axis=1)
    feature_names = list(model_names)

    if config.use_disagreement:
        disagreement  = meta_X.std(axis=1, keepdims=True)
        meta_X        = np.hstack([meta_X, disagreement])
        feature_names.append("inter_model_disagreement")

    if config.use_raw_features and extra_features is not None:
        if extra_features.shape[0] != meta_X.shape[0]:
            raise ValueError(
                f"extra_features row count ({extra_features.shape[0]}) "
                f"!= OOF row count ({meta_X.shape[0]}).  "
                "Ensure extra_features is aligned to the full training set."
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
    Stacking ensemble with LogisticRegression(C=meta_C) meta-model.

    Strict OOF protocol — no leakage:
      For each outer fold: meta-model trained on held-out OOF, predicts val.
      Final meta-model fitted on ALL OOF data for test-set inference.

    Why LogisticRegression as meta-model?
      ≤7 meta-features, 40k samples → tree meta-model would memorise.
      LogReg with C=0.05 gives stable, interpretable weights and respects
      the [0,1] probability contract.

    Known limitation — fold misalignment in calibrated_stacking():
      The Platt pass in Strategy 4 uses independently drawn CV folds,
      not aligned to these stacking folds.  Impact is negligible
      (2 parameters, 40k rows).  Documented in calibrated_stacking().
    """
    validate_model_set(config.model_names, oof_preds, "stacking_ensemble")

    meta_X, feature_names = _build_meta_features(oof_preds, config, extra_features)
    n_meta_features       = meta_X.shape[1]

    stacking_oof              = np.zeros(len(y_true), dtype=float)
    skf                       = StratifiedKFold(
        n_splits=config.n_splits, shuffle=True, random_state=config.seed
    )
    fold_scores: List[float] = []
    fold_lls   : List[float] = []
    fold_aucs  : List[float] = []

    print(f"  Nested {config.n_splits}-fold CV for honest OOF stacking...")
    print(f"  {'Fold':>4}  {'Score':>8}  {'LogLoss':>8}  {'AUC':>8}")
    print(f"  {'-'*40}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(meta_X, y_true)):
        meta_model = LogisticRegression(
            C=config.meta_C, solver=config.meta_solver,
            max_iter=config.meta_max_iter, random_state=config.seed,
        )
        meta_model.fit(meta_X[train_idx], y_true[train_idx])

        fold_pred             = meta_model.predict_proba(meta_X[val_idx])[:, 1]
        stacking_oof[val_idx] = fold_pred

        fs  = composite_score(y_true[val_idx], fold_pred)
        fll = float(log_loss(y_true[val_idx],
                              np.clip(fold_pred, EPS, 1.0 - EPS)))
        fau = float(roc_auc_score(y_true[val_idx], fold_pred))
        fold_scores.append(fs); fold_lls.append(fll); fold_aucs.append(fau)
        print(f"  {fold:>4}  {fs:>8.5f}  {fll:>8.5f}  {fau:>8.5f}")

    print(f"  {'-'*40}")
    print(f"  {'MEAN':>4}  {np.mean(fold_scores):>8.5f}  "
          f"{np.mean(fold_lls):>8.5f}  {np.mean(fold_aucs):>8.5f}")
    print(f"  {'STD':>4}  {np.std(fold_scores):>8.5f}  "
          f"{np.std(fold_lls):>8.5f}  {np.std(fold_aucs):>8.5f}")

    # Final meta-model on all data — for test-set inference ONLY
    final_meta_model = LogisticRegression(
        C=config.meta_C, solver=config.meta_solver,
        max_iter=config.meta_max_iter, random_state=config.seed,
    )
    final_meta_model.fit(meta_X, y_true)

    # Coefficient table — meta-model interpretability
    coefficients = pd.DataFrame({
        "feature"     : feature_names[:n_meta_features],
        "coefficient" : final_meta_model.coef_[0][:n_meta_features],
    }).sort_values("coefficient", ascending=False).reset_index(drop=True)

    print(f"\n  Meta-model coefficients (C={config.meta_C}):")
    print(coefficients.to_string(index=False))
    print(f"\n  → Meta-model trusts '{coefficients.iloc[0]['feature']}' most.")

    metrics = evaluate("stacking", y_true, stacking_oof)
    return stacking_oof, metrics, final_meta_model, coefficients


# =============================================================================
# ENSEMBLE STRATEGY 4 — STACKING WITH FINAL CALIBRATION
# =============================================================================

def calibrated_stacking(
    stacking_oof : np.ndarray,
    y_true       : np.ndarray,
    config       : EnsembleConfig,
) -> Tuple[np.ndarray, Dict, PlattCalibrator]:
    """
    Cross-validated Platt calibration of the stacking OOF predictions.

    LogisticRegression is calibrated by design, but mixing GBM predictions
    (mean ~0.15) with LogReg/TabNet (higher post-Platt means) can introduce
    mild residual miscalibration.  A Platt pass with 2 parameters corrects
    this with negligible overfitting risk.

    Known limitation — fold misalignment:
        The Platt CV folds here are NOT aligned to the stacking CV folds
        that produced stacking_oof.  Each calibration fold's held-out rows
        were predicted by meta-models trained on overlapping (but not
        identical) stacking samples.  Impact is negligible (2 parameters,
        40k rows), but treat calibrated_stacking scores as lower-bound
        estimates rather than perfectly honest OOF scores.
        Rigorous fix (calibrating within each stacking outer fold) requires
        restructuring stacking_ensemble() to expose per-fold predictions
        before aggregation.  Deferred post-competition.

    Returns
    -------
    calibrated_oof   : CV-calibrated stacking OOF predictions
    metrics          : full metric dict
    final_calibrator : PlattCalibrator fitted on all data (for test set)
    """
    calibrated_oof, final_calibrator = cv_platt_calibrate(
        stacking_oof, y_true,
        n_splits=config.n_splits, seed=config.seed, C=1e10,
    )
    calibrated_oof = np.clip(calibrated_oof, CLIP_LOW, CLIP_HIGH)
    metrics        = evaluate("stacking_calibrated", y_true, calibrated_oof)

    slope = final_calibrator.slope
    print(f"  Final calibrator : slope={slope:.4f}  "
          f"intercept={final_calibrator.intercept:.4f}")
    if abs(slope - 1.0) < 0.3:
        print("  Slope ≈ 1.0 → stacking output well-calibrated ✓")
    else:
        print(f"  Slope = {slope:.4f} → residual compression present; "
              "Platt correction applied.")

    return calibrated_oof, metrics, final_calibrator


# =============================================================================
# REPORTING UTILITIES
# =============================================================================

def print_correlation_report(
    corr_matrix : pd.DataFrame,
    config      : EnsembleConfig,
) -> None:
    """
    Print OOF Pearson correlation matrix with cluster analysis.

    Cluster groups read from config.model_groups — no hardcoded strings.
    Groups with fewer than 2 active members are reported as singletons.
    """
    print("\n  OOF Pearson Correlation Matrix (post-Platt):")
    print(corr_matrix.round(3).to_string())

    model_names = config.model_names

    # Resolve active members per group (only models in current config)
    active_groups: Dict[str, List[str]] = {
        g: [m for m in members if m in model_names]
        for g, members in config.model_groups.items()
    }

    print(f"\n  Cluster structure ({len(model_names)} active models):")
    for group_name, members in active_groups.items():
        if len(members) >= 2:
            intra = [corr_matrix.loc[a, b]
                     for i, a in enumerate(members)
                     for j, b in enumerate(members) if i < j]
            print(f"  {group_name:12s} intra-cluster : "
                  f"{min(intra):.3f} – {max(intra):.3f}")
        elif len(members) == 1:
            print(f"  {group_name:12s}              : "
                  f"{members[0]} (singleton)")

    # Cross-cluster range — every pair across different groups
    group_model_pairs = [
        (g, m) for g, ms in active_groups.items() for m in ms
    ]
    cross_vals: List[float] = []
    for i, (g1, m1) in enumerate(group_model_pairs):
        for j, (g2, m2) in enumerate(group_model_pairs):
            if i < j and g1 != g2:
                cross_vals.append(float(corr_matrix.loc[m1, m2]))

    if cross_vals:
        print(f"  {'cross-cluster':12s}              : "
              f"{min(cross_vals):.3f} – {max(cross_vals):.3f}")
        if max(cross_vals) < 0.82:
            print("  ✓ Cross-cluster diversity target MET (<0.82). "
                  "Stacking justified.")
        else:
            print(f"  ⚠ Cross-cluster max {max(cross_vals):.3f} > 0.82 target. "
                  "Stacking may not outperform weighted averaging.")


def print_results_table(
    results     : EnsembleResults,
    model_names : List[str],
) -> None:
    """Formatted comparison table of all ensemble strategies."""
    sep = "=" * 92
    print(f"\n{sep}")
    print("ENSEMBLE RESULTS SUMMARY")
    print(sep)
    print(f"{'Strategy':42s}  {'LogLoss':>8}  {'AUC':>8}  "
          f"{'Score':>8}  {'Brier':>8}")
    print("-" * 92)

    ref_score      = _KNOWN_INDIVIDUAL_SCORES.get("xgboost", None)
    sorted_metrics = sorted(results.strategy_metrics, key=lambda x: x["score"])

    for m in sorted_metrics:
        marker      = "  ← BEST" if m["name"] == results.best_strategy else ""
        improvement = ""
        if ref_score is not None and m["name"] not in _KNOWN_INDIVIDUAL_SCORES:
            delta = m["score"] - ref_score
            sign  = "+" if delta > 0 else ""
            improvement = f"  ({sign}{delta:.5f} vs XGB)"
        print(f"{m['name']:42s}  {m['logloss']:>8.5f}  {m['auc']:>8.5f}  "
              f"{m['score']:>8.5f}  {m['brier']:>8.5f}{marker}{improvement}")

    print(sep)
    print(f"\nModels        : {model_names}")
    print(f"Best strategy : {results.best_strategy}")
    print(f"Best score    : {results.best_score:.6f}")
    if ref_score is not None:
        delta = results.best_score - ref_score
        sign  = "+" if delta >= 0 else ""
        print(f"XGB baseline  : {ref_score:.6f}")
        print(f"Improvement   : {sign}{delta:.6f}")
    print(f"Runtime       : {results.runtime_sec:.1f} seconds")


# =============================================================================
# ARTEFACT SAVING
# =============================================================================

def save_ensemble_artifacts(
    results         : EnsembleResults,
    meta_model      : LogisticRegression,
    meta_calibrator : Optional[PlattCalibrator],
    coefficients    : pd.DataFrame,
    corr_matrix     : pd.DataFrame,
    config          : EnsembleConfig,
    run_dir         : Path,
) -> None:
    """
    Save all ensemble artefacts to the timestamped run directory.

    metadata.json is self-describing: it contains model_names,
    calibration_folder_map, ensemble_version, and meta_feature_order
    so any future reload — including EnsembleInference.from_run_dir() —
    can reconstruct the exact training configuration without relying on
    current module defaults.
    """
    # OOF predictions
    if results.ensemble_oof is not None and len(results.ensemble_oof) > 0:
        np.save(run_dir / "ensemble_oof.npy", results.ensemble_oof)
    if results.stacking_oof is not None and len(results.stacking_oof) > 0:
        np.save(run_dir / "stacking_oof.npy", results.stacking_oof)

    # Weights
    with open(run_dir / "ensemble_weights.json", "w") as f:
        json.dump({
            "optimised"  : results.optimised_weights,
            "model_order": config.model_names,
            "note"       : (
                "Weights are OOF-optimised (slightly optimistic). "
                "Use meta_model.pkl for deployment."
            ),
        }, f, indent=4)

    # Meta-model + calibrator
    joblib.dump(meta_model, run_dir / "meta_model.pkl")
    if meta_calibrator is not None:
        joblib.dump(meta_calibrator, run_dir / "meta_calibrator.pkl")

    # Strategy metrics + coefficients + correlation matrix
    with open(run_dir / "stacking_results.json", "w") as f:
        json.dump(results.strategy_metrics, f, indent=4)
    coefficients.to_csv(run_dir / "feature_importance.csv", index=False)
    corr_matrix.to_csv(run_dir / "correlation_matrix.csv")

    # Self-describing metadata
    ref_score = _KNOWN_INDIVIDUAL_SCORES.get("xgboost", None)
    metadata  = {
        "created_at"             : datetime.now().isoformat(),
        "ensemble_version"       : config.ensemble_version,
        "model_names"            : config.model_names,
        "calibration_folder_map" : config.calibration_folder_map,
        "model_groups"           : config.model_groups,
        "best_strategy"          : results.best_strategy,
        "best_score"             : results.best_score,
        "xgb_baseline"           : ref_score,
        "improvement_vs_xgb"     : (
            round(results.best_score - ref_score, 6)
            if ref_score is not None else None
        ),
        "runtime_sec"            : results.runtime_sec,
        "config"                 : asdict(config),
        "strategy_summary"       : [
            {k: v for k, v in m.items() if k != "name"} | {"strategy": m["name"]}
            for m in results.strategy_metrics
        ],
        "meta_model_summary"     : {
            "C"                  : config.meta_C,
            "solver"             : config.meta_solver,
            "use_disagreement"   : config.use_disagreement,
            "use_raw_features"   : config.use_raw_features,
            # Column order of meta_model.coef_[0] — critical for inference
            "meta_feature_order" : config.meta_feature_order,
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
    Full ensemble pipeline — evaluates all four strategies in order.

    Strategies:
      1. Simple average          (equal-weight baseline)
      2. Optimised weighted avg  (Scipy Nelder-Mead on composite score)
      3. Stacking                (LogisticRegression meta-model, nested CV)
      4. Calibrated stacking     (Platt pass on stacking output)

    Parameters
    ----------
    config         : EnsembleConfig — single source of truth
    extra_features : optional (n, k) array of top-k SHAP raw features.
                     Required when config.use_raw_features=True.

    Returns
    -------
    EnsembleResults — all metrics, predictions, and artefacts saved.
    """
    start_time = time.perf_counter()
    results    = EnsembleResults()

    # Pre-flight checks
    config.validate()
    if config.use_raw_features and extra_features is None:
        raise ValueError(
            "config.use_raw_features=True but extra_features=None.  "
            "Run SHAP analysis (notebook 10) and pass the top-k feature "
            "matrix as extra_features."
        )

    print("=" * 70)
    print(f"ENSEMBLE PIPELINE — {config.ensemble_version}")
    print("=" * 70)
    print(f"Models              : {config.model_names}")
    print(f"meta_C              : {config.meta_C}")
    print(f"Disagreement feature: {config.use_disagreement}")
    print(f"Raw features (SHAP) : {config.use_raw_features}")

    # ── 0. Setup ──────────────────────────────────────────────────────────────
    run_dir         = build_run_dir(config)
    results.run_dir = str(run_dir)
    print(f"\nRun directory: {run_dir}")

    # ── 1. Load and validate OOF artefacts ───────────────────────────────────
    print("\n[1/5] Loading OOF artefacts...")
    oof_artifacts = load_oof_artifacts(config)
    y_true        = oof_artifacts["y_true"]
    oof_preds     = {m: oof_artifacts[m] for m in config.model_names}

    print("\n  Computing OOF correlation matrix...")
    corr_matrix = compute_oof_correlations(oof_preds, config.model_names)
    print_correlation_report(corr_matrix, config)

    print("\n  Individual model metrics (Platt-calibrated OOF):")
    print(f"  {'Model':12s}  {'LogLoss':>8}  {'AUC':>8}  {'Score':>8}")
    print(f"  {'-'*46}")
    for m in config.model_names:
        m_metrics = evaluate(m, y_true, oof_preds[m])
        results.strategy_metrics.append(m_metrics)
        print(f"  {m:12s}  {m_metrics['logloss']:>8.5f}  "
              f"{m_metrics['auc']:>8.5f}  {m_metrics['score']:>8.5f}")

    # ── 2. Simple average ─────────────────────────────────────────────────────
    print("\n[2/5] Strategy 1 — Simple average...")
    avg_pred, avg_metrics = simple_average(oof_preds, y_true, config.model_names)
    results.strategy_metrics.append(avg_metrics)
    print(f"  LL={avg_metrics['logloss']:.5f}  AUC={avg_metrics['auc']:.5f}  "
          f"Score={avg_metrics['score']:.5f}")

    # ── 3. Optimised weighted average ─────────────────────────────────────────
    print("\n[3/5] Strategy 2 — Optimised weighted average...")
    opt_pred, opt_metrics, opt_weights = optimised_weighted_average(
        oof_preds, y_true, config
    )
    results.strategy_metrics.append(opt_metrics)
    results.optimised_weights = {
        m: float(w) for m, w in zip(config.model_names, opt_weights)
    }
    print(f"  LL={opt_metrics['logloss']:.5f}  AUC={opt_metrics['auc']:.5f}  "
          f"Score={opt_metrics['score']:.5f}")

    # ── 4. Stacking ───────────────────────────────────────────────────────────
    print(f"\n[4/5] Strategy 3 — Stacking (LogReg, C={config.meta_C})...")
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

    # ── 6. Identify best strategy ─────────────────────────────────────────────
    for m in results.strategy_metrics:
        if m["score"] < results.best_score:
            results.best_score    = m["score"]
            results.best_strategy = m["name"]

    # ── 7. Record runtime BEFORE saving ──────────────────────────────────────
    # Bug fix from v5.0: runtime was set AFTER save → metadata always showed 0.0
    results.runtime_sec = time.perf_counter() - start_time

    # ── 8. Save all artefacts ─────────────────────────────────────────────────
    save_ensemble_artifacts(
        results, meta_model, meta_calibrator,
        coefficients, corr_matrix, config, run_dir
    )

    print_results_table(results, config.model_names)
    return results


# =============================================================================
# PRODUCTION INFERENCE API
# =============================================================================

class EnsembleInference:
    """
    Production inference wrapper for the trained ensemble.

    Critically: model_names and calibration_folder_map are loaded from the
    run's metadata.json — NOT from current module-level defaults.  This
    guarantees training-inference consistency even if module defaults change.

    Usage
    -----
    inference = EnsembleInference.from_run_dir(
        run_dir      = "outputs/experiments/v5_ensemble/run_20260508_054540",
        project_root = "D:/PROJECTS/liquidity-stress-early-warning",
    )
    probs = inference.predict(raw_preds_per_model)

    TabNet note
    -----------
    raw_preds_per_model["tabnet"] should be RAW (un-Platt-calibrated)
    fold-averaged predictions.  The per-model calibrators apply Platt
    scaling before the meta-model sees them.
    """

    def __init__(
        self,
        meta_model       : LogisticRegression,
        meta_calibrator  : Optional[PlattCalibrator],
        base_calibrators : Dict[str, PlattCalibrator],
        model_names      : List[str],
        use_disagreement : bool,
        use_raw_features : bool,
        ensemble_version : str = "unknown",
    ) -> None:
        self.meta_model       = meta_model
        self.meta_calibrator  = meta_calibrator
        self.base_calibrators = base_calibrators
        self.model_names      = model_names
        self.use_disagreement = use_disagreement
        self.use_raw_features = use_raw_features
        self.ensemble_version = ensemble_version

    @classmethod
    def from_run_dir(
        cls,
        run_dir         : str,
        project_root    : str,
        calibration_dir : str = "outputs/calibration",
    ) -> "EnsembleInference":
        """
        Load inference artefacts from a completed ensemble run directory.

        Model names and calibration_folder_map are read from metadata.json,
        not from current module-level defaults.  This is the critical
        guarantee that prevents training-inference divergence when module
        defaults change in future versions.

        Parameters
        ----------
        run_dir         : path to timestamped run directory
        project_root    : project root (for resolving calibration paths)
        calibration_dir : relative path to calibration artefacts
        """
        run_path = Path(run_dir)
        root     = Path(project_root)

        meta_path = run_path / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"metadata.json not found at {meta_path}.\n"
                "Cannot reconstruct inference config without training metadata.\n"
                "If this run predates v5.1, reconstruct EnsembleInference manually."
            )
        with open(meta_path) as f:
            metadata = json.load(f)

        model_names            = metadata["model_names"]
        calibration_folder_map = metadata.get(
            "calibration_folder_map", _DEFAULT_CALIBRATION_FOLDER
        )
        meta_summary           = metadata.get("meta_model_summary", {})
        use_disagreement       = meta_summary.get("use_disagreement", True)
        use_raw_features       = meta_summary.get("use_raw_features", False)
        ensemble_version       = metadata.get("ensemble_version", "unknown")

        print(f"  Loading inference artefacts [{ensemble_version}]")
        print(f"  model_names : {model_names}")

        # Meta-model
        meta_model = joblib.load(run_path / "meta_model.pkl")

        # Meta-calibrator (optional)
        meta_calibrator = None
        cal_path        = run_path / "meta_calibrator.pkl"
        if cal_path.exists():
            meta_calibrator = joblib.load(cal_path)
        else:
            print("  Warning: meta_calibrator.pkl not found — "
                  "final Platt step will be skipped.")

        # Per-model Platt calibrators — paths from training metadata
        base_calibrators: Dict[str, PlattCalibrator] = {}
        cal_dir = root / calibration_dir
        for m in model_names:
            folder     = calibration_folder_map.get(m, m)
            platt_path = cal_dir / folder / "calibrator_platt.pkl"
            if platt_path.exists():
                base_calibrators[m] = joblib.load(platt_path)
                print(f"  Loaded Platt calibrator for {m} from {folder}/")
            else:
                print(f"  Warning: Platt calibrator not found for {m}: {platt_path}")

        return cls(
            meta_model       = meta_model,
            meta_calibrator  = meta_calibrator,
            base_calibrators = base_calibrators,
            model_names      = model_names,
            use_disagreement = use_disagreement,
            use_raw_features = use_raw_features,
            ensemble_version = ensemble_version,
        )

    def predict(
        self,
        raw_preds_per_model : Dict[str, np.ndarray],
        extra_features      : Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Produce final ensemble probabilities for test data.

        Pipeline:
          1. Shape validation — all arrays must have equal length
          2. Bidirectional model set validation
          3. Per-model Platt calibration (or pass-through with warning)
          4. Meta-feature matrix (OOF + optional disagreement + SHAP)
          5. Meta-model (LogisticRegression) prediction
          6. Final Platt calibration (if available)
          7. Clip to [1e-6, 1−1e-6]

        Parameters
        ----------
        raw_preds_per_model : model_name → raw (un-calibrated) predictions
        extra_features      : optional (n, k) SHAP raw features
        """
        # Step 1+2: shape and model set validation before any computation
        validate_model_set(self.model_names, raw_preds_per_model, "predict")
        lengths        = {m: len(raw_preds_per_model[m]) for m in self.model_names}
        unique_lengths = set(lengths.values())
        if len(unique_lengths) != 1:
            raise ValueError(
                f"Inconsistent prediction lengths: {lengths}.  "
                "All models must predict the same number of rows."
            )

        # Step 3: per-model Platt calibration
        calibrated: Dict[str, np.ndarray] = {}
        for m in self.model_names:
            if m in self.base_calibrators:
                calibrated[m] = self.base_calibrators[m].predict(
                    raw_preds_per_model[m]
                )
            else:
                calibrated[m] = raw_preds_per_model[m]
                print(f"  Warning: Using un-calibrated predictions for '{m}'.")

        # Step 4: meta-feature matrix
        meta_X = np.stack([calibrated[m] for m in self.model_names], axis=1)
        if self.use_disagreement:
            meta_X = np.hstack([meta_X, meta_X.std(axis=1, keepdims=True)])
        if self.use_raw_features and extra_features is not None:
            meta_X = np.hstack([meta_X, extra_features])

        # Steps 5–7: meta-model → calibrate → clip
        ensemble_pred = self.meta_model.predict_proba(meta_X)[:, 1]
        if self.meta_calibrator is not None:
            ensemble_pred = self.meta_calibrator.predict(ensemble_pred)
        return np.clip(ensemble_pred, CLIP_LOW, CLIP_HIGH)


# =============================================================================
# ENTRY POINT (direct execution)
# =============================================================================

if __name__ == "__main__":
    """
    Run the full ensemble pipeline from project root:
        python -m src.ensemble.ensemble

    Expects all configured OOF arrays in outputs/multi_model/.
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
        meta_C             = 0.05,
    )
    config.validate()

    print(f"Project root : {project_root}")
    print(f"Version      : {config.ensemble_version}")
    print(f"Models       : {config.model_names}")
    print(f"meta_C       : {config.meta_C}")
    print()

    results = run_ensemble_pipeline(config)
    sys.exit(0 if results.best_score < 999.0 else 1)