# src/inference/predict.py
"""
Inference Pipeline — Production v2.0
=====================================
Project  : AI4EAC Liquidity Stress Early Warning (Zindi Africa)
Module   : src/inference/predict.py
Author   : Henry Otsyula
Updated  : 2026-05-12  (v1 → v2.0: full 5-model rewrite)

Change log
----------
v1   : 3-model GBM-only draft.  Hardcoded MODEL_NAMES = ["lightgbm",
       "xgboost", "catboost"].  calibrate_test_predictions() mapped
       folders with model_name[:3] — broke for logreg/tabnet.
       ensemble_predict() ran the stacking meta-model, not the
       optimised weighted average that actually won.  Never used in
       production.

v2.0 : Complete rewrite.  Key changes:
       • All 5 models: lightgbm, xgboost, catboost, logreg, tabnet
       • Each model uses its OWN fitted preprocessor (critical: logreg
         and tabnet preprocessors include a fitted StandardScaler)
       • TabNet fold models loaded via .load_model() on .zip files —
         NOT joblib (would corrupt the model state silently)
       • Calibration folder map mirrors ensemble.py v5.1 exactly:
         lightgbm→lgb, xgboost→xgb, catboost→cat, logreg→logreg,
         tabnet→tabnet
       • Submission strategy: OPTIMISED WEIGHTED AVERAGE (score=0.19144)
         NOT stacking.  Weights loaded from ensemble_weights.json
         ["optimised"] at runtime — never hardcoded.
       • LogReg weight=0.000 but predictions are still generated and
         included in the weighted sum (zero contribution is correct
         behaviour, not a bug to skip around).
       • Full provenance: run metadata read from metadata.json for
         training-inference consistency guarantee.
       • Drift detection: OOF vs test prediction distribution compared;
         warns if mean shifts >0.05 — catches silent pipeline bugs.
       • 6 Zindi compliance assertions before any file write.
       • Structured logging (not bare print) for clean notebook/CLI use.
       • No assert statements in production paths (uses if/raise).
       • All clip operations use EPS = 1e-6 consistently.

Architecture (5-layer pipeline)
---------------------------------

  Raw test CSV (30,000 × 183)
        │
        ▼
  [Layer 1] Feature Engineering
        build_features() → 825 engineered features
        identical pipeline to training; no fit, pure transform
        │
        ▼ (per model — run in isolation, never cross-contaminate)
  [Layer 2] Model-specific Preprocessing
        PreprocessingPipeline.load(run_dir/preprocessor.pkl)
        .transform() ONLY — never .fit_transform() on test data
        GBMs:   scale_features=False  (pass-through, no scaler)
        LogReg: scale_features=True   (StandardScaler applied)
        TabNet: scale_features=True   (StandardScaler applied)
        │
        ▼
  [Layer 3] Fold Prediction Averaging
        Load 5 fold models per model family
        Predict proba[:,1] per fold → mean across folds
        TabNet: model.load_model(zip_path_no_ext) — NOT joblib
        Result: raw (uncalibrated) test prediction vector per model
        │
        ▼
  [Layer 4] Platt Calibration
        Load outputs/calibration/{folder}/calibrator_platt.pkl
        calibrator.predict(raw_preds) → calibrated [0,1] vector
        Slope reference values (from OOF analysis):
          lightgbm ≈ 1.0 (well-calibrated natively)
          xgboost  ≈ 1.0
          catboost ≈ 1.0
          logreg   slope=5.47, intercept=-4.43 (36% LL improvement)
          tabnet   slope=4.37, intercept=-3.52 (21% LL improvement)
        │
        ▼
  [Layer 5] Optimised Weighted Average Ensemble
        Weights loaded from ensemble_weights.json ["optimised"]:
          lightgbm : 0.17813
          xgboost  : 0.39721
          catboost : 0.38777
          logreg   : 0.00000  ← zero weight, still computed
          tabnet   : 0.03689
        final = Σ w_i × calibrated_i, clipped to [1e-6, 1-1e-6]
        OOF composite score: 0.19144
        │
        ▼
  Zindi submission CSV (30,000 × 4)
        ID, Target, TargetLogLoss, TargetRAUC  (all three identical)

Critical invariants (must never be violated)
----------------------------------------------
  • EPS = 1e-6 in all clip and ratio operations
  • preprocessor.transform() — never fit_transform() on test
  • TabNet loaded via .load_model(), not joblib
  • Each model uses its OWN preprocessor (never shared)
  • Weights read from artefact JSON — never hardcoded in logic
  • LogReg predictions computed even though weight=0.000
  • No assert in production paths — always if/raise ValueError
  • All 6 Zindi assertions must pass before file write

Usage
-----
  # CLI
  python -m src.inference.predict \\
      --ensemble-run outputs/experiments/v5_ensemble/run_20260508_054540 \\
      --output       submissions/submission_v5_ensemble.csv \\
      --test-path    data/raw/Test.csv \\
      --verbose

  # Notebook
  from src.inference.predict import InferencePipeline, InferenceConfig

  cfg = InferenceConfig(
      ensemble_run_dir = "outputs/experiments/v5_ensemble/run_20260508_054540",
      output_path      = "submissions/submission_v5_ensemble.csv",
  )
  pipeline = InferencePipeline(cfg)
  submission = pipeline.run()
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import warnings
from sklearn.exceptions import InconsistentVersionWarning
import numpy as np
import pandas as pd

class PlattCalibrator:
    """
    Backward-compatible Platt scaling calibrator.
    Supports legacy serialized objects.
    """

    def __init__(self, slope=None, intercept=None):
        self.slope = slope
        self.intercept = intercept

    def predict(self, x):
        # Backward compatibility
        slope = getattr(self, "slope", None)
        intercept = getattr(self, "intercept", None)

        # Legacy attribute names
        if slope is None:
            slope = getattr(self, "a", None)

        if intercept is None:
            intercept = getattr(self, "b", None)

        # Fallback safety
        if slope is None:
            slope = 1.0

        if intercept is None:
            intercept = 0.0

        z = slope * x + intercept
        return 1 / (1 + np.exp(-z))
    
# ---------------------------------------------------------------------------
# Project root — resolved relative to this file, not CWD.
# Guarantees the same resolution whether called as:
#   python -m src.inference.predict       (from project root)
#   python src/inference/predict.py       (from project root)
#   imported in a notebook                (from any working directory)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Logging — structured, level-configurable.  No bare print() in production.
# ---------------------------------------------------------------------------
_LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
logging.basicConfig(format=_LOG_FORMAT, datefmt="%H:%M:%S", level=logging.INFO)
log = logging.getLogger("inference")

# ---------------------------------------------------------------------------
# Global constants — must match training pipeline exactly
# ---------------------------------------------------------------------------
EPS       : float = 1e-6
CLIP_LOW  : float = EPS
CLIP_HIGH : float = 1.0 - EPS

EXPECTED_TEST_ROWS  : int   = 30_000
EXPECTED_TRAIN_ROWS : int   = 40_000
EXPECTED_POSITIVE_RATE     : float = 0.15
DRIFT_WARN_THRESHOLD: float = 0.05   # |test_mean - oof_mean| > this → warn
N_FOLDS             : int   = 5

# Models that required StandardScaler during training.
# Their preprocessor.pkl includes a fitted scaler.
# GBMs do NOT have a scaler — applying the wrong preprocessor
# to a GBM would silently pass raw-scale data and corrupt predictions.
_SCALE_SENSITIVE_MODELS = frozenset({"logreg", "tabnet"})

# Calibration subfolder mapping — mirrors ensemble.py v5.1 exactly.
# lightgbm → lgb/ preserves the abbreviated path from v4 artefacts.
_CALIBRATION_FOLDER_MAP: Dict[str, str] = {
    "lightgbm": "lgb",
    "xgboost" : "xgb",
    "catboost": "cat",
    "logreg"  : "logreg",
    "tabnet"  : "tabnet",
}

# Model family → training stage → config file
# These are the exact run directories where fitted artefacts live.
# catboost was not Optuna-tuned, so it remains in "baseline".
_MODEL_STAGE_MAP: Dict[str, Tuple[str, str]] = {
    "lightgbm": ("v2_feature_expansion", "configs/lightgbm_tuned.yaml"),
    "xgboost" : ("v2_feature_expansion", "configs/xgboost_tuned.yaml"),
    "catboost": ("baseline",             "configs/catboost_v2.yaml"),
    "logreg"  : ("v3_extended_models",   "configs/logreg_v1.yaml"),
    "tabnet"  : ("v3_extended_models",   "configs/tabnet_v1.yaml"),
}

# Ordered model list — order determines column order in any stacked array.
# Must match config.model_names from the ensemble training run.
_DEFAULT_MODEL_ORDER: List[str] = [
    "lightgbm",
    "xgboost",
    "catboost",
    "logreg",
    "tabnet",
]

# Reference OOF scores from training (for display and sanity checks only).
_KNOWN_OOF_SCORES: Dict[str, float] = {
    "lightgbm": 0.19557,
    "xgboost" : 0.19350,
    "catboost": 0.19430,
    "logreg"  : 0.26003,
    "tabnet"  : 0.25296,
    "ensemble": 0.19144,
}


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class InferenceConfig:
    """
    Single source of truth for the inference pipeline.

    All paths and behavioural flags live here.  No function reads
    module-level globals — everything is driven by this config.

    Parameters
    ----------
    ensemble_run_dir : path to the production ensemble run directory.
        Must contain: ensemble_weights.json, metadata.json.
    output_path : where to write the Zindi submission CSV.
    test_path : path to Test.csv.  Defaults to data/raw/Test.csv.
    oof_dir : path to the multi-model OOF arrays for drift detection.
    calibration_dir : root of the per-model Platt calibrator tree.
    project_root : absolute project root.  Defaults to the value
        resolved from this file's location.
    verbose : if True, logs per-fold prediction statistics.
    dry_run : if True, runs the full pipeline but does NOT write the
        submission CSV.  Useful for validation without side effects.
    """
    ensemble_run_dir : str  = ""
    output_path      : str  = ""
    test_path        : str  = "data/raw/Test.csv"
    oof_dir          : str  = "outputs/multi_model"
    calibration_dir  : str  = "outputs/calibration"
    project_root     : str  = str(PROJECT_ROOT)
    verbose          : bool = True
    dry_run          : bool = False

    def validate(self) -> None:
        """Fail fast on missing required fields before touching disk."""
        if not self.ensemble_run_dir:
            raise ValueError(
                "InferenceConfig.ensemble_run_dir is required.\n"
                "Example: 'outputs/experiments/v5_ensemble/run_20260508_054540'"
            )
        if not self.output_path:
            raise ValueError(
                "InferenceConfig.output_path is required.\n"
                "Example: 'submissions/submission_v5_ensemble.csv'"
            )
        run_path = Path(self.project_root) / self.ensemble_run_dir
        if not run_path.exists():
            raise FileNotFoundError(
                f"Ensemble run directory not found: {run_path}\n"
                "Run notebook 09_ensemble.ipynb first."
            )
        weights_path = run_path / "ensemble_weights.json"
        if not weights_path.exists():
            raise FileNotFoundError(
                f"ensemble_weights.json not found at {weights_path}.\n"
                "This file is required to determine the submission strategy."
            )

    @property
    def root(self) -> Path:
        return Path(self.project_root)

    @property
    def run_path(self) -> Path:
        return self.root / self.ensemble_run_dir

    @property
    def test_csv_path(self) -> Path:
        return self.root / self.test_path

    @property
    def oof_path(self) -> Path:
        return self.root / self.oof_dir

    @property
    def cal_root(self) -> Path:
        return self.root / self.calibration_dir


# =============================================================================
# STEP 1 — DATA LOADING
# =============================================================================

def load_test_data(cfg: InferenceConfig) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load Test.csv and return (feature_df, id_series).

    Applies the same dtype enforcement as load_data.py:
      - categorical columns cast to category dtype
      - numeric columns cast with pd.to_numeric(errors='coerce')
      - NO target column expected (Test.csv has 183 columns)

    Validates:
      - File exists
      - ID column present, no duplicates
      - Row count matches expected 30,000

    Returns
    -------
    test_df : DataFrame with raw columns (183), dtypes enforced.
              The 'ID' column is RETAINED for submission alignment.
    test_ids : Series of IDs in original row order — used for
               final submission alignment and must never be reordered.
    """
    path = cfg.test_csv_path
    if not path.exists():
        raise FileNotFoundError(
            f"Test.csv not found at {path}.\n"
            "Expected path: data/raw/Test.csv"
        )

    log.info("Loading test data from %s", path)
    test_df = pd.read_csv(path)
    test_df.columns = [c.strip() for c in test_df.columns]

    # Validate structure
    if "ID" not in test_df.columns:
        raise ValueError("Test.csv has no 'ID' column.")

    dup_ids = test_df["ID"].duplicated().sum()
    if dup_ids > 0:
        raise ValueError(f"Test.csv contains {dup_ids} duplicate IDs.")

    if len(test_df) != EXPECTED_TEST_ROWS:
        log.warning(
            "Expected %d test rows, found %d. "
            "Proceeding — check if this is an updated dataset.",
            EXPECTED_TEST_ROWS, len(test_df),
        )

    # Dtype enforcement — mirrors load_data.py _enforce_dtypes()
    _CATEGORICAL_COLS = [
        "gender", "region", "segment", "earning_pattern", "smartphone"
    ]
    _EXCLUDE = ["ID"]

    for col in _CATEGORICAL_COLS:
        if col in test_df.columns:
            test_df[col] = test_df[col].astype("category")

    numeric_cols = [
        c for c in test_df.columns
        if c not in _CATEGORICAL_COLS + _EXCLUDE
    ]
    test_df[numeric_cols] = test_df[numeric_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    test_ids = test_df["ID"].copy()
    log.info(
        "Test data loaded: %d rows × %d columns  |  "
        "ID range: %s … %s",
        len(test_df), len(test_df.columns),
        test_ids.iloc[0], test_ids.iloc[-1],
    )
    return test_df, test_ids


# =============================================================================
# STEP 2 — FEATURE ENGINEERING
# =============================================================================

def engineer_features(test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run feature engineering (v2.2.1) on the test set.

    Produces 825 engineered features from 183 raw columns — identical
    pipeline to training.  build_features() is a pure transform: it
    reads column values and computes derived features.  It never fits
    any statistic (mean, std, quantile) from the data it receives.

    The 'ID' column is dropped inside build_features() (or here if
    not handled upstream), ensuring it never enters the model inputs.

    Key properties preserved from training:
      - _fill_nulls() uses exclude='category' + per-column iteration
        (the v2.2.1 bug fix for pandas >= 1.3)
      - _clean_features() clips ratio features at the 99th percentile
        of the TRAINING distribution (the clip values are baked into
        the function, not re-computed here — no leakage)
      - log1p transformations applied to value/volume/highest_amount
        blocks exactly as during training

    Returns
    -------
    X_test : DataFrame of 825 engineered features, no ID column.
    """
    from src.features.feature_engineering import build_features, split_features_target

    log.info("Running feature engineering v2.2.1 ...")
    t0 = time.perf_counter()

    test_fe = build_features(test_df.copy())
    X_test, _ = split_features_target(test_fe)

    elapsed = time.perf_counter() - t0
    log.info(
        "Feature engineering complete: %d rows × %d features  (%.1fs)",
        len(X_test), len(X_test.columns), elapsed,
    )

    # Sanity check on expected feature count.
    # 825 is the known production feature count from training.
    # A mismatch here almost always means a version mismatch between
    # feature_engineering.py on disk and the version used during training.
    if len(X_test.columns) != 825:
        log.warning(
            "Expected 825 engineered features, got %d.  "
            "Verify that feature_engineering.py version matches training (v2.2.1).",
            len(X_test.columns),
        )

    return X_test


# =============================================================================
# STEP 3 — MODEL RUN DIRECTORY RESOLUTION
# =============================================================================

def resolve_model_run_dir(model_name: str, cfg: InferenceConfig) -> Path:
    """
    Resolve the most recent run directory for a given model family.

    Each model was trained in a specific experiment stage (see
    _MODEL_STAGE_MAP).  Within that stage, runs are timestamped.
    This function returns the most recent run (sorted lexicographically
    on the run_YYYYMMDD_HHMMSS timestamp).

    Hardcoded fallback: if the primary stage has no run directories,
    falls back to 'baseline' — this handles the catboost case where
    the model was never re-run in a later stage.

    Parameters
    ----------
    model_name : one of lightgbm, xgboost, catboost, logreg, tabnet.

    Returns
    -------
    run_dir : Path to the most recent run directory for this model.

    Raises
    ------
    FileNotFoundError if no run directory exists in any searched stage.
    """
    if model_name not in _MODEL_STAGE_MAP:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Known models: {list(_MODEL_STAGE_MAP.keys())}"
        )

    stage, _ = _MODEL_STAGE_MAP[model_name]
    experiments_root = cfg.root / "outputs" / "experiments"

    # Primary stage search
    model_dir = experiments_root / stage / model_name
    if model_dir.exists():
        run_dirs = sorted(model_dir.glob("run_*"), reverse=True)
        if run_dirs:
            log.info("  %s → stage=%s  run=%s", model_name, stage, run_dirs[0].name)
            return run_dirs[0]

    # Fallback to baseline (handles edge cases and catboost explicitly)
    baseline_dir = experiments_root / "baseline" / model_name
    if baseline_dir.exists():
        run_dirs = sorted(baseline_dir.glob("run_*"), reverse=True)
        if run_dirs:
            log.warning(
                "  %s: primary stage '%s' had no runs; "
                "using baseline run %s",
                model_name, stage, run_dirs[0].name,
            )
            return run_dirs[0]

    raise FileNotFoundError(
        f"No run directories found for '{model_name}'.\n"
        f"Searched:\n"
        f"  {experiments_root / stage / model_name}\n"
        f"  {experiments_root / 'baseline' / model_name}\n"
        "Run the training pipeline (run_all_models.py) for this model first."
    )


# =============================================================================
# STEP 4 — PER-MODEL FOLD LOADING AND PREDICTION
# =============================================================================

def _load_tabnet_fold(fold_path: Path):
    """
    Load a single TabNet fold model from its .zip artefact.

    TabNet serialises to .zip via pytorch_tabnet's internal save_model()
    mechanism.  Using joblib.load() on a .zip file would corrupt the
    model state silently (the object loads but predict_proba returns
    garbage).  Always use .load_model() on the path WITHOUT the .zip
    extension.

    Parameters
    ----------
    fold_path : path ending in tabnet_fold_N.zip

    Returns
    -------
    Fitted TabNetClassifier instance.
    """
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
    except ImportError as exc:
        raise ImportError(
            "pytorch_tabnet is not installed.  "
            "Install with: pip install pytorch-tabnet"
        ) from exc

    # TabNet's load_model() expects the path WITHOUT the .zip extension.
    path_no_ext = str(fold_path).replace(".zip", "")
    model = TabNetClassifier()
    model.load_model(path_no_ext)
    return model


def _load_fold_models(model_name: str, run_dir: Path) -> List:
    """
    Load all N_FOLDS fold models for a given model family.

    Fold model naming convention (from cv.py):
      lightgbm : models/lightgbm/lightgbm_fold_{k}.pkl
      xgboost  : models/xgboost/xgboost_fold_{k}.pkl
      catboost : models/catboost/catboost_fold_{k}.pkl
      logreg   : models/logreg/logreg_fold_{k}.pkl
      tabnet   : models/tabnet/tabnet_fold_{k}.zip   ← .zip not .pkl

    Raises ValueError if the expected number of fold models is not found.
    """
    model_dir = run_dir / "models" / model_name
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}\n"
            f"Expected structure: run_dir/models/{model_name}/"
        )

    is_tabnet = (model_name == "tabnet")
    ext = ".zip" if is_tabnet else ".pkl"
    pattern = f"{model_name}_fold_*{ext}"
    fold_paths = sorted(model_dir.glob(pattern))

    if len(fold_paths) != N_FOLDS:
        raise ValueError(
            f"Expected {N_FOLDS} fold models for '{model_name}', "
            f"found {len(fold_paths)} at {model_dir}.\n"
            f"Pattern searched: {pattern}\n"
            "Re-run training to regenerate all fold models."
        )

    models = []
    for fp in fold_paths:
        if is_tabnet:
            models.append(_load_tabnet_fold(fp))
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=InconsistentVersionWarning
                )
                models.append(joblib.load(fp))

    log.info(
        "    Loaded %d %s fold models from %s",
        len(models), model_name, model_dir.relative_to(PROJECT_ROOT),
    )
    return models


def _predict_single_fold(
    model,
    model_name: str,
    X_proc: np.ndarray,
) -> np.ndarray:
    """
    Generate class-1 probability predictions from a single fold model.

    All supported model families expose predict_proba(X)[:, 1].
    CatBoost requires data as Pool or numpy array — passing a DataFrame
    would trigger a warning; X_proc is already numpy (from preprocessor).

    Returns
    -------
    preds : shape (n_test,), values in [0, 1].
    """
    raw = model.predict_proba(X_proc)[:, 1]
    return raw.astype(np.float64)


def predict_model(
    model_name  : str,
    X_test      : pd.DataFrame,
    cfg         : InferenceConfig,
) -> np.ndarray:
    """
    Full per-model inference pipeline:
      1. Resolve run directory
      2. Load fitted preprocessor and transform test features
      3. Load all 5 fold models
      4. Predict per fold, log statistics, average across folds

    Returns
    -------
    raw_preds : shape (30000,), raw (un-calibrated) predictions,
                averaged across all N_FOLDS folds.
    """
    log.info("[%s] Starting inference ...", model_name.upper())

    # 1. Run directory
    run_dir = resolve_model_run_dir(model_name, cfg)

    # 2. Preprocessor — load fitted object, transform only.
    #    CRITICAL: logreg and tabnet preprocessors contain a fitted
    #    StandardScaler.  If a GBM preprocessor (no scaler) is used
    #    for logreg, the logistic regression receives raw-scale inputs
    #    and predictions collapse toward 0.5.
    preproc_path = run_dir / "preprocessor.pkl"
    if not preproc_path.exists():
        raise FileNotFoundError(
            f"preprocessor.pkl not found at {preproc_path}.\n"
            "This file must be the FITTED preprocessor from training."
        )
    from src.preprocessing.preprocessing import PreprocessingPipeline
    preproc = PreprocessingPipeline.load(str(preproc_path))

    # Backward compatibility patch for old pickles
    if not hasattr(preproc, "scale_features"):
        preproc.scale_features = False

    if not hasattr(preproc, "scaler_"):
        preproc.scaler_ = None

    # Verify scale_features flag matches expectation for this model family.
    # A mismatch (e.g. GBM preprocessor on logreg) would silently corrupt
    # predictions without raising an error.
    expects_scale = model_name in _SCALE_SENSITIVE_MODELS
    has_scale = getattr(preproc, "scale_features", False)
    if expects_scale and not has_scale:
        raise ValueError(
            f"'{model_name}' expects a preprocessor with scale_features=True "
            f"(includes StandardScaler), but the loaded preprocessor at "
            f"{preproc_path} has scale_features=False.\n"
            "You may have loaded a GBM preprocessor for a scale-sensitive model. "
            "Check resolve_model_run_dir() output."
        )
    if not expects_scale and has_scale:
        log.warning(
            "[%s] Preprocessor has scale_features=True but this model "
            "family does not require scaling.  Proceeding — verify the "
            "run directory is correct.",
            model_name,
        )

    # Align feature columns to the order the preprocessor was fitted on.
    # The feature_list.json in the run directory records the exact column
    # order seen during training.  This guards against any reordering that
    # might occur during feature engineering across different pandas versions.
    feature_list_path = run_dir / "feature_list.json"
    if feature_list_path.exists():
        with open(feature_list_path) as f:
            trained_features: List[str] = json.load(f)
        missing = [c for c in trained_features if c not in X_test.columns]
        if missing:
            raise ValueError(
                f"[{model_name}] {len(missing)} features present during training "
                f"are missing from the test set after feature engineering:\n"
                f"  First 10 missing: {missing[:10]}\n"
                "Version mismatch between feature_engineering.py on disk "
                "and the version used during training."
            )
        X_aligned = X_test[trained_features]
        log.info(
            "    Feature alignment: %d columns from feature_list.json",
            len(trained_features),
        )
    else:
        log.warning(
            "[%s] feature_list.json not found at %s.  "
            "Using X_test column order as-is — ensure this matches training.",
            model_name, run_dir,
        )
        X_aligned = X_test

    X_proc = preproc.transform(X_aligned)

    # Preserve DataFrame feature names for GBMs to avoid
    # sklearn/lightgbm feature-name warnings.
    # TabNet requires numpy input.
    if model_name == "tabnet":
        if isinstance(X_proc, pd.DataFrame):
            X_proc = X_proc.values
    log.info("    Preprocessed shape: %s  scale=%s", X_proc.shape, has_scale)

    # 3. Load fold models
    fold_models = _load_fold_models(model_name, run_dir)

    # 4. Per-fold prediction
    fold_preds: List[np.ndarray] = []
    for k, model in enumerate(fold_models):
        preds_k = _predict_single_fold(model, model_name, X_proc)
        fold_preds.append(preds_k)
        if cfg.verbose:
            log.info(
                "    Fold %d: min=%.4f  max=%.4f  mean=%.4f",
                k, preds_k.min(), preds_k.max(), preds_k.mean(),
            )

    raw_preds = np.mean(fold_preds, axis=0)
    log.info(
        "[%s] Raw avg: min=%.4f  max=%.4f  mean=%.4f  std=%.4f",
        model_name.upper(),
        raw_preds.min(), raw_preds.max(),
        raw_preds.mean(), raw_preds.std(),
    )
    return raw_preds


# =============================================================================
# STEP 5 — PLATT CALIBRATION
# =============================================================================

def calibrate_predictions(
    raw_preds  : np.ndarray,
    model_name : str,
    cfg        : InferenceConfig,
) -> np.ndarray:
    """
    Apply the Platt calibrator fitted during training (notebooks 07/07b).

    The calibrator transforms raw fold-averaged predictions into
    calibrated probabilities.  For GBMs this is a near-identity
    transformation (slope ≈ 1.0).  For LogReg and TabNet, the Platt
    pass provides substantial LogLoss improvement:
      LogReg : slope=5.47  36% LogLoss reduction
      TabNet : slope=4.37  21% LogLoss reduction

    Calibration folder mapping (mirrors ensemble.py v5.1):
      lightgbm → outputs/calibration/lgb/calibrator_platt.pkl
      xgboost  → outputs/calibration/xgb/calibrator_platt.pkl
      catboost → outputs/calibration/cat/calibrator_platt.pkl
      logreg   → outputs/calibration/logreg/calibrator_platt.pkl
      tabnet   → outputs/calibration/tabnet/calibrator_platt.pkl

    Raises
    ------
    FileNotFoundError if the calibrator file is missing.  Unlike the
    old predict.py, this version does NOT silently fall back to
    uncalibrated predictions — an absent calibrator is an artefact
    integrity failure that must be surfaced explicitly.
    """
    if model_name not in _CALIBRATION_FOLDER_MAP:
        raise ValueError(
            f"No calibration folder mapping for '{model_name}'.  "
            f"Known mappings: {_CALIBRATION_FOLDER_MAP}"
        )

    folder = _CALIBRATION_FOLDER_MAP[model_name]
    cal_path = cfg.cal_root / folder / "calibrator_platt.pkl"

    if not cal_path.exists():
        raise FileNotFoundError(
            f"Platt calibrator not found for '{model_name}' at:\n  {cal_path}\n"
            "Run the calibration notebook(s) to produce this artefact:\n"
            "  GBMs  → 07_calibration_analysis.ipynb\n"
            "  Other → 07b_logreg_tabnet_calibration.ipynb"
        )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=InconsistentVersionWarning
        )
        calibrator = joblib.load(cal_path)

        log.info(
            "[%s] Loaded calibrator type: %s",
            model_name.upper(),
            type(calibrator).__name__,
        )

    # Platt calibrators expect 2D input: (n_samples, 1)
    raw_preds_2d = raw_preds.reshape(-1, 1)

    # ------------------------------------------------------------------
    # Backward compatibility fix:
    # Older sklearn LogisticRegression pickles may miss attributes
    # like `multi_class` when loaded across sklearn versions.
    # ------------------------------------------------------------------
    if hasattr(calibrator, "predict_proba"):

        # Patch missing sklearn attributes from old pickles
        if not hasattr(calibrator, "multi_class"):
            calibrator.multi_class = "auto"

        if not hasattr(calibrator, "classes_"):
            calibrator.classes_ = np.array([0, 1])

        cal_preds = calibrator.predict_proba(raw_preds_2d)[:, 1]

    # custom Platt scaler with .predict()
    else:
        cal_preds = calibrator.predict(raw_preds_2d)

    cal_preds = np.asarray(cal_preds).reshape(-1)
    cal_preds = np.clip(cal_preds, CLIP_LOW, CLIP_HIGH)

    # Log Platt parameters for traceability.
    slope     = getattr(calibrator, "slope",     float("nan"))
    intercept = getattr(calibrator, "intercept", float("nan"))
    log.info(
        "[%s] Calibrated: min=%.4f  max=%.4f  mean=%.4f  "
        "(slope=%.3f  intercept=%.3f)",
        model_name.upper(),
        cal_preds.min(), cal_preds.max(), cal_preds.mean(),
        slope, intercept,
    )
    return cal_preds


# =============================================================================
# STEP 6 — ENSEMBLE: OPTIMISED WEIGHTED AVERAGE
# =============================================================================

def load_ensemble_weights(cfg: InferenceConfig) -> Dict[str, float]:
    """
    Load optimised ensemble weights from the production run artefact.

    Weights are read from ensemble_weights.json["optimised"] — never
    hardcoded.  This ensures that if a future run produces updated
    weights, the inference pipeline automatically picks them up without
    code changes.

    Also reads metadata.json to verify:
      - The model_names list in the run matches _DEFAULT_MODEL_ORDER
      - The best_strategy is 'optimised_weighted_average'

    Returns
    -------
    weights : dict mapping model_name → weight (floats summing to 1.0).

    Raises
    ------
    ValueError if the ensemble run's best strategy is not the weighted
    average — prevents accidentally deploying a stacking submission.
    """
    weights_path = cfg.run_path / "ensemble_weights.json"
    with open(weights_path) as f:
        weights_json = json.load(f)

    weights: Dict[str, float] = weights_json["optimised"]
    model_order: List[str]    = weights_json.get("model_order", list(weights.keys()))

    log.info("Ensemble weights loaded from %s", weights_path.name)
    for m in model_order:
        log.info("  %-12s : %.5f", m, weights.get(m, 0.0))

    # Verify the best strategy from metadata
    meta_path = cfg.run_path / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
        best_strategy = metadata.get("best_strategy", "")
        best_score    = metadata.get("best_score", float("nan"))
        if best_strategy != "optimised_weighted_average":
            log.warning(
                "Ensemble metadata reports best_strategy='%s' (score=%.5f), "
                "not 'optimised_weighted_average'.  "
                "Using optimised weights regardless — verify this is intentional.",
                best_strategy, best_score,
            )
        else:
            log.info(
                "Strategy confirmed: optimised_weighted_average  "
                "OOF composite score=%.5f",
                best_score,
            )

    # Validate weights sum to approximately 1.0
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-4:
        log.warning(
            "Ensemble weights sum to %.6f (expected 1.0).  "
            "Normalising automatically.",
            total,
        )
        weights = {m: w / total for m, w in weights.items()}

    return weights


def apply_ensemble(
    calibrated_preds : Dict[str, np.ndarray],
    weights          : Dict[str, float],
    model_names      : List[str],
) -> np.ndarray:
    """
    Compute the optimised weighted average of calibrated predictions.

    All models in model_names must be present in both calibrated_preds
    and weights.  Models with weight=0.000 (LogReg in the production
    configuration) contribute zero to the sum — their predictions are
    still required in calibrated_preds to maintain pipeline integrity.

    The weighted sum is clipped to [EPS, 1-EPS] to satisfy the Zindi
    log-loss constraint (log(0) is undefined).

    Parameters
    ----------
    calibrated_preds : model_name → Platt-calibrated prediction array
    weights          : model_name → ensemble weight (from JSON artefact)
    model_names      : ordered list of models to include

    Returns
    -------
    ensemble_preds : shape (n_test,), final submission probabilities.
    """
    # Bidirectional validation
    missing_preds   = [m for m in model_names if m not in calibrated_preds]
    missing_weights = [m for m in model_names if m not in weights]

    if missing_preds:
        raise ValueError(
            f"Missing calibrated predictions for: {missing_preds}.\n"
            "All models must complete inference before ensemble step."
        )
    if missing_weights:
        raise ValueError(
            f"No ensemble weight found for: {missing_weights}.\n"
            "Check that ensemble_weights.json contains all model entries."
        )

    pred_matrix = np.stack(
        [calibrated_preds[m] for m in model_names], axis=1
    )  # shape: (n_test, n_models)

    weight_vec = np.array(
        [weights[m] for m in model_names], dtype=np.float64
    )  # shape: (n_models,)

    ensemble_preds = pred_matrix @ weight_vec        # weighted dot product
    ensemble_preds = np.clip(ensemble_preds, CLIP_LOW, CLIP_HIGH)

    log.info(
        "Ensemble (weighted avg): min=%.4f  max=%.4f  mean=%.4f  std=%.4f",
        ensemble_preds.min(), ensemble_preds.max(),
        ensemble_preds.mean(), ensemble_preds.std(),
    )

    # Log effective weights (confirming LogReg zero contribution)
    for m, w in zip(model_names, weight_vec):
        contribution = w * calibrated_preds[m].mean()
        log.info(
            "  %-12s  w=%.5f  mean_cal=%.4f  contribution=%.4f",
            m, w, calibrated_preds[m].mean(), contribution,
        )

    return ensemble_preds


# =============================================================================
# STEP 7 — DISTRIBUTION DRIFT CHECK
# =============================================================================

def check_prediction_drift(
    ensemble_preds : np.ndarray,
    cfg            : InferenceConfig,
) -> None:
    """
    Compare test prediction distribution against OOF ensemble distribution.

    A large shift in mean prediction between OOF and test sets almost
    always indicates a pipeline bug rather than a genuine distributional
    shift:
      - Preprocessor applied twice (double-scaling)
      - Wrong preprocessor used for a model (scale/no-scale mismatch)
      - Feature engineering version mismatch (different feature count)
      - Test data loaded from the wrong file

    Thresholds:
      |Δmean| > 0.05  → WARNING (investigate)
      |Δmean| > 0.10  → ERROR (likely pipeline bug; do not submit)

    The OOF ensemble array lives at outputs/multi_model/ensemble_oof.npy
    (calibrated stacking OOF) or is computed from the optimised-weight
    blend of the per-model OOF arrays.  We use the run's ensemble_oof.npy
    if available; otherwise fall back to reading from run_path.
    """
    # Try the per-run ensemble OOF first
    oof_candidates = [
        cfg.run_path / "ensemble_oof.npy",
        cfg.oof_path / "ensemble_oof.npy",
    ]
    oof_path = next((p for p in oof_candidates if p.exists()), None)

    if oof_path is None:
        log.warning(
            "No ensemble_oof.npy found for drift check.  "
            "Skipping distribution validation."
        )
        return

    oof_preds = np.load(oof_path)

    # OOF positive rate sanity check
    oof_pos_rate = float(oof_preds.mean())
    if not 0.13 <= oof_pos_rate <= 0.17:
        log.warning(
            "OOF ensemble mean %.4f is outside expected range [0.13, 0.17].  "
            "Check that the correct OOF file was loaded.",
            oof_pos_rate,
        )

    test_mean = float(ensemble_preds.mean())
    oof_mean  = float(oof_preds.mean())
    delta     = abs(test_mean - oof_mean)

    log.info("Distribution drift check:")
    log.info("  OOF  mean : %.4f  std=%.4f", oof_mean, float(oof_preds.std()))
    log.info("  Test mean : %.4f  std=%.4f", test_mean, float(ensemble_preds.std()))
    log.info("  |Δmean|   : %.4f  (warn at %.2f)", delta, DRIFT_WARN_THRESHOLD)

    if delta > 0.10:
        raise ValueError(
            f"CRITICAL: Test prediction mean ({test_mean:.4f}) differs from "
            f"OOF mean ({oof_mean:.4f}) by {delta:.4f} — exceeds the 0.10 "
            "hard threshold.  This indicates a pipeline bug.  Do NOT submit.\n"
            "Common causes:\n"
            "  • Wrong preprocessor used for a model family\n"
            "  • Feature engineering version mismatch\n"
            "  • Predictions clipped incorrectly upstream"
        )
    elif delta > DRIFT_WARN_THRESHOLD:
        log.warning(
            "Test mean drifts from OOF mean by %.4f (threshold=%.2f).  "
            "Investigate before submitting.  "
            "Could indicate distribution shift or a pipeline issue.",
            delta, DRIFT_WARN_THRESHOLD,
        )
    else:
        log.info("  ✓  Distribution consistent  (|Δmean|=%.4f < %.2f)", delta, DRIFT_WARN_THRESHOLD)


# =============================================================================
# STEP 8 — ZINDI COMPLIANCE VALIDATION AND SUBMISSION BUILD
# =============================================================================

def validate_and_build_submission(
    test_ids       : pd.Series,
    ensemble_preds : np.ndarray,
) -> pd.DataFrame:
    """
    Run all 6 Zindi compliance assertions, then build the submission DataFrame.

    Zindi requirements for this competition:
      • Exactly 30,000 rows
      • ID column matches Test.csv IDs in original row order
      • Target, TargetLogLoss, TargetRAUC all identical columns
      • All predictions in [0, 1]
      • No NaN or Inf values

    These checks are implemented as if/raise ValueError (not assert)
    because assert is disabled by python -O and must never gate
    production file writes.

    Returns
    -------
    submission : DataFrame with columns [ID, Target, TargetLogLoss, TargetRAUC].
    """
    # ── Check 1: row count ──────────────────────────────────────────────────
    if len(ensemble_preds) != len(test_ids):
        raise ValueError(
            f"Prediction count ({len(ensemble_preds)}) does not match "
            f"test ID count ({len(test_ids)}).  "
            "Row alignment is broken."
        )

    # ── Check 2: expected row count ─────────────────────────────────────────
    if len(ensemble_preds) != EXPECTED_TEST_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_TEST_ROWS:,} prediction rows, got {len(ensemble_preds):,}.\n"
            "Check that Test.csv was loaded without row filtering."
        )

    # ── Check 3: no NaN ─────────────────────────────────────────────────────
    nan_count = int(np.isnan(ensemble_preds).sum())
    if nan_count > 0:
        raise ValueError(
            f"Ensemble predictions contain {nan_count} NaN value(s).  "
            "A NaN in the weighted average means at least one model "
            "produced NaN predictions.  Check all per-model fold outputs."
        )

    # ── Check 4: no Inf ─────────────────────────────────────────────────────
    inf_count = int(np.isinf(ensemble_preds).sum())
    if inf_count > 0:
        raise ValueError(
            f"Ensemble predictions contain {inf_count} Inf value(s).  "
            "Check for division by zero in feature engineering or preprocessing."
        )

    # ── Check 5: values in [0, 1] ───────────────────────────────────────────
    if ensemble_preds.min() < 0.0 or ensemble_preds.max() > 1.0:
        raise ValueError(
            f"Predictions outside [0, 1]: "
            f"min={ensemble_preds.min():.8f}  max={ensemble_preds.max():.8f}.\n"
            "Clipping should have been applied after the weighted average."
        )

    # ── Build submission ─────────────────────────────────────────────────────
    submission = pd.DataFrame({
        "ID"           : test_ids.values,
        "Target"       : ensemble_preds,
        "TargetLogLoss": ensemble_preds,   # Zindi requires three identical columns
        "TargetRAUC"   : ensemble_preds,
    })

    # ── Check 6: three columns identical ────────────────────────────────────
    if not (submission["Target"] == submission["TargetLogLoss"]).all():
        raise ValueError("Target != TargetLogLoss in submission.  "
                         "All three score columns must be identical.")
    if not (submission["Target"] == submission["TargetRAUC"]).all():
        raise ValueError("Target != TargetRAUC in submission.  "
                         "All three score columns must be identical.")

    log.info(
        "Submission validation passed (6/6 checks):\n"
        "  Rows          : %d\n"
        "  Mean pred      : %.4f  (train positive rate=%.2f)\n"
        "  Std pred       : %.4f\n"
        "  Min / Max      : %.6f / %.6f\n"
        "  Pred > 0.5     : %d  (%.1f%%)\n"
        "  Pred > 0.3     : %d  (%.1f%%)",
        len(submission),
        ensemble_preds.mean(),  EXPECTED_POSITIVE_RATE,
        ensemble_preds.std(),
        ensemble_preds.min(), ensemble_preds.max(),
        int((ensemble_preds > 0.5).sum()), (ensemble_preds > 0.5).mean() * 100,
        int((ensemble_preds > 0.3).sum()), (ensemble_preds > 0.3).mean() * 100,
    )
    return submission


def write_submission(
    submission  : pd.DataFrame,
    cfg         : InferenceConfig,
) -> Path:
    """
    Write the validated submission DataFrame to disk.

    Creates the output directory if it does not exist.  If a file
    already exists at the output path, it is overwritten (with a log
    warning so the operator is aware).

    Returns the resolved absolute output path.
    """
    out_path = cfg.root / cfg.output_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        log.warning("Output file already exists and will be overwritten: %s", out_path)

    if cfg.dry_run:
        log.info("[DRY RUN] Submission not written to disk.  Path would be: %s", out_path)
        return out_path

    submission.to_csv(out_path, index=False)
    file_size_kb = out_path.stat().st_size / 1024

    log.info("=" * 60)
    log.info("SUBMISSION WRITTEN")
    log.info("=" * 60)
    log.info("  Path     : %s", out_path)
    log.info("  Size     : %.1f KB", file_size_kb)
    log.info("  Rows     : %d", len(submission))
    log.info("  Columns  : %s", list(submission.columns))
    log.info("=" * 60)

    return out_path


# =============================================================================
# MAIN ORCHESTRATOR — InferencePipeline class
# =============================================================================

class InferencePipeline:
    """
    Orchestrates the full 5-model inference pipeline from raw test data
    to a Zindi-compliant submission CSV.

    Designed for both CLI and notebook usage.  All state is held in the
    config object; no global mutable state is used.

    Example (notebook)
    ------------------
    from src.inference.predict import InferencePipeline, InferenceConfig

    cfg = InferenceConfig(
        ensemble_run_dir = "outputs/experiments/v5_ensemble/run_20260508_054540",
        output_path      = "submissions/submission_v5_ensemble.csv",
    )
    pipeline = InferencePipeline(cfg)
    submission, report = pipeline.run()
    print(report)

    Example (CLI)
    -------------
    python -m src.inference.predict \\
        --ensemble-run outputs/experiments/v5_ensemble/run_20260508_054540 \\
        --output       submissions/submission_v5_ensemble.csv

    Attributes
    ----------
    cfg           : InferenceConfig
    model_names   : ordered list of model names (loaded from metadata.json)
    raw_preds     : dict of raw (uncalibrated) per-model predictions
    cal_preds     : dict of Platt-calibrated per-model predictions
    ensemble_preds: final weighted average prediction array
    submission    : final submission DataFrame
    report        : dict of provenance and diagnostic information
    """

    def __init__(self, cfg: InferenceConfig) -> None:
        cfg.validate()
        self.cfg            : InferenceConfig             = cfg
        self.model_names    : List[str]                   = []
        self.weights        : Dict[str, float]            = {}
        self.raw_preds      : Dict[str, np.ndarray]       = {}
        self.cal_preds      : Dict[str, np.ndarray]       = {}
        self.ensemble_preds : Optional[np.ndarray]        = None
        self.submission     : Optional[pd.DataFrame]      = None
        self.report         : Dict                        = {}
        self._start_time    : float                       = 0.0

    def _load_model_names_from_metadata(self) -> List[str]:
        """
        Read model_names from the ensemble run's metadata.json.

        This guarantees that the inference pipeline uses the exact same
        model set that was used during ensemble training — training-
        inference consistency is enforced by the artefact, not by
        module-level defaults.
        """
        meta_path = self.cfg.run_path / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)
            names = metadata.get("model_names", _DEFAULT_MODEL_ORDER)
            log.info("Model names from metadata.json: %s", names)
            return names
        log.warning(
            "metadata.json not found at %s.  "
            "Falling back to default model order: %s",
            meta_path, _DEFAULT_MODEL_ORDER,
        )
        return list(_DEFAULT_MODEL_ORDER)

    def run(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Execute the full inference pipeline.

        Stages:
          1.  Load and validate test data
          2.  Feature engineering (v2.2.1, 825 features)
          3.  Load ensemble weights and model list from run artefacts
          4.  Per-model inference (×5 models):
                a. Resolve run directory
                b. Load model-specific preprocessor
                c. Align and transform features
                d. Load 5 fold models (TabNet via .zip)
                e. Predict and average across folds
                f. Apply Platt calibration
          5.  Optimised weighted average ensemble
          6.  Distribution drift check vs OOF
          7.  Zindi compliance validation (6 assertions)
          8.  Write submission CSV

        Returns
        -------
        (submission_df, report_dict)
        """
        self._start_time = time.perf_counter()

        log.info("=" * 70)
        log.info("INFERENCE PIPELINE  v2.0  —  AI4EAC Liquidity Stress")
        log.info("=" * 70)
        log.info("Ensemble run : %s", self.cfg.ensemble_run_dir)
        log.info("Output       : %s", self.cfg.output_path)
        log.info("Dry run      : %s", self.cfg.dry_run)

        # ── Stage 1: Load test data ──────────────────────────────────────────
        log.info("\n[1/8] Loading test data ...")
        test_df, test_ids = load_test_data(self.cfg)

        # ── Stage 2: Feature engineering ────────────────────────────────────
        log.info("\n[2/8] Feature engineering ...")
        X_test = engineer_features(test_df)

        # ── Stage 3: Ensemble weights + model list ───────────────────────────
        log.info("\n[3/8] Loading ensemble configuration ...")
        self.model_names = self._load_model_names_from_metadata()
        self.weights     = load_ensemble_weights(self.cfg)

        # Verify all required models have weight entries
        missing_w = [m for m in self.model_names if m not in self.weights]
        if missing_w:
            raise ValueError(
                f"Ensemble weights missing for: {missing_w}.\n"
                "ensemble_weights.json must contain an entry for every model "
                "in the run's model_names list."
            )

        # ── Stage 4: Per-model inference ────────────────────────────────────
        log.info("\n[4/8] Per-model inference (%d models) ...", len(self.model_names))
        for model_name in self.model_names:
            log.info("")
            raw = predict_model(model_name, X_test, self.cfg)
            cal = calibrate_predictions(raw, model_name, self.cfg)
            self.raw_preds[model_name] = raw
            self.cal_preds[model_name] = cal

        # ── Stage 5: Weighted average ensemble ──────────────────────────────
        log.info("\n[5/8] Applying optimised weighted average ensemble ...")
        self.ensemble_preds = apply_ensemble(
            self.cal_preds, self.weights, self.model_names
        )

        # ── Stage 6: Distribution drift check ───────────────────────────────
        log.info("\n[6/8] Distribution drift check ...")
        check_prediction_drift(self.ensemble_preds, self.cfg)

        # ── Stage 7: Validation and submission build ─────────────────────────
        log.info("\n[7/8] Zindi compliance validation ...")
        self.submission = validate_and_build_submission(test_ids, self.ensemble_preds)

        # ── Stage 8: Write CSV ───────────────────────────────────────────────
        log.info("\n[8/8] Writing submission ...")
        out_path = write_submission(self.submission, self.cfg)

        # ── Build provenance report ──────────────────────────────────────────
        runtime = time.perf_counter() - self._start_time
        self.report = self._build_report(out_path, runtime)
        log.info("\nTotal runtime: %.1fs", runtime)

        return self.submission, self.report

    def _build_report(self, out_path: Path, runtime: float) -> Dict:
        """
        Assemble a provenance report for notebook display and archiving.

        Contains all information needed to reproduce or audit the
        submission: ensemble run directory, model list, weights,
        prediction distribution statistics, and output path.
        """
        preds = self.ensemble_preds
        report = {
            "timestamp"        : datetime.now().isoformat(),
            "ensemble_version" : "v5.1",
            "ensemble_run_dir" : self.cfg.ensemble_run_dir,
            "submission_path"  : str(out_path),
            "dry_run"          : self.cfg.dry_run,
            "model_names"      : self.model_names,
            "weights"          : self.weights,
            "n_rows"           : len(self.submission) if self.submission is not None else 0,
            "prediction_stats" : {
                "mean"  : float(preds.mean()),
                "std"   : float(preds.std()),
                "min"   : float(preds.min()),
                "max"   : float(preds.max()),
                "p10"   : float(np.percentile(preds, 10)),
                "p25"   : float(np.percentile(preds, 25)),
                "p50"   : float(np.percentile(preds, 50)),
                "p75"   : float(np.percentile(preds, 75)),
                "p90"   : float(np.percentile(preds, 90)),
                "pct_above_0p3" : float((preds > 0.3).mean()),
                "pct_above_0p5" : float((preds > 0.5).mean()),
            },
            "per_model_stats"  : {
                m: {
                    "raw_mean" : float(self.raw_preds[m].mean()),
                    "raw_max"  : float(self.raw_preds[m].max()),
                    "cal_mean" : float(self.cal_preds[m].mean()),
                    "cal_max"  : float(self.cal_preds[m].max()),
                    "weight"   : self.weights.get(m, 0.0),
                }
                for m in self.model_names
            },
            "known_oof_scores" : _KNOWN_OOF_SCORES,
            "runtime_sec"      : round(runtime, 1),
        }

        # Pretty-print summary for notebook display
        log.info("\n%s", "=" * 60)
        log.info("INFERENCE REPORT")
        log.info("=" * 60)
        log.info("Strategy        : optimised_weighted_average")
        log.info("OOF score (ref) : %.5f  (composite = 0.6×LL + 0.4×(1-AUC))",
                 _KNOWN_OOF_SCORES["ensemble"])
        log.info("Pred mean       : %.4f  (expected ≈ %.2f)",
                 report["prediction_stats"]["mean"], EXPECTED_POSITIVE_RATE)
        log.info("Runtime         : %.1fs", runtime)
        log.info("=" * 60)

        return report


# =============================================================================
# CONVENIENCE FUNCTION — for direct notebook use without class instantiation
# =============================================================================

def run_inference(
    ensemble_run_dir : str,
    output_path      : str,
    test_path        : str  = "data/raw/Test.csv",
    dry_run          : bool = False,
    verbose          : bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience wrapper for notebook use.

    Constructs an InferenceConfig and InferencePipeline, then runs the
    full pipeline.  Equivalent to calling InferencePipeline(cfg).run().

    Parameters
    ----------
    ensemble_run_dir : path (relative to project root) to the ensemble run.
    output_path      : where to write the Zindi submission CSV.
    test_path        : path to Test.csv.
    dry_run          : if True, validates without writing the CSV.
    verbose          : if True, logs per-fold statistics.

    Returns
    -------
    (submission_df, report_dict)

    Example
    -------
    submission, report = run_inference(
        ensemble_run_dir = "outputs/experiments/v5_ensemble/run_20260508_054540",
        output_path      = "submissions/submission_v5_ensemble.csv",
    )
    submission.head()
    """
    cfg = InferenceConfig(
        ensemble_run_dir = ensemble_run_dir,
        output_path      = output_path,
        test_path        = test_path,
        dry_run          = dry_run,
        verbose          = verbose,
    )
    pipeline = InferencePipeline(cfg)
    return pipeline.run()


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog        = "python -m src.inference.predict",
        description = (
            "AI4EAC Liquidity Stress — Final inference pipeline.\n"
            "Generates a Zindi-compliant submission CSV from the trained ensemble.\n\n"
            "Strategy: optimised weighted average (OOF composite=0.19144)\n"
            "Models: LightGBM(0.178) + XGBoost(0.397) + CatBoost(0.388) "
            "+ LogReg(0.000) + TabNet(0.037)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ensemble-run",
        required = True,
        metavar  = "PATH",
        help     = (
            "Relative path to the production ensemble run directory.\n"
            "Example: outputs/experiments/v5_ensemble/run_20260508_054540"
        ),
    )
    parser.add_argument(
        "--output",
        default = f"submissions/submission_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        metavar = "PATH",
        help    = "Output path for the Zindi submission CSV. (default: submissions/submission_TIMESTAMP.csv)",
    )
    parser.add_argument(
        "--test-path",
        default = "data/raw/Test.csv",
        metavar = "PATH",
        help    = "Path to Test.csv, relative to project root. (default: data/raw/Test.csv)",
    )
    parser.add_argument(
        "--dry-run",
        action  = "store_true",
        default = False,
        help    = "Run full pipeline validation without writing the submission CSV.",
    )
    parser.add_argument(
        "--quiet",
        action  = "store_true",
        default = False,
        help    = "Suppress per-fold prediction statistics.",
    )
    parser.add_argument(
        "--log-level",
        default = "INFO",
        choices = ["DEBUG", "INFO", "WARNING", "ERROR"],
        help    = "Logging verbosity. (default: INFO)",
    )
    return parser.parse_args()


def main() -> int:
    """
    CLI entry point.  Returns 0 on success, 1 on failure.

    Called by:
        python -m src.inference.predict [args]
        python src/inference/predict.py [args]
    """
    args = _parse_args()

    logging.getLogger().setLevel(args.log_level)

    cfg = InferenceConfig(
        ensemble_run_dir = args.ensemble_run,
        output_path      = args.output,
        test_path        = args.test_path,
        dry_run          = args.dry_run,
        verbose          = not args.quiet,
    )

    try:
        pipeline   = InferencePipeline(cfg)
        submission, report = pipeline.run()

        # Write the provenance report alongside the submission
        if not cfg.dry_run:
            report_path = (cfg.root / cfg.output_path).with_suffix(".report.json")
            with open(report_path, "w") as f:
                json.dump(report, f, indent=4, default=str)
            log.info("Provenance report saved: %s", report_path)

        return 0

    except (FileNotFoundError, ValueError) as exc:
        log.error("Pipeline failed: %s", exc)
        return 1
    except Exception as exc:
        log.exception("Unexpected error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())