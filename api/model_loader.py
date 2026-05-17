"""
Model Artefact Loader — Production v1.0
========================================
Project  : AI4EAC Liquidity Stress Early Warning (Zindi Africa)
Module   : api/model_loader.py
Author   : Henry Otsyula
Version  : 1.0.0

Responsibility
--------------
Load ALL production model artefacts exactly once at server startup and
expose them through a single, fully-typed ``LoadedEnsemble`` dataclass.
Every subsequent request reads from the in-memory cache — zero disk I/O
after startup.

Artefact inventory (loaded at startup)
---------------------------------------
  25 fold models      : 5 models × 5 folds
                        GBMs + LogReg → joblib .pkl
                        TabNet        → pytorch_tabnet .zip via .load_model()
   5 preprocessors    : one fitted PreprocessingPipeline per model
                        GBMs   : scale_features=False (no StandardScaler)
                        LogReg : scale_features=True  (StandardScaler fitted)
                        TabNet : scale_features=True  (StandardScaler fitted)
   5 Platt calibrators: one per model from outputs/calibration/{folder}/
                        Folder map: lightgbm→lgb, xgboost→xgb, catboost→cat,
                        logreg→logreg, tabnet→tabnet
   1 ensemble config  : weights + model_names + version from
                        outputs/experiments/v5_ensemble/run_20260508_054540/
   5 feature lists    : per-model feature_list.json (825 features each)
                        Used for column alignment at inference time

Caching strategy
----------------
``load_ensemble()`` is decorated with ``@lru_cache(maxsize=1)``.  The first
call (triggered by FastAPI's lifespan startup hook) blocks until all
artefacts are loaded.  All subsequent calls — including concurrent requests —
return the cached ``LoadedEnsemble`` immediately.  This is thread-safe under
the Python GIL and requires no explicit locking.

Arguments to ``load_ensemble`` are typed as ``str`` (not ``Path``) because
``lru_cache`` uses arguments as dict keys: using ``str`` avoids any edge case
where two ``Path`` objects representing the same directory compare unequal
due to symlink resolution differences on the deployment host.

Cache invalidation: call ``load_ensemble.cache_clear()`` in tests to force
a fresh load without restarting the interpreter.

Startup crash policy
---------------------
This module intentionally crashes the server on any artefact load failure.
A partially-loaded ensemble that silently serves wrong predictions is far
more dangerous than a server that refuses to start.  Crash messages include
the exact file path and the notebook reference needed to regenerate the
missing artefact.

Example of what would break silently without this policy:
  If TabNet's calibrator is missing and the load is skipped, TabNet's raw
  (un-Platt-calibrated) predictions would be passed to the weighted average.
  TabNet's Platt slope is 4.37 — without calibration its raw outputs are
  severely compressed toward 0.5.  With weight=0.037 this is a subtle but
  real corruption that would produce wrong probabilities on every request.

TabNet serialisation note
--------------------------
TabNet uses pytorch_tabnet's internal .zip serialisation via save_model() /
load_model().  Using joblib.load() on a .zip file loads raw bytes and
produces a broken object that calls without error but returns garbage.
ALWAYS use model.load_model(str(zip_path)) — never joblib for TabNet.
The .zip extension is appended internally by TabNet; pass the full path
including .zip to this module's _load_tabnet_fold() function.

Model stage → run directory mapping
-------------------------------------
lightgbm : outputs/experiments/v2_feature_expansion/lightgbm/run_*/
xgboost  : outputs/experiments/v2_feature_expansion/xgboost/run_*/
catboost : outputs/experiments/baseline/catboost/run_*/
logreg   : outputs/experiments/v3_extended_models/logreg/run_*/
tabnet   : outputs/experiments/v3_extended_models/tabnet/run_*/

The most recent run (lexicographic sort on run_YYYYMMDD_HHMMSS) is used
automatically.  Explicit run directory pinning is available via the
``LIQUIDITY_ENSEMBLE_RUN`` environment variable.

OOF performance reference (from training, for health endpoint)
----------------------------------------------------------------
LightGBM  composite=0.19557  weight=0.17813
XGBoost   composite=0.19350  weight=0.39721  ← dominant
CatBoost  composite=0.19430  weight=0.38777
LogReg    composite=0.26003  weight=0.00000  (diversity, zero blend weight)
TabNet    composite=0.25296  weight=0.03689
Ensemble  composite=0.19144  (optimised weighted average, OOF)

Changelog
---------
1.0.0  Initial production release.  Full 5-model loading with per-model
       preprocessor isolation, TabNet .zip handling, Platt calibrator
       backward-compat patching, startup integrity validation, and
       structured diagnostic logging.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.exceptions import InconsistentVersionWarning


# =============================================================================
# LOGGING
# =============================================================================

log = logging.getLogger("api.model_loader")

# Suppress sklearn version mismatch warnings globally in this module.
# These warnings fire when a model trained on sklearn 1.x is loaded on 1.y.
# They are informational only — the objects load and function correctly.
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


# =============================================================================
# PROJECT ROOT RESOLUTION
# =============================================================================

# api/model_loader.py is one level below the project root.
# predict.py is two levels below (src/inference/predict.py → parents[2]).
# This module is api/model_loader.py → parents[1].
_MODULE_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = _MODULE_DIR.parent


# =============================================================================
# MODULE-LEVEL CONSTANTS
# These must remain identical to src/inference/predict.py constants.
# Any divergence creates a training-inference inconsistency.
# =============================================================================

#: Number of cross-validation folds used during training.
N_FOLDS: int = 5

#: EPS for clipping predictions away from the log-loss singularity at 0/1.
EPS: float = 1e-6
CLIP_LOW: float = EPS
CLIP_HIGH: float = 1.0 - EPS

#: Models that required a fitted StandardScaler during training.
#: Their preprocessor.pkl contains a fitted scaler; GBM preprocessors do NOT.
#: Mixing these up silently corrupts predictions (LogReg/TabNet receive raw-scale).
_SCALE_SENSITIVE_MODELS: frozenset = frozenset({"logreg", "tabnet"})

#: Calibration subfolder map — mirrors ensemble.py v5.1 and predict.py exactly.
#: lightgbm→lgb/ preserves the abbreviated path from v4 artefacts.
_CALIBRATION_FOLDER_MAP: Dict[str, str] = {
    "lightgbm": "lgb",
    "xgboost":  "xgb",
    "catboost": "cat",
    "logreg":   "logreg",
    "tabnet":   "tabnet",
}

#: Model family → (experiment stage, config file).
#: CatBoost was not Optuna-tuned, so it remains in "baseline".
_MODEL_STAGE_MAP: Dict[str, Tuple[str, str]] = {
    "lightgbm": ("v2_feature_expansion", "configs/lightgbm_tuned.yaml"),
    "xgboost":  ("v2_feature_expansion", "configs/xgboost_tuned.yaml"),
    "catboost": ("baseline",             "configs/catboost_v2.yaml"),
    "logreg":   ("v3_extended_models",   "configs/logreg_v1.yaml"),
    "tabnet":   ("v3_extended_models",   "configs/tabnet_v1.yaml"),
}

#: Canonical model order — must match config.model_names from the ensemble run.
#: This order determines the column order of the meta-feature matrix.
_DEFAULT_MODEL_ORDER: List[str] = [
    "lightgbm",
    "xgboost",
    "catboost",
    "logreg",
    "tabnet",
]

#: Production ensemble run directory (relative to project root).
#: Override via LIQUIDITY_ENSEMBLE_RUN environment variable for staging.
_DEFAULT_ENSEMBLE_RUN: str = (
    "outputs/experiments/v5_ensemble/run_20260508_054540"
)

#: Root directory of all per-model Platt calibrator artefacts.
_CALIBRATION_DIR: str = "outputs/calibration"

#: Known OOF composite scores (0.6×LogLoss + 0.4×(1−AUC)) — for health reporting.
_KNOWN_OOF_SCORES: Dict[str, float] = {
    "lightgbm": 0.19557,
    "xgboost":  0.19350,
    "catboost": 0.19430,
    "logreg":   0.26003,
    "tabnet":   0.25296,
    "ensemble": 0.19144,
}

#: Production feature count expected from feature_engineering.py v2.2.1.
EXPECTED_FEATURE_COUNT: int = 825


# =============================================================================
# PLATT CALIBRATOR
# =============================================================================

class PlattCalibrator:
    """
    Backward-compatible Platt scaling calibrator.

    This class is defined here (not imported from src.inference.predict)
    to keep api/ self-contained and avoid pulling the full CLI machinery
    of predict.py into the server process.

    Supports two serialisation formats:
      - Legacy: attributes stored as self.slope / self.intercept (float)
      - Legacy alt: attributes stored as self.a / self.b (float)
      - sklearn LR wrapper: self._model is a fitted LogisticRegression

    The predict() method detects the format automatically and falls back
    to pass-through (slope=1, intercept=0) if neither is found — this
    matches the behaviour in predict.py's calibrate_predictions().

    Reference Platt parameters from OOF analysis (notebook 07/07b):
      lightgbm : slope ≈ 1.0  (well-calibrated, near-identity transform)
      xgboost  : slope ≈ 1.0
      catboost : slope ≈ 1.0
      logreg   : slope = 5.47,  intercept = −4.43  (36% LogLoss reduction)
      tabnet   : slope = 4.37,  intercept = −3.52  (21% LogLoss reduction)
    """

    def __init__(
        self,
        slope: Optional[float] = None,
        intercept: Optional[float] = None,
    ) -> None:
        self.slope     = slope
        self.intercept = intercept

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Apply sigmoid calibration: σ(slope × x + intercept).

        Accepts 1-D or 2-D input (2-D is reshaped automatically).
        Always returns a 1-D array.
        """
        x_flat = np.asarray(x).reshape(-1)

        # Resolution order for Platt parameters:
        #   1. self.slope / self.intercept  (standard format)
        #   2. self.a / self.b              (legacy alt format)
        #   3. 1.0 / 0.0                   (identity fallback)
        slope = (
            getattr(self, "slope", None)
            or getattr(self, "a", None)
            or 1.0
        )
        intercept = (
            getattr(self, "intercept", None)
            or getattr(self, "b", None)
            or 0.0
        )

        z = slope * x_flat + intercept
        return 1.0 / (1.0 + np.exp(-z))

    def __repr__(self) -> str:
        slope     = getattr(self, "slope", "?")
        intercept = getattr(self, "intercept", "?")
        return f"PlattCalibrator(slope={slope}, intercept={intercept})"


def _apply_calibrator(calibrator: Any, raw_preds: np.ndarray) -> np.ndarray:
    """
    Apply a Platt calibrator to raw fold-averaged predictions.

    Handles two calibrator types that may appear in serialised artefacts:

      Type A — sklearn LogisticRegression (fitted on reshape(-1,1) inputs):
        Detected by: hasattr(calibrator, "predict_proba")
        Applied via: calibrator.predict_proba(raw.reshape(-1,1))[:,1]
        Requires backward-compat patch for old pickles missing multi_class.

      Type B — Custom PlattCalibrator wrapper (defined above):
        Detected by: hasattr(calibrator, "predict") and not predict_proba
        Applied via: calibrator.predict(raw_preds)

    This dual-detection logic matches predict.py's calibrate_predictions()
    exactly and ensures the two modules never diverge in calibration behaviour.

    Parameters
    ----------
    calibrator : Any
        Loaded from outputs/calibration/{folder}/calibrator_platt.pkl.
    raw_preds  : np.ndarray, shape (n,)
        Raw fold-averaged predictions before calibration.

    Returns
    -------
    np.ndarray, shape (n,), clipped to [CLIP_LOW, CLIP_HIGH].
    """
    if hasattr(calibrator, "predict_proba"):
        # sklearn LogisticRegression — apply backward-compat patches
        if not hasattr(calibrator, "multi_class"):
            calibrator.multi_class = "auto"
        if not hasattr(calibrator, "classes_"):
            calibrator.classes_ = np.array([0, 1])
        cal_preds = calibrator.predict_proba(raw_preds.reshape(-1, 1))[:, 1]
    elif hasattr(calibrator, "predict"):
        cal_preds = calibrator.predict(raw_preds)
    else:
        log.warning(
            "Calibrator has neither predict_proba nor predict — "
            "using raw predictions without calibration."
        )
        cal_preds = raw_preds

    return np.clip(np.asarray(cal_preds).reshape(-1), CLIP_LOW, CLIP_HIGH)


# =============================================================================
# LOADED ENSEMBLE DATACLASS
# =============================================================================

@dataclass
class LoadedEnsemble:
    """
    Immutable container for all production inference artefacts.

    This is the single object stored in ``app.state.ensemble`` after
    startup.  Every field needed for the full 5-layer inference pipeline
    is present — no further disk I/O occurs after this object is created.

    Design note on per-model isolation
    ------------------------------------
    ``preprocessors`` and ``feature_lists`` are keyed per model (not shared)
    because each model was trained in an isolated run directory with its own
    fitted PreprocessingPipeline.  Although all five share the same 825-feature
    output from feature_engineering.py v2.2.1, their StandardScaler statistics
    (for LogReg/TabNet) differ from GBM preprocessors which have no scaler.
    Sharing a preprocessor across models would silently apply the wrong
    scaling to GBM inputs or strip the scaler from LogReg/TabNet inputs.

    ``fold_models[model_name]`` is a list of exactly N_FOLDS=5 models,
    in fold order (fold_0 through fold_4).  Fold averaging is performed in
    predictor.py by calling predict_proba on each and averaging.

    ``calibrators[model_name]`` holds the Platt calibrator fitted on the
    full OOF predictions for that model (notebooks 07 / 07b).  This is the
    same calibrator serialised in outputs/calibration/{folder}/calibrator_platt.pkl.

    ``weights`` maps model_name → optimised ensemble weight from the
    Scipy Nelder-Mead run (ensemble.py v5.1).  LogReg weight=0.0 is correct.

    Attributes
    ----------
    fold_models        : 5-key dict, each value a list of 5 model objects.
    preprocessors      : 5-key dict, each value a PreprocessingPipeline.
    calibrators        : 5-key dict, each value a Platt calibrator object.
    weights            : 5-key dict, model_name → float weight (sum ≈ 1.0).
    model_names        : ordered list loaded from metadata.json.
    feature_lists      : 5-key dict, model_name → list of 825 feature names.
    ensemble_version   : "v5.1" — from metadata.json.
    oof_score          : 0.19144 — best OOF composite score from metadata.
    load_time_sec      : wall-clock seconds from first call to return.
    loaded_at          : ISO-format UTC timestamp of the load completion.
    models_loaded_ok   : per-model bool — all True if load was successful.
    run_id             : ensemble run directory name (e.g. run_20260508_054540).
    """

    # ── Core inference artefacts ──────────────────────────────────────────────
    fold_models:    Dict[str, List[Any]]
    preprocessors:  Dict[str, Any]
    calibrators:    Dict[str, Any]
    weights:        Dict[str, float]
    model_names:    List[str]
    feature_lists:  Dict[str, List[str]]

    # ── Operational metadata ──────────────────────────────────────────────────
    ensemble_version:  str
    oof_score:         float
    load_time_sec:     float
    loaded_at:         str
    models_loaded_ok:  Dict[str, bool]
    run_id:            str

    # ── Derived summary (populated post-init) ─────────────────────────────────
    total_fold_models:     int = field(init=False)
    total_preprocessors:   int = field(init=False)
    total_calibrators:     int = field(init=False)

    def __post_init__(self) -> None:
        self.total_fold_models   = sum(len(v) for v in self.fold_models.values())
        self.total_preprocessors = len(self.preprocessors)
        self.total_calibrators   = len(self.calibrators)

    # ── Public helpers ────────────────────────────────────────────────────────

    def get_model_weight(self, model_name: str) -> float:
        """Return the ensemble weight for a model; 0.0 if not found."""
        return self.weights.get(model_name, 0.0)

    def is_fully_loaded(self) -> bool:
        """True if every model in model_names loaded all artefacts successfully."""
        return all(self.models_loaded_ok.get(m, False) for m in self.model_names)

    def summary(self) -> Dict[str, Any]:
        """
        Return a structured summary dict for the /health endpoint.

        This is the canonical data source for HealthResponse construction
        in app.py — no string parsing or re-computation needed there.
        """
        return {
            "ensemble_version":    self.ensemble_version,
            "oof_score":           self.oof_score,
            "run_id":              self.run_id,
            "model_names":         self.model_names,
            "weights":             self.weights,
            "total_fold_models":   self.total_fold_models,
            "total_preprocessors": self.total_preprocessors,
            "total_calibrators":   self.total_calibrators,
            "models_loaded_ok":    self.models_loaded_ok,
            "load_time_sec":       self.load_time_sec,
            "loaded_at":           self.loaded_at,
            "is_fully_loaded":     self.is_fully_loaded(),
        }

    def __repr__(self) -> str:
        status = "FULLY LOADED" if self.is_fully_loaded() else "DEGRADED"
        return (
            f"LoadedEnsemble("
            f"version={self.ensemble_version}, "
            f"score={self.oof_score:.5f}, "
            f"fold_models={self.total_fold_models}/25, "
            f"status={status}, "
            f"load_time={self.load_time_sec:.1f}s"
            f")"
        )


# =============================================================================
# PRIVATE HELPERS — RUN DIRECTORY RESOLUTION
# =============================================================================

def _resolve_model_run_dir(
    model_name:   str,
    project_root: Path,
) -> Path:
    """
    Resolve the most recent training run directory for a given model family.

    Search strategy (in order):
      1. Primary stage from _MODEL_STAGE_MAP (e.g. v2_feature_expansion)
      2. Fallback to "baseline" (catches catboost and edge cases)

    Within each stage, runs are sorted lexicographically on the
    run_YYYYMMDD_HHMMSS timestamp — the most recent run is selected.
    This is deterministic across platforms (no locale dependency).

    Parameters
    ----------
    model_name   : one of lightgbm, xgboost, catboost, logreg, tabnet.
    project_root : resolved absolute project root Path.

    Returns
    -------
    Path to the most recent run directory for this model.

    Raises
    ------
    ValueError
        If model_name is not in _MODEL_STAGE_MAP.
    FileNotFoundError
        If no run directory exists in any searched location.
    """
    if model_name not in _MODEL_STAGE_MAP:
        raise ValueError(
            f"Unknown model '{model_name}'.  "
            f"Known models: {list(_MODEL_STAGE_MAP.keys())}"
        )

    stage, _ = _MODEL_STAGE_MAP[model_name]
    experiments_root = project_root / "outputs" / "experiments"

    # ── Primary stage search ─────────────────────────────────────────────────
    primary_dir = experiments_root / stage / model_name
    if primary_dir.exists():
        run_dirs = sorted(primary_dir.glob("run_*"), reverse=True)
        if run_dirs:
            log.debug(
                "  [%s] run resolved: stage=%s  dir=%s",
                model_name, stage, run_dirs[0].name,
            )
            return run_dirs[0]

    # ── Fallback to baseline ─────────────────────────────────────────────────
    fallback_dir = experiments_root / "baseline" / model_name
    if fallback_dir.exists():
        run_dirs = sorted(fallback_dir.glob("run_*"), reverse=True)
        if run_dirs:
            log.warning(
                "  [%s] Primary stage '%s' had no runs. "
                "Using baseline run: %s",
                model_name, stage, run_dirs[0].name,
            )
            return run_dirs[0]

    raise FileNotFoundError(
        f"\n"
        f"No training run directory found for '{model_name}'.\n"
        f"Searched:\n"
        f"  {experiments_root / stage / model_name}\n"
        f"  {experiments_root / 'baseline' / model_name}\n"
        f"Re-run training:\n"
        f"  python -m src.orchestration.run_all_models "
        f"--configs configs/{model_name}_v*.yaml\n"
        f"Or check that the artefacts are present in the Docker image."
    )


# =============================================================================
# PRIVATE HELPERS — INDIVIDUAL ARTEFACT LOADERS
# =============================================================================

def _load_tabnet_fold(fold_path: Path) -> Any:
    """
    Load a single TabNet fold model from its .zip artefact.

    CRITICAL — do NOT use joblib.load() here.
    pytorch_tabnet serialises via its own internal mechanism into a .zip
    archive.  joblib.load() on a .zip loads raw bytes and returns a broken
    object that calls without error but produces garbage predictions.

    Always use TabNetClassifier().load_model(str(zip_path)).

    The .zip extension IS included in fold_path (the path ends in
    tabnet_fold_N.zip).  TabNet's load_model() accepts the full path
    including the extension — unlike some versions that strip it.
    If a FileNotFoundError is raised with the full path, retry without
    the .zip extension (see the try/except below).

    Parameters
    ----------
    fold_path : Path ending in tabnet_fold_N.zip.

    Returns
    -------
    Fitted TabNetClassifier instance.

    Raises
    ------
    ImportError  if pytorch_tabnet is not installed.
    FileNotFoundError if the .zip file is not present at fold_path.
    RuntimeError if load_model() raises an unexpected exception.
    """
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
    except ImportError as exc:
        raise ImportError(
            "\n"
            "pytorch_tabnet is not installed.  Install with:\n"
            "  pip install pytorch-tabnet torch\n"
            "For CPU-only (Docker, no GPU):\n"
            "  pip install pytorch-tabnet\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
        ) from exc

    if not fold_path.exists():
        raise FileNotFoundError(
            f"TabNet fold .zip not found: {fold_path}\n"
            "Re-run TabNet training:\n"
            "  python -m src.orchestration.run_all_models "
            "--configs configs/tabnet_v1.yaml"
        )

    model = TabNetClassifier()

    # TabNet's load_model() has version-dependent behaviour regarding the
    # .zip extension.  Try with the full path first; if it raises a generic
    # exception (not FileNotFoundError), retry without the extension.
    try:
        model.load_model(str(fold_path))
    except FileNotFoundError:
        # Some versions of pytorch_tabnet expect the path WITHOUT .zip
        path_no_ext = str(fold_path).replace(".zip", "")
        log.debug(
            "  TabNet load_model() failed with .zip extension; "
            "retrying without extension: %s", path_no_ext
        )
        model.load_model(path_no_ext)
    except Exception as exc:
        raise RuntimeError(
            f"TabNet load_model() raised an unexpected error for {fold_path}:\n"
            f"  {type(exc).__name__}: {exc}\n"
            "This may indicate a pytorch or pytorch_tabnet version mismatch.  "
            "Verify the environment matches the training environment:\n"
            "  conda activate liquidity_ml\n"
            "  python -c 'import pytorch_tabnet; print(pytorch_tabnet.__version__)'"
        ) from exc

    return model


def _load_fold_models(
    model_name: str,
    run_dir:    Path,
) -> List[Any]:
    """
    Load all N_FOLDS fold models for one model family from a run directory.

    Fold model file conventions (from cv.py save_cv_outputs):
      lightgbm / xgboost / catboost / logreg :
          run_dir/models/{name}/{name}_fold_{k}.pkl   (k = 0 .. 4)
      tabnet :
          run_dir/models/tabnet/tabnet_fold_{k}.zip   (pytorch_tabnet .zip)

    Validates that exactly N_FOLDS=5 files are found — not 4, not 6.
    An incomplete fold set means a training run was interrupted and the
    model would produce biased predictions.

    Parameters
    ----------
    model_name : model family identifier string.
    run_dir    : resolved path to the model's most recent run directory.

    Returns
    -------
    List of exactly 5 loaded model objects, in fold order (fold_0..fold_4).

    Raises
    ------
    FileNotFoundError if the models/ subdirectory does not exist.
    ValueError if the fold count is not exactly N_FOLDS.
    """
    is_tabnet = model_name == "tabnet"
    ext       = ".zip" if is_tabnet else ".pkl"
    model_dir = run_dir / "models" / model_name

    if not model_dir.exists():
        raise FileNotFoundError(
            f"\n"
            f"Model directory not found: {model_dir}\n"
            f"Expected structure:\n"
            f"  {run_dir}/models/{model_name}/{model_name}_fold_0..4{ext}\n"
            f"The run directory exists but the models/ subdirectory is absent.  "
            f"Re-run training for '{model_name}'."
        )

    pattern    = f"{model_name}_fold_*{ext}"
    fold_paths = sorted(model_dir.glob(pattern))

    if len(fold_paths) != N_FOLDS:
        raise ValueError(
            f"\n"
            f"Expected exactly {N_FOLDS} fold models for '{model_name}', "
            f"found {len(fold_paths)}.\n"
            f"Directory: {model_dir}\n"
            f"Pattern searched: {pattern}\n"
            f"Found: {[fp.name for fp in fold_paths]}\n"
            f"An incomplete fold set indicates an interrupted training run.  "
            f"Re-run training to regenerate all {N_FOLDS} folds."
        )

    loaded_models = []
    for k, fp in enumerate(fold_paths):
        if is_tabnet:
            m = _load_tabnet_fold(fp)
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
                m = joblib.load(fp)
        loaded_models.append(m)
        log.debug("    fold %d loaded: %s", k, fp.name)

    log.info(
        "  [%s] %d fold models loaded from %s/",
        model_name, len(loaded_models), model_dir.relative_to(PROJECT_ROOT),
    )
    return loaded_models


def _load_preprocessor(
    model_name: str,
    run_dir:    Path,
) -> Any:
    """
    Load the fitted PreprocessingPipeline for one model.

    Critical contract — per-model preprocessor isolation:
      Each model's run directory contains its OWN fitted preprocessor.
      LogReg and TabNet preprocessors include a fitted StandardScaler
      (scale_features=True).  GBM preprocessors do NOT have a scaler.
      Sharing preprocessors across model families MUST NEVER happen —
      it would silently apply the wrong scaling:
        GBM preprocessor on LogReg → LogReg receives raw-scale inputs →
        ElasticNet penalty applied inconsistently → predictions collapse.
        LogReg preprocessor on XGBoost → XGBoost receives StandardScaled
        inputs → trees still work but feature importance is distorted.

    Backward-compatibility patches applied (matches predict.py exactly):
      - If scale_features attribute is missing: set to False
      - If scaler_ attribute is missing: set to None
    These patches handle preprocessors serialised before v2.0.

    Scale flag validation:
      Raises ValueError if the loaded preprocessor's scale_features flag
      disagrees with what's expected for this model family.  This converts
      a silent wrong-preprocessor bug into an explicit startup failure.

    Parameters
    ----------
    model_name : model family identifier.
    run_dir    : resolved path to the model's most recent run directory.

    Returns
    -------
    Fitted PreprocessingPipeline, ready for .transform() calls.

    Raises
    ------
    FileNotFoundError if preprocessor.pkl is absent.
    ValueError if scale_features flag is inconsistent with model family.
    """
    preproc_path = run_dir / "preprocessor.pkl"

    if not preproc_path.exists():
        raise FileNotFoundError(
            f"\n"
            f"preprocessor.pkl not found: {preproc_path}\n"
            f"This must be the FITTED PreprocessingPipeline from training.\n"
            f"Re-run training for '{model_name}' to regenerate this artefact."
        )

    # Import here (not at module level) to keep api/ decoupled from src/
    # at import time.  The server process has src/ on its PYTHONPATH.
    from src.preprocessing.preprocessing import PreprocessingPipeline  # noqa: PLC0415

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        preproc = PreprocessingPipeline.load(str(preproc_path))

    # ── Backward-compat patches ────────────────────────────────────────────
    if not hasattr(preproc, "scale_features"):
        preproc.scale_features = False
        log.debug(
            "  [%s] Applied compat patch: scale_features=False "
            "(attribute absent in loaded pickle)",
            model_name,
        )
    if not hasattr(preproc, "scaler_"):
        preproc.scaler_ = None
        log.debug(
            "  [%s] Applied compat patch: scaler_=None "
            "(attribute absent in loaded pickle)",
            model_name,
        )

    # ── Scale flag consistency check ──────────────────────────────────────
    expects_scale = model_name in _SCALE_SENSITIVE_MODELS
    has_scale     = bool(getattr(preproc, "scale_features", False))

    if expects_scale and not has_scale:
        raise ValueError(
            f"\n"
            f"Preprocessor mismatch for '{model_name}':\n"
            f"  Expected scale_features=True  (this model requires StandardScaler)\n"
            f"  Found    scale_features=False  at {preproc_path}\n"
            f"A GBM preprocessor was likely loaded for '{model_name}'.\n"
            f"Check that _resolve_model_run_dir() returned the correct run directory.\n"
            f"Expected stage: {_MODEL_STAGE_MAP[model_name][0]}"
        )
    if not expects_scale and has_scale:
        log.warning(
            "[%s] Preprocessor has scale_features=True but this model "
            "family does not require scaling.  Predictions will still be "
            "generated — verify the run directory is correct.  "
            "Run dir: %s",
            model_name,
            run_dir.name,
        )

    n_features = len(preproc.feature_list) if preproc.feature_list else "unknown"
    log.info(
        "  [%s] Preprocessor loaded: features=%s  scale=%s  "
        "clip_cols=%d  constant_cols=%d",
        model_name,
        n_features,
        has_scale,
        len(getattr(preproc, "clip_cols_", [])),
        len(getattr(preproc, "constant_cols_", [])),
    )
    return preproc


def _load_calibrator(
    model_name:   str,
    project_root: Path,
) -> Any:
    """
    Load the Platt calibrator for one model from the calibration artefact tree.

    Calibrator path pattern:
      outputs/calibration/{folder}/calibrator_platt.pkl
      where folder = _CALIBRATION_FOLDER_MAP[model_name]
      (lightgbm→lgb, xgboost→xgb, catboost→cat, logreg→logreg, tabnet→tabnet)

    Legacy serialisation handling:
      The calibrator_platt.pkl files saved by notebooks 07 / 07b may be
      either sklearn LogisticRegression objects (wrapped to accept scalar
      inputs) or custom PlattCalibrator instances.  Before loading, we
      inject our PlattCalibrator class into __main__ so joblib can resolve
      the type reference during unpickling.  This pattern mirrors predict.py.

    Parameters
    ----------
    model_name   : model family identifier.
    project_root : resolved absolute project root Path.

    Returns
    -------
    Loaded calibrator object (sklearn LR or PlattCalibrator).

    Raises
    ------
    KeyError         if model_name has no entry in _CALIBRATION_FOLDER_MAP.
    FileNotFoundError if calibrator_platt.pkl is absent.
    """
    if model_name not in _CALIBRATION_FOLDER_MAP:
        raise KeyError(
            f"No calibration folder mapping for '{model_name}'.  "
            f"Known mappings: {_CALIBRATION_FOLDER_MAP}"
        )

    folder   = _CALIBRATION_FOLDER_MAP[model_name]
    cal_path = project_root / _CALIBRATION_DIR / folder / "calibrator_platt.pkl"

    if not cal_path.exists():
        raise FileNotFoundError(
            f"\n"
            f"Platt calibrator not found for '{model_name}':\n"
            f"  Expected: {cal_path}\n"
            f"Regenerate with:\n"
            f"  GBMs  → run 07_calibration_analysis.ipynb\n"
            f"  Other → run 11_calibration_analysis_v2_tabnet_logreg.ipynb\n"
            f"Then re-build the Docker image."
        )

    # Inject PlattCalibrator into __main__ so joblib can resolve the type
    # if the calibrator was serialised as a PlattCalibrator instance.
    import __main__ as _main  # noqa: PLC0415
    _main.PlattCalibrator = PlattCalibrator

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        calibrator = joblib.load(cal_path)

    # Log calibrator type and Platt parameters for startup diagnostics.
    cal_type  = type(calibrator).__name__
    slope     = getattr(calibrator, "slope",     getattr(calibrator, "a", "?"))
    intercept = getattr(calibrator, "intercept", getattr(calibrator, "b", "?"))

    # For sklearn LR calibrators, extract slope/intercept from coef_/intercept_
    if hasattr(calibrator, "coef_") and slope == "?":
        try:
            slope     = float(calibrator.coef_[0][0])
            intercept = float(calibrator.intercept_[0])
        except (IndexError, AttributeError):
            pass

    log.info(
        "  [%s] Calibrator loaded: type=%s  slope=%s  intercept=%s",
        model_name, cal_type,
        f"{slope:.4f}" if isinstance(slope, float) else slope,
        f"{intercept:.4f}" if isinstance(intercept, float) else intercept,
    )
    return calibrator


def _load_feature_list(
    model_name: str,
    run_dir:    Path,
) -> List[str]:
    """
    Load the ordered feature list for one model from its run directory.

    feature_list.json records the exact column order that the model was
    trained on.  Using this list at inference time guarantees that the
    feature matrix passed to each model has columns in the identical order
    to training — even if pandas column ordering changes across versions.

    If feature_list.json is absent (pre-v2.0 run), a warning is issued
    and an empty list is returned.  The predictor handles an empty list
    by falling back to the natural column order of the engineered DataFrame.

    Validates:
      - Feature count matches EXPECTED_FEATURE_COUNT (825).
        Mismatch indicates a feature_engineering.py version divergence
        between the run that produced this artefact and the current codebase.

    Parameters
    ----------
    model_name : model family identifier (for log messages).
    run_dir    : resolved path to the model's most recent run directory.

    Returns
    -------
    List of 825 feature name strings (or empty list if file absent).
    """
    fl_path = run_dir / "feature_list.json"

    if not fl_path.exists():
        log.warning(
            "[%s] feature_list.json not found at %s.  "
            "Predictor will use natural column order — verify feature "
            "engineering version matches training.",
            model_name, run_dir.name,
        )
        return []

    with open(fl_path) as f:
        feature_list: List[str] = json.load(f)

    if len(feature_list) != EXPECTED_FEATURE_COUNT:
        log.warning(
            "[%s] feature_list.json has %d features; expected %d.  "
            "This may indicate a feature_engineering.py version mismatch.  "
            "Training used v2.2.1 which produces %d features.",
            model_name,
            len(feature_list),
            EXPECTED_FEATURE_COUNT,
            EXPECTED_FEATURE_COUNT,
        )
    else:
        log.debug(
            "  [%s] feature_list.json: %d features verified",
            model_name, len(feature_list),
        )

    return feature_list


def _load_ensemble_config(
    project_root:  Path,
    ensemble_run:  str,
) -> Tuple[Dict[str, float], List[str], str, float, str]:
    """
    Load ensemble weights, model order, and version metadata.

    Reads two artefact files from the production ensemble run directory:
      ensemble_weights.json  → weights["optimised"] dict + model_order
      metadata.json          → model_names, ensemble_version, best_score,
                               best_strategy

    Verifies:
      1. best_strategy == "optimised_weighted_average"
         (warns if not — prevents accidentally serving a stacking submission)
      2. Weights sum to approximately 1.0
         (normalises automatically if drift detected, with warning)
      3. All model names in metadata.json are present in the weights dict
         (bidirectional check — same logic as predict.py)

    Parameters
    ----------
    project_root  : resolved absolute project root Path.
    ensemble_run  : relative path to ensemble run directory (str).

    Returns
    -------
    Tuple of:
      weights           : Dict[str, float]  — model_name → weight
      model_names       : List[str]         — canonical ordered model list
      ensemble_version  : str               — e.g. "v5.1"
      oof_score         : float             — best composite score (0.19144)
      run_id            : str               — run directory name

    Raises
    ------
    FileNotFoundError if either JSON file is absent.
    ValueError if the weight dict is malformed.
    """
    run_path = project_root / ensemble_run

    if not run_path.exists():
        raise FileNotFoundError(
            f"\n"
            f"Ensemble run directory not found: {run_path}\n"
            f"Set LIQUIDITY_ENSEMBLE_RUN env var to override the default:\n"
            f"  export LIQUIDITY_ENSEMBLE_RUN=outputs/experiments/v5_ensemble/run_YYYYMMDD_HHMMSS\n"
            f"Or run notebook 09_ensemble.ipynb to generate a new run."
        )

    # ── Load ensemble_weights.json ─────────────────────────────────────────
    weights_path = run_path / "ensemble_weights.json"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"\n"
            f"ensemble_weights.json not found: {weights_path}\n"
            f"This file is required to determine per-model blend weights.\n"
            f"Re-run ensemble.py (notebook 09_ensemble.ipynb) to regenerate."
        )

    with open(weights_path) as f:
        weights_json = json.load(f)

    weights: Dict[str, float]    = weights_json.get("optimised", {})
    model_order: List[str]       = weights_json.get("model_order", list(weights.keys()))

    if not weights:
        raise ValueError(
            f"ensemble_weights.json has an empty 'optimised' dict: {weights_path}\n"
            "Expected format: {{\"optimised\": {{\"lightgbm\": 0.178, ...}}}}"
        )

    # ── Load metadata.json ─────────────────────────────────────────────────
    meta_path = run_path / "metadata.json"
    if not meta_path.exists():
        log.warning(
            "metadata.json not found at %s.  "
            "Using model_order from ensemble_weights.json and default version.",
            meta_path,
        )
        return (
            weights,
            model_order,
            "v5.1",
            _KNOWN_OOF_SCORES["ensemble"],
            run_path.name,
        )

    with open(meta_path) as f:
        metadata = json.load(f)

    model_names      = metadata.get("model_names", model_order)
    ensemble_version = metadata.get("ensemble_version", "v5.1")
    oof_score        = float(metadata.get("best_score", _KNOWN_OOF_SCORES["ensemble"]))
    best_strategy    = metadata.get("best_strategy", "")

    # ── Strategy verification ──────────────────────────────────────────────
    if best_strategy and best_strategy != "optimised_weighted_average":
        log.warning(
            "Ensemble metadata reports best_strategy='%s' (score=%.5f), "
            "not 'optimised_weighted_average'.  "
            "Serving optimised weights regardless — verify this is intentional.",
            best_strategy, oof_score,
        )
    else:
        log.info(
            "  Ensemble strategy confirmed: optimised_weighted_average  "
            "OOF score=%.5f",
            oof_score,
        )

    # ── Log weights table ──────────────────────────────────────────────────
    log.info("  Ensemble weights:")
    for m in model_names:
        w = weights.get(m, 0.0)
        note = " ← dominant" if w == max(weights.values()) else ""
        zero = " (zero weight — diversity only)" if w == 0.0 else ""
        log.info("    %-12s : %.5f%s%s", m, w, note, zero)

    # ── Weight sum validation ──────────────────────────────────────────────
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-4:
        log.warning(
            "Ensemble weights sum to %.6f (expected 1.0).  "
            "Normalising automatically.",
            total,
        )
        weights = {m: w / total for m, w in weights.items()}

    # ── Bidirectional weight–model_names check ─────────────────────────────
    missing_weights = [m for m in model_names if m not in weights]
    extra_weights   = [m for m in weights     if m not in model_names]
    if missing_weights:
        raise ValueError(
            f"Model names from metadata.json have no weight entry: {missing_weights}\n"
            f"ensemble_weights.json must contain all models in model_names."
        )
    if extra_weights:
        log.warning(
            "ensemble_weights.json contains models not in metadata model_names: %s.  "
            "Extra entries will be ignored.",
            extra_weights,
        )

    return weights, model_names, ensemble_version, oof_score, run_path.name


# =============================================================================
# INTEGRITY VALIDATION
# =============================================================================

def _validate_ensemble_integrity(ensemble: LoadedEnsemble) -> None:
    """
    Post-load integrity validation — called once inside load_ensemble().

    Checks performed (all use if/raise — never assert):
      1. All model_names have fold_models, preprocessors, calibrators entries.
      2. Each model has exactly N_FOLDS fold models.
      3. Weights sum to approximately 1.0 (within 1e-4).
      4. Weights and model_names are bidirectionally consistent.
      5. Feature list lengths match EXPECTED_FEATURE_COUNT (warns if not).
      6. All models_loaded_ok flags are True.

    Parameters
    ----------
    ensemble : Fully constructed LoadedEnsemble (all fields set).

    Raises
    ------
    ValueError with a specific message on any structural failure.
    """
    errors: List[str] = []

    # Check 1 — all models have all artefact types
    for m in ensemble.model_names:
        if m not in ensemble.fold_models:
            errors.append(f"Missing fold_models entry for '{m}'")
        if m not in ensemble.preprocessors:
            errors.append(f"Missing preprocessors entry for '{m}'")
        if m not in ensemble.calibrators:
            errors.append(f"Missing calibrators entry for '{m}'")

    # Check 2 — fold count
    for m, folds in ensemble.fold_models.items():
        if len(folds) != N_FOLDS:
            errors.append(
                f"'{m}' has {len(folds)} fold models; expected {N_FOLDS}"
            )

    # Check 3 — weight sum
    total = sum(ensemble.weights.values())
    if abs(total - 1.0) > 1e-4:
        errors.append(
            f"Ensemble weights sum to {total:.6f}; expected 1.0 (±1e-4)"
        )

    # Check 4 — bidirectional weight ↔ model_names
    for m in ensemble.model_names:
        if m not in ensemble.weights:
            errors.append(f"No weight entry for model '{m}'")

    # Check 5 — feature list lengths (warning, not error)
    for m, fl in ensemble.feature_lists.items():
        if fl and len(fl) != EXPECTED_FEATURE_COUNT:
            log.warning(
                "[INTEGRITY] '%s' feature list has %d features; "
                "expected %d (feature_engineering.py v2.2.1).",
                m, len(fl), EXPECTED_FEATURE_COUNT,
            )

    # Check 6 — all loaded_ok flags
    failed_models = [m for m, ok in ensemble.models_loaded_ok.items() if not ok]
    if failed_models:
        errors.append(
            f"These models did not load cleanly: {failed_models}"
        )

    if errors:
        error_block = "\n  ".join(errors)
        raise ValueError(
            f"\n"
            f"LoadedEnsemble integrity validation FAILED ({len(errors)} error(s)):\n"
            f"  {error_block}\n"
            f"The server cannot safely serve predictions in this state.  "
            f"Fix the above errors and restart."
        )

    log.info(
        "[INTEGRITY] All checks passed: "
        "%d models × %d folds | %d preprocessors | %d calibrators",
        len(ensemble.model_names),
        N_FOLDS,
        len(ensemble.preprocessors),
        len(ensemble.calibrators),
    )


# =============================================================================
# PUBLIC API — STARTUP DIAGNOSTICS
# =============================================================================

def _log_startup_banner(model_names: List[str], project_root: Path) -> None:
    """Log a formatted startup banner before artefact loading begins."""
    log.info("=" * 72)
    log.info("AI4EAC LIQUIDITY STRESS — MODEL LOADER  v1.0.0")
    log.info("=" * 72)
    log.info("Project root   : %s", project_root)
    log.info("Models to load : %s", model_names)
    log.info(
        "Artefact target: %d fold models + %d preprocessors + %d calibrators",
        len(model_names) * N_FOLDS,
        len(model_names),
        len(model_names),
    )
    log.info("-" * 72)


def _log_startup_summary(ensemble: LoadedEnsemble) -> None:
    """Log a formatted summary table after all artefacts are loaded."""
    log.info("=" * 72)
    log.info("ENSEMBLE LOADED SUCCESSFULLY")
    log.info("=" * 72)
    log.info(
        "%-12s | %-6s | %-8s | %-6s | %-8s | %s",
        "Model", "Folds", "Scale", "Weight", "OOF Score", "Status",
    )
    log.info("-" * 72)
    for m in ensemble.model_names:
        n_folds   = len(ensemble.fold_models.get(m, []))
        scale     = getattr(ensemble.preprocessors.get(m), "scale_features", False)
        weight    = ensemble.weights.get(m, 0.0)
        oof       = _KNOWN_OOF_SCORES.get(m, float("nan"))
        ok        = ensemble.models_loaded_ok.get(m, False)
        status    = "OK" if ok else "FAILED"
        log.info(
            "%-12s | %-6d | %-8s | %-6.5f | %-8.5f | %s",
            m.upper(), n_folds, str(scale), weight, oof, status,
        )
    log.info("-" * 72)
    log.info("Total fold models  : %d / %d", ensemble.total_fold_models, len(ensemble.model_names) * N_FOLDS)
    log.info("Total preprocessors: %d / %d", ensemble.total_preprocessors, len(ensemble.model_names))
    log.info("Total calibrators  : %d / %d", ensemble.total_calibrators,   len(ensemble.model_names))
    log.info("Ensemble version   : %s", ensemble.ensemble_version)
    log.info("OOF score (ref)    : %.5f  (0.6×LogLoss + 0.4×(1−AUC))", ensemble.oof_score)
    log.info("Load time          : %.2fs", ensemble.load_time_sec)
    log.info("Loaded at          : %s UTC", ensemble.loaded_at)
    log.info("=" * 72)


# =============================================================================
# MAIN PUBLIC FUNCTION — @lru_cache SINGLETON LOADER
# =============================================================================

@lru_cache(maxsize=1)
def load_ensemble(project_root: Optional[str] = None) -> LoadedEnsemble:
    """
    Load all production model artefacts and return a ``LoadedEnsemble``.

    This function is the ONLY entry point for accessing ensemble artefacts
    at inference time.  It is decorated with ``@lru_cache(maxsize=1)`` so
    the expensive disk I/O runs exactly once — on the first call from
    FastAPI's lifespan startup hook.  All subsequent calls return the cached
    result immediately, including concurrent requests from multiple threads.

    Argument is typed as ``Optional[str]`` (not ``Path``) because lru_cache
    uses arguments as hash keys.  Internally, the string is converted to a
    ``Path`` object immediately.

    Environment variable override
    ------------------------------
    Set ``LIQUIDITY_ENSEMBLE_RUN`` to override the default production run
    directory.  Useful for staging environments or A/B testing:

        export LIQUIDITY_ENSEMBLE_RUN=outputs/experiments/v5_ensemble/run_20260509_044838
        uvicorn api.app:app --host 0.0.0.0 --port 8000

    Loading order
    -------------
    1.  Resolve project root and ensemble run directory.
    2.  Load ensemble config: weights, model_names, version from JSON.
    3.  For each model in model_names (order preserved from metadata.json):
        a.  Resolve run directory (most recent matching run_*).
        b.  Load N_FOLDS fold models.
        c.  Load fitted preprocessor (with scale_features validation).
        d.  Load Platt calibrator (with type detection).
        e.  Load feature list (with count validation).
    4.  Construct LoadedEnsemble dataclass.
    5.  Run integrity validation (post-load checks).
    6.  Log formatted summary table.
    7.  Return the validated LoadedEnsemble.

    Crash policy
    ------------
    Any exception during loading propagates immediately and crashes the
    server.  This is intentional — a partially-loaded ensemble serving
    wrong predictions is far more dangerous than a server that refuses
    to start with a clear error message.

    Test usage
    ----------
    To force a fresh load in tests (e.g. after swapping artefacts):
        from api.model_loader import load_ensemble
        load_ensemble.cache_clear()
        ensemble = load_ensemble("/path/to/test/project")

    Parameters
    ----------
    project_root : str, optional
        Absolute path to the project root directory.  Defaults to the
        directory two levels above this file (i.e. the project root
        when the standard ``api/model_loader.py`` layout is used).

    Returns
    -------
    LoadedEnsemble
        Immutable, fully-validated ensemble artefact container.

    Raises
    ------
    FileNotFoundError  : any required artefact file is missing.
    ValueError         : integrity validation fails post-load.
    ImportError        : pytorch_tabnet is not installed.
    RuntimeError       : TabNet load_model() raises an unexpected error.
    """
    start_time = time.perf_counter()

    # ── 0. Resolve project root ────────────────────────────────────────────
    root: Path = (
        Path(project_root).resolve()
        if project_root is not None
        else PROJECT_ROOT
    )

    # ── 0b. Resolve ensemble run directory ────────────────────────────────
    # Environment variable override takes precedence over the compiled default.
    ensemble_run: str = os.environ.get("LIQUIDITY_ENSEMBLE_RUN", _DEFAULT_ENSEMBLE_RUN)
    log.info("Ensemble run directory: %s", ensemble_run)

    # ── 1. Load ensemble configuration ────────────────────────────────────
    log.info("Loading ensemble configuration from metadata.json / ensemble_weights.json ...")
    weights, model_names, ensemble_version, oof_score, run_id = _load_ensemble_config(
        project_root=root,
        ensemble_run=ensemble_run,
    )

    _log_startup_banner(model_names, root)

    # ── 2. Per-model artefact loading ──────────────────────────────────────
    fold_models:     Dict[str, List[Any]] = {}
    preprocessors:   Dict[str, Any]       = {}
    calibrators:     Dict[str, Any]       = {}
    feature_lists:   Dict[str, List[str]] = {}
    models_loaded_ok: Dict[str, bool]     = {}

    for model_name in model_names:
        log.info("")
        log.info("─── Loading artefacts for: %s ───", model_name.upper())

        # a. Resolve run directory
        run_dir = _resolve_model_run_dir(model_name, root)
        log.info("  Run directory: %s", run_dir.name)

        # b. Fold models
        fold_models[model_name] = _load_fold_models(model_name, run_dir)

        # c. Preprocessor
        preprocessors[model_name] = _load_preprocessor(model_name, run_dir)

        # d. Platt calibrator
        calibrators[model_name] = _load_calibrator(model_name, root)

        # e. Feature list
        feature_lists[model_name] = _load_feature_list(model_name, run_dir)

        models_loaded_ok[model_name] = True

    # ── 3. Compute load time before constructing the dataclass ────────────
    # Record NOW (before post-init __post_init__ and integrity validation)
    # so load_time_sec reflects actual disk I/O, not validation overhead.
    load_time_sec = round(time.perf_counter() - start_time, 3)

    # ── 4. Construct LoadedEnsemble ────────────────────────────────────────
    ensemble = LoadedEnsemble(
        fold_models      = fold_models,
        preprocessors    = preprocessors,
        calibrators      = calibrators,
        weights          = weights,
        model_names      = model_names,
        feature_lists    = feature_lists,
        ensemble_version = ensemble_version,
        oof_score        = oof_score,
        load_time_sec    = load_time_sec,
        loaded_at        = datetime.now(tz=timezone.utc).isoformat(),
        models_loaded_ok = models_loaded_ok,
        run_id           = run_id,
    )

    # ── 5. Post-load integrity validation ─────────────────────────────────
    log.info("")
    log.info("[INTEGRITY] Running post-load validation ...")
    _validate_ensemble_integrity(ensemble)

    # ── 6. Log summary table ───────────────────────────────────────────────
    _log_startup_summary(ensemble)

    return ensemble


# =============================================================================
# PUBLIC UTILITY FUNCTIONS (used by app.py and predictor.py)
# =============================================================================

def get_artefact_inventory(ensemble: LoadedEnsemble) -> Dict[str, Any]:
    """
    Return a structured artefact inventory for the /health endpoint.

    This is the canonical data source for populating ``ArtefactCounts``
    in schemas.py.  No re-computation occurs — all values are derived
    from the already-loaded ``LoadedEnsemble``.

    Parameters
    ----------
    ensemble : The loaded ensemble returned by ``load_ensemble()``.

    Returns
    -------
    Dict matching the fields of ``api.schemas.ArtefactCounts``:
      fold_models_loaded       : int  — total fold models (expected: 25)
      preprocessors_loaded     : int  — preprocessor count (expected: 5)
      calibrators_loaded       : int  — calibrator count (expected: 5)
      ensemble_weights_loaded  : bool — True if weights dict is non-empty
      total_artefact_size_mb   : None — expensive to compute; excluded
    """
    return {
        "fold_models_loaded":      ensemble.total_fold_models,
        "preprocessors_loaded":    ensemble.total_preprocessors,
        "calibrators_loaded":      ensemble.total_calibrators,
        "ensemble_weights_loaded": bool(ensemble.weights),
        "total_artefact_size_mb":  None,
    }


def get_per_model_status(ensemble: LoadedEnsemble) -> Dict[str, bool]:
    """
    Return per-model load status for the /health endpoint.

    Used to populate ``HealthResponse.models_loaded`` in schemas.py.

    Returns
    -------
    Dict mapping model_name → bool (True = all artefacts loaded successfully).
    """
    return dict(ensemble.models_loaded_ok)


def is_ready(ensemble: Optional[LoadedEnsemble]) -> bool:
    """
    Return True if the ensemble is fully loaded and ready to serve.

    Designed for use in health checks and readiness probes.  Returns False
    if the ensemble is None (not yet loaded) or if any model failed to load.

    Parameters
    ----------
    ensemble : LoadedEnsemble from app.state.ensemble, or None.

    Returns
    -------
    bool
    """
    if ensemble is None:
        return False
    return ensemble.is_fully_loaded()


def compute_uptime_seconds(ensemble: LoadedEnsemble) -> float:
    """
    Compute seconds elapsed since the ensemble was loaded.

    Uses the ``loaded_at`` ISO timestamp stored in the dataclass.
    Called by app.py to populate ``HealthResponse.uptime_seconds``.

    Parameters
    ----------
    ensemble : The loaded ensemble returned by ``load_ensemble()``.

    Returns
    -------
    float — elapsed seconds since the load completed.
    """
    loaded_dt = datetime.fromisoformat(ensemble.loaded_at)
    now_dt    = datetime.now(tz=timezone.utc)

    # loaded_at is always timezone-aware (UTC); handle both aware and naive now()
    if loaded_dt.tzinfo is None:
        loaded_dt = loaded_dt.replace(tzinfo=timezone.utc)

    return max(0.0, (now_dt - loaded_dt).total_seconds())


# =============================================================================
# INLINE SMOKE TEST
# Run directly to verify artefact availability before starting the server:
#     conda activate liquidity_ml
#     cd D:\PROJECTS\liquidity-stress-early-warning
#     python -m api.model_loader
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    log.info("Running model_loader standalone smoke test ...")

    try:
        ensemble = load_ensemble()
    except Exception as exc:
        log.error("SMOKE TEST FAILED: %s", exc)
        sys.exit(1)

    # Verify the ensemble is fully loaded
    if not ensemble.is_fully_loaded():
        log.error(
            "SMOKE TEST FAILED: Ensemble is not fully loaded.  "
            "Check models_loaded_ok: %s",
            ensemble.models_loaded_ok,
        )
        sys.exit(1)

    # Verify artefact counts
    expected_folds = len(ensemble.model_names) * N_FOLDS
    if ensemble.total_fold_models != expected_folds:
        log.error(
            "SMOKE TEST FAILED: Expected %d fold models, got %d",
            expected_folds,
            ensemble.total_fold_models,
        )
        sys.exit(1)

    # Log the inventory
    inventory = get_artefact_inventory(ensemble)
    log.info("Artefact inventory:")
    for k, v in inventory.items():
        log.info("  %-30s : %s", k, v)

    # Verify lru_cache works — second call must return same object
    ensemble2 = load_ensemble()
    if ensemble is not ensemble2:
        log.error(
            "SMOKE TEST FAILED: lru_cache is not working — "
            "second call returned a different object."
        )
        sys.exit(1)
    log.info("lru_cache verified: same object returned on second call ✓")

    log.info("")
    log.info("SMOKE TEST PASSED ✓")
    log.info("%s", ensemble)
    sys.exit(0)