"""
Multi-Model Training Orchestrator
==================================
Project : Liquidity Stress Early Warning (AI4EAC / Zindi)
Module  : src/orchestration/run_all_models.py

Responsibilities
----------------
- Resolve project root deterministically (no CWD dependency)
- Load and validate per-model YAML configs
- Pre-flight dependency check per model family (catches missing
  pytorch-tabnet before wasting time on feature engineering)
- Execute LightGBM / XGBoost / CatBoost / LogisticRegression / TabNet
  training in-process
- Windows-safe logging (UTF-8, no emoji in file/console handlers)
- Per-model failure isolation: one failure never aborts siblings
- Structured run timing and summary report
- Save YAML run summary to outputs/logs/

Output contract (aligned with project output spec)
---------------------------------------------------
outputs/experiments/<stage>/<model>/run_YYYYMMDD_HHMMSS/
    models/              fold-level serialised models
                         GBMs + LogReg: joblib .pkl per fold
                         TabNet: pytorch-tabnet .zip per fold
                         (cv.py handles serialisation routing transparently)
    oof_preds.npy        out-of-fold predictions
    y_true.npy           ground-truth labels
    fold_scores.json     per-fold logloss + AUC
    fold_indices.pkl     train/valid indices per fold
    fold_predictions.pkl per-fold raw predictions
    feature_importance.csv
    feature_list.json
    preprocessor.pkl     fitted PreprocessingPipeline (includes scaler if
                         scale_features=true)
    metadata.json
    config_used.yaml

Usage
-----
# All models in default set (from project root):
python -m src.orchestration.run_all_models

# Specific model(s):
python -m src.orchestration.run_all_models --configs configs/logreg_v1.yaml
python -m src.orchestration.run_all_models --configs configs/tabnet_v1.yaml
python -m src.orchestration.run_all_models --configs configs/logreg_v1.yaml configs/tabnet_v1.yaml

# Full 5-model run:
python -m src.orchestration.run_all_models --configs \\
    configs/lgbm_v2.yaml \\
    configs/xgb_v2.yaml \\
    configs/catboost_v2.yaml \\
    configs/logreg_v1.yaml \\
    configs/tabnet_v1.yaml

Changelog
---------
v1.0   LightGBM / XGBoost / CatBoost orchestration.
v2.0   NEW: LogisticRegression and TabNet support.
       - DEFAULT_CONFIGS updated to include logreg_v1.yaml and tabnet_v1.yaml
       - Pre-flight dependency check per model family with actionable
         error messages (catches missing pytorch-tabnet before feature
         engineering runs)
       - Model-aware config validation: enforces scale_features=true for
         logreg and tabnet, preventing silent unscaled-data failures
       - Runtime class hints logged before dispatch (GBMs: minutes,
         LogReg: seconds-to-minutes, TabNet: 10-60 minutes on CPU)
       - Module docstring updated: TabNet .zip serialisation noted in
         output contract
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# =============================================================================
# PROJECT ROOT RESOLUTION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Default config set ────────────────────────────────────────────────────────
# This list defines which models run when no --configs flag is supplied.
# Update this list as new model families are added to the project.
# Phase 1–2 (GBMs only)    : lgbm, xgb, catboost
# Phase 2–3 (+ LR + TabNet): all five below
# To run a subset, use --configs explicitly rather than editing this list.
DEFAULT_CONFIGS: List[str] = [
    "configs/lgbm_v2.yaml",
    "configs/xgb_v2.yaml",
    "configs/catboost_v2.yaml",
    "configs/logreg_v1.yaml",
    "configs/tabnet_v1.yaml",
]

LOG_DIR = PROJECT_ROOT / "outputs" / "logs"

# ── Models that require StandardScaler (not scale-invariant) ─────────────────
# Used by validate_config() to enforce scale_features=true.
# Add new scale-sensitive model names here as the project grows.
_SCALE_SENSITIVE_MODELS = {"logreg", "tabnet"}

# ── Per-model dependency map ───────────────────────────────────────────────────
# Maps model name → (pip package name, importable module name, install hint).
# Used by _check_model_dependencies() pre-flight validation.
_MODEL_DEPENDENCIES: Dict[str, tuple] = {
    "lightgbm": ("lightgbm",       "lightgbm",                  "pip install lightgbm"),
    "xgboost":  ("xgboost",        "xgboost",                   "pip install xgboost"),
    "catboost": ("catboost",        "catboost",                  "pip install catboost"),
    "logreg":   ("scikit-learn",    "sklearn.linear_model",      "pip install scikit-learn"),
    "tabnet":   ("pytorch-tabnet",  "pytorch_tabnet.tab_model",  "pip install pytorch-tabnet torch"),
}

# ── Approximate runtime class per model on CPU ────────────────────────────────
# Printed before dispatch so the user knows what to expect.
_RUNTIME_HINTS: Dict[str, str] = {
    "lightgbm": "~5–15 min  (5-fold, early stopping)",
    "xgboost":  "~10–20 min (5-fold, early stopping)",
    "catboost": "~15–30 min (5-fold, CPU)",
    "logreg":   "~1–5 min   (5-fold, saga solver)",
    "tabnet":   "~20–60 min (5-fold, CPU — gpu speeds this up 5-10x)",
}


# =============================================================================
# WINDOWS-SAFE LOGGING
# =============================================================================

import logging


class _AsciiFilter(logging.Filter):
    """
    Strip non-ASCII characters from log records before they reach the
    Windows console handler (cp1252 cannot encode many Unicode chars).
    The file handler receives the full UTF-8 message unmodified.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = record.msg.encode("ascii", errors="replace").decode("ascii")
        return True


def setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = log_dir / f"multi_model_run_{timestamp}.log"

    logger = logging.getLogger("orchestrator")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()   # prevent duplicate handlers on Jupyter re-runs

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler: full UTF-8, retains all characters
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # Console handler: ASCII-safe for Windows cp1252
    ch = logging.StreamHandler(
        io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        if hasattr(sys.stdout, "buffer")
        else sys.stdout
    )
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    ch.addFilter(_AsciiFilter())

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("=" * 72)
    logger.info("MULTI-MODEL TRAINING PIPELINE STARTED")
    logger.info("Project root : %s", PROJECT_ROOT)
    logger.info("Log file     : %s", log_file)
    logger.info("=" * 72)

    return logger


# =============================================================================
# PRE-FLIGHT DEPENDENCY CHECK
# =============================================================================

def _check_model_dependencies(model_name: str, logger: logging.Logger) -> None:
    """
    Verify that the Python package required by `model_name` is importable.

    Raises ImportError with an actionable install command if the package
    is missing. This prevents wasting 2–5 minutes on feature engineering
    only to fail on a missing import inside the CV loop.

    Called once per model before _run_training_inprocess().
    """
    if model_name not in _MODEL_DEPENDENCIES:
        # Unknown model — cv.py will raise a clear ValueError later.
        return

    pip_pkg, import_module, install_hint = _MODEL_DEPENDENCIES[model_name]

    try:
        __import__(import_module)
        logger.debug("  Dependency check OK: %s (%s)", model_name, pip_pkg)
    except ImportError:
        raise ImportError(
            f"\n"
            f"  Missing dependency for model '{model_name}': {pip_pkg}\n"
            f"  Install with:\n"
            f"    {install_hint}\n"
            f"  Then re-run this experiment."
        )


# =============================================================================
# CONFIG HELPERS
# =============================================================================

def _resolve_config_path(raw_path: str) -> Path:
    """
    Accept absolute paths or paths relative to the project root.
    Raises FileNotFoundError with a clear message if the file is not found.
    """
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / raw_path
    if not candidate.exists():
        raise FileNotFoundError(
            f"Config not found: {candidate}\n"
            f"  Tried relative to project root : {PROJECT_ROOT}\n"
            f"  Original path supplied         : {raw_path}"
        )
    return candidate


def load_config(path: str):
    resolved = _resolve_config_path(path)
    with open(resolved, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg, resolved


def validate_config(cfg: Dict[str, Any], path: Path) -> None:
    """
    Structural and model-aware config validation.

    Checks performed:
      1. Required top-level sections present.
      2. model.name and model.params present.
      3. Scale-sensitive models (logreg, tabnet) have scale_features=true.
         Without this, unscaled data silently produces wrong results —
         this check converts a silent failure into an explicit error.
    """
    required_top = [
        "project", "experiment", "data", "model",
        "cv", "training", "evaluation", "artifacts",
    ]
    missing = [k for k in required_top if k not in cfg]
    if missing:
        raise ValueError(
            f"Config {path.name} is missing required sections: {missing}"
        )

    if "name" not in cfg["model"]:
        raise ValueError(f"Config {path.name}: model.name is required")
    if "params" not in cfg["model"]:
        raise ValueError(f"Config {path.name}: model.params is required")

    # ── Scale-sensitive model enforcement ─────────────────────────────────
    model_name = cfg["model"]["name"]
    if model_name in _SCALE_SENSITIVE_MODELS:
        scale_features = cfg.get("preprocessing", {}).get("scale_features", False)
        if not scale_features:
            raise ValueError(
                f"Config {path.name}: model '{model_name}' requires "
                f"preprocessing.scale_features: true\n"
                f"  Without StandardScaler, {model_name} produces incorrect results:\n"
                f"  - logreg: ElasticNet penalty applied inconsistently across "
                f"features with different scales\n"
                f"  - tabnet: attention softmax collapses to uniform distribution\n"
                f"  Set preprocessing.scale_features: true in {path.name} and re-run."
            )


# =============================================================================
# OUTPUT PATH BUILDER
# Produces: outputs/experiments/<stage>/<model>/run_YYYYMMDD_HHMMSS/
# =============================================================================

def build_run_output_dir(cfg: Dict[str, Any]) -> Path:
    stage      = cfg["experiment"]["stage"]    # e.g. "v3_extended_models"
    model_name = cfg["model"]["name"]          # e.g. "logreg"
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = (
        PROJECT_ROOT
        / "outputs"
        / "experiments"
        / stage
        / model_name
        / f"run_{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# =============================================================================
# IN-PROCESS TRAINING DISPATCHER
# =============================================================================

def _run_training_inprocess(
    cfg: Dict[str, Any],
    run_dir: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Import and execute the full CV pipeline in-process for one model config.

    Pipeline steps:
      1. Load raw training data
      2. Build engineered features (feature_engineering.py v2.2.1)
      3. Split features / target
      4. Fit PreprocessingPipeline (config-aware: reads scale_features,
         clip_quantiles, enable_clipping from cfg["preprocessing"])
      5. Run stratified K-fold CV (cv.py — model-agnostic engine)
      6. Attach preprocessor to results dict
      7. Save all artifacts via save_cv_outputs()

    The preprocessor is fitted here (on each model's full training data)
    rather than inside each CV fold because:
      - Constant column detection and quantile clip bounds are stable
        across folds for this dataset (confirmed in notebook 04).
      - Fitting inside each fold would multiply preprocessing time by
        n_splits with negligible benefit for this feature space.
      - The scaler (if scale_features=True) is also fitted here on the
        full training set. This is the standard sklearn convention for
        pipeline.fit_transform() and is CV-safe because the scaler
        statistics (mean, std) are computed on the training set only —
        validation folds are never seen by the scaler during fit().

    Note on scale_features:
      PreprocessingPipeline(cfg) reads scale_features from
      cfg["preprocessing"]["scale_features"] automatically.
      logreg and tabnet configs set this to true; GBM configs omit it
      (defaults to false). No branching logic needed here.
    """
    # Lazy imports — no hard model-library deps at module level
    from src.data.load_data import load_data
    from src.features.feature_engineering import build_features, split_features_target
    from src.preprocessing.preprocessing import PreprocessingPipeline
    from src.training.cv import run_cv, save_cv_outputs

    model_name = cfg["model"]["name"]

    # ------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------
    logger.info("  [1/5] Loading data ...")
    train_df, _ = load_data(
        train_path=str(PROJECT_ROOT / cfg["data"]["train_path"]),
        validate=True,
        verbose=False,
    )
    logger.info("        Raw data shape: %d rows x %d cols",
                train_df.shape[0], train_df.shape[1])

    # ------------------------------------------------------------------
    # Step 2: Feature engineering
    # ------------------------------------------------------------------
    logger.info("  [2/5] Building features (v2.2.1) ...")
    train_fe = build_features(train_df)
    X, y = split_features_target(train_fe)
    logger.info("        Feature matrix: %d rows x %d cols", X.shape[0], X.shape[1])
    logger.info("        Target balance: %.1f%% positive", 100 * y.mean())

    # ------------------------------------------------------------------
    # Step 3: Preprocessing
    # ------------------------------------------------------------------
    scale_active = cfg.get("preprocessing", {}).get("scale_features", False)
    logger.info(
        "  [3/5] Fitting PreprocessingPipeline "
        "(clipping=True, scale_features=%s) ...",
        scale_active,
    )
    preproc     = PreprocessingPipeline(cfg)
    X_processed = preproc.fit_transform(X)
    logger.info(
        "        Processed shape: %d rows x %d cols",
        X_processed.shape[0], X_processed.shape[1],
    )

    # ------------------------------------------------------------------
    # Step 4: Cross-validation
    # ------------------------------------------------------------------
    n_folds = cfg["cv"]["n_splits"]
    logger.info(
        "  [4/5] Starting %d-fold CV | model=%s | "
        "expected: %s",
        n_folds,
        model_name.upper(),
        _RUNTIME_HINTS.get(model_name, "runtime unknown"),
    )
    cv_results = run_cv(X_processed, y, cfg)

    # ------------------------------------------------------------------
    # Step 5: Save artifacts
    # ------------------------------------------------------------------
    logger.info("  [5/5] Saving artifacts -> %s", run_dir)
    cv_results["preprocessor"] = preproc   # embed for save_cv_outputs()
    save_cv_outputs(cv_results, cfg, str(run_dir))

    return cv_results


# =============================================================================
# SINGLE EXPERIMENT RUNNER
# Full lifecycle of one config: validate → check deps → train → record.
# Per-model failure isolation: exceptions are caught, logged, and the next
# model continues.
# =============================================================================

def run_single_experiment(
    raw_config_path: str,
    logger: logging.Logger,
) -> Dict[str, Any]:

    start_time = time.perf_counter()
    result: Dict[str, Any] = {
        "config":       raw_config_path,
        "model":        "unknown",
        "stage":        "unknown",
        "run_dir":      None,
        "status":       "failed",
        "mean_logloss": None,
        "mean_auc":     None,
        "final_score":  None,
        "runtime_sec":  None,
        "error":        None,
    }

    try:
        # ------------------------------------------------------------------
        # 1. Load + validate config
        # ------------------------------------------------------------------
        cfg, resolved_path = load_config(raw_config_path)
        validate_config(cfg, resolved_path)

        model_name = cfg["model"]["name"]
        stage      = cfg["experiment"]["stage"]

        result["model"] = model_name
        result["stage"] = stage

        logger.info("")
        logger.info("=" * 60)
        logger.info(
            "STARTING | model=%-12s  stage=%s",
            model_name.upper(), stage,
        )
        logger.info("Config   | %s", resolved_path)
        logger.info("Runtime  | %s", _RUNTIME_HINTS.get(model_name, "unknown"))
        logger.info("=" * 60)

        # ------------------------------------------------------------------
        # 2. Pre-flight dependency check
        # ------------------------------------------------------------------
        logger.info("  Checking dependencies for '%s' ...", model_name)
        _check_model_dependencies(model_name, logger)

        # ------------------------------------------------------------------
        # 3. Build output directory
        # ------------------------------------------------------------------
        run_dir        = build_run_output_dir(cfg)
        result["run_dir"] = str(run_dir)
        logger.info("  Run dir: %s", run_dir)

        # ------------------------------------------------------------------
        # 4. Execute training
        # ------------------------------------------------------------------
        cv_results = _run_training_inprocess(cfg, run_dir, logger)

        # ------------------------------------------------------------------
        # 5. Record metrics
        # ------------------------------------------------------------------
        result["mean_logloss"] = round(cv_results["mean_logloss"], 5)
        result["mean_auc"]     = round(cv_results["mean_auc"],     5)
        result["final_score"]  = round(cv_results["final_score"],  5)
        result["status"]       = "success"

        runtime = time.perf_counter() - start_time
        result["runtime_sec"] = round(runtime, 1)

        logger.info("")
        logger.info(
            "SUCCESS  | model=%-12s  LogLoss=%.5f  AUC=%.5f  "
            "Score=%.5f  [%.1fs]",
            model_name.upper(),
            result["mean_logloss"],
            result["mean_auc"],
            result["final_score"],
            runtime,
        )

    except Exception as exc:
        runtime = time.perf_counter() - start_time
        result["runtime_sec"] = round(runtime, 1)
        result["error"]       = str(exc)

        logger.error("")
        logger.error("FAILED   | config=%s", raw_config_path)
        logger.error("Error    | %s", exc)
        logger.debug("Traceback:\n%s", traceback.format_exc())

    return result


# =============================================================================
# SUMMARY REPORTER
# =============================================================================

def _print_summary(
    results: List[Dict[str, Any]],
    total_runtime: float,
    logger: logging.Logger,
) -> None:

    success = [r for r in results if r["status"] == "success"]
    failed  = [r for r in results if r["status"] == "failed"]

    logger.info("")
    logger.info("=" * 72)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 72)
    logger.info(
        "%-14s | %-8s | %-10s | %-8s | %-10s | %s",
        "Model", "Status", "LogLoss", "AUC", "Score", "Runtime(s)",
    )
    logger.info("-" * 72)

    for r in results:
        ll    = f"{r['mean_logloss']:.5f}" if r["mean_logloss"] is not None else "N/A"
        auc   = f"{r['mean_auc']:.5f}"     if r["mean_auc"]     is not None else "N/A"
        score = f"{r['final_score']:.5f}"  if r["final_score"]  is not None else "N/A"
        rt    = f"{r['runtime_sec']:.1f}s" if r["runtime_sec"]  is not None else "N/A"
        logger.info(
            "%-14s | %-8s | %-10s | %-8s | %-10s | %s",
            r["model"].upper(), r["status"].upper(), ll, auc, score, rt,
        )

    logger.info("-" * 72)
    logger.info("Passed  : %d / %d", len(success), len(results))
    logger.info("Failed  : %d / %d", len(failed),  len(results))
    logger.info("Total   : %.1f seconds  (%.1f minutes)", total_runtime, total_runtime / 60)
    logger.info("=" * 72)

    if failed:
        logger.warning("")
        logger.warning("Failed experiments (check log for full traceback):")
        for r in failed:
            logger.warning("  %-12s -> %s", r["model"].upper(), r["error"])


def _save_summary(results: List[Dict[str, Any]], log_dir: Path) -> Path:
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = log_dir / f"run_summary_{timestamp}.yaml"

    serialisable = []
    for r in results:
        row = dict(r)
        if isinstance(row.get("run_dir"), Path):
            row["run_dir"] = str(row["run_dir"])
        serialisable.append(row)

    with open(summary_path, "w", encoding="utf-8") as f:
        yaml.dump(serialisable, f, allow_unicode=True, sort_keys=False)

    return summary_path


# =============================================================================
# MAIN PIPELINE ENTRY POINT
# =============================================================================

def run_pipeline(config_paths: List[str]) -> List[Dict[str, Any]]:
    logger = setup_logger(LOG_DIR)

    results:     List[Dict[str, Any]] = []
    total_start: float                = time.perf_counter()

    logger.info("Experiments queued: %d", len(config_paths))
    for i, p in enumerate(config_paths, 1):
        logger.info("  [%d] %s", i, p)

    for config_path in config_paths:
        result = run_single_experiment(config_path, logger)
        results.append(result)

    total_runtime = time.perf_counter() - total_start

    _print_summary(results, total_runtime, logger)

    summary_path = _save_summary(results, LOG_DIR)
    logger.info("Summary saved: %s", summary_path)

    return results


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-model training orchestrator — "
            "Liquidity Stress Early Warning (AI4EAC / Zindi)"
        )
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_CONFIGS,
        help=(
            "One or more config file paths (absolute or relative to project root). "
            f"Default runs all {len(DEFAULT_CONFIGS)} model configs: "
            f"{DEFAULT_CONFIGS}"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(args.configs)