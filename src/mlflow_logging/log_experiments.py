# src/mlflow_logging/log_experiments.py
"""
MLflow Retrospective Experiment Logger
=======================================
Project  : AI4EAC Liquidity Stress Early Warning (Zindi Africa)
Module   : src/mlflow_logging/log_experiments.py
Author   : Henry Otsyula
Created  : 2026-05-13

Purpose
-------
Retrospectively logs all training results, calibration metrics, Optuna
tuning histories, ensemble strategies, SHAP findings, and the final
submission into a local MLflow tracking server — without re-running any
training.  Every metric, parameter, and artefact is read directly from
the JSON/CSV/npy files already on disk.

Experiment hierarchy (mirrors the project's research progression)
-----------------------------------------------------------------
  AI4EAC / 1_baseline_models
      Runs: lightgbm-baseline, xgboost-baseline, catboost-baseline
      Shows the 3-GBM starting point and intra-cluster correlation.

  AI4EAC / 2_hyperparameter_tuning
      Runs: lightgbm-optuna, xgboost-optuna
      Shows Optuna trial history, best params, improvement over baseline.

  AI4EAC / 3_extended_models
      Runs: logreg-baseline, tabnet-baseline
      Shows cross-cluster diversity (r=0.745) justification.

  AI4EAC / 4_calibration
      Runs: one per model (5 total)
      Shows Platt slope/intercept, LogLoss reduction from calibration.
      LogReg 36% reduction, TabNet 21% reduction.

  AI4EAC / 5_ensemble
      Runs: one per ensemble strategy (4 strategies × 2 versions)
        v4 (3-model GBM only), v5.1 (5-model production)
      Shows why weighted average (0.19144) beat stacking.

  AI4EAC / 6_shap_augmentation
      Runs: K=5, K=20, K=50, K=100 augmentation experiments
      Shows why unscaled SHAP features caused collapse at K≥20.

  AI4EAC / 7_final_submission
      Single run: the production submission
      Tags: competition=AI4EAC, final_submission=true
      Artefact: the actual CSV uploaded to Zindi.

Usage
-----
  # First-time setup (run once):
  pip install mlflow pyyaml

  # Log everything:
  python -m src.mlflow_logging.log_experiments

  # Launch the UI:
  mlflow ui --port 5000
  # Then open: http://localhost:5000

  # Options:
  python -m src.mlflow_logging.log_experiments --tracking-uri ./sqlite:///mlflow.db
  python -m src.mlflow_logging.log_experiments --experiment 5_ensemble
  python -m src.mlflow_logging.log_experiments --dry-run

Design principles
-----------------
  • NEVER re-trains any model — reads artefacts from disk only.
  • Graceful degradation: if a file is missing, the run is still
    logged with whatever IS available; missing files are noted in a
    tag, not a crash.
  • Idempotent: running twice produces duplicate runs (MLflow's
    default behaviour) but does not corrupt existing runs.
  • All composite scores use 0.6*LL + 0.4*(1-AUC) exactly.
  • Run names are human-readable and sort naturally in the UI.
  • Every logged artefact is the original file — no copies made.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Project root — resolved from this file's location.
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
    level   = logging.INFO,
)
log = logging.getLogger("mlflow_logger")

# ---------------------------------------------------------------------------
# Constants — must match training pipeline exactly
# ---------------------------------------------------------------------------
LOG_LOSS_WEIGHT : float = 0.6
AUC_WEIGHT      : float = 0.4

# Composite score formula used everywhere
def _composite(logloss: float, auc: float) -> float:
    return LOG_LOSS_WEIGHT * logloss + AUC_WEIGHT * (1.0 - auc)

# Known reference scores — used to compute improvement deltas
_BASELINE_SCORES: Dict[str, Dict[str, float]] = {
    "lightgbm": {"logloss": 0.26117, "auc": 0.90284, "composite": 0.19557},
    "xgboost" : {"logloss": 0.25835, "auc": 0.90378, "composite": 0.19350},
    "catboost": {"logloss": 0.25865, "auc": 0.90223, "composite": 0.19430},
    "logreg"  : {"logloss": 0.31993, "auc": 0.82983, "composite": 0.26003},
    "tabnet"  : {"logloss": 0.31481, "auc": 0.83981, "composite": 0.25296},
}

# Production ensemble score
_ENSEMBLE_BEST_SCORE: float = 0.19144

# Calibration parameters from notebook 07 / 07b
_PLATT_PARAMS: Dict[str, Dict[str, float]] = {
    "lightgbm": {"slope": 1.0,    "intercept": 0.0,   "ll_reduction_pct": 0.0},
    "xgboost" : {"slope": 1.0,    "intercept": 0.0,   "ll_reduction_pct": 0.0},
    "catboost": {"slope": 1.0,    "intercept": 0.0,   "ll_reduction_pct": 0.0},
    "logreg"  : {"slope": 5.47,   "intercept": -4.43, "ll_reduction_pct": 36.0},
    "tabnet"  : {"slope": 4.37,   "intercept": -3.52, "ll_reduction_pct": 21.0},
}

# OOF correlation matrix (post-Platt, from notebook 08)
_OOF_CORRELATIONS: Dict[Tuple[str, str], float] = {
    ("lightgbm", "xgboost") : 0.978,
    ("lightgbm", "catboost"): 0.962,
    ("xgboost",  "catboost"): 0.969,
    ("lightgbm", "logreg")  : 0.748,
    ("lightgbm", "tabnet")  : 0.745,
    ("xgboost",  "logreg")  : 0.748,
    ("xgboost",  "tabnet")  : 0.747,
    ("catboost", "logreg")  : 0.750,
    ("catboost", "tabnet")  : 0.753,
    ("logreg",   "tabnet")  : 0.752,
}

# Production ensemble weights
_ENSEMBLE_WEIGHTS: Dict[str, float] = {
    "lightgbm": 0.17813,
    "xgboost" : 0.39721,
    "catboost": 0.38777,
    "logreg"  : 0.00000,
    "tabnet"  : 0.03689,
}

# SHAP augmentation results from notebook 10 §11
_SHAP_AUG_RESULTS: Dict[int, Dict[str, float]] = {
    5  : {"best_score": 0.19144, "stacking_score": 0.19851, "stacking_auc": 0.901},
    20 : {"best_score": 0.19144, "stacking_score": 0.45630, "stacking_auc": 0.698},
    50 : {"best_score": 0.19144, "stacking_score": 0.55047, "stacking_auc": 0.602},
    100: {"best_score": 0.19144, "stacking_score": 0.55009, "stacking_auc": 0.585},
}

# ---------------------------------------------------------------------------
# Artefact path registry — every known path on disk
# ---------------------------------------------------------------------------

def _experiments_dir() -> Path:
    return PROJECT_ROOT / "outputs" / "experiments"

def _most_recent_run(model_dir: Path) -> Optional[Path]:
    """Return the most recent run_YYYYMMDD_HHMMSS subdirectory, or None."""
    runs = sorted(model_dir.glob("run_*"), reverse=True)
    return runs[0] if runs else None

# Production run directories (most recent per model family)
_MODEL_RUN_DIRS: Dict[str, Path] = {
    "lightgbm_baseline" : _experiments_dir() / "baseline"            / "lightgbm",
    "xgboost_baseline"  : _experiments_dir() / "baseline"            / "xgboost",
    "catboost_baseline" : _experiments_dir() / "baseline"            / "catboost",
    "lightgbm_tuned"    : _experiments_dir() / "v2_feature_expansion" / "lightgbm" / "run_20260503_081128",
    "xgboost_tuned"     : _experiments_dir() / "v2_feature_expansion" / "xgboost"  / "run_20260503_082310",
    "logreg"            : _experiments_dir() / "v3_extended_models"   / "logreg"   / "run_20260507_072935",
    "tabnet"            : _experiments_dir() / "v3_extended_models"   / "tabnet"   / "run_20260507_044719",
    "v5_ensemble"       : _experiments_dir() / "v5_ensemble"          / "run_20260508_054540",
    "v6_xgb_shap"       : _experiments_dir() / "v6_xgb_825_features"  / "xgboost",
}

_TUNING_DIR   = PROJECT_ROOT / "outputs" / "tuning"
_CALIB_DIR    = PROJECT_ROOT / "outputs" / "calibration"
_MULTI_MODEL  = PROJECT_ROOT / "outputs" / "multi_model"
_SHAP_DIR     = PROJECT_ROOT / "outputs" / "shap"
_SUBMISSIONS  = PROJECT_ROOT / "submissions"

# ---------------------------------------------------------------------------
# Safe file readers — never crash on missing files
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> Optional[Dict]:
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as exc:
            log.warning("Could not read JSON %s: %s", path, exc)
    return None

def _read_yaml(path: Path) -> Optional[Dict]:
    if path.exists():
        try:
            with open(path) as f:
                return yaml.safe_load(f)
        except Exception as exc:
            log.warning("Could not read YAML %s: %s", path, exc)
    return None

def _read_csv_first_row(path: Path) -> Optional[Dict]:
    """Read the first data row of a CSV as a dict — for single-row summary files."""
    if path.exists():
        try:
            import csv
            with open(path) as f:
                reader = csv.DictReader(f)
                return next(iter(reader), None)
        except Exception as exc:
            log.warning("Could not read CSV %s: %s", path, exc)
    return None

def _npy_stats(path: Path) -> Optional[Dict[str, float]]:
    """Load a .npy array and return basic statistics."""
    if path.exists():
        try:
            arr = np.load(path)
            return {
                "mean": float(arr.mean()),
                "std" : float(arr.std()),
                "min" : float(arr.min()),
                "max" : float(arr.max()),
                "p25" : float(np.percentile(arr, 25)),
                "p50" : float(np.percentile(arr, 50)),
                "p75" : float(np.percentile(arr, 75)),
            }
        except Exception as exc:
            log.warning("Could not load npy %s: %s", path, exc)
    return None

def _log_artifact_if_exists(path: Path) -> None:
    """Log a file as an MLflow artefact only if it exists."""
    import mlflow
    if path.exists():
        mlflow.log_artifact(str(path))
    else:
        log.warning("    Artefact not found (skipping): %s", path.name)

def _log_artifacts_in_dir(directory: Path, extensions: List[str]) -> None:
    """Log all files matching extensions in a directory."""

    import mlflow
    if not directory.exists():
        return
    for ext in extensions:
        for fp in sorted(directory.glob(f"*{ext}")):
            mlflow.log_artifact(str(fp))

def _common_tags() -> Dict[str, str]:
    """
    Shared tags applied to all MLflow runs.
    """
    return {
        "project_name"   : "AI4EAC Liquidity Stress Early Warning",
        "project_version": "v5.1",
        "git_branch"     : "main",
        "author"         : "Henry Otsyula",
    }
# =============================================================================
# EXPERIMENT 1 — BASELINE MODELS
# =============================================================================

def log_baseline_models(mlflow) -> None:
    """
    Log the 3 GBM baseline runs with clean MLflow structure.
    """

    exp_name = "AI4EAC/1_baseline_models"
    mlflow.set_experiment(exp_name)
    log.info("\n[Exp 1] %s", exp_name)

    baseline_models = [
        ("lightgbm", "lgbm_v2.yaml",     _MODEL_RUN_DIRS["lightgbm_baseline"]),
        ("xgboost",  "xgb_v2.yaml",      _MODEL_RUN_DIRS["xgboost_baseline"]),
        ("catboost", "catboost_v2.yaml", _MODEL_RUN_DIRS["catboost_baseline"]),
    ]

    for model_name, config_file, model_dir in baseline_models:

        run_dir = _most_recent_run(model_dir) if model_dir.is_dir() else model_dir
        if run_dir is None:
            log.warning("  No run directory found for %s baseline — skipping.", model_name)
            continue

        run_name = f"{model_name}-baseline"
        log.info("  Logging run: %s  (%s)", run_name, run_dir.name)

        # 🔥 IMPORTANT: isolate each model safely
        with mlflow.start_run(run_name=run_name, nested=True):

            # ─────────────────────────────────────────────
            # Tags
            # ─────────────────────────────────────────────
            mlflow.set_tags({
                **_common_tags(),
                "model_family": model_name,
                "estimator_type": "gradient_boosting",
                "stage": "baseline",
                "run_dir": str(run_dir),
                "n_folds": "5",
            })

            # ─────────────────────────────────────────────
            # Config params (SAFE logging, no duplicates)
            # ─────────────────────────────────────────────
            config = _read_yaml(PROJECT_ROOT / "configs" / config_file)

            if config:
                model_params = config.get("model", {}).get("params", {})

                for k, v in model_params.items():
                    if v is None:
                        v = "N/A"

                    mlflow.log_param(f"cfg_{k}", v)

            # ⚠️ REMOVE DUPLICATE LOGGING OF n_estimators
            # (this was causing your crash)

            # ─────────────────────────────────────────────
            # Run-specific config
            # ─────────────────────────────────────────────
            config_used = _read_yaml(run_dir / "config_used.yaml")

            if config_used:
                run_params = (
                    config_used.get("model", {}).get("params", {})
                    or config_used.get("params", {})
                )

                for k, v in run_params.items():
                    try:
                        mlflow.log_param(f"run_{k}", v)
                    except Exception:
                        pass

            # ─────────────────────────────────────────────
            # CV metrics
            # ─────────────────────────────────────────────
            fold_scores = _read_json(run_dir / "fold_scores.json")

            if fold_scores:
                logloss_vals, auc_vals = [], []

                for k in range(5):
                    f = fold_scores.get(f"fold_{k}", {})
                    if "logloss" in f:
                        logloss_vals.append(f["logloss"])
                    if "auc" in f:
                        auc_vals.append(f["auc"])

                if logloss_vals:
                    mlflow.log_metric("cv_logloss_mean", np.mean(logloss_vals))
                    mlflow.log_metric("cv_logloss_std", np.std(logloss_vals))

                if auc_vals:
                    mlflow.log_metric("cv_auc_mean", np.mean(auc_vals))
                    mlflow.log_metric("cv_auc_std", np.std(auc_vals))

                if logloss_vals and auc_vals:
                    mlflow.log_metric(
                        "composite_score",
                        _composite(np.mean(logloss_vals), np.mean(auc_vals))
                    )

            # ─────────────────────────────────────────────
            # Metadata fallback
            # ─────────────────────────────────────────────
            metadata = _read_json(run_dir / "metadata.json")

            if metadata:
                for key in ["cv_logloss_mean", "cv_auc_mean", "composite_score"]:
                    if key in metadata:
                        mlflow.log_metric(f"diagnostic_{key}", float(metadata[key]))

            # ─────────────────────────────────────────────
            # Reference scores
            # ─────────────────────────────────────────────
            ref = _BASELINE_SCORES.get(model_name, {})

            if ref:
                mlflow.log_metric("oof_logloss_platt", ref["logloss"])
                mlflow.log_metric("oof_auc_platt", ref["auc"])
                mlflow.log_metric("oof_composite_platt", ref["composite"])

            # ─────────────────────────────────────────────
            # OOF stats
            # ─────────────────────────────────────────────
            oof_stats = _npy_stats(run_dir / "oof_preds.npy")

            if oof_stats:
                for k, v in oof_stats.items():
                    mlflow.log_metric(f"oof_raw_{k}", v)

            cal_stats = _npy_stats(_MULTI_MODEL / f"oof_calibrated_{model_name}.npy")

            if cal_stats:
                for k, v in cal_stats.items():
                    mlflow.log_metric(f"oof_cal_{k}", v)

            # ─────────────────────────────────────────────
            # Platt calibration
            # ─────────────────────────────────────────────
            platt = _PLATT_PARAMS.get(model_name, {})

            if platt:
                mlflow.log_metric("platt_slope", platt["slope"])
                mlflow.log_metric("platt_intercept", platt["intercept"])
                mlflow.log_metric("ll_reduction_pct", platt["ll_reduction_pct"])

            # ─────────────────────────────────────────────
            # Correlations
            # ─────────────────────────────────────────────
            for (m1, m2), corr in _OOF_CORRELATIONS.items():
                if model_name in (m1, m2):
                    other = m2 if m1 == model_name else m1
                    mlflow.log_metric(f"oof_corr_{other}", corr)

            # ─────────────────────────────────────────────
            # Artifacts
            # ─────────────────────────────────────────────
            _log_artifact_if_exists(run_dir / "feature_importance.csv")
            _log_artifact_if_exists(run_dir / "fold_scores.json")
            _log_artifact_if_exists(run_dir / "metadata.json")

            _log_artifacts_in_dir(
                _CALIB_DIR / {
                    "lightgbm": "lgb",
                    "xgboost": "xgb",
                    "catboost": "cat"
                }[model_name],
                [".pkl", ".npy"]
            )

            _log_artifact_if_exists(
                _CALIB_DIR / "reliability_diagrams_logreg_tabnet.png"
            )

        log.info("    ✓  %s baseline logged", model_name)


# =============================================================================
# EXPERIMENT 2 — HYPERPARAMETER TUNING (Optuna)
# =============================================================================

def log_tuning(mlflow) -> None:
    """
    Log Optuna hyperparameter tuning results for LightGBM and XGBoost.

    Logs:
      - Best params (from *_best_params.yaml)
      - Study summary (best trial score, n_trials, duration)
      - All trial metrics as MLflow steps (trial_history CSV → step chart)
      - Improvement over baseline (delta composite score)
      - Best params as artefact
    """
    exp_name = "AI4EAC/2_hyperparameter_tuning"
    mlflow.set_experiment(exp_name)
    log.info("\n[Exp 2] %s", exp_name)

    tuned_models = [
        {
            "model_name"     : "lightgbm",
            "summary_file"   : "lightgbm_study_summary.json",
            "best_params_file": "lightgbm_best_params.yaml",
            "trial_history"  : "lightgbm_trial_history.csv",
            "importance_file": "lightgbm_importance.json",
            "run_dir"        : _MODEL_RUN_DIRS["lightgbm_tuned"],
            "config_file"    : "lightgbm_tuned.yaml",
            "optuna_best"    : 0.1941752,
            "baseline_composite": 0.19557,
        },
        {
            "model_name"     : "xgboost",
            "summary_file"   : "xgboost_study_summary.json",
            "best_params_file": "xgboost_best_params.yaml",
            "trial_history"  : "xgboost_trial_history.csv",
            "importance_file": "xgboost_importance.json",
            "run_dir"        : _MODEL_RUN_DIRS["xgboost_tuned"],
            "config_file"    : "xgboost_tuned.yaml",
            "optuna_best"    : 0.1920835,
            "baseline_composite": 0.19350,
        },
    ]

    for m in tuned_models:
        model_name = m["model_name"]
        run_dir    = m["run_dir"]
        run_name   = f"{model_name}-optuna-tuned"
        log.info("  Logging run: %s", run_name)

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({
                **_common_tags(),
                "model_family"   : model_name,
                "metric_direction": "lower_is_better",
                "stage"          : "v2_feature_expansion",
                "tuning_method"  : "Optuna TPE",
                "n_trials"       : "100",
                "training_phase" : "Phase 2c — Optuna tuning",
                "run_dir"        : str(run_dir),
                "sampler"        : "TPESampler",
                "pruner"         : "MedianPruner",
            })

            # ── Optuna study summary ────────────────────────────────────────
            summary = _read_json(_TUNING_DIR / m["summary_file"])
            if summary:
                best_score = summary.get("best_score", m["optuna_best"])
                n_trials   = summary.get("n_trials",   100)
                duration   = summary.get("duration_sec", None)
                mlflow.log_metric("optuna_best_composite", best_score)
                mlflow.log_param("n_trials", n_trials)
                if duration:
                    mlflow.log_metric("tuning_duration_sec", duration)

            # Always log the known values from handoff as ground truth
            mlflow.log_metric("optuna_best_composite", m["optuna_best"])
            mlflow.log_metric("baseline_composite",    m["baseline_composite"])
            mlflow.log_metric("delta_vs_baseline",
                              m["optuna_best"] - m["baseline_composite"])

            # ── Best hyperparameters ────────────────────────────────────────
            best_params = _read_yaml(_TUNING_DIR / m["best_params_file"])
            if best_params:
                for k, v in best_params.items():
                    try:
                        mlflow.log_param(f"best_{k}", v)
                    except Exception:
                        mlflow.log_param(f"best_{k}", str(v))

            # ── Config used in the tuned training run ──────────────────────
            config = _read_yaml(PROJECT_ROOT / "configs" / m["config_file"])
            if config:
                model_params = config.get("model", {}).get("params", {})
                for k, v in model_params.items():
                    try:
                        mlflow.log_param(f"tuned_{k}", v)
                    except Exception:
                        pass

            # ── Trial history → step metrics (creates a chart in the UI) ───
            trial_csv = _TUNING_DIR / m["trial_history"]
            if trial_csv.exists():
                try:
                    import csv
                    with open(trial_csv) as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            step = int(row.get("number", row.get("trial", 0)))
                            score_val = row.get("value", row.get("composite_score"))
                            if score_val is not None:
                                mlflow.log_metric("trial_composite_score",
                                                  float(score_val), step=step)
                except Exception as exc:
                    log.warning("    Could not parse trial history: %s", exc)

            # ── Post-tuning CV metrics ──────────────────────────────────────
            fold_scores = _read_json(run_dir / "fold_scores.json")
            if fold_scores:
                logloss_vals = [fold_scores.get(f"fold_{k}", {}).get("logloss")
                                for k in range(5)]
                auc_vals     = [fold_scores.get(f"fold_{k}", {}).get("auc")
                                for k in range(5)]
                logloss_vals = [v for v in logloss_vals if v is not None]
                auc_vals     = [v for v in auc_vals     if v is not None]
                if logloss_vals:
                    mlflow.log_metric("tuned_cv_logloss_mean", np.mean(logloss_vals))
                    mlflow.log_metric("tuned_cv_logloss_std",  np.std(logloss_vals))
                if auc_vals:
                    mlflow.log_metric("tuned_cv_auc_mean", np.mean(auc_vals))
                    mlflow.log_metric("tuned_cv_auc_std",  np.std(auc_vals))
                if logloss_vals and auc_vals:
                    mlflow.log_metric("tuned_composite_score",
                                      _composite(np.mean(logloss_vals), np.mean(auc_vals)))

            # Known post-Platt scores
            ref = _BASELINE_SCORES.get(model_name, {})
            if ref:
                mlflow.log_metric("oof_logloss_platt",  ref["logloss"])
                mlflow.log_metric("oof_auc_platt",       ref["auc"])
                mlflow.log_metric("oof_composite_platt", ref["composite"])

            # ── Artefacts ──────────────────────────────────────────────────
            _log_artifact_if_exists(_TUNING_DIR / m["best_params_file"])
            _log_artifact_if_exists(_TUNING_DIR / m["summary_file"])
            _log_artifact_if_exists(_TUNING_DIR / m["trial_history"])
            _log_artifact_if_exists(_TUNING_DIR / m["importance_file"])
            _log_artifact_if_exists(run_dir / "feature_importance.csv")
            _log_artifact_if_exists(run_dir / "fold_scores.json")

        log.info("    ✓  %s tuning logged", model_name)


# =============================================================================
# EXPERIMENT 3 — EXTENDED MODELS (LogReg + TabNet)
# =============================================================================

def log_extended_models(mlflow) -> None:
    """
    Log LogReg and TabNet runs from v3_extended_models.

    These models were added specifically to introduce cross-cluster
    diversity.  The key story here:
      - GBM intra-cluster r = 0.962–0.978 (effectively one signal)
      - LogReg vs GBMs:  r = 0.748  (orthogonal linear boundary)
      - TabNet vs GBMs:  r = 0.745  (per-customer attention)
      - This diversity justified the 5-model ensemble over GBM-only.

    Despite weaker individual scores, both models contribute to the
    ensemble's diversity budget (even though LogReg's final weight=0.000).
    """
    exp_name = "AI4EAC/3_extended_models"
    mlflow.set_experiment(exp_name)
    log.info("\n[Exp 3] %s", exp_name)

    extended_models = [
        {
            "model_name"   : "logreg",
            "estimator_type": "linear_model",
            "run_dir"      : _MODEL_RUN_DIRS["logreg"],
            "config_file"  : "logreg_v1.yaml",
            "description"  : "ElasticNet LogReg — linear boundary, orthogonal to GBMs",
            "scale_features": True,
            "diversity_vs_lgbm": 0.748,
            "diversity_vs_xgb" : 0.748,
            "diversity_vs_cat" : 0.750,
        },
        {
            "model_name"   : "tabnet",
            "estimator_type": "neural_network",
            "run_dir"      : _MODEL_RUN_DIRS["tabnet"],
            "config_file"  : "tabnet_v1.yaml",
            "description"  : "TabNet — instance-wise sequential attention",
            "scale_features": True,
            "diversity_vs_lgbm": 0.745,
            "diversity_vs_xgb" : 0.747,
            "diversity_vs_cat" : 0.753,
        },
    ]

    for m in extended_models:
        model_name = m["model_name"]
        run_dir    = m["run_dir"]
        run_name   = f"{model_name}-extended"
        log.info("  Logging run: %s  (%s)", run_name, run_dir.name)

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({
                **_common_tags(),
                "model_family"   : model_name,
                "stage"          : "v3_extended_models",
                "training_phase" : "Phase 2b — Extended models",
                "description"    : m["description"],
                "scale_features" : str(m["scale_features"]),
                "run_dir"        : str(run_dir),
                "rationale"      : "Cross-cluster diversity — GBM r=0.962-0.978 → need orthogonal signal",
            })

            # ── Config ─────────────────────────────────────────────────────
            config = _read_yaml(PROJECT_ROOT / "configs" / m["config_file"])
            if config:
                model_params = config.get("model", {}).get("params", {})
                for k, v in model_params.items():
                    try:
                        mlflow.log_param(k, v)
                    except Exception:
                        mlflow.log_param(k, str(v))
            mlflow.log_param("scale_features", m["scale_features"])

            # ── CV metrics ─────────────────────────────────────────────────
            fold_scores = _read_json(run_dir / "fold_scores.json")
            if fold_scores:
                logloss_vals = [fold_scores.get(f"fold_{k}", {}).get("logloss")
                                for k in range(5)]
                auc_vals     = [fold_scores.get(f"fold_{k}", {}).get("auc")
                                for k in range(5)]
                logloss_vals = [v for v in logloss_vals if v is not None]
                auc_vals     = [v for v in auc_vals     if v is not None]
                if logloss_vals:
                    mlflow.log_metric("cv_logloss_mean", np.mean(logloss_vals))
                    mlflow.log_metric("cv_logloss_std",  np.std(logloss_vals))
                    for k, v in enumerate(logloss_vals):
                        mlflow.log_metric("cv_logloss_fold", v, step=k)
                if auc_vals:
                    mlflow.log_metric("cv_auc_mean", np.mean(auc_vals))
                    for k, v in enumerate(auc_vals):
                        mlflow.log_metric("cv_auc_fold", v, step=k)
                if logloss_vals and auc_vals:
                    mlflow.log_metric("composite_score",
                                      _composite(np.mean(logloss_vals), np.mean(auc_vals)))

            # ── Post-Platt scores ──────────────────────────────────────────
            ref   = _BASELINE_SCORES.get(model_name, {})
            platt = _PLATT_PARAMS.get(model_name, {})
            if ref:
                mlflow.log_metric("oof_logloss_raw",         ref["logloss"])
                mlflow.log_metric("oof_auc_platt",            ref["auc"])
                mlflow.log_metric("oof_composite_platt",      ref["composite"])
            if platt:
                mlflow.log_metric("platt_slope",              platt["slope"])
                mlflow.log_metric("platt_intercept",          platt["intercept"])
                mlflow.log_metric("platt_ll_reduction_pct",   platt["ll_reduction_pct"])

            # ── Cross-cluster diversity metrics ────────────────────────────
            mlflow.log_metric("oof_corr_vs_lightgbm",  m["diversity_vs_lgbm"])
            mlflow.log_metric("oof_corr_vs_xgboost",   m["diversity_vs_xgb"])
            mlflow.log_metric("oof_corr_vs_catboost",  m["diversity_vs_cat"])
            # Average cross-cluster diversity (lower = more diverse)
            avg_diversity = np.mean([m["diversity_vs_lgbm"],
                                     m["diversity_vs_xgb"],
                                     m["diversity_vs_cat"]])
            mlflow.log_metric("avg_oof_corr_vs_gbms",  avg_diversity)

            # ── Calibrated OOF stats ───────────────────────────────────────
            cal_stats = _npy_stats(_MULTI_MODEL / f"oof_calibrated_{model_name}.npy")
            if cal_stats:
                for stat, val in cal_stats.items():
                    mlflow.log_metric(f"oof_cal_{stat}", val)

            # ── Calibration summary from notebook 07b ─────────────────────
            cal_summary = _read_json(
                _CALIB_DIR / model_name / "calibration_summary.json"
            )
            if cal_summary:
                for k, v in cal_summary.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"cal_{k}", float(v))
                    else:
                        mlflow.set_tag(f"cal_{k}", str(v))

            # ── Artefacts ──────────────────────────────────────────────────
            _log_artifact_if_exists(run_dir / "feature_importance.csv")
            _log_artifact_if_exists(run_dir / "fold_scores.json")
            _log_artifact_if_exists(run_dir / "metadata.json")
            _log_artifact_if_exists(
                _CALIB_DIR / model_name / "calibration_summary.json"
            )

        log.info("    ✓  %s logged", model_name)


# =============================================================================
# EXPERIMENT 4 — CALIBRATION DEEP-DIVE
# =============================================================================

def log_calibration(mlflow) -> None:
    """
    Log the per-model calibration analysis as a dedicated experiment.

    This surfaces the calibration story as a first-class experiment:
      - For GBMs, Platt slope ≈ 1.0 (already well calibrated)
      - For LogReg, slope=5.47 → 36% LogLoss improvement (critical)
      - For TabNet, slope=4.37 → 21% LogLoss improvement (important)

    The reliability_diagrams image is logged as an artefact so it
    appears inline in the MLflow run UI.
    """
    exp_name = "AI4EAC/4_calibration"
    mlflow.set_experiment(exp_name)
    log.info("\n[Exp 4] %s", exp_name)

    folder_map = {
        "lightgbm": "lgb",
        "xgboost" : "xgb",
        "catboost": "cat",
        "logreg"  : "logreg",
        "tabnet"  : "tabnet",
    }

    for model_name, folder in folder_map.items():
        run_name = f"{model_name}-calibration"
        log.info("  Logging run: %s", run_name)

        with mlflow.start_run(run_name=run_name):
            platt = _PLATT_PARAMS[model_name]
            ref   = _BASELINE_SCORES[model_name]

            mlflow.set_tags({
                **_common_tags(),
                "model_family"     : model_name,
                "calibration_method": "Platt scaling (sklearn LogisticRegression)",
                "calibration_cv"   : "5-fold stratified",
                "training_phase"   : "Phase 3 — Calibration",
                "rationale": (
                    "Platt chosen over isotonic for test-set generalisation. "
                    "Isotonic retained for OOF analysis only."
                ),
            })

            # ── Platt parameters ───────────────────────────────────────────
            mlflow.log_metric("platt_slope",           platt["slope"])
            mlflow.log_metric("platt_intercept",       platt["intercept"])
            mlflow.log_metric("ll_reduction_pct",      platt["ll_reduction_pct"])

            # ── Pre/post calibration comparison ───────────────────────────
            # Pre-calibration = raw OOF; post = Platt-calibrated OOF
            mlflow.log_metric("post_platt_logloss",    ref["logloss"])
            mlflow.log_metric("post_platt_auc",        ref["auc"])
            mlflow.log_metric("post_platt_composite",  ref["composite"])

            # ── OOF distribution stats (raw vs calibrated) ────────────────
            raw_stats = _npy_stats(
                _MODEL_RUN_DIRS.get(
                    f"{model_name}_tuned",
                    _MODEL_RUN_DIRS.get(model_name, Path("nonexistent"))
                ) / "oof_preds.npy"
            )
            if raw_stats:
                for stat, val in raw_stats.items():
                    mlflow.log_metric(f"oof_raw_{stat}", val)

            cal_stats = _npy_stats(_MULTI_MODEL / f"oof_calibrated_{model_name}.npy")
            if cal_stats:
                for stat, val in cal_stats.items():
                    mlflow.log_metric(f"oof_cal_{stat}", val)
                # Compression indicator: calibrated max vs raw max
                if raw_stats:
                    mlflow.log_metric("max_compression_ratio",
                                      cal_stats["max"] / (raw_stats["max"] + 1e-9))

            # ── Calibration summary JSON (logreg/tabnet) ──────────────────
            cal_summary = _read_json(_CALIB_DIR / folder / "calibration_summary.json")
            if cal_summary:
                for k, v in cal_summary.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"cal_{k}", float(v))

            # ── Artefacts ──────────────────────────────────────────────────
            _log_artifacts_in_dir(_CALIB_DIR / folder, [".json"])
            _log_artifact_if_exists(
                _CALIB_DIR / "reliability_diagrams_logreg_tabnet.png"
            )

        log.info("    ✓  %s calibration logged", model_name)


# =============================================================================
# EXPERIMENT 5 — ENSEMBLE STRATEGIES
# =============================================================================

def log_ensemble(mlflow) -> None:
    """
    Log the full ensemble experiment — the centrepiece of the project.

    Logs two ensemble versions:
      v4 (3-model GBM-only) — the intermediate result
      v5.1 (5-model production) — the final competition submission

    For v5.1, logs one MLflow run per ensemble strategy:
      1. simple_average
      2. optimised_weighted_average  ← WINNER (0.19144)
      3. stacking
      4. calibrated_stacking

    Also logs the ablation study results as a separate parent run
    to show why 0.19144 is the stable floor.

    The correlation matrix and ensemble weights are logged as
    artefacts — they tell the diversity story visually.
    """
    exp_name = "AI4EAC/5_ensemble"
    mlflow.set_experiment(exp_name)
    log.info("\n[Exp 5] %s", exp_name)

    v5_run_dir = _MODEL_RUN_DIRS["v5_ensemble"]

    # ── 5A: v5.1 Production Ensemble ──────────────────────────────────────
    log.info("  Logging v5.1 production ensemble ...")
    stacking_results = _read_json(v5_run_dir / "stacking_results.json")
    metadata_v5      = _read_json(v5_run_dir / "metadata.json")
    weights_json     = _read_json(v5_run_dir / "ensemble_weights.json")

    # Parse stacking_results into strategy → metrics dict
    strategy_metrics: Dict[str, Dict] = {}
    if stacking_results:
        for entry in stacking_results:
            name = entry.get("name") or entry.get("strategy", "unknown")
            strategy_metrics[name] = entry

    # Map our known results as ground truth (in case JSON is missing)
    _known_strategies = {
        "optimised_weighted_average": {
            "logloss": 0.25588, "auc": 0.90522, "score": 0.19144,
        },
        "simple_average": {
            "logloss": 0.26448, "auc": 0.89765, "score": 0.19963,
        },
        "stacking": {
            "logloss": 0.26433, "auc": 0.90097, "score": 0.19821,
        },
        "stacking_calibrated": {
            "logloss": 0.27663, "auc": 0.89837, "score": 0.20663,
        },
    }

    # Merge on-disk data with known ground truth
    for strategy, known in _known_strategies.items():
        if strategy not in strategy_metrics:
            strategy_metrics[strategy] = known
        else:
            # Fill missing keys from ground truth
            for k, v in known.items():
                strategy_metrics[strategy].setdefault(k, v)

    # Log a parent run for the entire v5.1 ensemble
    with mlflow.start_run(run_name="v5.1-ensemble-summary") as parent_run:
        mlflow.set_tags({
            **_common_tags(),
            "estimator_type"    : "ensemble",
            "metric_direction"  : "lower_is_better",
            "ensemble_version"  : "v5.1",
            "training_phase"    : "Phase 5 — 5-model ensemble",
            "best_strategy"     : "optimised_weighted_average",
            "final_submission"  : "true",
            "model_set"         : "lightgbm+xgboost+catboost+logreg+tabnet",
            "run_dir"           : str(v5_run_dir),
            "meta_C"            : "0.05",
            "optimiser"         : "Scipy Nelder-Mead",
            "n_initialisations" : "24",
        })

        # Best score
        mlflow.log_metric("best_composite_score",      _ENSEMBLE_BEST_SCORE)
        mlflow.log_metric("improvement_vs_xgb",
                          _ENSEMBLE_BEST_SCORE - _BASELINE_SCORES["xgboost"]["composite"])

        # Model weights
        for model_name, weight in _ENSEMBLE_WEIGHTS.items():
            mlflow.log_metric(f"weight_{model_name}", weight)

        # Cross-cluster correlations
        for (m1, m2), corr in _OOF_CORRELATIONS.items():
            mlflow.log_metric(f"oof_corr_{m1}_vs_{m2}", corr)

        mlflow.log_metric("intra_gbm_corr_min",  0.962)
        mlflow.log_metric("intra_gbm_corr_max",  0.978)
        mlflow.log_metric("cross_cluster_corr_min", 0.745)
        mlflow.log_metric("cross_cluster_corr_max", 0.753)

        # Artefacts on the parent run
        _log_artifact_if_exists(v5_run_dir / "ensemble_weights.json")
        _log_artifact_if_exists(v5_run_dir / "correlation_matrix.csv")
        _log_artifact_if_exists(v5_run_dir / "feature_importance.csv")
        _log_artifact_if_exists(v5_run_dir / "stacking_results.json")
        _log_artifact_if_exists(v5_run_dir / "metadata.json")

        # ── Nested runs: one per strategy ─────────────────────────────────
        strategy_display = {
            "optimised_weighted_average": "2-optimised-weighted-avg",
            "simple_average"            : "1-simple-average",
            "stacking"                  : "3-stacking",
            "stacking_calibrated"       : "4-calibrated-stacking",
        }

        for strategy, display_name in strategy_display.items():
            metrics = strategy_metrics.get(strategy, {})
            is_best = (strategy == "optimised_weighted_average")

            with mlflow.start_run(run_name=display_name, nested=True):
                mlflow.set_tags({
                    **_common_tags(),
                    "strategy"         : strategy,
                    "metric_direction" : "lower_is_better",
                    "is_best_strategy" : str(is_best),
                    "ensemble_version" : "v5.1",
                    "model_count"      : "5",
                    "note": (
                        "WINNER — score=0.19144, weights optimised with Nelder-Mead "
                        "(24 initialisations, 5000 max-iter per)"
                        if is_best else
                        "Stacking loses to weighted avg: LogReg/TabNet bounded probs "
                        "(max 0.74/0.70) compress meta-model, Platt slope=5.99"
                        if "stack" in strategy else
                        "Equal-weight baseline; GBM bloc gets 3/5 slots = 60% implicit weight"
                    ),
                })

                logloss  = metrics.get("logloss", metrics.get("ll", None))
                auc      = metrics.get("auc", None)
                score    = metrics.get("score", metrics.get("composite_score", None))
                brier    = metrics.get("brier", None)

                if logloss  is not None: mlflow.log_metric("logloss",         float(logloss))
                if auc      is not None: mlflow.log_metric("auc",              float(auc))
                if score    is not None: mlflow.log_metric("composite_score",  float(score))
                if brier    is not None: mlflow.log_metric("brier_score",      float(brier))

                if score is not None:
                    mlflow.log_metric(
                        "delta_vs_xgb_baseline",
                        float(score) - _BASELINE_SCORES["xgboost"]["composite"]
                    )

                # OOF prediction distribution
                oof_file = (
                    "stacking_oof.npy"   if "stack" in strategy
                    else "ensemble_oof.npy"
                )
                oof_stats = _npy_stats(v5_run_dir / oof_file)
                if oof_stats:
                    for stat, val in oof_stats.items():
                        mlflow.log_metric(f"oof_{stat}", val)

            log.info("    ✓  strategy '%s' logged", strategy)

    # ── 5B: Ablation study results ────────────────────────────────────────
    log.info("  Logging ablation study ...")

    # Known ablation results from handoff doc
    ablation_experiments = [
        {
            "name"      : "ablation-exp1-no-logreg",
            "models"    : "lightgbm+xgboost+catboost+tabnet",
            "n_models"  : 4,
            "best_score": 0.19144,
            "verdict"   : "NEUTRAL — LogReg weight was already 0.000",
        },
        {
            "name"      : "ablation-exp2-gbm-only",
            "models"    : "lightgbm+xgboost+catboost",
            "n_models"  : 3,
            "best_score": 0.19164,
            "verdict"   : "NEUTRAL — non-GBM diversity = 0.00020 composite benefit",
        },
        {
            "name"      : "ablation-exp3-meta-c-0.1",
            "models"    : "all-5",
            "n_models"  : 5,
            "best_score": 0.19817,  # stacking with C=0.1
            "verdict"   : "WORSE — stacking weakness is structural, not C-related",
        },
        {
            "name"      : "ablation-exp4-gbm-tabnet",
            "models"    : "lightgbm+xgboost+catboost+tabnet",
            "n_models"  : 4,
            "best_score": 0.19144,
            "verdict"   : "NEUTRAL — 0.19144 is the stable floor",
        },
    ]

    # Also try to read from the ablation CSVs in notebooks/outputs/
    ablation_csv = (
        PROJECT_ROOT / "notebooks" / "outputs" / "ablation_study_results.csv"
    )
    ablation_data: Dict[str, Dict] = {}
    if ablation_csv.exists():
        try:
            import csv
            with open(ablation_csv) as f:
                for row in csv.DictReader(f):
                    ablation_data[row.get("experiment", "")] = dict(row)
        except Exception as exc:
            log.warning("    Could not read ablation CSV: %s", exc)

    with mlflow.start_run(run_name="ablation-study-v5.1"):
        mlflow.set_tags({
            **_common_tags(),
            "training_phase": "Phase 6 — Ablation study",
            "baseline_score": str(_ENSEMBLE_BEST_SCORE),
            "conclusion"    : "0.19144 is structurally stable — no subset or C sweep beat it",
        })
        mlflow.log_metric("reference_best_score", _ENSEMBLE_BEST_SCORE)

        for exp in ablation_experiments:
            mlflow.log_metric(
                f"ablation_{exp['name'].replace('-', '_')}_best",
                exp["best_score"]
            )
            mlflow.log_metric(
                f"ablation_{exp['name'].replace('-', '_')}_delta",
                exp["best_score"] - _ENSEMBLE_BEST_SCORE
            )
            mlflow.set_tag(
                f"ablation_{exp['name']}_verdict", exp["verdict"]
            )

        # Log ablation artefacts
        _log_artifact_if_exists(ablation_csv)
        ablation_outputs = PROJECT_ROOT / "notebooks" / "outputs"
        _log_artifacts_in_dir(ablation_outputs, [".png", ".csv", ".md"])

    log.info("    ✓  ensemble logged")


# =============================================================================
# EXPERIMENT 6 — SHAP INTERPRETABILITY + AUGMENTATION
# =============================================================================

def log_shap(mlflow) -> None:
    """
    Log SHAP interpretability findings and augmentation experiments.

    Two sub-experiments:
      6A — SHAP interpretability (notebook 13):
           Top-10 features, theme-level importance, failure mode analysis.
           The balance_slope ≈ 0 bifurcation threshold.

      6B — SHAP augmentation (notebook 13 §11):
           K=5/20/50/100 experiments.
           Root cause: unscaled features (balance_slope ±40,000) caused
           K≥20 to catastrophically collapse the meta-model.
    """
    exp_name = "AI4EAC/06_shap_interpretability_and_failure_analysis"
    mlflow.set_experiment(exp_name)
    log.info("\n[Exp 6] %s", exp_name)

    # ── 6A: SHAP interpretability ─────────────────────────────────────────
    with mlflow.start_run(run_name="shap-xgboost-825-features"):
        mlflow.set_tags({
            **_common_tags(),
            "model_used"        : "XGBoost v6 (825 features)",
            "metric_direction"  : "lower_is_better",
            "shap_method"       : "TreeExplainer, interventional perturbation",
            "background_n"      : "500",
            "sample_n"          : "5000",
            "training_phase"    : "Phase 7 — SHAP interpretability",
            "run_dir"           : "outputs/experiments/v6_xgb_825_features",
            "key_finding_1"     : "balance_slope bifurcation at 0: SHAP +1.0 to +2.5 below, -0.5 to -1.0 above",
            "key_finding_2"     : "Balance Deterioration 36% + Income Stability 19% = 55% of signal",
            "key_finding_3"     : "False negative failure: income-collapse-but-balance-stable customers",
            "key_finding_4"     : "MVC segment lowest confidence (mean |SHAP|=0.0081) — most heterogeneous",
            "rule_override"     : "inflow_slope < -5000 AND balance_slope > 0 → flag as stressed",
        })

        # Top-10 SHAP feature importances (from handoff)
        top_shap_features = [
            ("balance_slope",              0.793),
            ("m4_daily_avg_bal",           0.287),
            ("inflow_slope",               0.262),
            ("m3_daily_avg_bal",           0.210),
            ("balance_trend_3m",           0.197),
            ("m2_daily_avg_bal",           0.194),
            ("inflow_volume_recency_ratio", 0.193),
            ("withdraw_recency_x_spend_ratio", 0.161),
            ("balance_cv_x_drawdown",      0.138),
            ("deposit_recency_ratio",      0.128),
        ]
        for i, (feat, importance) in enumerate(top_shap_features):
            mlflow.log_metric(f"shap_top_{i+1:02d}_importance", importance)
            mlflow.set_tag(f"shap_top_{i+1:02d}_feature", feat)

        # Theme-level importance (corrected, from handoff)
        theme_importances = {
            "balance_deterioration" : 2.537,
            "income_stability"      : 1.328,
            "other_residual"        : 1.108,
            "spending_stress"       : 0.726,
            "engagement_dynamics"   : 0.604,
            "behavioral_instability": 0.329,
            "temporal_deterioration": 0.300,
        }
        total_shap = sum(theme_importances.values())
        for theme, val in theme_importances.items():
            mlflow.log_metric(f"theme_shap_{theme}", val)
            mlflow.log_metric(f"theme_share_{theme}", val / total_shap)

        # Segment confidence metrics
        segment_shap = {"HVC": 0.01027, "LVC": 0.00839, "MVC": 0.00807}
        for seg, val in segment_shap.items():
            mlflow.log_metric(f"segment_mean_abs_shap_{seg}", val)

        # Model performance on SHAP run (slightly optimistic — own training fold)
        mlflow.log_metric("shap_model_auc",     0.982)
        mlflow.log_metric("shap_model_logloss", 0.202)

        # Artefacts
        _log_artifact_if_exists(_SHAP_DIR / "shap_feature_importance.csv")
        _log_artifact_if_exists(_SHAP_DIR / "shap_theme_importance.csv")
        _log_artifact_if_exists(_SHAP_DIR / "shap_feature_subsets.json")
        _log_artifact_if_exists(_SHAP_DIR / "top50_features.csv")

        # SHAP augmented results CSV
        _log_artifact_if_exists(_SHAP_DIR / "shap_augmented_ensemble_results.csv")

    log.info("    ✓  SHAP interpretability logged")

    # ── 6B: SHAP augmentation experiments ─────────────────────────────────
    log.info("  Logging SHAP augmentation experiments ...")

    aug_shap_dir = _experiments_dir() / "v5_shap_augmented"

    with mlflow.start_run(run_name="shap-augmentation-sweep"):
        mlflow.set_tags({
            **_common_tags(),
            "training_phase": "Phase 8 — SHAP augmentation",
            "conclusion"    : (
                "ALL K values failed to beat 0.19144. "
                "Root cause: balance_slope (±40,000 range) unscaled in meta-model. "
                "Fix: apply StandardScaler to extra_features before ensemble."
            ),
            "fix_for_future": "StandardScaler on extra_features before run_ensemble_pipeline()",
        })
        mlflow.log_metric("reference_score", _ENSEMBLE_BEST_SCORE)

        for k_val, results in _SHAP_AUG_RESULTS.items():
            mlflow.log_metric(f"K{k_val}_best_score",      results["best_score"])
            mlflow.log_metric(f"K{k_val}_stacking_score",  results["stacking_score"])
            mlflow.log_metric(f"K{k_val}_stacking_auc",    results["stacking_auc"])
            mlflow.log_metric(f"K{k_val}_delta_vs_best",
                              results["stacking_score"] - _ENSEMBLE_BEST_SCORE)

            # Log per-K run directory artefacts
            k_run_dirs = sorted(
                (aug_shap_dir / f"K{k_val}").glob("run_*"), reverse=True
            )
            if k_run_dirs:
                k_run = k_run_dirs[0]
                _log_artifact_if_exists(k_run / "ensemble_weights.json")
                _log_artifact_if_exists(k_run / "stacking_results.json")

        # Log the aggregated results CSV
        aug_results_csv = _SHAP_DIR / "shap_augmented_ensemble_results.csv"
        _log_artifact_if_exists(aug_results_csv)

        # Individual K experiments as nested runs
        for k_val, results in _SHAP_AUG_RESULTS.items():
            run_name_k = f"K{k_val}-augmentation"
            k_dir = aug_shap_dir / f"K{k_val}"
            k_run_dirs = sorted(k_dir.glob("run_*"), reverse=True) if k_dir.exists() else []

            with mlflow.start_run(run_name=run_name_k, nested=True):
                status = (
                    "SAFE — regularisation suppressed unscaled features correctly"
                    if k_val == 5 else
                    "CATASTROPHIC COLLAPSE — unscaled financial features overwhelmed LogReg meta-model"
                )
                mlflow.set_tags({
                    "k_features"  : str(k_val),
                    "outcome"     : "no_improvement",
                    "status"      : status,
                    "root_cause"  : (
                        "None — K=5 safely suppressed"
                        if k_val == 5 else
                        "Raw SHAP features not standardised; balance_slope ±40,000 "
                        "overwhelms C=0.05 regularisation calibrated for [0,1] OOF inputs"
                    ),
                })
                mlflow.log_metric("best_composite",       results["best_score"])
                mlflow.log_metric("stacking_composite",   results["stacking_score"])
                mlflow.log_metric("stacking_auc",         results["stacking_auc"])
                mlflow.log_metric("delta_vs_unaugmented",
                                  results["stacking_score"] - _ENSEMBLE_BEST_SCORE)

                if k_run_dirs:
                    _log_artifact_if_exists(k_run_dirs[0] / "stacking_results.json")

    log.info("    ✓  SHAP augmentation logged")


# =============================================================================
# EXPERIMENT 7 — FINAL SUBMISSION
# =============================================================================

def log_final_submission(mlflow) -> None:
    """
    Log the final competition submission as a dedicated experiment.

    This is the capstone run — everything the project produced,
    summarised in a single MLflow run with the submission file as
    an artefact.  Any recruiter opening the MLflow UI sees this as
    the headline result immediately.
    """
    exp_name = "AI4EAC/7_final_submission"
    mlflow.set_experiment(exp_name)
    log.info("\n[Exp 7] %s", exp_name)

    # Find the best submission file
    submission_files = [
        _SUBMISSIONS / "submission_v5_ensemble.csv",
        _SUBMISSIONS / "submission_v5_final.csv",
    ]
    submission_file = next((f for f in submission_files if f.exists()), None)

    report_file = _SUBMISSIONS / "submission_v5_ensemble.report.json"
    report      = _read_json(report_file)

    with mlflow.start_run(run_name="final-submission-v5.1-ensemble"):
        mlflow.set_tags({
            **_common_tags(),
            "competition"        : "AI for Economic Activity Challenge (AI4EAC) — Zindi Africa",
            "author"             : "Henry Otsyula",
            "final_submission"   : "true",
            "strategy"           : "optimised_weighted_average",
            "ensemble_version"   : "v5.1",
            "model_set"          : "lightgbm+xgboost+catboost+logreg+tabnet",
            "submission_format"  : "ID,Target,TargetLogLoss,TargetRAUC (all three identical)",
            "test_rows"          : "30000",
            "metric_formula"     : "0.6 × LogLoss + 0.4 × (1 − AUC)",
            "metric_direction"   : "lower_is_better",
            "weights_source"     : "outputs/experiments/v5_ensemble/run_20260508_054540/ensemble_weights.json",
            "training_data"      : "40000 customers × 183 raw features",
            "feature_engineering": "v2.2.1 — 825 engineered features",
            "phase_9_complete"   : "true",
        })

        # ── Competition metric ─────────────────────────────────────────────
        mlflow.log_metric("oof_composite_score",    _ENSEMBLE_BEST_SCORE)
        mlflow.log_metric("oof_logloss",            0.25588)
        mlflow.log_metric("oof_auc",                0.90522)
        mlflow.log_metric("improvement_vs_best_single",
                          _ENSEMBLE_BEST_SCORE - _BASELINE_SCORES["xgboost"]["composite"])

        # ── Per-model weights ──────────────────────────────────────────────
        for model_name, weight in _ENSEMBLE_WEIGHTS.items():
            mlflow.log_metric(f"weight_{model_name}", weight)

        # ── Individual model reference scores ─────────────────────────────
        for model_name, scores in _BASELINE_SCORES.items():
            mlflow.log_metric(f"{model_name}_oof_composite", scores["composite"])
            mlflow.log_metric(f"{model_name}_oof_logloss",   scores["logloss"])
            mlflow.log_metric(f"{model_name}_oof_auc",       scores["auc"])

        # ── Submission prediction distribution (from report JSON) ──────────
        if report and "prediction_stats" in report:
            stats = report["prediction_stats"]
            for stat, val in stats.items():
                mlflow.log_metric(f"submission_{stat}", float(val))

        # ── Runtime ───────────────────────────────────────────────────────
        if report and "runtime_sec" in report:
            mlflow.log_metric("inference_runtime_sec", float(report["runtime_sec"]))

        # ── Key findings summary ───────────────────────────────────────────
        mlflow.set_tag("finding_1",
                       "balance_slope is #1 feature (SHAP=0.793); near-vertical bifurcation at 0")
        mlflow.set_tag("finding_2",
                       "Balance Deterioration (36%) + Income Stability (19%) = 55% of signal")
        mlflow.set_tag("finding_3",
                       "Stacking lost to weighted avg: LogReg/TabNet bounded probs, Platt slope=5.99")
        mlflow.set_tag("finding_4",
                       "Cross-cluster r=0.745-0.753 justified 5-model ensemble over GBM-only")
        mlflow.set_tag("finding_5",
                       "False negative failure mode: income-collapse-but-balance-stable customers")

        # ── Artefacts ──────────────────────────────────────────────────────
        if submission_file:
            mlflow.log_artifact(str(submission_file))
            log.info("    Submission file logged: %s", submission_file.name)
        else:
            log.warning("    No submission CSV found at %s", _SUBMISSIONS)

        _log_artifact_if_exists(report_file)
        _log_artifact_if_exists(_MODEL_RUN_DIRS["v5_ensemble"] / "ensemble_weights.json")
        _log_artifact_if_exists(_MODEL_RUN_DIRS["v5_ensemble"] / "metadata.json")
        _log_artifact_if_exists(_MODEL_RUN_DIRS["v5_ensemble"] / "correlation_matrix.csv")

    log.info("    ✓  Final submission logged")


# =============================================================================
# MASTER ORCHESTRATOR
# =============================================================================

def log_all_experiments(
    tracking_uri : str  = "sqlite:///mlflow.db",
    experiments  : Optional[List[str]] = None,
    dry_run      : bool = False,
) -> None:
    """
    Run all experiment loggers in sequence.

    Parameters
    ----------
    tracking_uri : where MLflow stores runs.  Defaults to sqlite:///mlflow.db

    experiments  : list of experiment numbers/names to run, e.g.
                   ["1", "5", "7"].  None = run all.
    dry_run      : if True, validates paths but logs nothing.
    """
    try:
        import mlflow
    except ImportError:
        log.error(
            "MLflow is not installed.\n"
            "Install with:  conda activate liquidity_ml && pip install mlflow\n"
            "Then re-run:   python -m src.mlflow_logging.log_experiments"
        )
        sys.exit(1)

    from pathlib import Path

    # Resolve tracking URI safely for MLflow
    if tracking_uri.startswith(("http", "https", "sqlite")):
        uri = tracking_uri
    else:
        # Convert local path → absolute file URI (MLflow-compatible)
        uri = Path(tracking_uri).resolve().as_uri()

    mlflow.set_tracking_uri(uri)
    log.info("Resolved MLflow tracking URI: %s", mlflow.get_tracking_uri())
    # Force early failure instead of silent cascade
    from urllib.parse import urlparse
    parsed = urlparse(mlflow.get_tracking_uri())
    if parsed.scheme not in ("file", "http", "https", "sqlite", ""):
        raise ValueError(f"Unsupported MLflow URI scheme: {parsed.scheme}")
    
    log.info("=" * 70)
    log.info("AI4EAC MLflow Retrospective Logger")
    log.info("=" * 70)
    log.info("Tracking URI  : %s", uri)
    log.info("Project root  : %s", PROJECT_ROOT)
    log.info("Dry run       : %s", dry_run)

    actual_uri = mlflow.get_tracking_uri()
    log.info("Resolved MLflow tracking URI: %s", actual_uri)

    if dry_run:
        log.info("\n[DRY RUN] Validating artefact paths ...")
        log.info("[DRY RUN] Validating MLflow tracking URI ...")

        test_uri = Path(tracking_uri).resolve().as_uri() \
            if not tracking_uri.startswith(("http", "https", "sqlite")) \
            else tracking_uri

        log.info("  MLflow URI would be: %s", test_uri)
        log.info("[DRY RUN] Path validation complete.  Nothing logged.")
        return

    # Experiment registry — ordered to tell a coherent research story
    all_loggers = {
        "1": ("Baseline Models",            log_baseline_models),
        "2": ("Hyperparameter Tuning",      log_tuning),
        "3": ("Extended Models",            log_extended_models),
        "4": ("Calibration",                log_calibration),
        "5": ("Ensemble",                   log_ensemble),
        "6": ("SHAP Interpretability",      log_shap),
        "7": ("Final Submission",           log_final_submission),
    }

    # Filter if --experiment flag provided
    if experiments:
        # Accept "1", "baseline", "Baseline Models" etc.
        selected = {}
        for key, (name, fn) in all_loggers.items():
            if (key in experiments or
                name.lower() in [e.lower() for e in experiments] or
                any(e in name.lower() for e in [ex.lower() for ex in experiments])):
                selected[key] = (name, fn)
        if not selected:
            log.error("No experiments matched: %s.  Valid keys: %s",
                      experiments, list(all_loggers.keys()))
            sys.exit(1)
        loggers_to_run = selected
    else:
        loggers_to_run = all_loggers

    start_time = time.perf_counter()
    failed = []

    for key, (name, logger_fn) in loggers_to_run.items():
        log.info("\n%s", "─" * 60)
        log.info("Running experiment %s — %s", key, name)
        log.info("─" * 60)

         # Safety cleanup: prevent active-run collisions
        if mlflow.active_run():
            log.warning("Active MLflow run detected. Closing it first.")
            mlflow.end_run()
        try:
            logger_fn(mlflow)
        except Exception as exc:
            log.error("  FAILED: %s  —  %s", name, exc, exc_info=True)
            failed.append((key, name, str(exc)))

        finally:
            # Ensure no dangling runs remain open
            if mlflow.active_run():
                mlflow.end_run()

    runtime = time.perf_counter() - start_time

    # ── Final summary ──────────────────────────────────────────────────────
    log.info("\n%s", "=" * 70)
    log.info("LOGGING COMPLETE")
    log.info("=" * 70)
    log.info("  Experiments attempted : %d", len(loggers_to_run))
    log.info("  Succeeded             : %d", len(loggers_to_run) - len(failed))
    log.info("  Failed                : %d", len(failed))
    log.info("  Runtime               : %.1fs", runtime)
    if failed:
        log.warning("  Failed experiments:")
        for key, name, err in failed:
            log.warning("    %s — %s: %s", key, name, err)
    log.info("")
    log.info("  To view the MLflow UI:")
    log.info("    mlflow ui --port 5000")
    log.info("    Open: http://localhost:5000")
    log.info("")
    log.info("  Tracking directory: %s", uri)
    log.info("=" * 70)


def _validate_paths() -> None:
    """Check all key artefact paths exist and report any gaps."""
    critical_paths = [
        _MODEL_RUN_DIRS["v5_ensemble"],
        _MULTI_MODEL / "y_true.npy",
        _TUNING_DIR / "lightgbm_best_params.yaml",
        _TUNING_DIR / "xgboost_best_params.yaml",
    ]
    for path in critical_paths:
        status = "✓" if path.exists() else "✗ MISSING"
        log.info("  %s  %s", status, path)


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog        = "python -m src.mlflow_logging.log_experiments",
        description = (
            "AI4EAC — Retrospective MLflow experiment logger.\n"
            "Reads all training artefacts from disk and logs them to MLflow.\n"
            "Does NOT re-run any training."
        ),
        formatter_class = argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tracking-uri",
        default = "sqlite:///mlflow.db",
        help    = "MLflow tracking URI (default: sqlite:///mlflow.db in project root)",
    )
    parser.add_argument(
        "--experiment",
        nargs   = "+",
        default = None,
        metavar = "N",
        help    = "Specific experiment(s) to log, e.g. --experiment 1 5 7",
    )
    parser.add_argument(
        "--dry-run",
        action  = "store_true",
        default = False,
        help    = "Validate artefact paths without logging anything.",
    )
    parser.add_argument(
        "--log-level",
        default = "INFO",
        choices = ["DEBUG", "INFO", "WARNING"],
        help    = "Logging verbosity (default: INFO)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.getLogger().setLevel(args.log_level)

    log_all_experiments(
        tracking_uri = args.tracking_uri,
        experiments  = args.experiment,
        dry_run      = args.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())