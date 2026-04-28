"""
Multi-Model Training Orchestrator
==================================
Project : Liquidity Stress Early Warning (AI4EAC / Zindi)
Module  : src/orchestration/run_all_models.py

Responsibilities
----------------
- Resolve project root deterministically (no CWD dependency)
- Load and validate per-model YAML configs
- Execute LightGBM / XGBoost / CatBoost training in-process
- Windows-safe logging (UTF-8, no emoji in file/console handlers)
- Per-model failure isolation: one failure never aborts siblings
- Structured run timing and summary report
- Save YAML run summary to outputs/logs/

Output contract (aligned with project output spec)
---------------------------------------------------
outputs/experiments/<stage>/<model>/run_YYYYMMDD_HHMMSS/
    models/              fold-level serialised models
    oof_preds.npy        out-of-fold predictions
    y_true.npy           ground-truth labels
    fold_scores.json     per-fold logloss + AUC
    fold_indices.pkl     train/valid indices per fold
    fold_predictions.pkl per-fold raw predictions
    feature_importance.csv
    feature_list.json
    preprocessor.pkl
    metadata.json
    config_used.yaml

Usage
-----
# From project root:
python -m src.orchestration.run_all_models

# Single model (override):
python -m src.orchestration.run_all_models --configs configs/lgbm_v2.yaml
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

DEFAULT_CONFIGS: List[str] = [
    "configs/lgbm_v2.yaml",
    "configs/xgb_v2.yaml",
    "configs/catboost_v2.yaml",
]

LOG_DIR = PROJECT_ROOT / "outputs" / "logs"


# =============================================================================
# WINDOWS-SAFE LOGGING
# =============================================================================

import logging


class _AsciiFilter(logging.Filter):
    """Strip non-ASCII characters from records before they reach the
    Windows console handler. The file handler receives the full message."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = record.msg.encode("ascii", errors="replace").decode("ascii")
        return True


def setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"multi_model_run_{timestamp}.log"

    logger = logging.getLogger("orchestrator")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()          # prevent duplicate handlers on re-runs

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- File handler: full UTF-8, keeps all characters ---
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # --- Console handler: ASCII-safe for Windows cp1252 ---
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
# CONFIG HELPERS
# =============================================================================

def _resolve_config_path(raw_path: str) -> Path:
    """
    Accept either absolute paths or paths relative to project root.
    Raises FileNotFoundError with a clear message if not found.
    """
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / raw_path
    if not candidate.exists():
        raise FileNotFoundError(
            f"Config not found: {candidate}\n"
            f"  Tried relative to project root: {PROJECT_ROOT}\n"
            f"  Original path supplied        : {raw_path}"
        )
    return candidate


def load_config(path: str) -> Dict[str, Any]:
    resolved = _resolve_config_path(path)
    with open(resolved, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg, resolved


def validate_config(cfg: Dict[str, Any], path: Path) -> None:
    required_top = ["project", "experiment", "data", "model", "cv",
                    "training", "evaluation", "artifacts"]
    missing = [k for k in required_top if k not in cfg]
    if missing:
        raise ValueError(
            f"Config {path.name} is missing required sections: {missing}"
        )
    if "name" not in cfg["model"]:
        raise ValueError(f"Config {path.name}: model.name is required")
    if "params" not in cfg["model"]:
        raise ValueError(f"Config {path.name}: model.params is required")


# =============================================================================
# OUTPUT PATH BUILDER
# Produces: outputs/experiments/<stage>/<model>/run_YYYYMMDD_HHMMSS/
# =============================================================================

def build_run_output_dir(cfg: Dict[str, Any]) -> Path:
    stage      = cfg["experiment"]["stage"]          # e.g. "baseline"
    model_name = cfg["model"]["name"]                # e.g. "lightgbm"
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
# Calls the CV engine and artifact saver directly — no subprocess spawn.
# =============================================================================

def _run_training_inprocess(
    cfg: Dict[str, Any],
    run_dir: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Import and execute the CV pipeline in-process.
    Returns the cv_results dict for summary reporting.
    """

    # --- Lazy imports so the orchestrator itself has no hard model deps ---
    from src.data.load_data import load_data
    from src.features.feature_engineering import build_features, split_features_target
    from src.preprocessing.preprocessing import PreprocessingPipeline
    from src.training.cv import run_cv, save_cv_outputs

    logger.info("  Loading data ...")
    train_df, _ = load_data(
        train_path=str(PROJECT_ROOT / cfg["data"]["train_path"]),
        validate=True,
        verbose=False,
    )

    logger.info("  Building features ...")
    train_fe = build_features(train_df)
    X, y = split_features_target(train_fe, target_col=cfg["data"]["target"])

    logger.info("  Fitting preprocessor ...")
    preproc = PreprocessingPipeline(cfg)
    X_processed = preproc.fit_transform(X, y)

    logger.info(
        "  Feature matrix: %d rows x %d cols", X_processed.shape[0], X_processed.shape[1]
    )

    logger.info("  Starting cross-validation ...")
    cv_results = run_cv(X_processed, y, cfg)

    logger.info("  Saving artifacts to: %s", run_dir)
    # Attach preprocessor to results so save_cv_outputs can persist it
    cv_results["preprocessor"] = preproc

    save_cv_outputs(cv_results, cfg, str(run_dir))

    return cv_results


# =============================================================================
# SINGLE EXPERIMENT RUNNER
# Wraps the full lifecycle of one config file with timing + error isolation.
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
        logger.info("STARTING | model=%-12s  stage=%s", model_name.upper(), stage)
        logger.info("Config   | %s", resolved_path)
        logger.info("=" * 60)

        # ------------------------------------------------------------------
        # 2. Build output directory
        # ------------------------------------------------------------------
        run_dir = build_run_output_dir(cfg)
        result["run_dir"] = str(run_dir)
        logger.info("Run dir  | %s", run_dir)

        # ------------------------------------------------------------------
        # 3. Execute training
        # ------------------------------------------------------------------
        cv_results = _run_training_inprocess(cfg, run_dir, logger)

        # ------------------------------------------------------------------
        # 4. Record metrics
        # ------------------------------------------------------------------
        result["mean_logloss"] = round(cv_results["mean_logloss"], 5)
        result["mean_auc"]     = round(cv_results["mean_auc"], 5)
        result["final_score"]  = round(cv_results["final_score"], 5)
        result["status"]       = "success"

        runtime = time.perf_counter() - start_time
        result["runtime_sec"] = round(runtime, 1)

        logger.info("")
        logger.info(
            "SUCCESS  | model=%-12s  LogLoss=%.5f  AUC=%.5f  Score=%.5f  [%.1fs]",
            model_name.upper(),
            result["mean_logloss"],
            result["mean_auc"],
            result["final_score"],
            runtime,
        )

    except Exception as exc:
        runtime = time.perf_counter() - start_time
        result["runtime_sec"] = round(runtime, 1)
        result["error"] = str(exc)

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
    logger.info("%-14s | %-8s | %-10s | %-8s | %-10s | %s",
                "Model", "Status", "LogLoss", "AUC", "Score", "Runtime(s)")
    logger.info("-" * 72)

    for r in results:
        ll     = f"{r['mean_logloss']:.5f}" if r["mean_logloss"] else "N/A"
        auc    = f"{r['mean_auc']:.5f}"     if r["mean_auc"]     else "N/A"
        score  = f"{r['final_score']:.5f}"  if r["final_score"]  else "N/A"
        rt     = f"{r['runtime_sec']:.1f}s" if r["runtime_sec"]  else "N/A"
        logger.info(
            "%-14s | %-8s | %-10s | %-8s | %-10s | %s",
            r["model"].upper(), r["status"].upper(), ll, auc, score, rt,
        )

    logger.info("-" * 72)
    logger.info("Passed  : %d / %d", len(success), len(results))
    logger.info("Failed  : %d / %d", len(failed),  len(results))
    logger.info("Total   : %.1f seconds", total_runtime)
    logger.info("=" * 72)

    if failed:
        logger.warning("Failed experiments:")
        for r in failed:
            logger.warning("  %s -> %s", r["config"], r["error"])


def _save_summary(results: List[Dict[str, Any]], log_dir: Path) -> Path:
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = log_dir / f"run_summary_{timestamp}.yaml"

    # Convert Path objects to strings for YAML serialisation
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

    results: List[Dict[str, Any]] = []
    total_start = time.perf_counter()

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
        description="Multi-model training orchestrator — Liquidity Stress Early Warning"
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_CONFIGS,
        help="One or more config file paths (absolute or relative to project root). "
             f"Defaults: {DEFAULT_CONFIGS}",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(args.configs)