"""
Production-Grade Multi-Model Training Orchestrator
==================================================

Responsibilities:
- Execute multiple model configs sequentially
- Logging (file + console)
- Failure handling (continue on error)
- Runtime tracking
- Output structure validation
- Reproducible experiment execution

Usage:
python -m src.orchestration.run_all_models
"""

import os
import sys
import time
import yaml
import logging
import traceback
import subprocess
from datetime import datetime
from typing import List, Dict


# =========================================================
# CONFIGURATION
# =========================================================

CONFIG_PATHS = [
    "configs/lgbm_v2.yaml",
    "configs/xgb_v2.yaml",
    "configs/catboost_v2.yaml"
]

LOG_DIR = "outputs/logs"


# =========================================================
# LOGGING SETUP
# =========================================================

def setup_logger() -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)

    log_file = os.path.join(
        LOG_DIR,
        f"multi_model_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logger = logging.getLogger("orchestrator")
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("=" * 80)
    logger.info("🚀 MULTI-MODEL TRAINING STARTED")
    logger.info("=" * 80)

    return logger


# =========================================================
# UTILITIES
# =========================================================

def load_config(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r") as f:
        return yaml.safe_load(f)


def validate_config(config: Dict, path: str):
    required_keys = ["model", "paths", "experiment"]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"{path} missing required key: {key}")

    if "name" not in config["model"]:
        raise ValueError(f"{path} missing model.name")


def extract_run_info(config: Dict) -> str:
    stage = config["experiment"]["stage"]
    model = config["model"]["name"]
    return f"{stage}/{model}"


# =========================================================
# EXECUTION ENGINE
# =========================================================

def run_single_experiment(config_path: str, logger: logging.Logger) -> Dict:
    start_time = time.time()

    try:
        config = load_config(config_path)
        validate_config(config, config_path)

        run_name = extract_run_info(config)

        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 RUNNING: {run_name}")
        logger.info(f"📄 Config: {config_path}")
        logger.info(f"{'='*60}")

        # Execute training script
        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.training.train_baseline",
                "--config",
                config_path
            ],
            check=True
        )

        runtime = time.time() - start_time

        logger.info(f"✅ SUCCESS: {run_name}")
        logger.info(f"⏱ Runtime: {runtime:.2f} sec")

        return {
            "config": config_path,
            "status": "success",
            "runtime": runtime,
            "error": None
        }

    except Exception as e:
        runtime = time.time() - start_time

        logger.error(f"❌ FAILED: {config_path}")
        logger.error(traceback.format_exc())

        return {
            "config": config_path,
            "status": "failed",
            "runtime": runtime,
            "error": str(e)
        }


# =========================================================
# MAIN ORCHESTRATOR
# =========================================================

def run_pipeline(config_paths: List[str]):
    logger = setup_logger()

    results = []
    total_start = time.time()

    for config_path in config_paths:
        result = run_single_experiment(config_path, logger)
        results.append(result)

    total_runtime = time.time() - total_start

    # =====================================================
    # SUMMARY
    # =====================================================

    logger.info("\n" + "=" * 80)
    logger.info("📊 FINAL SUMMARY")
    logger.info("=" * 80)

    success_count = sum(r["status"] == "success" for r in results)
    failure_count = sum(r["status"] == "failed" for r in results)

    for r in results:
        logger.info(
            f"{r['config']} | {r['status'].upper()} | {r['runtime']:.2f}s"
        )

    logger.info("-" * 80)
    logger.info(f"✅ Success: {success_count}")
    logger.info(f"❌ Failed : {failure_count}")
    logger.info(f"⏱ Total Runtime: {total_runtime:.2f} sec")
    logger.info("=" * 80)

    # Save summary
    summary_path = os.path.join(
        LOG_DIR,
        f"run_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    )

    with open(summary_path, "w") as f:
        yaml.dump(results, f)

    logger.info(f"📁 Summary saved: {summary_path}")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    run_pipeline(CONFIG_PATHS)