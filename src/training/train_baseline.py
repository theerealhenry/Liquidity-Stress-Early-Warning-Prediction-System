"""
Baseline Training Pipeline (DEPRECATED)
========================================
Project : Liquidity Stress Early Warning (AI4EAC / Zindi)
Module  : src/training/train_baseline.py

DEPRECATION NOTICE
------------------
This script is SUPERSEDED by src/orchestration/run_all_models.py and is
retained only as a historical record of an earlier architecture.

DO NOT run this script for new experiments. It is out of sync with the
current pipeline in several ways:

  1. It calls run_cv() with kwargs (return_fold_indices, return_fold_predictions)
     that no longer exist in the current cv.py — these are now handled
     internally by run_cv() and always returned.

  2. It calls save_cv_outputs() with `output_dir=run_dir` which no longer
     matches the current function signature (`run_dir` as third positional).

  3. It runs PreprocessingPipeline outside the CV loop, fitting the scaler
     on the full training set before splitting folds.  This is a data-leakage
     risk.  run_all_models.py fits the preprocessor inside each CV fold.

  4. It does not support LogisticRegression or TabNet (added in cv.py v2.0).

  5. It does not read scale_features from config, so scaled models would
     receive unscaled data.

The canonical training command is:
    python -m src.orchestration.run_all_models --configs configs/<model>.yaml

This file will be removed in a future cleanup commit once all experiments
have been migrated to the orchestration layer.
"""

# =============================================================================
# CODE PRESERVED BELOW FOR HISTORICAL REFERENCE ONLY
# =============================================================================

import os
import argparse
import yaml
import pandas as pd
import numpy as np
import random
import json
from datetime import datetime
import joblib

from src.features.feature_engineering import build_features, split_features_target
from src.preprocessing.preprocessing import PreprocessingPipeline
from src.training.cv import run_cv, save_cv_outputs


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_data(path: str):
    df = pd.read_csv(path)
    print(f"Loaded data: {df.shape}")
    return df


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def build_run_dir(config: dict):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_root = config["paths"]["experiment_root"]
    version         = config["project"]["version"]
    model_name      = config["model"]["name"]
    run_dir = os.path.join(experiment_root, version, model_name, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return timestamp, run_dir


def main(config_path: str):
    raise RuntimeError(
        "\n\n"
        "train_baseline.py is DEPRECATED and cannot be run.\n"
        "Use the production orchestrator instead:\n\n"
        "    python -m src.orchestration.run_all_models "
        "--configs configs/<model>.yaml\n\n"
        "See the DEPRECATION NOTICE at the top of this file for details."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)