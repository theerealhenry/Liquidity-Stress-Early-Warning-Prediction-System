"""
Baseline Training Pipeline (Experiment-Driven)
============================================================

Features:
- Structured experiment tracking
- Model-aware output routing
- Full reproducibility (OOF, y_true, folds, metadata)
- Calibration-ready outputs
- Multi-model compatible

Run:
python -m src.training.train_baseline --config configs/baseline.yaml
"""

import os
import argparse
import yaml
import pandas as pd
import numpy as np
import random
import json
from datetime import datetime
import joblib

# Project imports
from src.features.feature_engineering import build_features, split_features_target
from src.preprocessing.preprocessing import PreprocessingPipeline
from src.training.cv import run_cv, save_cv_outputs


# =========================================================
# UTILS
# =========================================================

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
    """
    Builds structured experiment directory:
    outputs/experiments/{version}/{model}/run_{timestamp}
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    experiment_root = config["paths"]["experiment_root"]
    version = config["project"]["version"]
    model_name = config["model"]["name"]

    run_dir = os.path.join(
        experiment_root,
        version,
        model_name,
        f"run_{timestamp}"
    )

    os.makedirs(run_dir, exist_ok=True)

    return timestamp, run_dir


# =========================================================
# MAIN
# =========================================================

def main(config_path: str):

    print("=" * 60)
    print("📦 LOADING CONFIG")
    print("=" * 60)

    config = load_config(config_path)

    seed = config["project"]["seed"]
    set_seed(seed)

    # -----------------------------------------------------
    # BUILD RUN DIRECTORY
    # -----------------------------------------------------
    run_id, run_dir = build_run_dir(config)

    print(f"\n🧪 RUN ID: {run_id}")
    print(f"📁 Output Dir: {run_dir}")

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------
    print("\n📥 LOADING DATA")
    train_df = load_data(config["data"]["train_path"])

    # -----------------------------------------------------
    # FEATURE ENGINEERING
    # -----------------------------------------------------
    print("\n🧠 RUNNING FEATURE ENGINEERING")
    train_df = build_features(train_df)
    print(f"After feature engineering: {train_df.shape}")

    # -----------------------------------------------------
    # SPLIT TARGET
    # -----------------------------------------------------
    print("\n🎯 SPLITTING FEATURES & TARGET")
    X, y = split_features_target(train_df)

    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts(normalize=True)}")

    # Save target
    np.save(os.path.join(run_dir, "y_true.npy"), y.values)

    # Save feature schema
    save_json(X.columns.tolist(), os.path.join(run_dir, "feature_list.json"))

    # -----------------------------------------------------
    # PREPROCESSING
    # -----------------------------------------------------
    print("\n🧹 RUNNING PREPROCESSING")

    preprocessor = PreprocessingPipeline(
        clip_quantiles=tuple(config["preprocessing"]["clip_quantiles"])
    )

    X_processed = preprocessor.fit_transform(X)

    print(f"Processed shape: {X_processed.shape}")

    preprocessor.save(os.path.join(run_dir, "preprocessor.pkl"))

    # -----------------------------------------------------
    # CROSS VALIDATION
    # -----------------------------------------------------
    print("\n🚀 STARTING MODEL TRAINING")

    results = run_cv(
        X=X_processed,
        y=y,
        config=config,
        return_fold_indices=True,
        return_fold_predictions=True
    )

    # -----------------------------------------------------
    # SAVE CV OUTPUTS
    # -----------------------------------------------------
    print("\n💾 SAVING OUTPUTS")

    save_cv_outputs(
        results=results,
        config=config,
        output_dir=run_dir
    )

    # Save fold indices
    joblib.dump(
        results["fold_indices"],
        os.path.join(run_dir, "fold_indices.pkl")
    )

    # Save fold predictions
    joblib.dump(
        results["fold_predictions"],
        os.path.join(run_dir, "fold_predictions.pkl")
    )

    # -----------------------------------------------------
    # METADATA
    # -----------------------------------------------------
    metadata = {
        "run_id": run_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "version": config["project"]["version"],
        "model": config["model"]["name"],
        "n_samples": int(X.shape[0]),
        "n_features_before": int(X.shape[1]),
        "n_features_after": int(X_processed.shape[1]),
        "target_mean": float(y.mean()),
        "metrics": {
            "logloss": float(results["mean_logloss"]),
            "auc": float(results["mean_auc"]),
            "final_score": float(results["final_score"]),
        }
    }

    save_json(metadata, os.path.join(run_dir, "metadata.json"))

    # -----------------------------------------------------
    # SAVE CONFIG SNAPSHOT
    # -----------------------------------------------------
    with open(os.path.join(run_dir, "config_used.yaml"), "w") as f:
        yaml.dump(config, f)

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE")
    print("=" * 60)

    print(f"Final LogLoss: {results['mean_logloss']:.5f}")
    print(f"Final AUC:     {results['mean_auc']:.5f}")
    print(f"Final Score:   {results['final_score']:.5f}")


# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    args = parser.parse_args()

    main(args.config)