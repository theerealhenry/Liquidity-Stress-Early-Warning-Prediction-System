"""
Baseline Training Pipeline (Elite Production-Grade)
============================================================

Enhancements:
- Full reproducibility (run_id, metadata, config snapshot)
- Calibration-ready outputs (OOF + y_true + fold indices)
- Fold-wise predictions storage
- Feature schema tracking
- Robust artifact structure
- Future-ready for ensemble + multi-model

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


def create_run_dir(base_dir: str):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    return run_id, run_dir


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
    # CREATE RUN DIRECTORY (EXPERIMENT TRACKING)
    # -----------------------------------------------------
    base_output_dir = config["training"]["output_dir"]
    run_id, run_dir = create_run_dir(base_output_dir)

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

    # Save raw target (critical for calibration)
    np.save(os.path.join(run_dir, "y_true.npy"), y.values)

    # Save feature schema
    feature_list = X.columns.tolist()
    save_json(feature_list, os.path.join(run_dir, "feature_list.json"))

    # -----------------------------------------------------
    # PREPROCESSING
    # -----------------------------------------------------
    print("\n🧹 RUNNING PREPROCESSING")

    preprocessor = PreprocessingPipeline(
        clip_quantiles=tuple(config["preprocessing"]["clip_quantiles"])
    )

    X_processed = preprocessor.fit_transform(X)

    print(f"Processed shape: {X_processed.shape}")

    # Save preprocessor
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
        return_fold_predictions=True   # 👈 NEW
    )

    # -----------------------------------------------------
    # SAVE CV OUTPUTS
    # -----------------------------------------------------
    print("\n💾 SAVING OUTPUTS")

    save_cv_outputs(results, config, output_dir=run_dir)

    # Save fold indices
    joblib.dump(
        results["fold_indices"],
        os.path.join(run_dir, "fold_indices.pkl")
    )

    # Save fold predictions (critical for deep error analysis)
    joblib.dump(
        results["fold_predictions"],
        os.path.join(run_dir, "fold_predictions.pkl")
    )

    # -----------------------------------------------------
    # METADATA (EXPERIMENT TRACKING)
    # -----------------------------------------------------
    metadata = {
        "run_id": run_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "n_samples": int(X.shape[0]),
        "n_features_before": int(X.shape[1]),
        "n_features_after": int(X_processed.shape[1]),
        "target_mean": float(y.mean()),
        "model_type": config["model"]["name"],
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
# CLI ENTRY
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML"
    )

    args = parser.parse_args()

    main(args.config)