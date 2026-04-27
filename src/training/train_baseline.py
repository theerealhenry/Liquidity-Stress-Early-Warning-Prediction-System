"""
Baseline Training Pipeline (Production-Grade)
============================================================

Upgrades:
- Saves y_true.npy (critical for calibration)
- Saves fold indices (reproducibility)
- Saves metadata.json (experiment tracking)
- Saves feature_list.json
- Fully reproducible artifacts

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

    output_dir = config["training"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

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

    # Save y_true immediately (CRITICAL)
    y_path = os.path.join(output_dir, "y_true.npy")
    np.save(y_path, y.values)
    print(f"💾 y_true saved: {y_path}")

    # Save feature list
    feature_list_path = os.path.join(output_dir, "feature_list.json")
    save_json(X.columns.tolist(), feature_list_path)

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
    if config["artifacts"]["save_models"]:
        preprocessor_path = os.path.join(output_dir, "preprocessor.pkl")
        preprocessor.save(preprocessor_path)
        print(f"💾 Preprocessor saved: {preprocessor_path}")

    # -----------------------------------------------------
    # CROSS VALIDATION
    # -----------------------------------------------------
    print("\n🚀 STARTING MODEL TRAINING")

    results = run_cv(
        X=X_processed,
        y=y,
        config=config,
        return_fold_indices=True   # 🔥 NEW (you must support this in cv.py)
    )

    # -----------------------------------------------------
    # SAVE CV OUTPUTS
    # -----------------------------------------------------
    print("\n💾 SAVING OUTPUTS")
    save_cv_outputs(results, config)

    # Save fold indices (CRITICAL)
    fold_path = os.path.join(output_dir, "fold_indices.pkl")
    joblib.dump(results["fold_indices"], fold_path)
    print(f"💾 Fold indices saved: {fold_path}")

    # -----------------------------------------------------
    # METADATA (ELITE LEVEL)
    # -----------------------------------------------------
    metadata = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "n_samples": int(X.shape[0]),
        "n_features_before": int(X.shape[1]),
        "n_features_after": int(X_processed.shape[1]),
        "target_mean": float(y.mean()),
        "cv_folds": config["training"]["n_folds"],
        "model_type": config["model"]["name"],
        "metrics": {
            "logloss": float(results["mean_logloss"]),
            "auc": float(results["mean_auc"]),
            "final_score": float(results["final_score"]),
        }
    }

    metadata_path = os.path.join(output_dir, "metadata.json")
    save_json(metadata, metadata_path)
    print(f"📊 Metadata saved: {metadata_path}")

    # -----------------------------------------------------
    # SAVE CONFIG COPY
    # -----------------------------------------------------
    if config["artifacts"]["save_config_copy"]:
        config_path_out = os.path.join(output_dir, "config_used.yaml")
        with open(config_path_out, "w") as f:
            yaml.dump(config, f)
        print(f"📄 Config saved: {config_path_out}")

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