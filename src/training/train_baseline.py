"""
Baseline Training Pipeline
============================================

Responsibilities:
- Load configuration (YAML)
- Load data
- Run feature engineering
- Run preprocessing
- Execute cross-validation
- Save outputs (OOF, models, metrics, importance)
- Ensure full reproducibility

Run:
python src/training/train_baseline.py --config configs/baseline.yaml
"""

import os
import argparse
import yaml
import pandas as pd
import numpy as np
import random

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


# =========================================================
# MAIN
# =========================================================

def main(config_path: str):

    print("=" * 60)
    print("📦 LOADING CONFIG")
    print("=" * 60)

    config = load_config(config_path)

    set_seed(config["project"]["seed"])

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

    # -----------------------------------------------------
    # PREPROCESSING
    # -----------------------------------------------------
    print("\n🧹 RUNNING PREPROCESSING")

    preprocessor = PreprocessingPipeline(
        clip_quantiles=tuple(config["preprocessing"]["clip_quantiles"])
    )

    X_processed = preprocessor.fit_transform(X)

    print(f"Processed shape: {X_processed.shape}")

    # Save preprocessor for inference
    if config["artifacts"]["save_models"]:
        os.makedirs(config["training"]["output_dir"], exist_ok=True)
        preprocessor_path = os.path.join(
            config["training"]["output_dir"], "preprocessor.pkl"
        )
        preprocessor.save(preprocessor_path)
        print(f"💾 Preprocessor saved: {preprocessor_path}")

    # -----------------------------------------------------
    # CROSS VALIDATION
    # -----------------------------------------------------
    print("\n🚀 STARTING MODEL TRAINING")

    results = run_cv(
        X=X_processed,
        y=y,
        config=config
    )

    # -----------------------------------------------------
    # SAVE OUTPUTS
    # -----------------------------------------------------
    print("\n💾 SAVING OUTPUTS")
    save_cv_outputs(results, config)

    # Save config copy for reproducibility
    if config["artifacts"]["save_config_copy"]:
        config_path_out = os.path.join(
            config["training"]["output_dir"], "config_used.yaml"
        )
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
        help="D:/PROJECTS/liquidity-stress-early-warning/configs/baseline.yaml"
    )

    args = parser.parse_args()

    main(args.config)