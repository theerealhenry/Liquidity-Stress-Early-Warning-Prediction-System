# src/inference/predict.py
"""
Inference Pipeline — Final Submission Generation
=================================================
Project : AI4EAC Liquidity Stress Early Warning (Zindi Africa)
Module  : src/inference/predict.py

Produces a Zindi-compliant submission CSV from the trained ensemble.

Output format (required):
    ID,Target,TargetLogLoss,TargetRAUC
    ID_XYZ,0.45,0.45,0.45

Usage
-----
    python -m src.inference.predict \
        --ensemble-run outputs/experiments/v4_ensemble/run_YYYYMMDD_HHMMSS \
        --stage v2_feature_expansion \
        --output submissions/submission_v5_tuned_ensemble.csv
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# =============================================================================
# FOLD PREDICTION AVERAGING
# =============================================================================

def predict_test_per_model(
    test_df: pd.DataFrame,
    stage: str,
    model_name: str,
    config_path: str,
) -> np.ndarray:
    """
    Load all fold models for a given stage + model_name.
    Apply feature engineering + preprocessing, then average fold predictions.

    Returns raw (uncalibrated) test predictions averaged across folds.
    """
    import yaml
    from src.data.load_data import load_data
    from src.features.feature_engineering import build_features, split_features_target
    from src.preprocessing.preprocessing import PreprocessingPipeline

    with open(PROJECT_ROOT / config_path) as f:
        cfg = yaml.safe_load(f)

    # Feature engineering on test set (same pipeline as train)
    test_fe = build_features(test_df.copy())
    X_test, _ = split_features_target(test_fe)

    # Find the most recent run directory for this stage/model
    base_dir = (
        PROJECT_ROOT / "outputs" / "experiments" / stage / model_name
    )
    run_dirs = sorted(base_dir.glob("run_*"), reverse=True)
    if not run_dirs:
        raise FileNotFoundError(
            f"No run directories found at {base_dir}. "
            "Run training first via run_all_models.py"
        )
    run_dir = run_dirs[0]
    print(f"  Using run: {run_dir.name}")

    # Load preprocessor fitted during training
    preproc = PreprocessingPipeline.load(str(run_dir / "preprocessor.pkl"))
    X_test_proc = preproc.transform(X_test)

    # Load fold models and average predictions
    model_dir = run_dir / "models" / model_name
    fold_paths = sorted(model_dir.glob(f"{model_name}_fold_*.pkl"))
    if not fold_paths:
        raise FileNotFoundError(f"No fold models found at {model_dir}")

    fold_preds = []
    for fold_path in fold_paths:
        model = joblib.load(fold_path)
        pred = model.predict_proba(X_test_proc)[:, 1]
        fold_preds.append(pred)
        print(f"    Fold {len(fold_preds)-1}: pred range "
              f"[{pred.min():.4f}, {pred.max():.4f}]")

    raw_preds = np.mean(fold_preds, axis=0)
    print(f"  {model_name} raw avg: [{raw_preds.min():.4f}, {raw_preds.max():.4f}]")
    return raw_preds


# =============================================================================
# PLATT CALIBRATION APPLICATION
# =============================================================================

def calibrate_test_predictions(
    raw_preds: np.ndarray,
    model_name: str,
) -> np.ndarray:
    """Apply the Platt calibrator fitted on OOF predictions."""
    cal_path = (
        PROJECT_ROOT / "outputs" / "calibration"
        / model_name[:3]  # lgb / xgb / cat
        / "calibrator_platt.pkl"
    )
    if not cal_path.exists():
        print(f"  WARNING: No Platt calibrator found for {model_name} at {cal_path}")
        print(f"  Using uncalibrated predictions.")
        return raw_preds

    calibrator = joblib.load(cal_path)
    cal_preds = calibrator.predict(raw_preds)
    print(f"  {model_name} calibrated: [{cal_preds.min():.4f}, {cal_preds.max():.4f}]")
    return cal_preds


# =============================================================================
# ENSEMBLE INFERENCE
# =============================================================================

def ensemble_predict(
    calibrated_preds: Dict[str, np.ndarray],
    ensemble_run_dir: str,
) -> np.ndarray:
    """
    Apply the trained ensemble (meta-model + final calibrator) to
    calibrated base model predictions.

    This mirrors exactly what EnsembleInference.predict() does.
    """
    from src.ensemble.ensemble import EnsembleInference, EnsembleConfig

    run_dir = Path(ensemble_run_dir)

    # Load weights to verify which strategy was best
    weights_path = run_dir / "ensemble_weights.json"
    if weights_path.exists():
        with open(weights_path) as f:
            weights = json.load(f)
        print(f"\n  Optimised weights: {weights.get('optimised', {})}")

    # Load meta-model and calibrator
    meta_model = joblib.load(run_dir / "meta_model.pkl")
    meta_calibrator = None
    cal_path = run_dir / "meta_calibrator.pkl"
    if cal_path.exists():
        meta_calibrator = joblib.load(cal_path)
        print("  Meta-calibrator: loaded")

    # Build meta-features (OOF preds + disagreement)
    MODEL_NAMES = ["lightgbm", "xgboost", "catboost"]
    meta_X = np.stack([calibrated_preds[m] for m in MODEL_NAMES], axis=1)

    # Add disagreement feature (if used during training)
    disagreement = meta_X.std(axis=1, keepdims=True)
    meta_X = np.hstack([meta_X, disagreement])

    # Meta-model prediction
    ensemble_pred = meta_model.predict_proba(meta_X)[:, 1]

    # Final calibration
    if meta_calibrator is not None:
        ensemble_pred = meta_calibrator.predict(ensemble_pred)

    ensemble_pred = np.clip(ensemble_pred, 1e-6, 1 - 1e-6)
    print(f"\n  Final ensemble: [{ensemble_pred.min():.4f}, {ensemble_pred.max():.4f}]")
    print(f"  Mean pred: {ensemble_pred.mean():.4f} (train positive rate ~0.15)")
    return ensemble_pred


# =============================================================================
# SUBMISSION BUILDER
# =============================================================================

def build_submission(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    output_path: str,
) -> pd.DataFrame:
    """
    Build Zindi-compliant submission CSV.

    Required format:
        ID, Target, TargetLogLoss, TargetRAUC
        (TargetLogLoss and TargetRAUC must be identical to Target)
    """
    submission = pd.DataFrame({
        "ID"           : test_df["ID"].values,
        "Target"       : predictions,
        "TargetLogLoss": predictions,   # competition requires duplicated columns
        "TargetRAUC"   : predictions,
    })

    # Sanity checks
    assert not submission["Target"].isna().any(), "NaN predictions detected"
    assert (submission["Target"] >= 0).all() and (submission["Target"] <= 1).all(), \
        "Predictions outside [0, 1]"
    assert len(submission) == len(test_df), "Row count mismatch"

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out, index=False)

    print(f"\n{'='*60}")
    print(f"SUBMISSION SAVED: {out}")
    print(f"{'='*60}")
    print(f"Rows          : {len(submission):,}")
    print(f"Mean pred     : {predictions.mean():.4f}")
    print(f"Std pred      : {predictions.std():.4f}")
    print(f"Pred > 0.5    : {(predictions > 0.5).sum():,} ({(predictions > 0.5).mean()*100:.1f}%)")
    print(f"{'='*60}")

    return submission


# =============================================================================
# SANITY CHECK: compare OOF vs test distribution
# =============================================================================

def check_prediction_drift(
    oof_path: str,
    test_preds: np.ndarray,
) -> None:
    """
    Compare OOF prediction distribution vs test prediction distribution.
    Large drift suggests the test set is OOD or there is a pipeline bug.
    Threshold: warn if mean differs by > 0.05.
    """
    oof_preds = np.load(oof_path)
    oof_mean  = oof_preds.mean()
    test_mean = test_preds.mean()
    diff      = abs(oof_mean - test_mean)

    print(f"\n  Distribution check:")
    print(f"    OOF  mean: {oof_mean:.4f}")
    print(f"    Test mean: {test_mean:.4f}")
    print(f"    Difference: {diff:.4f}")

    if diff > 0.05:
        print("  ⚠ WARNING: Large distribution shift. Check for pipeline inconsistency.")
    else:
        print("  ✔ Distribution consistent.")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_inference(
    ensemble_run_dir: str,
    stage: str,
    output_path: str,
) -> pd.DataFrame:

    print("=" * 60)
    print("INFERENCE PIPELINE")
    print("=" * 60)

    # ── Load test data ────────────────────────────────────────────────
    print("\n[1/5] Loading test data...")
    test_df = pd.read_csv(PROJECT_ROOT / "data/raw/Test.csv")
    test_df.columns = [c.strip() for c in test_df.columns]
    print(f"  Test shape: {test_df.shape}")

    # ── Per-model predictions ─────────────────────────────────────────
    model_configs = {
        "lightgbm": (stage, "configs/lightgbm_tuned.yaml"),
        "xgboost" : (stage, "configs/xgboost_tuned.yaml"),
        "catboost" : ("baseline", "configs/catboost_v2.yaml"),
        # CatBoost not tuned — uses baseline run
    }

    raw_preds: Dict[str, np.ndarray] = {}
    calibrated_preds: Dict[str, np.ndarray] = {}

    print("\n[2/5] Generating per-model test predictions...")
    for model_name, (model_stage, config_path) in model_configs.items():
        print(f"\n  {model_name.upper()}:")
        try:
            raw = predict_test_per_model(test_df, model_stage, model_name, config_path)
            raw_preds[model_name] = raw
        except FileNotFoundError:
            # Fall back to baseline stage for untuned models
            raw = predict_test_per_model(test_df, "baseline", model_name, config_path)
            raw_preds[model_name] = raw

    # ── Calibrate ─────────────────────────────────────────────────────
    print("\n[3/5] Applying Platt calibration...")
    for model_name, raw in raw_preds.items():
        calibrated_preds[model_name] = calibrate_test_predictions(raw, model_name)

    # ── Ensemble ──────────────────────────────────────────────────────
    print("\n[4/5] Applying ensemble meta-model...")
    final_preds = ensemble_predict(calibrated_preds, ensemble_run_dir)

    # ── Distribution sanity check ─────────────────────────────────────
    oof_path = Path(ensemble_run_dir) / "ensemble_oof.npy"
    if oof_path.exists():
        check_prediction_drift(str(oof_path), final_preds)

    # ── Build submission ──────────────────────────────────────────────
    print("\n[5/5] Building submission...")
    submission = build_submission(test_df, final_preds, output_path)

    return submission


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference pipeline — Liquidity Stress Early Warning"
    )
    parser.add_argument(
        "--ensemble-run",
        required=True,
        help="Path to ensemble run directory (e.g. outputs/experiments/v4_ensemble/run_20260430_074450)",
    )
    parser.add_argument(
        "--stage",
        default="v2_feature_expansion",
        help="Experiment stage for tuned models (default: v2_feature_expansion)",
    )
    parser.add_argument(
        "--output",
        default=f"submissions/submission_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        help="Output path for submission CSV",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_inference(
        ensemble_run_dir=args.ensemble_run,
        stage=args.stage,
        output_path=args.output,
    )