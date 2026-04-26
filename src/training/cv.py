"""
Cross-Validation Engine
=========================================

Capabilities:
- Stratified K-Fold CV
- OOF predictions (ensemble-ready)
- Fold-level + global metrics (LogLoss, ROC-AUC)
- Model training (LightGBM baseline)
- Feature importance aggregation
- Model persistence
- Fully reproducible

Design Principles:
- CV-safe (no leakage)
- Deterministic
- Modular (easy to extend to XGBoost / CatBoost)
- Config-driven
"""

import os
import json
import numpy as np
import pandas as pd

from typing import Dict, Any
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score

import lightgbm as lgb


# =========================================================
# METRICS
# =========================================================

def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return {
        "logloss": log_loss(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_pred)
    }


# =========================================================
# MODEL FACTORY
# =========================================================

def get_model(model_name: str, params: Dict[str, Any]):
    if model_name == "lightgbm":
        return lgb.LGBMClassifier(**params)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


# =========================================================
# CROSS VALIDATION
# =========================================================

def run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    config: Dict[str, Any],
) -> Dict[str, Any]:

    seed = config["project"]["seed"]
    cv_cfg = config["cv"]
    model_cfg = config["model"]
    training_cfg = config["training"]
    eval_cfg = config["evaluation"]

    n_splits = cv_cfg["n_splits"]

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=cv_cfg["shuffle"],
        random_state=seed
    )

    # OOF predictions
    oof_preds = np.zeros(len(X))

    # Storage
    fold_scores = []
    models = []
    feature_importance = []

    print("=" * 60)
    print("🚀 STARTING CROSS-VALIDATION")
    print("=" * 60)

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        print(f"\n🔹 Fold {fold}")

        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = get_model(model_cfg["name"], model_cfg["params"])

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric=["binary_logloss", "auc"],
            callbacks=[
                lgb.early_stopping(training_cfg["early_stopping_rounds"]),
                lgb.log_evaluation(training_cfg["verbose_eval"])
            ]
        )

        # Predict probabilities
        preds = model.predict_proba(X_valid)[:, 1]
        oof_preds[valid_idx] = preds

        # Metrics
        metrics = compute_metrics(y_valid, preds)
        fold_scores.append(metrics)

        print(f"Fold {fold} → LogLoss: {metrics['logloss']:.5f} | AUC: {metrics['roc_auc']:.5f}")

        # Store model
        models.append(model)

        # Feature importance
        fold_importance = pd.DataFrame({
            "feature": X.columns,
            "importance": model.feature_importances_,
            "fold": fold
        })

        feature_importance.append(fold_importance)

    print("\n" + "=" * 60)
    print("📊 CV RESULTS")
    print("=" * 60)

    # Aggregate metrics
    mean_logloss = np.mean([f["logloss"] for f in fold_scores])
    mean_auc = np.mean([f["roc_auc"] for f in fold_scores])

    print(f"Mean LogLoss: {mean_logloss:.5f}")
    print(f"Mean AUC:     {mean_auc:.5f}")

    # Weighted score (competition metric)
    final_score = (
        eval_cfg["metrics"]["logloss"]["weight"] * mean_logloss +
        eval_cfg["metrics"]["roc_auc"]["weight"] * (1 - mean_auc)
    )

    print(f"Final Weighted Score: {final_score:.5f}")

    # Aggregate feature importance
    feature_importance_df = pd.concat(feature_importance)
    feature_importance_df = (
        feature_importance_df
        .groupby("feature")["importance"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    return {
        "oof_preds": oof_preds,
        "models": models,
        "fold_scores": fold_scores,
        "mean_logloss": mean_logloss,
        "mean_auc": mean_auc,
        "final_score": final_score,
        "feature_importance": feature_importance_df
    }


# =========================================================
# SAVE ARTIFACTS
# =========================================================

def save_cv_outputs(results: Dict[str, Any], config: Dict[str, Any]):
    output_dir = config["training"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Save OOF
    np.save(os.path.join(output_dir, "oof_preds.npy"), results["oof_preds"])

    # Save fold scores
    with open(os.path.join(output_dir, "fold_scores.json"), "w") as f:
        json.dump(results["fold_scores"], f, indent=4)

    # Save feature importance
    results["feature_importance"].to_csv(
        os.path.join(output_dir, "feature_importance.csv"),
        index=False
    )

    # Save models
    if config["training"]["save_model"]:
        import joblib
        model_dir = os.path.join(output_dir, "models")
        os.makedirs(model_dir, exist_ok=True)

        for i, model in enumerate(results["models"]):
            joblib.dump(model, os.path.join(model_dir, f"model_fold_{i}.pkl"))

    print(f"\n💾 Outputs saved to: {output_dir}")