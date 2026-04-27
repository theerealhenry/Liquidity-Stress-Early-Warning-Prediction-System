"""
Advanced Cross-Validation Engine (Model-Agnosti)
==================================================================

Capabilities:
- Stratified K-Fold CV
- OOF predictions (ensemble-ready)
- Fold-level + global metrics (LogLoss, ROC-AUC)
- Supports LightGBM, XGBoost, CatBoost
- Feature importance aggregation
- Model persistence
- Reproducibility artifacts (fold indices, fold predictions)

Design Principles:
- Fully model-agnostic
- CV-safe (no leakage)
- Deterministic
- Config-driven
- Ensemble-ready
"""

import os
import json
import numpy as np
import pandas as pd

from typing import Dict, Any
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score

# Models
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier


# =========================================================
# METRICS
# =========================================================

def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return {
        "logloss": float(log_loss(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_pred))
    }


# =========================================================
# MODEL FACTORY
# =========================================================

def get_model(model_name: str, params: Dict[str, Any]):
    if model_name == "lightgbm":
        return lgb.LGBMClassifier(**params)

    elif model_name == "xgboost":
        return xgb.XGBClassifier(**params)

    elif model_name == "catboost":
        return CatBoostClassifier(**params, verbose=0)

    else:
        raise ValueError(f"Unsupported model: {model_name}")


# =========================================================
# TRAINING WRAPPER
# =========================================================

def train_model(model, model_name, X_train, y_train, X_valid, y_valid, training_cfg):

    if model_name == "lightgbm":
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

    elif model_name == "xgboost":
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=training_cfg["verbose_eval"],
            early_stopping_rounds=training_cfg["early_stopping_rounds"]
        )

    elif model_name == "catboost":
        model.fit(
            X_train,
            y_train,
            eval_set=(X_valid, y_valid),
            use_best_model=True,
            early_stopping_rounds=training_cfg["early_stopping_rounds"],
            verbose=training_cfg["verbose_eval"]
        )

    return model


# =========================================================
# PREDICTION SAFE HANDLER
# =========================================================

def predict_proba_safe(model, model_name, X):
    if model_name in ["lightgbm", "xgboost", "catboost"]:
        return model.predict_proba(X)[:, 1]
    else:
        raise ValueError(f"Unsupported model: {model_name}")


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

def get_feature_importance(model, model_name, feature_names):

    if model_name in ["lightgbm", "xgboost"]:
        importance = model.feature_importances_

    elif model_name == "catboost":
        importance = model.get_feature_importance()

    else:
        importance = np.zeros(len(feature_names))

    return pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    })


# =========================================================
# CROSS VALIDATION
# =========================================================

def run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    config: Dict[str, Any],
    return_fold_indices: bool = False,
    return_fold_predictions: bool = False
) -> Dict[str, Any]:

    seed = config["project"]["seed"]
    cv_cfg = config["cv"]
    model_cfg = config["model"]
    training_cfg = config["training"]
    eval_cfg = config["evaluation"]

    model_name = model_cfg["name"]
    model_params = model_cfg["params"]

    skf = StratifiedKFold(
        n_splits=cv_cfg["n_splits"],
        shuffle=cv_cfg["shuffle"],
        random_state=seed
    )

    oof_preds = np.zeros(len(X))
    fold_scores = []
    models = []
    feature_importance = []

    fold_indices = []
    fold_predictions = []

    print("=" * 60)
    print(f"🚀 STARTING CV | MODEL: {model_name.upper()}")
    print("=" * 60)

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):

        print(f"\n🔹 Fold {fold}")

        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        if return_fold_indices:
            fold_indices.append({
                "fold": fold,
                "train_idx": train_idx.tolist(),
                "valid_idx": valid_idx.tolist()
            })

        model = get_model(model_name, model_params)

        model = train_model(
            model,
            model_name,
            X_train,
            y_train,
            X_valid,
            y_valid,
            training_cfg
        )

        preds = predict_proba_safe(model, model_name, X_valid)
        oof_preds[valid_idx] = preds

        if return_fold_predictions:
            fold_predictions.append({
                "fold": fold,
                "valid_idx": valid_idx.tolist(),
                "preds": preds.tolist()
            })

        metrics = compute_metrics(y_valid, preds)
        fold_scores.append(metrics)

        print(f"Fold {fold} → LogLoss: {metrics['logloss']:.5f} | AUC: {metrics['roc_auc']:.5f}")

        models.append(model)

        fi = get_feature_importance(model, model_name, X.columns)
        fi["fold"] = fold
        feature_importance.append(fi)

    # =====================================================
    # AGGREGATION
    # =====================================================

    mean_logloss = np.mean([f["logloss"] for f in fold_scores])
    mean_auc = np.mean([f["roc_auc"] for f in fold_scores])

    final_score = (
        eval_cfg["metrics"]["logloss"]["weight"] * mean_logloss +
        eval_cfg["metrics"]["roc_auc"]["weight"] * (1 - mean_auc)
    )

    feature_importance_df = (
        pd.concat(feature_importance)
        .groupby("feature")["importance"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    print("\n" + "=" * 60)
    print("📊 CV RESULTS")
    print("=" * 60)
    print(f"Mean LogLoss: {mean_logloss:.5f}")
    print(f"Mean AUC:     {mean_auc:.5f}")
    print(f"Final Score:  {final_score:.5f}")

    results = {
        "oof_preds": oof_preds,
        "models": models,
        "fold_scores": fold_scores,
        "mean_logloss": mean_logloss,
        "mean_auc": mean_auc,
        "final_score": final_score,
        "feature_importance": feature_importance_df,
        "model_name": model_name
    }

    if return_fold_indices:
        results["fold_indices"] = fold_indices

    if return_fold_predictions:
        results["fold_predictions"] = fold_predictions

    return results


# =========================================================
# SAVE ARTIFACTS
# =========================================================

def save_cv_outputs(
    results: Dict[str, Any],
    config: Dict[str, Any],
    output_dir: str
):

    os.makedirs(output_dir, exist_ok=True)

    model_name = results["model_name"]

    # OOF
    np.save(os.path.join(output_dir, f"oof_preds_{model_name}.npy"), results["oof_preds"])

    # Fold metrics
    with open(os.path.join(output_dir, f"fold_scores_{model_name}.json"), "w") as f:
        json.dump(results["fold_scores"], f, indent=4)

    # Feature importance
    results["feature_importance"].to_csv(
        os.path.join(output_dir, f"feature_importance_{model_name}.csv"),
        index=False
    )

    # Fold indices
    if "fold_indices" in results:
        with open(os.path.join(output_dir, f"fold_indices_{model_name}.json"), "w") as f:
            json.dump(results["fold_indices"], f)

    # Fold predictions
    if "fold_predictions" in results:
        with open(os.path.join(output_dir, f"fold_predictions_{model_name}.json"), "w") as f:
            json.dump(results["fold_predictions"], f)

    # Models
    if config.get("artifacts", {}).get("save_models", True):
        import joblib

        model_dir = os.path.join(output_dir, "models", model_name)
        os.makedirs(model_dir, exist_ok=True)

        for i, model in enumerate(results["models"]):
            joblib.dump(model, os.path.join(model_dir, f"{model_name}_fold_{i}.pkl"))

    print(f"\n💾 Outputs saved to: {output_dir}")