"""
Cross-Validation Engine
========================
Project : Liquidity Stress Early Warning (AI4EAC / Zindi)
Module  : src/training/cv.py

Capabilities
------------
- Stratified K-Fold CV with full reproducibility
- OOF predictions (ensemble-ready format)
- Per-fold + global metrics (LogLoss, ROC-AUC, composite score)
- Model-agnostic: LightGBM / XGBoost / CatBoost via unified factory
- CatBoost categorical feature pass-through
- Full artifact contract aligned to output spec
- Deterministic across runs (seed propagated from config)

Artifact contract
-----------------
run_dir/
    models/<model_name>/    fold-level serialised models
    oof_preds.npy
    y_true.npy
    fold_scores.json
    fold_indices.pkl
    fold_predictions.pkl
    feature_importance.csv
    feature_list.json
    preprocessor.pkl
    metadata.json
    config_used.yaml
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return {
        "logloss": float(log_loss(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_pred)),
    }


def compute_composite_score(
    mean_logloss: float,
    mean_auc: float,
    eval_cfg: Dict[str, Any],
) -> float:
    """
    Weighted composite: 0.6 * LogLoss + 0.4 * (1 - AUC)
    Lower is better. Mirrors the Zindi competition scoring formula.
    """
    w_ll  = eval_cfg["metrics"]["logloss"]["weight"]
    w_auc = eval_cfg["metrics"]["roc_auc"]["weight"]
    return w_ll * mean_logloss + w_auc * (1.0 - mean_auc)


# =============================================================================
# MODEL FACTORY
# =============================================================================

def get_model(model_name: str, params: Dict[str, Any]) -> Any:
    if model_name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(**params)

    elif model_name == "xgboost":
        import xgboost as xgb
        # Remove keys that newer XGBoost versions reject
        clean = {k: v for k, v in params.items() if k != "use_label_encoder"}
        return xgb.XGBClassifier(**clean)

    elif model_name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**params)

    else:
        raise ValueError(
            f"Unsupported model: '{model_name}'. "
            "Supported: lightgbm, xgboost, catboost"
        )


# =============================================================================
# MODEL TRAINING WRAPPER
# Each branch passes only the kwargs that the respective library accepts.
# =============================================================================

def train_model(
    model: Any,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    training_cfg: Dict[str, Any],
    cat_features: Optional[List[str]] = None,
) -> Any:

    es_rounds   = training_cfg["early_stopping_rounds"]
    verbose_n   = training_cfg["verbose_eval"]

    if model_name == "lightgbm":
        import lightgbm as lgb
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric=["binary_logloss", "auc"],
            callbacks=[
                lgb.early_stopping(es_rounds, verbose=False),
                lgb.log_evaluation(verbose_n),
            ],
        )

    elif model_name == "xgboost":
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=verbose_n,
            early_stopping_rounds=es_rounds,
        )

    elif model_name == "catboost":
        
        active_cats: List[str] = []
        if cat_features:
            active_cats = [c for c in cat_features if c in X_train.columns]

        model.fit(
            X_train, y_train,
            eval_set=(X_valid, y_valid),
            cat_features=active_cats if active_cats else None,
            use_best_model=True,
            early_stopping_rounds=es_rounds,
            verbose=verbose_n,
        )

    return model


# =============================================================================
# SAFE PREDICTION
# =============================================================================

def predict_proba(model: Any, model_name: str, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

def get_feature_importance(
    model: Any,
    model_name: str,
    feature_names: List[str],
) -> pd.DataFrame:
    try:
        if model_name in ("lightgbm", "xgboost"):
            importance = model.feature_importances_
        elif model_name == "catboost":
            importance = model.get_feature_importance()
        else:
            importance = np.zeros(len(feature_names))
    except Exception:
        importance = np.zeros(len(feature_names))

    return pd.DataFrame({"feature": feature_names, "importance": importance})


# =============================================================================
# CROSS-VALIDATION ENGINE
# =============================================================================

def run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    config: Dict[str, Any],
) -> Dict[str, Any]:

    seed         = config["project"]["seed"]
    cv_cfg       = config["cv"]
    model_cfg    = config["model"]
    training_cfg = config["training"]
    eval_cfg     = config["evaluation"]

    model_name   = model_cfg["name"]
    model_params = model_cfg["params"]
    cat_features = model_cfg.get("cat_features", None)  # CatBoost only

    skf = StratifiedKFold(
        n_splits=cv_cfg["n_splits"],
        shuffle=cv_cfg["shuffle"],
        random_state=seed,
    )

    oof_preds          = np.zeros(len(X), dtype=np.float32)
    fold_scores:       List[Dict] = []
    models:            List[Any]  = []
    fi_frames:         List[pd.DataFrame] = []
    fold_indices:      List[Dict] = []
    fold_predictions:  List[Dict] = []

    print(f"\n{'='*60}")
    print(f"CV ENGINE | model={model_name.upper()}  folds={cv_cfg['n_splits']}")
    print(f"{'='*60}")

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):

        fold_start = time.perf_counter()
        print(f"\n[Fold {fold}] training...")

        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = get_model(model_name, model_params)
        model = train_model(
            model, model_name,
            X_train, y_train, X_valid, y_valid,
            training_cfg, cat_features,
        )

        preds = predict_proba(model, model_name, X_valid)
        oof_preds[valid_idx] = preds

        metrics = compute_metrics(y_valid.values, preds)
        fold_scores.append({**metrics, "fold": fold})

        fold_time = time.perf_counter() - fold_start
        print(
            f"[Fold {fold}] LogLoss={metrics['logloss']:.5f}  "
            f"AUC={metrics['roc_auc']:.5f}  ({fold_time:.1f}s)"
        )

        models.append(model)

        fi = get_feature_importance(model, model_name, list(X.columns))
        fi["fold"] = fold
        fi_frames.append(fi)

        fold_indices.append({
            "fold":      fold,
            "train_idx": train_idx.tolist(),
            "valid_idx": valid_idx.tolist(),
        })
        fold_predictions.append({
            "fold":      fold,
            "valid_idx": valid_idx.tolist(),
            "preds":     preds.tolist(),
        })

    # -------------------------------------------------------------------------
    # AGGREGATION
    # -------------------------------------------------------------------------
    mean_logloss = float(np.mean([f["logloss"] for f in fold_scores]))
    mean_auc     = float(np.mean([f["roc_auc"] for f in fold_scores]))
    std_logloss  = float(np.std( [f["logloss"] for f in fold_scores]))
    std_auc      = float(np.std( [f["roc_auc"] for f in fold_scores]))

    final_score = compute_composite_score(mean_logloss, mean_auc, eval_cfg)

    fi_agg = (
        pd.concat(fi_frames)
        .groupby("feature")["importance"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"importance": "mean_importance"})
    )

    print(f"\n{'='*60}")
    print(f"CV RESULTS | {model_name.upper()}")
    print(f"{'='*60}")
    print(f"LogLoss : {mean_logloss:.5f}  (+/- {std_logloss:.5f})")
    print(f"AUC     : {mean_auc:.5f}  (+/- {std_auc:.5f})")
    print(f"Score   : {final_score:.5f}  (0.6*LL + 0.4*(1-AUC))")
    print(f"{'='*60}\n")

    return {
        "model_name":         model_name,
        "model_params":       model_params,
        "oof_preds":          oof_preds,
        "y_true":             y.values,
        "models":             models,
        "fold_scores":        fold_scores,
        "mean_logloss":       mean_logloss,
        "mean_auc":           mean_auc,
        "std_logloss":        std_logloss,
        "std_auc":            std_auc,
        "final_score":        final_score,
        "feature_importance": fi_agg,
        "feature_names":      list(X.columns),
        "fold_indices":       fold_indices,
        "fold_predictions":   fold_predictions,
    }


# =============================================================================
# ARTIFACT SAVER
# Aligned to: outputs/experiments/<stage>/<model>/run_YYYYMMDD_HHMMSS/
# =============================================================================

def save_cv_outputs(
    results: Dict[str, Any],
    config:  Dict[str, Any],
    run_dir: str,
) -> None:

    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_name = results["model_name"]

    # -------------------------------------------------------------------------
    # OOF predictions + ground truth
    # -------------------------------------------------------------------------
    np.save(out / "oof_preds.npy",  results["oof_preds"])
    np.save(out / "y_true.npy",     results["y_true"])

    # -------------------------------------------------------------------------
    # Fold-level scores
    # -------------------------------------------------------------------------
    summary = {
        "model":        model_name,
        "mean_logloss": results["mean_logloss"],
        "mean_auc":     results["mean_auc"],
        "std_logloss":  results["std_logloss"],
        "std_auc":      results["std_auc"],
        "final_score":  results["final_score"],
        "fold_scores":  results["fold_scores"],
    }
    with open(out / "fold_scores.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    # -------------------------------------------------------------------------
    # Fold indices + predictions (pickle for numpy-array compatibility)
    # -------------------------------------------------------------------------
    with open(out / "fold_indices.pkl", "wb") as f:
        pickle.dump(results["fold_indices"], f)

    with open(out / "fold_predictions.pkl", "wb") as f:
        pickle.dump(results["fold_predictions"], f)

    # -------------------------------------------------------------------------
    # Feature importance + feature list
    # -------------------------------------------------------------------------
    results["feature_importance"].to_csv(
        out / "feature_importance.csv", index=False
    )
    with open(out / "feature_list.json", "w", encoding="utf-8") as f:
        json.dump(results["feature_names"], f, indent=2)

    # -------------------------------------------------------------------------
    # Serialised models (one pkl per fold)
    # -------------------------------------------------------------------------
    if config.get("artifacts", {}).get("save_models", True):
        import joblib
        model_dir = out / "models" / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        for i, model in enumerate(results["models"]):
            joblib.dump(model, model_dir / f"{model_name}_fold_{i}.pkl")

    # -------------------------------------------------------------------------
    # Preprocessor
    # -------------------------------------------------------------------------
    if "preprocessor" in results:
        import joblib
        joblib.dump(results["preprocessor"], out / "preprocessor.pkl")

    # -------------------------------------------------------------------------
    # Metadata (human-readable run record)
    # -------------------------------------------------------------------------
    metadata = {
        "model":          model_name,
        "stage":          config["experiment"]["stage"],
        "version":        config["experiment"].get("version", ""),
        "n_folds":        config["cv"]["n_splits"],
        "n_features":     len(results["feature_names"]),
        "mean_logloss":   results["mean_logloss"],
        "mean_auc":       results["mean_auc"],
        "final_score":    results["final_score"],
        "seed":           config["project"]["seed"],
        "created_at":     pd.Timestamp.now().isoformat(),
    }
    with open(out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    # -------------------------------------------------------------------------
    # Config snapshot (reproduces this exact run)
    # -------------------------------------------------------------------------
    with open(out / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)

    print(f"Artifacts saved -> {out}")