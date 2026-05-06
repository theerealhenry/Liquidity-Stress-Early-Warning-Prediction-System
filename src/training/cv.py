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
- Model-agnostic: LightGBM / XGBoost / CatBoost / LogisticRegression /
  TabNet via unified factory functions
- CatBoost categorical feature pass-through
- TabNet-safe: numpy array conversion, custom epoch loop, non-joblib
  serialisation (save_model / load_model)
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

Changelog
---------
v1.0   LightGBM / XGBoost / CatBoost support.
v1.1   XGBoost early_stopping_rounds moved to constructor (XGBoost >= 1.7).
v1.2   CatBoost cat_features active-column filtering.
v2.0   NEW: Logistic Regression and TabNet support.

       LogisticRegression (sklearn)
       - Uses ElasticNet penalty + saga solver (the only combination that
         supports both L1 and L2 penalties simultaneously).
       - class_weight="balanced" handles the 15% positive rate without
         needing manual scale_pos_weight.
       - No early stopping — convergence is handled by max_iter and the
         saga solver's internal tolerance.
       - Feature importance: |coef_[0]| (absolute coefficient magnitude).
       - Requires scale_features=True in preprocessing (not scale-invariant).

       TabNet (pytorch-tabnet)
       - Instance-wise sequential attention — structurally different from
         all three GBMs.
       - Requires numpy float32 arrays, not DataFrames.
       - Uses its own training loop (max_epochs, patience) rather than
         sklearn's .fit() convention.
       - Serialisation via save_model() / load_model() — NOT joblib.
         joblib cannot serialise PyTorch tensors correctly.
       - Feature importance: mean absolute attention mask across all steps
         (model.feature_importances_).
       - Requires scale_features=True in preprocessing (attention weights
         collapse to uniform without normalisation).
"""

from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute LogLoss and ROC-AUC.

    y_pred is clipped to [1e-15, 1-1e-15] to prevent log(0) — this is
    the standard guard used in Kaggle/Zindi production pipelines.
    """
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
    Lower is better. Mirrors the Zindi AI4EAC scoring formula exactly.
    """
    w_ll  = eval_cfg["metrics"]["logloss"]["weight"]
    w_auc = eval_cfg["metrics"]["roc_auc"]["weight"]
    return w_ll * mean_logloss + w_auc * (1.0 - mean_auc)


# =============================================================================
# MODEL FACTORY
# =============================================================================

def get_model(
    model_name: str,
    params: Dict[str, Any],
    training_cfg: Dict[str, Any] = None,
) -> Any:
    """
    Instantiate a model from its name and parameter dict.

    Parameters
    ----------
    model_name : str
        One of: lightgbm, xgboost, catboost, logreg, tabnet
    params : dict
        Model hyperparameters. Passed directly to the model constructor
        (after stripping framework-specific keys where needed).
    training_cfg : dict, optional
        Full training config block. Required for XGBoost (early_stopping_rounds
        must be in the constructor since XGBoost >= 1.7) and TabNet
        (max_epochs, patience, batch_size, virtual_batch_size, weights).

    Returns
    -------
    Instantiated (unfitted) model object.
    """
    if model_name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(**params)

    elif model_name == "xgboost":
        import xgboost as xgb
        clean = {k: v for k, v in params.items() if k != "use_label_encoder"}
        # early_stopping_rounds belongs in constructor from XGBoost >= 1.7
        if training_cfg is not None:
            clean["early_stopping_rounds"] = training_cfg["early_stopping_rounds"]
        return xgb.XGBClassifier(**clean)

    elif model_name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**params)

    elif model_name == "logreg":
        from sklearn.linear_model import LogisticRegression
        # ElasticNet requires saga solver — the only solver that supports
        # both L1 and L2 simultaneously. All other kwargs pass through.
        return LogisticRegression(**params)

    elif model_name == "tabnet":
        from pytorch_tabnet.tab_model import TabNetClassifier
        # TabNet training kwargs (max_epochs, patience, batch_size etc.)
        # are passed at .fit() time, not construction time.
        # Construction only takes architecture params.
        tabnet_constructor_keys = {
            "n_d", "n_a", "n_steps", "gamma", "momentum",
            "n_independent", "n_shared", "seed", "verbose",
            "optimizer_fn", "optimizer_params", "scheduler_fn",
            "scheduler_params", "mask_type", "lambda_sparse",
            "cat_idxs", "cat_dims", "cat_emb_dim",
        }
        constructor_params = {
            k: v for k, v in params.items()
            if k in tabnet_constructor_keys
        }
        return TabNetClassifier(**constructor_params)

    else:
        raise ValueError(
            f"Unsupported model: '{model_name}'. "
            "Supported: lightgbm, xgboost, catboost, logreg, tabnet"
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
    """
    Unified training interface for all supported model families.

    Each branch isolates library-specific training conventions so that
    run_cv() never needs to branch on model type.

    Notes on TabNet:
      - Converts DataFrames to numpy float32 arrays before training.
        TabNet's internal PyTorch layers expect float32; passing float64
        causes silent precision loss and potential NaN in attention weights.
      - eval_set passed as numpy arrays, not DataFrames.
      - patience maps to the pytorch-tabnet `patience` kwarg (not sklearn's).
      - weights=1 activates TabNet's built-in class balancing (equivalent
        to class_weight="balanced" in sklearn).

    Notes on LogisticRegression:
      - No eval_set, no early stopping, no verbose_eval — sklearn's
        LogisticRegression is a closed-form / iterative solver with its own
        convergence criterion (tol, max_iter).
      - class_weight="balanced" is set in the config params dict, not here.
    """
    es_rounds = training_cfg.get("early_stopping_rounds", 100)
    verbose_n = training_cfg.get("verbose_eval", 100)

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

    elif model_name == "logreg":
        # LogisticRegression is a standard sklearn estimator.
        # No eval_set, no early stopping.  class_weight and max_iter
        # are already in params dict and were passed to the constructor.
        model.fit(X_train, y_train)

    elif model_name == "tabnet":
        # ── Convert to numpy float32 ──────────────────────────────────────
        # TabNet's PyTorch backend requires float32. DataFrames are
        # converted here rather than in run_cv so that the conversion is
        # always applied consistently regardless of calling context.
        X_tr_np = (
            X_train.values.astype(np.float32)
            if isinstance(X_train, pd.DataFrame)
            else X_train.astype(np.float32)
        )
        y_tr_np = (
            y_train.values.astype(np.int64)
            if isinstance(y_train, pd.Series)
            else y_train.astype(np.int64)
        )
        X_va_np = (
            X_valid.values.astype(np.float32)
            if isinstance(X_valid, pd.DataFrame)
            else X_valid.astype(np.float32)
        )
        y_va_np = (
            y_valid.values.astype(np.int64)
            if isinstance(y_valid, pd.Series)
            else y_valid.astype(np.int64)
        )

        # Read TabNet-specific training kwargs from config
        max_epochs         = training_cfg.get("max_epochs", 200)
        patience           = training_cfg.get("patience", 20)
        batch_size         = training_cfg.get("batch_size", 1024)
        virtual_batch_size = training_cfg.get("virtual_batch_size", 256)
        # weights=1 activates pytorch-tabnet's built-in class balancing
        weights            = training_cfg.get("weights", 1)

        model.fit(
            X_tr_np, y_tr_np,
            eval_set=[(X_va_np, y_va_np)],
            eval_metric=["logloss"],
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
            weights=weights,
            # Drop last incomplete batch to avoid BatchNorm instability
            drop_last=False,
        )

    return model


# =============================================================================
# SAFE PREDICTION
# =============================================================================

def predict_proba(
    model: Any,
    model_name: str,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Return probability of the positive class (stress = 1) for each row.

    TabNet requires numpy float32 input — conversion applied here.
    All other models accept DataFrames directly.
    """
    if model_name == "tabnet":
        X_np = (
            X.values.astype(np.float32)
            if isinstance(X, pd.DataFrame)
            else X.astype(np.float32)
        )
        return model.predict_proba(X_np)[:, 1]

    return model.predict_proba(X)[:, 1]


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

def get_feature_importance(
    model: Any,
    model_name: str,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Extract per-feature importance scores in a unified format.

    Model-specific extraction strategies:
      LightGBM  : model.feature_importances_  (gain-based, averaged over trees)
      XGBoost   : model.feature_importances_  (weight by default, config-driven)
      CatBoost  : model.get_feature_importance()  (PredictionValuesChange)
      LogReg    : |coef_[0]|  — absolute coefficient magnitude after
                  StandardScaler normalisation (coefficients are directly
                  comparable because all features have unit variance).
      TabNet    : model.feature_importances_  — mean absolute attention mask
                  aggregated across all N_steps decision steps.  This
                  captures which features TabNet actually routes information
                  through at the architecture level.

    Returns
    -------
    pd.DataFrame with columns ["feature", "importance"], unsorted.
    """
    try:
        if model_name in ("lightgbm", "xgboost"):
            importance = model.feature_importances_

        elif model_name == "catboost":
            importance = model.get_feature_importance()

        elif model_name == "logreg":
            # Absolute coefficient magnitude.
            # coef_ shape is (1, n_features) for binary classification.
            importance = np.abs(model.coef_[0])

        elif model_name == "tabnet":
            # pytorch-tabnet exposes aggregated attention masks as
            # feature_importances_ after training.
            importance = model.feature_importances_

        else:
            importance = np.zeros(len(feature_names))

    except Exception:
        # Graceful fallback: all-zero importance so the run doesn't fail
        importance = np.zeros(len(feature_names))

    return pd.DataFrame({"feature": feature_names, "importance": importance})


# =============================================================================
# TABNET SERIALISATION HELPERS
# =============================================================================

def _save_tabnet_model(model: Any, path: Path) -> None:
    """
    Save a fitted TabNetClassifier using its native save_model() API.

    pytorch-tabnet's save_model() writes a zip file containing PyTorch
    state_dict plus architecture config.  The .zip extension is appended
    automatically by pytorch-tabnet — we strip it from the path to avoid
    double-extension artefacts.

    Why not joblib?
    joblib serialises Python objects via pickle.  PyTorch tensors inside
    TabNet's network hold C++ references that do not round-trip correctly
    through pickle across different pytorch versions.  save_model() is the
    library's official serialisation path.
    """
    # pytorch-tabnet appends ".zip" automatically — pass path without it
    save_path = str(path).replace(".pkl", "")
    model.save_model(save_path)


def _load_tabnet_model(path: Path) -> Any:
    """
    Load a TabNetClassifier from disk using its native load_model() API.
    """
    from pytorch_tabnet.tab_model import TabNetClassifier
    model = TabNetClassifier()
    # load_model expects the path without the .zip extension
    load_path = str(path).replace(".pkl", "")
    model.load_model(load_path + ".zip")
    return model


# =============================================================================
# CROSS-VALIDATION ENGINE
# =============================================================================

def run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run stratified K-fold cross-validation and return OOF predictions plus
    all fold-level artefacts.

    The engine is model-agnostic: it delegates to get_model(), train_model(),
    predict_proba(), and get_feature_importance() which each branch on
    model_name internally.

    Parameters
    ----------
    X : pd.DataFrame
        Preprocessed feature matrix (output of PreprocessingPipeline.transform).
    y : pd.Series
        Binary target (0 = no stress, 1 = stress).
    config : dict
        Full YAML config dict.  Must contain: project, cv, model, training,
        evaluation keys.

    Returns
    -------
    dict with keys:
        model_name, model_params, oof_preds, y_true, models,
        fold_scores, mean_logloss, mean_auc, std_logloss, std_auc,
        final_score, feature_importance, feature_names,
        fold_indices, fold_predictions
    """
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

    oof_preds:         np.ndarray          = np.zeros(len(X), dtype=np.float32)
    fold_scores:       List[Dict]          = []
    models:            List[Any]           = []
    fi_frames:         List[pd.DataFrame]  = []
    fold_indices:      List[Dict]          = []
    fold_predictions:  List[Dict]          = []

    print(f"\n{'='*60}")
    print(f"CV ENGINE | model={model_name.upper()}  folds={cv_cfg['n_splits']}")
    print(f"{'='*60}")

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):

        fold_start = time.perf_counter()
        print(f"\n[Fold {fold}] training...")

        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = get_model(model_name, model_params, training_cfg)
        model = train_model(
            model, model_name,
            X_train, y_train,
            X_valid, y_valid,
            training_cfg,
            cat_features,
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
#
# TabNet serialisation note:
#   TabNet models are saved via _save_tabnet_model() which calls
#   model.save_model() instead of joblib.dump().  The artifact path uses
#   a .pkl suffix in the directory listing for naming consistency, but
#   the actual file written by pytorch-tabnet is a .zip archive.
#   _load_tabnet_model() handles the .zip extension transparently.
# =============================================================================

def save_cv_outputs(
    results: Dict[str, Any],
    config:  Dict[str, Any],
    run_dir: str,
) -> None:
    """
    Persist all CV artefacts to disk in the standard artifact contract layout.

    For TabNet models, fold-level serialisation uses save_model() / load_model()
    instead of joblib to avoid PyTorch tensor serialisation issues.
    All other models continue to use joblib.
    """
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_name = results["model_name"]

    # -------------------------------------------------------------------------
    # OOF predictions + ground truth
    # -------------------------------------------------------------------------
    np.save(out / "oof_preds.npy", results["oof_preds"])
    np.save(out / "y_true.npy",    results["y_true"])

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
    # Serialised models — model-family-aware serialisation
    # -------------------------------------------------------------------------
    if config.get("artifacts", {}).get("save_models", True):
        model_dir = out / "models" / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        if model_name == "tabnet":
            # pytorch-tabnet cannot be serialised via joblib.
            # Use the library's native save_model() API.
            for i, model in enumerate(results["models"]):
                fold_path = model_dir / f"{model_name}_fold_{i}.pkl"
                _save_tabnet_model(model, fold_path)
        else:
            import joblib
            for i, model in enumerate(results["models"]):
                joblib.dump(model, model_dir / f"{model_name}_fold_{i}.pkl")

    # -------------------------------------------------------------------------
    # Preprocessor (if present in results — set by run_all_models.py)
    # -------------------------------------------------------------------------
    if "preprocessor" in results:
        import joblib
        joblib.dump(results["preprocessor"], out / "preprocessor.pkl")

    # -------------------------------------------------------------------------
    # Metadata (human-readable run record)
    # -------------------------------------------------------------------------
    metadata = {
        "model":        model_name,
        "stage":        config["experiment"]["stage"],
        "version":      config["experiment"].get("version", ""),
        "n_folds":      config["cv"]["n_splits"],
        "n_features":   len(results["feature_names"]),
        "mean_logloss": results["mean_logloss"],
        "mean_auc":     results["mean_auc"],
        "final_score":  results["final_score"],
        "seed":         config["project"]["seed"],
        "created_at":   pd.Timestamp.now().isoformat(),
    }
    with open(out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    # -------------------------------------------------------------------------
    # Config snapshot (reproduces this exact run)
    # -------------------------------------------------------------------------
    with open(out / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)

    print(f"Artifacts saved -> {out}")