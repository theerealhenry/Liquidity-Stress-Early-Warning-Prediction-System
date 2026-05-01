"""
Hyperparameter Tuning Pipeline — Optuna
========================================
Project : AI4EAC Liquidity Stress Early Warning (Zindi Africa)
Module  : src/tuning/tune.py

Design philosophy
-----------------
This module tunes LightGBM and XGBoost — the two best-performing base models
(composite scores 0.19557 and 0.19350 respectively on calibrated OOF). CatBoost
is excluded from tuning for two reasons: (1) CPU tuning is prohibitively slow
at 150 trials; (2) post-calibration CatBoost matches the other two models,
meaning tuning gains would be marginal relative to compute cost.

Ensemble motivation
-------------------
The ensemble pipeline (Phase B) identified that inter-model correlation of
0.962–0.978 made stacking ineffective. This tuning run deliberately targets
STRUCTURAL DIVERSITY — searching hyperparameter ranges that produce models
with genuinely different decision boundaries. A tuned LightGBM with
num_leaves=128, subsample=0.6 learns a fundamentally different model from a
tuned XGBoost with max_depth=4, min_child_weight=20. Lower correlation enables
stacking to add value in the post-tuning ensemble re-run.

Objective function
------------------
The competition's exact weighted composite score:
    score = 0.6 * LogLoss + 0.4 * (1 - AUC)
Lower is better. This is evaluated on 5-fold stratified CV OOF predictions
with the full feature engineering + preprocessing pipeline, matching exactly
what happens during training.

Key design decisions
--------------------
1. PRUNING: MedianPruner eliminates unpromising trials early (after fold 2),
   cutting total compute by ~40% without sacrificing result quality.

2. CALIBRATION IN OBJECTIVE: The objective applies Platt calibration to OOF
   predictions before computing Log Loss. This is critical — an uncalibrated
   objective would optimise for ranking (AUC) at the expense of probability
   quality (Log Loss). The competition weights Log Loss at 60%.

3. WARM STARTING: Best parameters from baseline configs are injected as the
   first trial via enqueue_trial(). This guarantees the search starts from a
   known-good region and the tuned model cannot be worse than baseline.

4. SCALE_POS_WEIGHT: Corrected from 5.5 (approximation) to 5.667 (exact:
   34000/6000) based on verified positive rate of 15% in training data.

5. SEPARATE STUDIES PER MODEL: Each model gets its own Optuna study with its
   own SQLite storage. This allows independent resumption, parallel execution
   on different machines, and clean comparison of study histories.

6. ARTIFACT CONTRACT: Tuning outputs land in the v2_feature_expansion stage
   directory (not baseline), because tuned models represent an improvement
   over the baseline runs, not a separate experiment type.

Output contract
---------------
outputs/experiments/v2_feature_expansion/{model}/run_YYYYMMDD_HHMMSS/
    [same artifact contract as baseline cv runs]

outputs/tuning/
    {model}_study.db          Optuna SQLite storage (resumable)
    {model}_best_params.yaml  Best hyperparameters
    {model}_study_summary.json All trial results
    {model}_importance.json   Parameter importance scores
    tuning_metadata.json      Full run record

Usage
-----
# Tune both models (default 100 trials each):
python -m src.tuning.tune

# Tune single model with custom trial count:
python -m src.tuning.tune --model lightgbm --n-trials 150

# Resume an interrupted study:
python -m src.tuning.tune --model xgboost --resume

# Quick smoke test (5 trials):
python -m src.tuning.tune --n-trials 5 --fast
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
import yaml
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# PROJECT ROOT RESOLUTION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# =============================================================================
# CONSTANTS — derived from verified data analysis
# =============================================================================

SEED              : int   = 42
N_SPLITS          : int   = 5
LOG_LOSS_WEIGHT   : float = 0.6
AUC_WEIGHT        : float = 0.4
EPS               : float = 1e-15
CLIP_LOW          : float = 1e-6
CLIP_HIGH         : float = 1.0 - 1e-6

# Verified from distribution check: 34000 negatives / 6000 positives
SCALE_POS_WEIGHT  : float = 34000 / 6000   # 5.6667

TUNABLE_MODELS    : List[str] = ["lightgbm", "xgboost"]

# Directories
TUNING_DIR        : Path = PROJECT_ROOT / "outputs" / "tuning"
EXPERIMENT_DIR    : Path = PROJECT_ROOT / "outputs" / "experiments" / "v2_feature_expansion"


# =============================================================================
# BASELINE PARAMETERS (warm start — guaranteed not to regress)
# Derived from configs/lgbm_v2.yaml and configs/xgb_v2.yaml
# =============================================================================

BASELINE_PARAMS: Dict[str, Dict] = {
    "lightgbm": {
        "learning_rate"    : 0.03,
        "num_leaves"       : 64,
        "min_child_samples": 50,
        "subsample"        : 0.8,
        "colsample_bytree" : 0.8,
        "reg_alpha"        : 0.5,
        "reg_lambda"       : 1.0,
        "max_bin"          : 255,
    },
    "xgboost": {
        "learning_rate"    : 0.03,
        "max_depth"        : 6,
        "min_child_weight" : 50,
        "subsample"        : 0.8,
        "colsample_bytree" : 0.8,
        "reg_alpha"        : 0.5,
        "reg_lambda"       : 1.0,
        "max_bin"          : 256,
    },
}


# =============================================================================
# METRICS
# =============================================================================

def composite_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Competition objective: 0.6 * LogLoss + 0.4 * (1 - AUC). Lower = better."""
    y_clipped = np.clip(y_pred, EPS, 1 - EPS)
    ll  = log_loss(y_true, y_clipped)
    auc = roc_auc_score(y_true, y_pred)
    return LOG_LOSS_WEIGHT * ll + AUC_WEIGHT * (1.0 - auc)


def platt_calibrate_oof(
    oof_preds: np.ndarray,
    y_true: np.ndarray,
    n_splits: int = 5,
) -> np.ndarray:
    """
    Cross-validated Platt calibration applied to OOF predictions.
    Produces honest calibrated OOF estimate for Log Loss evaluation.
    This is critical: the objective must optimise calibrated Log Loss,
    not raw Log Loss, since that is what the competition measures.
    """
    calibrated = np.zeros_like(oof_preds, dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    for train_idx, val_idx in skf.split(oof_preds, y_true):
        cal = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        cal.fit(oof_preds[train_idx].reshape(-1, 1), y_true[train_idx])
        calibrated[val_idx] = cal.predict_proba(
            oof_preds[val_idx].reshape(-1, 1)
        )[:, 1]

    return np.clip(calibrated, CLIP_LOW, CLIP_HIGH)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_training_data(config_path: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load and prepare training data using the full pipeline.
    Mirrors exactly what happens during training to prevent any
    objective-to-inference discrepancy.
    """
    sys.path.insert(0, str(PROJECT_ROOT))

    with open(PROJECT_ROOT / config_path) as f:
        cfg = yaml.safe_load(f)

    from src.data.load_data import load_data
    from src.features.feature_engineering import build_features, split_features_target
    from src.preprocessing.preprocessing import PreprocessingPipeline

    train_path = str(PROJECT_ROOT / cfg["data"]["train_path"])
    train_df, _ = load_data(train_path=train_path, validate=False, verbose=False)

    fe_df = build_features(train_df)
    X, y  = split_features_target(fe_df)

    preproc  = PreprocessingPipeline(config=cfg)
    X_processed = preproc.fit_transform(X)

    return X_processed, y.values.astype(int)


# =============================================================================
# SEARCH SPACES
# Deliberately wider than baseline to encourage structural diversity.
# Comments justify each bound based on domain knowledge and existing results.
# =============================================================================

def lightgbm_search_space(trial: optuna.Trial) -> Dict[str, Any]:
    """
    LightGBM hyperparameter search space.

    Diversity targets vs baseline (num_leaves=64, subsample=0.8):
    - num_leaves: extend to 256 (more complex trees → different splits)
    - subsample: extend down to 0.5 (higher randomness → less correlation with XGB)
    - min_child_samples: extend to 200 (stronger regularisation region)
    - reg_alpha/lambda: wider range to explore sparse vs dense solutions
    """
    params = {
        # Learning dynamics
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.005, 0.1, log=True
        ),
        "n_estimators": trial.suggest_int(
            "n_estimators", 500, 3000, step=100
        ),

        # Tree structure — primary diversity driver vs XGBoost
        "num_leaves": trial.suggest_int(
            "num_leaves", 16, 256, log=True
        ),
        "max_depth": trial.suggest_int(
            "max_depth", -1, 12
            # -1 = unlimited; allowing constrained depths creates different models
        ),
        "min_child_samples": trial.suggest_int(
            "min_child_samples", 10, 200, log=True
        ),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 1e-4, 1e-1, log=True
        ),

        # Sampling — second diversity driver
        "subsample": trial.suggest_float(
            "subsample", 0.5, 1.0
        ),
        "subsample_freq": trial.suggest_int(
            "subsample_freq", 1, 5
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.5, 1.0
        ),

        # Regularisation
        "reg_alpha": trial.suggest_float(
            "reg_alpha", 1e-3, 10.0, log=True
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", 1e-3, 10.0, log=True
        ),

        # Histogram binning
        "max_bin": trial.suggest_categorical(
            "max_bin", [127, 255, 511]
        ),

        # Fixed — not worth tuning for this problem
        "boosting_type"   : "gbdt",
        "objective"       : "binary",
        "metric"          : ["binary_logloss", "auc"],
        "scale_pos_weight": SCALE_POS_WEIGHT,   # verified: 34000/6000
        "verbosity"       : -1,
        "random_state"    : SEED,
        "n_jobs"          : -1,
    }
    return params


def xgboost_search_space(trial: optuna.Trial) -> Dict[str, Any]:
    """
    XGBoost hyperparameter search space.

    Diversity targets vs LightGBM:
    - max_depth: constrained (4–8) vs LightGBM's leaf-wise growth.
      Depth-wise growth with lower depth = different decision boundary.
    - min_child_weight: lower range (5–100) explores more granular splits.
    - gamma: XGBoost-specific regularisation LightGBM doesn't have.
    - grow_policy: depthwise vs lossguide (lossguide mimics leaf-wise).
    """
    grow_policy = trial.suggest_categorical(
        "grow_policy", ["depthwise", "lossguide"]
    )

    params = {
        # Learning dynamics
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.005, 0.1, log=True
        ),
        "n_estimators": trial.suggest_int(
            "n_estimators", 500, 3000, step=100
        ),

        # Tree structure — primary diversity driver vs LightGBM
        "max_depth": trial.suggest_int(
            "max_depth", 3, 10
            # Depth-wise growth with depth 3–5 = very different from LightGBM
        ),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 5.0, 100.0, log=True
        ),
        "gamma": trial.suggest_float(
            "gamma", 0.0, 5.0
            # Minimum loss reduction for split — XGBoost-specific lever
        ),
        "grow_policy": grow_policy,

        # Max leaves only active for lossguide
        "max_leaves": trial.suggest_int(
            "max_leaves", 16, 256, log=True
        ) if grow_policy == "lossguide" else 0,

        # Sampling
        "subsample": trial.suggest_float(
            "subsample", 0.5, 1.0
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.5, 1.0
        ),
        "colsample_bylevel": trial.suggest_float(
            "colsample_bylevel", 0.5, 1.0
            # XGBoost-specific — LightGBM equivalent doesn't exist
        ),

        # Regularisation
        "reg_alpha": trial.suggest_float(
            "reg_alpha", 1e-3, 10.0, log=True
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", 1e-3, 10.0, log=True
        ),

        # Histogram
        "max_bin": trial.suggest_categorical(
            "max_bin", [128, 256, 512]
        ),

        # Fixed
        "objective"            : "binary:logistic",
        "eval_metric"          : ["logloss", "auc"],
        "tree_method"          : "hist",
        "scale_pos_weight"     : SCALE_POS_WEIGHT,
        "early_stopping_rounds": 100,
        "verbosity"            : 0,
        "random_state"         : SEED,
        "n_jobs"               : -1,
    }
    return params


# =============================================================================
# OBJECTIVE FUNCTIONS
# =============================================================================

class LightGBMObjective:
    """
    Optuna objective for LightGBM.

    Implements __call__ for compatibility with optuna.study.optimize().
    Stores X, y internally to avoid repeated loading across trials.
    Uses pruning callbacks to terminate unpromising trials after fold 2.
    """

    def __init__(
        self,
        X           : pd.DataFrame,
        y           : np.ndarray,
        calibrate   : bool = True,
        n_splits    : int  = N_SPLITS,
        es_rounds   : int  = 100,
        verbose_eval: int  = 0,
    ) -> None:
        self.X            = X
        self.y            = y
        self.calibrate    = calibrate
        self.n_splits     = n_splits
        self.es_rounds    = es_rounds
        self.verbose_eval = verbose_eval

    def __call__(self, trial: optuna.Trial) -> float:
        import lightgbm as lgb

        params = lightgbm_search_space(trial)

        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=SEED
        )
        oof_preds = np.zeros(len(self.y), dtype=float)

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(self.X, self.y)
        ):
            X_tr, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_tr, y_val = self.y[train_idx], self.y[val_idx]

            n_est = params.pop("n_estimators")
            model = lgb.LGBMClassifier(n_estimators=n_est, **params)
            params["n_estimators"] = n_est   # restore

            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric=["binary_logloss", "auc"],
                callbacks=[
                    lgb.early_stopping(self.es_rounds, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )

            fold_pred = model.predict_proba(X_val)[:, 1]
            oof_preds[val_idx] = fold_pred

            # ── Pruning: report intermediate score after each fold ────────
            fold_score = composite_score(y_val, fold_pred)
            trial.report(fold_score, step=fold)

            if trial.should_prune():
                raise optuna.TrialPruned()

        # ── Final score: calibrated composite ────────────────────────────
        if self.calibrate:
            oof_calibrated = platt_calibrate_oof(oof_preds, self.y)
            return composite_score(self.y, oof_calibrated)
        else:
            return composite_score(self.y, oof_preds)


class XGBoostObjective:
    """Optuna objective for XGBoost. Mirror of LightGBMObjective."""

    def __init__(
        self,
        X           : pd.DataFrame,
        y           : np.ndarray,
        calibrate   : bool = True,
        n_splits    : int  = N_SPLITS,
        es_rounds   : int  = 100,
        verbose_eval: int  = 0,
    ) -> None:
        self.X            = X
        self.y            = y
        self.calibrate    = calibrate
        self.n_splits     = n_splits
        self.es_rounds    = es_rounds
        self.verbose_eval = verbose_eval

    def __call__(self, trial: optuna.Trial) -> float:
        import xgboost as xgb

        params = xgboost_search_space(trial)

        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=SEED
        )
        oof_preds = np.zeros(len(self.y), dtype=float)

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(self.X, self.y)
        ):
            X_tr, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_tr, y_val = self.y[train_idx], self.y[val_idx]

            model = xgb.XGBClassifier(**params)

            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=self.verbose_eval,
            )

            fold_pred = model.predict_proba(X_val)[:, 1]
            oof_preds[val_idx] = fold_pred

            # ── Pruning ───────────────────────────────────────────────────
            fold_score = composite_score(y_val, fold_pred)
            trial.report(fold_score, step=fold)

            if trial.should_prune():
                raise optuna.TrialPruned()

        if self.calibrate:
            oof_calibrated = platt_calibrate_oof(oof_preds, self.y)
            return composite_score(self.y, oof_calibrated)
        else:
            return composite_score(self.y, oof_preds)


# =============================================================================
# STUDY MANAGEMENT
# =============================================================================

def create_or_load_study(
    model_name  : str,
    storage_path: Path,
    resume      : bool = False,
    n_startup   : int  = 10,
    n_warmup    : int  = 5,
) -> optuna.Study:
    """
    Create a new Optuna study or load an existing one for resumption.

    Sampler: TPE (Tree-structured Parzen Estimator)
    - Builds a probabilistic model of the objective surface
    - Samples new trials by maximising Expected Improvement
    - Superior to random search after n_startup trials

    Pruner: MedianPruner
    - After n_warmup completed trials, prunes any trial whose intermediate
      score is worse than the median of completed trials at the same step
    - Eliminates ~40% of compute on unpromising hyperparameter regions
    - n_warmup=5 ensures enough trials complete before pruning activates
    """
    storage_uri = f"sqlite:///{storage_path}"
    study_name  = f"{model_name}_tuning"

    sampler = TPESampler(
        n_startup_trials=n_startup,
        seed=SEED,
        multivariate=True,    # models correlations between parameters
        group=True,           # groups correlated params for better sampling
    )

    pruner = MedianPruner(
        n_startup_trials=n_warmup,
        n_warmup_steps=2,     # report scores from fold 2 onwards
        interval_steps=1,
    )

    if resume and storage_path.exists():
        print(f"  Resuming study '{study_name}' from {storage_path}")
        study = optuna.load_study(
            study_name=study_name,
            storage=storage_uri,
            sampler=sampler,
            pruner=pruner,
        )
        completed = len([t for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE])
        print(f"  Resumed: {completed} completed trials found")
    else:
        if storage_path.exists() and not resume:
            # Remove stale storage to start fresh
            storage_path.unlink()

        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            storage=storage_uri,
            sampler=sampler,
            pruner=pruner,
        )
        print(f"  Created new study '{study_name}'")

    return study


def inject_baseline_trial(
    study      : optuna.Study,
    model_name : str,
) -> None:
    """
    Inject baseline parameters as the first trial (warm start).

    This guarantees:
    1. The tuned model cannot be worse than baseline
    2. The TPE sampler starts from a known-good region
    3. The study history always contains a reproducible reference point

    Optuna's enqueue_trial() tells the sampler to use these exact parameters
    for the next trial before switching to TPE-guided sampling.
    """
    baseline = BASELINE_PARAMS[model_name].copy()
    study.enqueue_trial(baseline)
    print(f"  Warm start: baseline parameters enqueued as trial 0")


# =============================================================================
# RESULTS ANALYSIS
# =============================================================================

def analyze_study(
    study      : optuna.Study,
    model_name : str,
) -> Dict[str, Any]:
    """
    Extract and summarise study results.

    Returns:
    - best_params: dict of optimal hyperparameters
    - param_importance: ranked parameter importance scores
    - trial_summary: all trial results as a DataFrame
    - improvement: percentage improvement over baseline trial
    """
    best_trial    = study.best_trial
    all_trials    = study.trials

    # ── Parameter importance ──────────────────────────────────────────
    try:
        importance = optuna.importance.get_param_importances(study)
    except Exception:
        importance = {}

    # ── Trial summary ─────────────────────────────────────────────────
    trial_rows = []
    for t in all_trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            row = {"trial": t.number, "score": t.value}
            row.update(t.params)
            trial_rows.append(row)

    trial_df = pd.DataFrame(trial_rows).sort_values("score")

    # ── Improvement vs baseline (trial 0) ─────────────────────────────
    baseline_trials = [
        t for t in all_trials
        if t.number == 0 and t.state == optuna.trial.TrialState.COMPLETE
    ]
    if baseline_trials:
        baseline_score = baseline_trials[0].value
        improvement    = (baseline_score - best_trial.value) / baseline_score * 100
    else:
        baseline_score = None
        improvement    = None

    completed = len([t for t in all_trials
                    if t.state == optuna.trial.TrialState.COMPLETE])
    pruned    = len([t for t in all_trials
                    if t.state == optuna.trial.TrialState.PRUNED])

    return {
        "model_name"     : model_name,
        "best_score"     : best_trial.value,
        "best_params"    : best_trial.params,
        "best_trial_num" : best_trial.number,
        "baseline_score" : baseline_score,
        "improvement_pct": improvement,
        "n_completed"    : completed,
        "n_pruned"       : pruned,
        "n_total"        : len(all_trials),
        "param_importance": importance,
        "trial_df"       : trial_df,
    }


def print_study_summary(analysis: Dict[str, Any]) -> None:
    """Print a formatted summary of study results."""
    m = analysis["model_name"].upper()
    print(f"\n{'='*65}")
    print(f"TUNING SUMMARY — {m}")
    print(f"{'='*65}")
    print(f"Best score    : {analysis['best_score']:.6f}")
    if analysis["baseline_score"]:
        print(f"Baseline score: {analysis['baseline_score']:.6f}")
        print(f"Improvement   : {analysis['improvement_pct']:.2f}%")
    print(f"Best trial    : #{analysis['best_trial_num']}")
    print(f"Completed     : {analysis['n_completed']} / {analysis['n_total']}")
    print(f"Pruned        : {analysis['n_pruned']} / {analysis['n_total']}")
    print(f"\nBest hyperparameters:")
    for k, v in sorted(analysis["best_params"].items()):
        baseline_val = BASELINE_PARAMS.get(
            analysis["model_name"], {}
        ).get(k, "—")
        changed = " ← changed" if str(v) != str(baseline_val) else ""
        print(f"  {k:30s}: {str(v):15s} (baseline: {baseline_val}){changed}")

    if analysis["param_importance"]:
        print(f"\nParameter importance (top 8):")
        for k, v in list(analysis["param_importance"].items())[:8]:
            bar = "█" * int(v * 30)
            print(f"  {k:30s}: {v:.4f}  {bar}")
    print(f"{'='*65}")


# =============================================================================
# ARTIFACT SAVING
# =============================================================================

def save_tuning_artifacts(
    analysis   : Dict[str, Any],
    study      : optuna.Study,
    tuning_dir : Path,
    model_name : str,
) -> None:
    """Save all tuning artifacts to the tuning directory."""
    tuning_dir.mkdir(parents=True, exist_ok=True)

    # ── Best parameters as YAML ───────────────────────────────────────
    best_params_path = tuning_dir / f"{model_name}_best_params.yaml"
    with open(best_params_path, "w") as f:
        yaml.dump(analysis["best_params"], f, default_flow_style=False)
    print(f"  Best params   : {best_params_path}")

    # ── Study summary as JSON ─────────────────────────────────────────
    summary_data = {
        k: v for k, v in analysis.items()
        if k not in ("trial_df", "param_importance")
    }
    summary_data["param_importance"] = {
        k: float(v) for k, v in analysis.get("param_importance", {}).items()
    }
    summary_path = tuning_dir / f"{model_name}_study_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=4, default=str)
    print(f"  Study summary : {summary_path}")

    # ── Trial history as CSV ──────────────────────────────────────────
    if not analysis["trial_df"].empty:
        trial_path = tuning_dir / f"{model_name}_trial_history.csv"
        analysis["trial_df"].to_csv(trial_path, index=False)
        print(f"  Trial history : {trial_path}")

    # ── Parameter importance ──────────────────────────────────────────
    imp_path = tuning_dir / f"{model_name}_importance.json"
    with open(imp_path, "w") as f:
        json.dump(
            {k: float(v) for k, v in analysis.get("param_importance", {}).items()},
            f, indent=4
        )

    print(f"  All artifacts saved to: {tuning_dir}")


def build_tuned_config(
    model_name  : str,
    best_params : Dict[str, Any],
    base_config : Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge tuned hyperparameters into the base config to produce
    a ready-to-use config for re-training the tuned model.

    The resulting config can be passed directly to run_cv() and
    save_cv_outputs() without any modification.
    """
    tuned_config = base_config.copy()

    # Deep copy model params and update with tuned values
    tuned_model_params = base_config.get("model", {}).get("params", {}).copy()
    tuned_model_params.update(best_params)

    # Correct scale_pos_weight with verified value
    tuned_model_params["scale_pos_weight"] = SCALE_POS_WEIGHT

    tuned_config["model"] = {
        "name"  : model_name,
        "params": tuned_model_params,
    }

    # Update experiment stage to v2_feature_expansion
    tuned_config["experiment"] = {
        **base_config.get("experiment", {}),
        "stage"  : "v2_feature_expansion",
        "version": "v2_tuned",
    }

    return tuned_config


# =============================================================================
# MAIN TUNING FUNCTION
# =============================================================================

def tune_model(
    model_name  : str,
    n_trials    : int  = 100,
    resume      : bool = False,
    fast        : bool = False,
    calibrate   : bool = True,
) -> Dict[str, Any]:
    """
    Run hyperparameter tuning for a single model.

    Parameters
    ----------
    model_name  : "lightgbm" or "xgboost"
    n_trials    : number of Optuna trials (default 100)
    resume      : resume interrupted study from SQLite storage
    fast        : smoke test mode — reduces n_splits to 3, es_rounds to 50
    calibrate   : whether to apply Platt calibration in objective

    Returns
    -------
    dict with analysis results and best config
    """
    if model_name not in TUNABLE_MODELS:
        raise ValueError(
            f"model_name must be one of {TUNABLE_MODELS}, got '{model_name}'"
        )

    print(f"\n{'='*65}")
    print(f"TUNING {model_name.upper()}")
    print(f"{'='*65}")
    print(f"  Trials      : {n_trials}")
    print(f"  CV splits   : {N_SPLITS if not fast else 3}")
    print(f"  Calibration : {calibrate}")
    print(f"  Warm start  : baseline params as trial 0")
    print(f"  Pruner      : MedianPruner (n_warmup=5, n_warmup_steps=2)")

    start_time = time.perf_counter()

    # ── Load data ─────────────────────────────────────────────────────
    print(f"\n  Loading data...")
    config_path = f"configs/{model_name.replace('lightgbm', 'lgbm')}_v2.yaml"
    if model_name == "lightgbm":
        config_path = "configs/lgbm_v2.yaml"
    elif model_name == "xgboost":
        config_path = "configs/xgb_v2.yaml"

    X, y = load_training_data(config_path)
    print(f"  Data loaded: {X.shape[0]:,} rows × {X.shape[1]:,} features")
    print(f"  Positive rate: {y.mean():.4f} ({y.sum():,} positives)")

    # ── Load base config for later merging ────────────────────────────
    with open(PROJECT_ROOT / config_path) as f:
        base_config = yaml.safe_load(f)

    # ── Create study ──────────────────────────────────────────────────
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    storage_path = TUNING_DIR / f"{model_name}_study.db"
    study = create_or_load_study(model_name, storage_path, resume=resume)
    inject_baseline_trial(study, model_name)

    # ── Build objective ───────────────────────────────────────────────
    n_splits_eff = 3 if fast else N_SPLITS
    es_rounds    = 50 if fast else 100

    if model_name == "lightgbm":
        objective = LightGBMObjective(
            X, y,
            calibrate=calibrate,
            n_splits=n_splits_eff,
            es_rounds=es_rounds,
        )
    else:
        objective = XGBoostObjective(
            X, y,
            calibrate=calibrate,
            n_splits=n_splits_eff,
            es_rounds=es_rounds,
        )

    # ── Progress callback ─────────────────────────────────────────────
    def progress_callback(study: optuna.Study, trial: optuna.Trial) -> None:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            n_done = len([t for t in study.trials
                         if t.state == optuna.trial.TrialState.COMPLETE])
            print(
                f"  Trial {trial.number:3d} | "
                f"Score={trial.value:.5f} | "
                f"Best={study.best_value:.5f} | "
                f"Completed={n_done}/{n_trials}"
            )
        elif trial.state == optuna.trial.TrialState.PRUNED:
            print(f"  Trial {trial.number:3d} | PRUNED")

    # ── Optimise ──────────────────────────────────────────────────────
    print(f"\n  Starting optimisation...")
    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[progress_callback],
        show_progress_bar=False,
        gc_after_trial=True,        # free memory between trials
    )

    runtime = time.perf_counter() - start_time

    # ── Analyse results ───────────────────────────────────────────────
    analysis = analyze_study(study, model_name)
    analysis["runtime_sec"] = runtime
    print_study_summary(analysis)

    # ── Build tuned config ────────────────────────────────────────────
    tuned_config = build_tuned_config(
        model_name, analysis["best_params"], base_config
    )
    analysis["tuned_config"] = tuned_config

    # ── Save artifacts ────────────────────────────────────────────────
    print(f"\n  Saving artifacts...")
    save_tuning_artifacts(analysis, study, TUNING_DIR, model_name)

    # Save tuned config as YAML for direct use in orchestrator
    tuned_config_path = PROJECT_ROOT / "configs" / f"{model_name}_tuned.yaml"
    with open(tuned_config_path, "w") as f:
        yaml.dump(tuned_config, f, default_flow_style=False, allow_unicode=True)
    print(f"  Tuned config  : {tuned_config_path}")

    print(f"\n  Total runtime : {runtime:.1f}s ({runtime/60:.1f} min)")

    return analysis


# =============================================================================
# DIVERSITY CHECK
# =============================================================================

def check_post_tuning_diversity(
    lgbm_analysis : Dict[str, Any],
    xgb_analysis  : Dict[str, Any],
    X             : pd.DataFrame,
    y             : np.ndarray,
) -> None:
    """
    After tuning both models, estimate whether their predictions are
    more diverse than the baseline (correlation 0.978).

    Trains each tuned model on a single fold and computes prediction
    correlation — a fast proxy for the full CV correlation.

    A correlation drop from 0.978 to <0.95 indicates stacking may be
    viable in the post-tuning ensemble re-run.
    """
    import lightgbm as lgb
    import xgboost as xgb

    print(f"\n{'='*65}")
    print("POST-TUNING DIVERSITY CHECK")
    print(f"{'='*65}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, val_idx = next(iter(skf.split(X, y)))

    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    # LightGBM tuned
    lgbm_params = lgbm_analysis["best_params"].copy()
    n_est_lgbm  = lgbm_params.pop("n_estimators", 1000)
    lgbm_model  = lgb.LGBMClassifier(
        n_estimators=n_est_lgbm,
        scale_pos_weight=SCALE_POS_WEIGHT,
        verbosity=-1,
        random_state=SEED,
        **{k: v for k, v in lgbm_params.items()
           if k not in ("objective", "metric", "boosting_type", "n_jobs",
                        "scale_pos_weight", "verbosity", "random_state")}
    )
    lgbm_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    )
    lgbm_pred = lgbm_model.predict_proba(X_val)[:, 1]

    # XGBoost tuned
    xgb_params = xgb_analysis["best_params"].copy()
    xgb_model  = xgb.XGBClassifier(
        scale_pos_weight=SCALE_POS_WEIGHT,
        verbosity=0,
        random_state=SEED,
        early_stopping_rounds=50,
        **{k: v for k, v in xgb_params.items()
           if k not in ("objective", "eval_metric", "tree_method", "n_jobs",
                        "scale_pos_weight", "verbosity", "random_state",
                        "early_stopping_rounds")}
    )
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    xgb_pred = xgb_model.predict_proba(X_val)[:, 1]

    corr = float(np.corrcoef(lgbm_pred, xgb_pred)[0, 1])
    baseline_corr = 0.978   # from notebook 08

    print(f"  Baseline correlation (pre-tuning)  : {baseline_corr:.4f}")
    print(f"  Tuned correlation (single fold)    : {corr:.4f}")
    print(f"  Change                             : {corr - baseline_corr:+.4f}")

    if corr < 0.95:
        print(f"  → Stacking VIABLE in post-tuning ensemble re-run")
    elif corr < 0.97:
        print(f"  → Stacking MARGINAL — worth testing in ensemble re-run")
    else:
        print(f"  → Stacking still unlikely to beat averaging")
        print(f"    Consider adding raw features to meta-model (use_raw_features=True)")

    print(f"{'='*65}")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning — Liquidity Stress Early Warning"
    )
    parser.add_argument(
        "--model",
        choices=TUNABLE_MODELS + ["all"],
        default="all",
        help="Model to tune (default: all)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of Optuna trials per model (default: 100)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted study from SQLite storage",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Smoke test: 3-fold CV, 50 early stopping rounds",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Skip Platt calibration in objective (not recommended)",
    )
    parser.add_argument(
        "--diversity-check",
        action="store_true",
        help="Run post-tuning diversity check after both models are tuned",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    models_to_tune = (
        TUNABLE_MODELS if args.model == "all" else [args.model]
    )

    calibrate = not args.no_calibrate

    print(f"\nTuning pipeline started")
    print(f"  Models    : {models_to_tune}")
    print(f"  Trials    : {args.n_trials} per model")
    print(f"  Calibrate : {calibrate}")
    print(f"  Fast mode : {args.fast}")
    print(f"  Resume    : {args.resume}")
    print(f"  Seed      : {SEED}")
    print(f"  Output    : {TUNING_DIR}")

    analyses = {}
    for model_name in models_to_tune:
        analysis = tune_model(
            model_name=model_name,
            n_trials=args.n_trials,
            resume=args.resume,
            fast=args.fast,
            calibrate=calibrate,
        )
        analyses[model_name] = analysis

    # ── Post-tuning diversity check ───────────────────────────────────
    if args.diversity_check and len(analyses) == 2:
        print("\nRunning post-tuning diversity check...")
        # Reload data once for diversity check
        X, y = load_training_data("configs/lgbm_v2.yaml")
        check_post_tuning_diversity(
            analyses["lightgbm"], analyses["xgboost"], X, y
        )

    # ── Final summary across models ───────────────────────────────────
    if len(analyses) > 1:
        print(f"\n{'='*65}")
        print("FINAL TUNING COMPARISON")
        print(f"{'='*65}")
        print(f"{'Model':15s}  {'Best Score':>12}  {'Improvement':>12}  "
              f"{'Trials':>8}")
        print("-" * 55)
        for m, a in analyses.items():
            imp = f"{a['improvement_pct']:.2f}%" if a["improvement_pct"] else "N/A"
            print(f"{m:15s}  {a['best_score']:>12.6f}  {imp:>12}  "
                  f"{a['n_completed']:>8}")
        print(f"\nTuned configs saved to configs/{{model}}_tuned.yaml")
        print("Next: retrain tuned models via orchestrator, then re-run ensemble")
        print(f"{'='*65}")

    # ── Save tuning metadata ──────────────────────────────────────────
    metadata = {
        "created_at"    : datetime.now().isoformat(),
        "models_tuned"  : models_to_tune,
        "n_trials"      : args.n_trials,
        "calibrate"     : calibrate,
        "seed"          : SEED,
        "scale_pos_weight": SCALE_POS_WEIGHT,
        "results"       : {
            m: {
                "best_score"     : a["best_score"],
                "baseline_score" : a.get("baseline_score"),
                "improvement_pct": a.get("improvement_pct"),
                "n_completed"    : a["n_completed"],
                "n_pruned"       : a["n_pruned"],
            }
            for m, a in analyses.items()
        },
    }
    meta_path = TUNING_DIR / "tuning_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4, default=str)
    print(f"\nMetadata saved: {meta_path}")


if __name__ == "__main__":
    main()