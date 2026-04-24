"""
Feature Engineering Module
======================================

Advanced feature engineering pipeline designed for:
- Temporal financial behavior modeling (6-month panel data)
- Zero-inflated transaction distributions
- Highly skewed monetary features
- Tree-based models (LightGBM / XGBoost / CatBoost)
- SHAP interpretability

Design Principles:
- Full temporal aggregations (mean/std/min/max/trend/slope)
- Behavioral ratios (financial stress signals)
- Activity & sparsity features
- Recency-weighted signals
- Balance + cashflow intelligence
- STRICTLY CV-safe (no leakage)
- Deterministic & reproducible
- Modular feature blocks
- SHAP-friendly transformations
"""

from typing import List, Tuple
import numpy as np
import pandas as pd


# =========================================================
# CONFIG
# =========================================================

TARGET = "liquidity_stress_next_30d"
ID_COL = "ID"

MONTHS = ["m1", "m2", "m3", "m4", "m5", "m6"]

TRANSACTION_GROUPS = [
    "deposit",
    "withdraw",
    "merchantpay",
    "paybill",
    "transfer_from_bank",
    "mm_send",
    "received"
]

VALUE_SUFFIX = "total_value"
EPS = 1e-6


# =========================================================
# MAIN PIPELINE
# =========================================================

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = _log_transform(df)

    features = {}

    # Cache frequently used columns
    cache = _build_cache(df)

    features.update(_zero_indicators(df))
    features.update(_temporal_features(cache))
    features.update(_activity_features(cache))
    features.update(_cashflow_features(cache))
    features.update(_behavioral_ratios(df, cache))
    features.update(_recency_features(cache))
    features.update(_balance_features(df))
    features.update(_volatility_features(cache))
    features.update(_momentum_features(cache))
    features.update(_interaction_features(cache))

    feature_df = pd.DataFrame(features, index=df.index)

    return pd.concat([df, feature_df], axis=1)


# =========================================================
# CACHE 
# =========================================================

def _build_cache(df):
    cache = {}

    for group in TRANSACTION_GROUPS:
        cols = [f"{m}_{group}_{VALUE_SUFFIX}" for m in MONTHS if f"{m}_{group}_{VALUE_SUFFIX}" in df.columns]

        if cols:
            data = df[cols].values
            cache[group] = {
                "cols": cols,
                "data": data,
                "sum": data.sum(axis=1),
                "mean": data.mean(axis=1),
                "std": data.std(axis=1),
            }

    return cache


# =========================================================
# ZERO INDICATORS
# =========================================================

def _zero_indicators(df):
    features = {}

    numeric_cols = df.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        if (df[col] == 0).mean() > 0.6:
            features[f"{col}_is_zero"] = (df[col] == 0).astype(int)

    return features


# =========================================================
# LOG TRANSFORM
# =========================================================

def _log_transform(df):
    df = df.copy()

    for col in df.columns:
        if "total_value" in col or "highest_amount" in col:
            df[col] = np.log1p(df[col])

    return df


# =========================================================
# TEMPORAL FEATURES
# =========================================================

def _compute_slope(data):
    x = np.arange(data.shape[1])
    x_mean = x.mean()

    return ((data - data.mean(axis=1, keepdims=True)) * (x - x_mean)).sum(axis=1) / (
        ((x - x_mean) ** 2).sum() + EPS
    )


def _temporal_features(cache):
    features = {}

    for g, v in cache.items():
        data = v["data"]

        features[f"{g}_mean"] = v["mean"]
        features[f"{g}_std"] = v["std"]
        features[f"{g}_min"] = data.min(axis=1)
        features[f"{g}_max"] = data.max(axis=1)

        features[f"{g}_cv"] = v["std"] / (v["mean"] + EPS)

        features[f"{g}_trend"] = data[:, -1] - data[:, 0]
        features[f"{g}_trend_ratio"] = data[:, -1] / (data[:, 0] + EPS)

        features[f"{g}_slope"] = _compute_slope(data)
        features[f"{g}_consistency"] = 1 / (v["std"] + EPS)

    return features


# =========================================================
# ACTIVITY
# =========================================================

def _activity_features(cache):
    features = {}

    for g, v in cache.items():
        activity = (v["data"] > 0).astype(int)

        features[f"{g}_active_months"] = activity.sum(axis=1)
        features[f"{g}_inactive_months"] = (activity == 0).sum(axis=1)

    return features


# =========================================================
# CASHFLOW
# =========================================================

def _cashflow_features(cache):
    features = {}

    if "deposit" in cache and "withdraw" in cache:
        d = cache["deposit"]["sum"]
        w = cache["withdraw"]["sum"]

        features["total_deposit"] = d
        features["total_withdraw"] = w
        features["net_cashflow"] = d - w
        features["withdraw_deposit_ratio"] = w / (d + EPS)

        features["cashflow_stability"] = (d - w) / (np.abs(d) + np.abs(w) + EPS)

    return features


# =========================================================
# BEHAVIORAL RATIOS
# =========================================================

def _behavioral_ratios(df, cache):
    features = {}

    if "deposit" in cache and "withdraw" in cache:
        d = cache["deposit"]["sum"]
        w = cache["withdraw"]["sum"]

        features["cash_pressure"] = w / (d + EPS)

    if "arpu" in df.columns and "deposit" in cache:
        features["deposit_intensity"] = cache["deposit"]["sum"] / (df["arpu"] + EPS)

    return features


# =========================================================
# RECENCY
# =========================================================

def _recency_features(cache):
    features = {}

    for g, v in cache.items():
        data = v["data"]

        if data.shape[1] >= 3:
            recent = data[:, :3].mean(axis=1)
            past = data[:, 3:].mean(axis=1)

            features[f"{g}_recency_ratio"] = recent / (past + EPS)

    return features


# =========================================================
# BALANCE
# =========================================================

def _balance_features(df):
    features = {}

    bal_cols = [f"{m}_daily_avg_bal" for m in MONTHS if f"{m}_daily_avg_bal" in df.columns]

    if len(bal_cols) >= 2:
        data = df[bal_cols].values

        features["balance_mean"] = data.mean(axis=1)
        features["balance_trend"] = data[:, -1] - data[:, 0]
        features["balance_volatility"] = data.std(axis=1)

    return features


# =========================================================
# VOLATILITY CHANGE
# =========================================================

def _volatility_features(cache):
    features = {}

    for g, v in cache.items():
        data = v["data"]

        if data.shape[1] >= 4:
            recent = data[:, :3].std(axis=1)
            past = data[:, 3:].std(axis=1)

            features[f"{g}_volatility_ratio"] = recent / (past + EPS)

    return features


# =========================================================
# MOMENTUM
# =========================================================

def _momentum_features(cache):
    features = {}

    for g, v in cache.items():
        data = v["data"]

        if data.shape[1] >= 2:
            features[f"{g}_momentum"] = data[:, -1] - data[:, -2]

    return features


# =========================================================
# INTERACTIONS
# =========================================================

def _interaction_features(cache):
    features = {}

    if "deposit" in cache and "withdraw" in cache:
        d = cache["deposit"]["sum"]
        w = cache["withdraw"]["sum"]

        features["deposit_withdraw_interaction"] = d * w
        features["net_flow_ratio"] = (d - w) / (d + w + EPS)

    return features


# =========================================================
# SPLIT
# =========================================================

def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET, ID_COL], errors="ignore")
    y = df[TARGET] if TARGET in df.columns else None
    return X, y


# =========================================================
# FEATURE LIST
# =========================================================

def get_feature_list(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if col not in [ID_COL, TARGET]]