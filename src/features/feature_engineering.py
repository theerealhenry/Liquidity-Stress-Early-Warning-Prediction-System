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

TARGET = "liquidity_stress_next_30d"
ID_COL = "ID"

MONTHS = ["m1", "m2", "m3", "m4", "m5", "m6"]

TRANSACTION_GROUPS = [
    "deposit", "withdraw", "merchantpay",
    "paybill", "transfer_from_bank",
    "mm_send", "received"
]

VALUE_SUFFIX = "total_value"
EPS = 1e-6


# =========================================================
# MAIN PIPELINE
# =========================================================

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    raw_df = df.copy()
    log_df = _log_transform(df.copy())

    raw_cache = _build_cache(raw_df)
    log_cache = _build_cache(log_df)

    features = {}

    # =========================
    # RAW FEATURES (PRIMARY)
    # =========================
    features.update(_temporal_features(raw_cache))
    features.update(_activity_features(raw_cache))
    features.update(_cashflow_features(raw_cache))
    features.update(_behavioral_ratios(raw_df, raw_cache))
    features.update(_recency_features(raw_cache))
    features.update(_balance_features(raw_df))
    features.update(_volatility_features(raw_cache))
    features.update(_momentum_features(raw_cache))
    features.update(_interaction_features(raw_cache))

    # =========================
    # HIGH-IMPACT FEATURES
    # =========================
    features.update(_cross_group_features(raw_cache))
    features.update(_balance_pressure_features(raw_df, raw_cache))
    features.update(_acceleration_features(raw_cache))
    features.update(_drawdown_features(raw_df, raw_cache))
    features.update(_consistency_features(raw_cache))
    features.update(_activity_switch_features(raw_cache))
    features.update(_peak_intensity_features(raw_cache))
    features.update(_cashflow_volatility(raw_cache))

    # =========================
    # LOG FEATURES (SECONDARY)
    # =========================
    features.update(_temporal_features(log_cache, "_log"))
    features.update(_volatility_features(log_cache, "_log"))
    features.update(_momentum_features(log_cache, "_log"))
    features.update(_recency_features(log_cache, "_log"))

    # =========================
    # ZERO INDICATORS
    # =========================
    features.update(_zero_indicators(raw_df))

    feature_df = pd.DataFrame(features, index=df.index)

    _validate_no_leakage(feature_df)
    feature_df = _clean_features(feature_df)

    print(f"Generated {feature_df.shape[1]} features")

    return pd.concat([df, feature_df], axis=1)


# =========================================================
# CORE UTILS
# =========================================================

def _build_cache(df):
    cache = {}

    for g in TRANSACTION_GROUPS:
        cols = [f"{m}_{g}_{VALUE_SUFFIX}" for m in MONTHS if f"{m}_{g}_{VALUE_SUFFIX}" in df.columns]

        if cols:
            data = df[cols].values.astype(float)

            cache[g] = {
                "data": data,
                "sum": data.sum(axis=1),
                "mean": data.mean(axis=1),
                "std": data.std(axis=1),
                "recent": data[:, 0],   # m1
                "old": data[:, -1],     # m6
            }

    return cache


def _log_transform(df):
    for col in df.columns:
        if "total_value" in col or "highest_amount" in col:
            df[col] = np.log1p(df[col].clip(lower=0))
    return df


def _validate_no_leakage(df):
    leaked = [c for c in df.columns if TARGET in c]
    if leaked:
        raise ValueError(f"🚨 Leakage detected: {leaked}")


def _clean_features(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    nunique = df.nunique()
    df = df.loc[:, nunique > 1]

    return df


# =========================================================
# FEATURE BLOCKS
# =========================================================

def _compute_slope(data):
    x = np.arange(data.shape[1])[::-1]  
    x_mean = x.mean()
    return ((data - data.mean(axis=1, keepdims=True)) * (x - x_mean)).sum(axis=1) / (
        ((x - x_mean) ** 2).sum() + EPS
    )


def _temporal_features(cache, suffix=""):
    features = {}

    for g, v in cache.items():
        data = v["data"]

        features[f"{g}_mean{suffix}"] = v["mean"]
        features[f"{g}_std{suffix}"] = v["std"]

        
        features[f"{g}_trend{suffix}"] = v["recent"] - v["old"]

        features[f"{g}_slope{suffix}"] = _compute_slope(data)

    return features


def _activity_features(cache):
    features = {}
    for g, v in cache.items():
        activity = (v["data"] > 0).astype(int)
        features[f"{g}_active_months"] = activity.sum(axis=1)
    return features


def _activity_switch_features(cache):
    features = {}
    for g, v in cache.items():
        activity = (v["data"] > 0).astype(int)
        features[f"{g}_activity_switch"] = np.abs(np.diff(activity, axis=1)).sum(axis=1)
    return features


def _cashflow_features(cache):
    features = {}
    if "deposit" in cache and "withdraw" in cache:
        d = cache["deposit"]["sum"]
        w = cache["withdraw"]["sum"]

        features["net_cashflow"] = d - w
        features["withdraw_deposit_ratio"] = w / (d + 1)

    return features


def _cashflow_volatility(cache):
    features = {}
    if "deposit" in cache and "withdraw" in cache:
        features["cashflow_volatility"] = (
            cache["deposit"]["std"] - cache["withdraw"]["std"]
        )
    return features


def _behavioral_ratios(df, cache):
    features = {}
    if "arpu" in df.columns and "deposit" in cache:
        features["deposit_intensity"] = cache["deposit"]["sum"] / (df["arpu"] + 1)
    return features


def _recency_features(cache, suffix=""):
    features = {}
    for g, v in cache.items():
        data = v["data"]
        if data.shape[1] >= 3:
            recent = data[:, :3].mean(axis=1)
            past = data[:, 3:].mean(axis=1)
            features[f"{g}_recency_ratio{suffix}"] = recent / (past + 1)
    return features


def _balance_features(df):
    features = {}
    cols = [f"{m}_daily_avg_bal" for m in MONTHS if f"{m}_daily_avg_bal" in df.columns]

    if cols:
        data = df[cols].values

        features["balance_trend"] = data[:, 0] - data[:, -1]  # FIXED
        features["balance_volatility"] = data.std(axis=1)

    return features


def _volatility_features(cache, suffix=""):
    features = {}
    for g, v in cache.items():
        data = v["data"]
        if data.shape[1] >= 4:
            features[f"{g}_volatility_ratio{suffix}"] = (
                data[:, :3].std(axis=1) / (data[:, 3:].std(axis=1) + 1)
            )
    return features


def _momentum_features(cache, suffix=""):
    features = {}
    for g, v in cache.items():
        data = v["data"]
        if data.shape[1] >= 2:
            
            features[f"{g}_momentum{suffix}"] = data[:, 0] - data[:, 1]
    return features


def _acceleration_features(cache):
    features = {}
    for g, v in cache.items():
        data = v["data"]
        if data.shape[1] >= 3:
            accel = (data[:, 0] - data[:, 1]) - (data[:, 1] - data[:, 2])
            features[f"{g}_acceleration"] = accel
    return features


def _interaction_features(cache):
    features = {}
    if "deposit" in cache and "withdraw" in cache:
        d = cache["deposit"]["sum"]
        w = cache["withdraw"]["sum"]
        features["net_flow_ratio"] = (d - w) / (d + w + 1)
    return features


def _cross_group_features(cache):
    features = {}
    if all(k in cache for k in ["deposit", "withdraw", "merchantpay", "paybill", "received"]):
        inflow = cache["deposit"]["sum"] + cache["received"]["sum"]
        spend = cache["withdraw"]["sum"] + cache["merchantpay"]["sum"] + cache["paybill"]["sum"]

        features["spend_to_inflow"] = spend / (inflow + 1)
    return features


def _balance_pressure_features(df, cache):
    features = {}
    if "withdraw" in cache and "m1_daily_avg_bal" in df.columns:
        features["balance_to_spend_pressure"] = (
            df["m1_daily_avg_bal"] / (cache["withdraw"]["sum"] + 1)
        )
    return features


def _drawdown_features(df, cache):
    features = {}

    cols = [f"{m}_daily_avg_bal" for m in MONTHS if f"{m}_daily_avg_bal" in df.columns]

    if cols:
        data = df[cols].values
        peak = data.max(axis=1)
        current = data[:, 0]  # FIXED (m1)

        features["balance_drawdown"] = (peak - current) / (peak + 1)

    return features


def _consistency_features(cache):
    features = {}
    for g, v in cache.items():
        features[f"{g}_consistency"] = 1 / (v["std"] + 1)
    return features


def _peak_intensity_features(cache):
    features = {}
    for g, v in cache.items():
        features[f"{g}_peak_to_mean"] = v["data"].max(axis=1) / (v["mean"] + 1)
    return features


def _zero_indicators(df):
    features = {}
    for col in df.select_dtypes(include=["number"]).columns:
        if col not in [TARGET, ID_COL]:
            ratio = (df[col] == 0).mean()
            if 0.6 < ratio < 0.98:
                features[f"{col}_is_zero"] = (df[col] == 0).astype(np.int8)
    return features


# =========================================================
# SPLIT
# =========================================================

def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET, ID_COL], errors="ignore")
    y = df[TARGET] if TARGET in df.columns else None
    return X, y


def get_feature_list(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in [TARGET, ID_COL]]