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

    # Apply log transform FIRST
    df = _log_transform(df)

    new_features = []

    new_features.append(_handle_zero_inflation(df))
    new_features.append(_create_temporal_features(df))
    new_features.append(_create_activity_features(df))
    new_features.append(_create_cashflow_features(df))
    new_features.append(_create_behavioral_ratios(df))
    new_features.append(_create_recency_features(df))
    new_features.append(_create_balance_features(df))
    new_features.append(_create_interaction_features(df)) 
    new_features.append(_create_volatility_features(df))    
    new_features.append(_create_momentum_features(df))

    feature_df = pd.concat(new_features, axis=1)

    df = pd.concat([df, feature_df], axis=1)

    return df.copy()


# =========================================================
# ZERO-INFLATION
# =========================================================

def _handle_zero_inflation(df):
    features = {}
    numeric_cols = df.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        zero_ratio = (df[col] == 0).mean()

        if zero_ratio > 0.6:
            features[f"{col}_is_zero"] = (df[col] == 0).astype(int)

    return pd.DataFrame(features)


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
# SLOPE
# =========================================================

def _compute_slope(data):
    x = np.arange(data.shape[1])
    x_mean = x.mean()

    slopes = ((data - data.mean(axis=1, keepdims=True)) * (x - x_mean)).sum(axis=1) / (
        ((x - x_mean) ** 2).sum() + EPS
    )

    return slopes


# =========================================================
# TEMPORAL FEATURES
# =========================================================

def _create_temporal_features(df):
    features = {}

    for group in TRANSACTION_GROUPS:

        cols = [
            f"{m}_{group}_{VALUE_SUFFIX}"
            for m in MONTHS
            if f"{m}_{group}_{VALUE_SUFFIX}" in df.columns
        ]

        if len(cols) >= 2:
            data = df[cols].values

            mean = data.mean(axis=1)
            std = data.std(axis=1)

            features[f"{group}_mean"] = mean
            features[f"{group}_std"] = std
            features[f"{group}_min"] = data.min(axis=1)
            features[f"{group}_max"] = data.max(axis=1)

            features[f"{group}_cv"] = std / (mean + EPS)

            # TREND (old → recent)
            features[f"{group}_trend"] = data[:, -1] - data[:, 0]
            features[f"{group}_trend_ratio"] = data[:, -1] / (data[:, 0] + EPS)

            # SLOPE
            features[f"{group}_slope"] = _compute_slope(data)

            # consistency
            features[f"{group}_consistency"] = 1 / (std + EPS)

    return pd.DataFrame(features)


# =========================================================
# ACTIVITY FEATURES
# =========================================================

def _create_activity_features(df):
    features = {}

    for group in TRANSACTION_GROUPS:

        cols = [
            f"{m}_{group}_{VALUE_SUFFIX}"
            for m in MONTHS
            if f"{m}_{group}_{VALUE_SUFFIX}" in df.columns
        ]

        if cols:
            activity = (df[cols] > 0).astype(int)

            features[f"{group}_active_months"] = activity.sum(axis=1)
            features[f"{group}_inactive_months"] = (activity == 0).sum(axis=1)

            # activity change
            features[f"{group}_activity_change"] = activity.iloc[:, 0] - activity.iloc[:, -1]

    return pd.DataFrame(features)


# =========================================================
# CASHFLOW
# =========================================================

def _create_cashflow_features(df):
    features = {}

    deposit_cols = [f"{m}_deposit_{VALUE_SUFFIX}" for m in MONTHS if f"{m}_deposit_{VALUE_SUFFIX}" in df.columns]
    withdraw_cols = [f"{m}_withdraw_{VALUE_SUFFIX}" for m in MONTHS if f"{m}_withdraw_{VALUE_SUFFIX}" in df.columns]

    if deposit_cols and withdraw_cols:
        total_deposit = df[deposit_cols].sum(axis=1)
        total_withdraw = df[withdraw_cols].sum(axis=1)

        features["total_deposit"] = total_deposit
        features["total_withdraw"] = total_withdraw
        features["net_cashflow"] = total_deposit - total_withdraw
        features["withdraw_deposit_ratio"] = total_withdraw / (total_deposit + EPS)

    return pd.DataFrame(features)


# =========================================================
# BEHAVIORAL RATIOS
# =========================================================

def _create_behavioral_ratios(df):
    features = {}

    if "arpu" in df.columns:

        deposit_cols = [f"{m}_deposit_{VALUE_SUFFIX}" for m in MONTHS if f"{m}_deposit_{VALUE_SUFFIX}" in df.columns]
        withdraw_cols = [f"{m}_withdraw_{VALUE_SUFFIX}" for m in MONTHS if f"{m}_withdraw_{VALUE_SUFFIX}" in df.columns]

        if deposit_cols:
            total_deposit = df[deposit_cols].sum(axis=1)
            features["deposit_intensity"] = total_deposit / (df["arpu"] + EPS)

        if withdraw_cols:
            total_withdraw = df[withdraw_cols].sum(axis=1)
            features["withdraw_intensity"] = total_withdraw / (df["arpu"] + EPS)

    return pd.DataFrame(features)


# =========================================================
# RECENCY
# =========================================================

def _create_recency_features(df):
    features = {}

    for group in TRANSACTION_GROUPS:

        m1_col = f"m1_{group}_{VALUE_SUFFIX}"
        history_cols = [
            f"{m}_{group}_{VALUE_SUFFIX}"
            for m in MONTHS[1:]
            if f"{m}_{group}_{VALUE_SUFFIX}" in df.columns
        ]

        if m1_col in df.columns and history_cols:
            features[f"{group}_recency_ratio"] = (
                df[m1_col] / (df[history_cols].mean(axis=1) + EPS)
            )

    return pd.DataFrame(features)


# =========================================================
# BALANCE (FIXED TREND)
# =========================================================

def _create_balance_features(df):
    features = {}

    bal_cols = [f"{m}_daily_avg_bal" for m in MONTHS if f"{m}_daily_avg_bal" in df.columns]

    if len(bal_cols) >= 2:
        data = df[bal_cols].values

        features["balance_mean"] = data.mean(axis=1)
        features["balance_trend"] = data[:, -1] - data[:, 0]
        features["balance_volatility"] = data.std(axis=1)

    return pd.DataFrame(features)


# =========================================================
# INTERACTION FEATURES
# =========================================================

def _create_interaction_features(df):
    features = {}

    if "total_deposit" in df.columns and "withdraw_deposit_ratio" in df.columns:
        features["deposit_withdraw_interaction"] = (
            df["total_deposit"] * df["withdraw_deposit_ratio"]
        )

    return pd.DataFrame(features)


# =========================================================
# VOLATILITY CHANGE FEATURES
# =========================================================

def _create_volatility_features(df):
    features = {}

    for group in TRANSACTION_GROUPS:

        cols = [
            f"{m}_{group}_{VALUE_SUFFIX}"
            for m in MONTHS
            if f"{m}_{group}_{VALUE_SUFFIX}" in df.columns
        ]

        if len(cols) >= 4:
            data = df[cols].values

            recent_std = data[:, :3].std(axis=1)
            past_std = data[:, 3:].std(axis=1)

            features[f"{group}_volatility_ratio"] = recent_std / (past_std + EPS)

    return pd.DataFrame(features)


# =========================================================
# MOMENTUM FEATURES
# =========================================================

def _create_momentum_features(df):
    features = {}

    for group in TRANSACTION_GROUPS:

        m1 = f"m1_{group}_{VALUE_SUFFIX}"
        m2 = f"m2_{group}_{VALUE_SUFFIX}"

        if m1 in df.columns and m2 in df.columns:
            features[f"{group}_momentum"] = df[m1] - df[m2]

    return pd.DataFrame(features)


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