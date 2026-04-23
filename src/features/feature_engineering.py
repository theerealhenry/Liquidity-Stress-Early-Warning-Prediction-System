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
# CONFIGURATION
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
VOLUME_SUFFIX = "volume"

EPS = 1e-6


# =========================================================
# MAIN PIPELINE
# =========================================================

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = _handle_zero_inflation(df)
    df = _log_transform(df)

    df = _create_temporal_aggregations(df)
    df = _create_trend_features(df)
    df = _create_volatility_features(df)

    df = _create_activity_features(df)
    df = _create_cashflow_features(df)
    df = _create_behavioral_ratios(df)

    df = _create_recency_features(df)
    df = _create_balance_features(df)

    return df


# =========================================================
# ZERO-INFLATION
# =========================================================

def _handle_zero_inflation(df):
    numeric_cols = df.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        zero_ratio = (df[col] == 0).mean()

        if zero_ratio > 0.6:
            df[f"{col}_is_zero"] = (df[col] == 0).astype(int)

    return df


# =========================================================
# LOG TRANSFORM
# =========================================================

def _log_transform(df):
    for col in df.columns:
        if "total_value" in col or "highest_amount" in col:
            df[col] = np.log1p(df[col])
    return df


# =========================================================
# TEMPORAL AGGREGATIONS 
# =========================================================

def _create_temporal_aggregations(df):
    for group in TRANSACTION_GROUPS:

        cols = [f"{m}_{group}_{VALUE_SUFFIX}" for m in MONTHS if f"{m}_{group}_{VALUE_SUFFIX}" in df.columns]

        if len(cols) >= 2:
            df[f"{group}_mean"] = df[cols].mean(axis=1)
            df[f"{group}_std"] = df[cols].std(axis=1)
            df[f"{group}_min"] = df[cols].min(axis=1)
            df[f"{group}_max"] = df[cols].max(axis=1)

    return df


# =========================================================
# TREND FEATURES
# =========================================================

def _create_trend_features(df):
    time_index = np.arange(len(MONTHS))

    for group in TRANSACTION_GROUPS:

        cols = [f"{m}_{group}_{VALUE_SUFFIX}" for m in MONTHS if f"{m}_{group}_{VALUE_SUFFIX}" in df.columns]

        if len(cols) >= 2:
            values = df[cols].values

            # Linear trend (slope)
            slope = np.polyfit(time_index, values.T, 1)[0]
            df[f"{group}_slope"] = slope

            # Simple trend
            df[f"{group}_trend"] = df[cols[0]] - df[cols[-1]]

            # Trend ratio
            df[f"{group}_trend_ratio"] = df[cols[0]] / (df[cols[-1]] + EPS)

    return df


# =========================================================
# VOLATILITY
# =========================================================

def _create_volatility_features(df):
    for group in TRANSACTION_GROUPS:

        cols = [f"{m}_{group}_{VALUE_SUFFIX}" for m in MONTHS if f"{m}_{group}_{VALUE_SUFFIX}" in df.columns]

        if len(cols) >= 3:
            mean = df[cols].mean(axis=1)
            std = df[cols].std(axis=1)

            df[f"{group}_cv"] = std / (mean + EPS)

    return df


# =========================================================
# ACTIVITY FEATURES
# =========================================================

def _create_activity_features(df):
    for group in TRANSACTION_GROUPS:

        cols = [f"{m}_{group}_{VALUE_SUFFIX}" for m in MONTHS if f"{m}_{group}_{VALUE_SUFFIX}" in df.columns]

        if cols:
            activity = (df[cols] > 0).astype(int)

            df[f"{group}_active_months"] = activity.sum(axis=1)
            df[f"{group}_inactive_months"] = (activity == 0).sum(axis=1)

    return df


# =========================================================
# CASHFLOW FEATURES
# =========================================================

def _create_cashflow_features(df):
    deposit_cols = [f"{m}_deposit_{VALUE_SUFFIX}" for m in MONTHS if f"{m}_deposit_{VALUE_SUFFIX}" in df.columns]
    withdraw_cols = [f"{m}_withdraw_{VALUE_SUFFIX}" for m in MONTHS if f"{m}_withdraw_{VALUE_SUFFIX}" in df.columns]

    if deposit_cols and withdraw_cols:
        df["total_deposit"] = df[deposit_cols].sum(axis=1)
        df["total_withdraw"] = df[withdraw_cols].sum(axis=1)

        df["net_cashflow"] = df["total_deposit"] - df["total_withdraw"]
        df["withdraw_deposit_ratio"] = df["total_withdraw"] / (df["total_deposit"] + EPS)

    return df


# =========================================================
# BEHAVIORAL RATIOS
# =========================================================

def _create_behavioral_ratios(df):

    if "total_deposit" in df.columns and "total_withdraw" in df.columns:
        df["cash_pressure"] = df["total_withdraw"] / (df["total_deposit"] + EPS)

    if "total_deposit" in df.columns:
        df["deposit_intensity"] = df["total_deposit"] / (df["arpu"] + EPS)

    if "total_withdraw" in df.columns:
        df["withdraw_intensity"] = df["total_withdraw"] / (df["arpu"] + EPS)

    return df


# =========================================================
# RECENCY FEATURES
# =========================================================

def _create_recency_features(df):
    for group in TRANSACTION_GROUPS:

        m1_col = f"m1_{group}_{VALUE_SUFFIX}"
        history_cols = [f"{m}_{group}_{VALUE_SUFFIX}" for m in MONTHS[1:] if f"{m}_{group}_{VALUE_SUFFIX}" in df.columns]

        if m1_col in df.columns and history_cols:
            df[f"{group}_recency_ratio"] = df[m1_col] / (df[history_cols].mean(axis=1) + EPS)

    return df


# =========================================================
# BALANCE FEATURES
# =========================================================

def _create_balance_features(df):
    bal_cols = [f"{m}_daily_avg_bal" for m in MONTHS if f"{m}_daily_avg_bal" in df.columns]

    if len(bal_cols) >= 2:
        df["balance_mean"] = df[bal_cols].mean(axis=1)
        df["balance_trend"] = df[bal_cols[0]] - df[bal_cols[-1]]
        df["balance_volatility"] = df[bal_cols].std(axis=1)

    return df


# =========================================================
# FINAL SPLIT
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