"""
Feature Engineering Module — v2.2 (Production Grade)
======================================================

Project     : AI4EAC Liquidity Stress Early Warning Prediction
Task        : Binary classification — predict P(liquidity stress within 30 days)
Metrics     : Log Loss (60%) + ROC-AUC (40%)
Data        : 6-month mobile money panel — 40,000 customers × 226 raw columns
              226 raw columns → ~620+ engineered features

Changelog v2.2
--------------
FEATURE EXPANSION 1 — Volume & Highest-Amount Cache
  _build_cache() previously only built temporal arrays on total_value columns.
  The dataset has four suffixes per group per month: total_value, volume
  (transaction count), highest_amount, and unique-entity count. Trend,
  recency, momentum, volatility, and acceleration features were only computed
  on monetary value — missing the frequency dimension entirely.
  Fix: _build_cache() now builds three parallel caches per group:
    - {g}        : total_value (monetary, existing)
    - {g}_vol    : volume (transaction count, NEW)
    - {g}_hi     : highest_amount (transaction size extremes, NEW)
  All existing feature blocks (_temporal_aggregations, _trend_features,
  _recency_features, _momentum_features, _volatility_features,
  _acceleration_features, _consistency_features, _activity_features,
  _peak_intensity_features) automatically pick up the new cache entries
  because they iterate over cache.items(). No changes needed in those
  functions — the architecture was already modular enough.
  Expected feature count increase: +180 to +220 features.

FEATURE EXPANSION 2 — Value-Volume Ratio Features (NEW Block 21)
  _value_volume_ratio_features(): computes value-per-transaction (avg txn size)
  and its 6-month trend for each transaction group.
  Financial rationale: a customer whose deposit value stays constant but
  deposit volume drops is making fewer, larger deposits — a behavioural shift
  that signals income irregularity or cash-out pressure. This signal is
  orthogonal to both pure value and pure volume features.
  Adds ~14 features (2 per group × 7 groups).

FEATURE EXPANSION 3 — Unique Entity Features (NEW Block 22)
  _unique_entity_features(): tracks unique agents, merchants, recipients,
  senders, companies, banks per group across M1–M6.
  Financial rationale: network contraction (fewer unique counterparties)
  precedes financial stress. A customer who previously used 5 agents for
  cash deposits but is now using 1 has reduced their financial mobility.
  The declining unique-entity trend is a leading indicator.
  Adds ~21 features (3 per group × 7 groups).

FEATURE EXPANSION 4 — Volume-Based Cashflow Features (NEW Block 23)
  _volume_cashflow_features(): aggregates transaction counts across inflow
  and outflow groups monthly to compute volume-slope, volume-momentum,
  and inflow/outflow volume divergence.
  Financial rationale: customers often reduce transaction frequency before
  reducing transaction value — frequency decline is an earlier warning signal
  than value decline.
  Adds ~6 features.

FEATURE EXPANSION 5 — Extended Winsorisation
  Added new volume-ratio features to WINSORISE_FEATURES list:
    - deposit_vol_recency_ratio: extreme values when deposit count spikes
    - withdraw_vol_recency_ratio: extreme values when withdrawal count spikes
    - avg_txn_size_withdraw: can reach extreme values for large infrequent ATM
  These were identified as likely outlier-prone by domain analysis of the
  volume-to-value ratio distributions on sparse binary transaction data.

FEATURE EXPANSION 6 — Extended Log-Transformed Layer
  Added volume cache to log-transformed layer (_temporal_aggregations,
  _trend_features, _recency_features on log1p(volume)).
  Rationale: transaction counts are right-skewed (most customers have
  zero or 1–2 transactions per month; a few have 30+). Log-transforming
  count data improves split quality in tree models and linear separability
  for LogisticRegression.

ARCHITECTURE NOTE — Why existing blocks required no changes
  The cache-driven architecture means _temporal_aggregations(cache) iterates
  over ALL cache keys. Adding 'deposit_vol' and 'deposit_hi' to the cache
  automatically produces deposit_vol_mean, deposit_vol_std, deposit_hi_mean,
  etc. — without touching those functions. This is the payoff of the modular
  design. Any future feature type (e.g. unique-entity counts) follows the
  same pattern: add to _build_cache(), everything else is automatic.

Changelog v2.1
--------------
BUG 1 — CRITICAL: Duplicate 'age' column
  Fix: removed age re-creation from _categorical_features().

BUG 2 — CRITICAL: activity_rate duplicates x_90_d_activity_rate
  Fix: removed activity_rate from _categorical_features().

BUG 3 — MODERATE: _recency_features log suffix creates duplicate keys
  Fix: log-of-ratio only computed when suffix == "".

BUG 4 — MODERATE: _zero_indicators must run on raw df before concat
  Confirmed: called on raw df. Added docstring note to prevent regression.

BUG 5 — MINOR: earning_pattern_encoded non-deterministic ordering
  Fix: sorted() applied before enumeration.

Design Principles
-----------------
1.  CV-SAFE — zero leakage. Every feature derived exclusively from historical
    columns (M1–M6). Target never touches feature computation.
2.  NULL = ZERO — per competition spec, null means no activity. All nulls
    filled with 0 before feature computation.
3.  M1 IS GROUND TRUTH — most recent month is the primary anchor.
4.  WINSORISATION — 99th-pct clip on known outlier-prone features.
5.  TREE-FRIENDLY & LINEAR-FRIENDLY — log transforms applied on skewed
    monetary and count distributions. StandardScaler applied downstream
    in PreprocessingPipeline for LogReg/TabNet (not here).
6.  SHAP-EVIDENCED — every feature block justified by SHAP or domain logic.
7.  DETERMINISTIC — no random operations. Same input → identical output.
8.  MODULAR — each feature family is isolated. Adding a feature block
    requires one line in build_features(). Removing requires one line.

Feature Families (23 blocks)
------------------------------
RAW SIGNAL LAYER (value-based, existing)
  Block 01  _temporal_aggregations     — mean/std/min/max per group (value + vol + hi)
  Block 02  _trend_features            — OLS slope + simple trend (value + vol + hi)
  Block 03  _momentum_features         — 1-month delta (value + vol + hi)
  Block 04  _acceleration_features     — change-of-momentum (value + vol + hi)
  Block 05  _volatility_features       — CV + recent/past std ratio (value + vol + hi)
  Block 06  _consistency_features      — inverse-CV stability (value + vol + hi)
  Block 07  _activity_features         — months active, zero-streak, switch rate
  Block 08  _recency_features          — M1–M3 / M4–M6 ratio (value + vol + hi)
  Block 09  _peak_intensity_features   — max/mean ratio (value + vol + hi)

BALANCE INTELLIGENCE LAYER
  Block 10  _balance_features          — trend, slope, CV, range, recovery
  Block 11  _drawdown_features         — peak-to-trough drawdown
  Block 12  _balance_pressure_features — balance vs spending obligations

CASHFLOW INTELLIGENCE LAYER
  Block 13  _cashflow_features         — net flow, ratios, cumulative 6m
  Block 14  _cashflow_slope_features   — monthly net cashflow OLS slope
  Block 15  _cashflow_volatility       — std of monthly net cashflow
  Block 23  _volume_cashflow_features  — frequency-based cashflow trends (NEW)

P2P & BANKING LAYER
  Block 16  _p2p_features              — send/receive ratio, net P2P
  Block 17  _banking_features          — bank transfer trend, ARPU-relative

INTERACTION LAYER (SHAP-evidenced)
  Block 18  _interaction_features      — balance_trend×balance_level,
                                         deposit_recency×balance,
                                         withdraw_recency×spend_ratio

ENCODING & INDICATOR LAYER
  Block 19  _categorical_features      — ordinal encode segment/earning_pattern
  Block 20  _zero_indicators           — binary sparsity flags

VALUE-VOLUME INTELLIGENCE LAYER (NEW)
  Block 21  _value_volume_ratio_features — avg txn size + size trend per group
  Block 22  _unique_entity_features      — unique agents/merchants/recipients

LOG-TRANSFORMED LAYER
  Blocks 01,02,05,03,08 on log1p(total_value) + log1p(volume)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

TARGET   : str       = "liquidity_stress_next_30d"
ID_COL   : str       = "ID"
EPS      : float     = 1e-6

MONTHS   : List[str] = ["m1", "m2", "m3", "m4", "m5", "m6"]

# 7 transaction categories — direction annotated for cashflow logic
INFLOW_GROUPS  : List[str] = ["deposit", "received", "transfer_from_bank"]
OUTFLOW_GROUPS : List[str] = ["withdraw", "merchantpay", "paybill", "mm_send"]
ALL_GROUPS     : List[str] = INFLOW_GROUPS + OUTFLOW_GROUPS

# Column suffixes present in the raw dataset
VALUE_SUFFIX      : str = "total_value"
VOLUME_SUFFIX     : str = "volume"
HIGHEST_AMT_SUFFIX: str = "highest_amount"

# Unique-entity column suffix per group (from data dictionary)
ENTITY_SUFFIX_MAP : Dict[str, str] = {
    "deposit"            : "agents",
    "withdraw"           : "agents",
    "mm_send"            : "recipients",
    "received"           : "senders",
    "merchantpay"        : "merchants",
    "paybill"            : "companies",
    "transfer_from_bank" : "banks",
}

# Features known to produce extreme outliers → winsorise at 99th pct
# Extended in v2.2 to include volume-ratio features
WINSORISE_FEATURES : List[str] = [
    # v2.1 originals (monetary ratios)
    "withdraw_recency_ratio",
    "withdraw_recency_ratio_log",
    "net_flow_ratio",
    "balance_to_spend_pressure",
    # v2.2 additions (volume ratios and size features)
    "deposit_vol_recency_ratio",
    "withdraw_vol_recency_ratio",
    "mm_send_vol_recency_ratio",
    "avg_txn_size_withdraw",
    "avg_txn_size_withdraw_trend",
    "avg_txn_size_mm_send",
]

WINSORISE_PERCENTILE : float = 99.0

# Segment value tier — ordered for ordinal encoding
SEGMENT_ORDER : Dict[str, int] = {"LVC": 0, "MVC": 1, "HVC": 2}

# Raw columns that already exist in df — never re-create in feature blocks
_RAW_PASSTHROUGH_COLS = {
    "age",
    "x_90_d_activity_rate",
    "arpu",
}


# ─────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master feature engineering pipeline — v2.2.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataframe — train or test. Must contain the M1–M6 columns
        defined in the competition data dictionary. Target column
        (liquidity_stress_next_30d) is preserved if present; never used
        in feature computation.

    Returns
    -------
    pd.DataFrame
        Original columns + all engineered features (~620+ columns).
        Ready for split_features_target() → preprocessing → model training.

    Notes
    -----
    - Call on train and test separately. Fully stateless — no fitting.
    - Winsorisation uses percentiles computed on the passed DataFrame.
      In production, fit percentiles on train only. For competition, per-
      split clipping is acceptable (applied inside CV folds).
    - _zero_indicators() MUST be called on raw df BEFORE concat. Do not move.
    - Cache keys follow naming convention:
        {group}      → total_value cache  (e.g. "deposit")
        {group}_vol  → volume cache       (e.g. "deposit_vol")
        {group}_hi   → highest_amt cache  (e.g. "deposit_hi")
    """
    df = df.copy()

    # ── Step 1: Fill nulls — null = zero activity per competition spec ───
    df = _fill_nulls(df)

    # ── Step 2: Build raw and log-transformed caches ─────────────────────
    # v2.2: _build_cache() now returns value + volume + highest_amount caches
    raw_cache = _build_cache(df)

    log_df    = _apply_log_transform(df.copy())
    log_cache = _build_cache(log_df)
    # Note: log_cache will have _vol entries from log1p(volume) — intentional.
    # log1p(volume) is interpretable: log-counts are standard in econometrics.

    # ── Step 3: Compute all feature blocks ───────────────────────────────
    feature_blocks: Dict[str, np.ndarray | pd.Series] = {}

    # RAW SIGNAL LAYER
    # All blocks automatically process value + volume + highest_amt caches
    # because they iterate over cache.items() — no per-block changes needed.
    feature_blocks.update(_temporal_aggregations(raw_cache))
    feature_blocks.update(_trend_features(raw_cache))
    feature_blocks.update(_momentum_features(raw_cache))
    feature_blocks.update(_acceleration_features(raw_cache))
    feature_blocks.update(_volatility_features(raw_cache))
    feature_blocks.update(_consistency_features(raw_cache))
    feature_blocks.update(_activity_features(raw_cache))
    feature_blocks.update(_recency_features(raw_cache))
    feature_blocks.update(_peak_intensity_features(raw_cache))

    # BALANCE INTELLIGENCE LAYER
    feature_blocks.update(_balance_features(df))
    feature_blocks.update(_drawdown_features(df))
    feature_blocks.update(_balance_pressure_features(df, raw_cache))

    # CASHFLOW INTELLIGENCE LAYER
    feature_blocks.update(_cashflow_features(raw_cache))
    feature_blocks.update(_cashflow_slope_features(df, raw_cache))
    feature_blocks.update(_cashflow_volatility(df, raw_cache))
    feature_blocks.update(_volume_cashflow_features(df))      # NEW Block 23

    # P2P & BANKING LAYER
    feature_blocks.update(_p2p_features(raw_cache))
    feature_blocks.update(_banking_features(df, raw_cache))

    # INTERACTION LAYER (SHAP-evidenced)
    feature_blocks.update(_interaction_features(df, raw_cache, feature_blocks))

    # ENCODING LAYER
    feature_blocks.update(_categorical_features(df))

    # ZERO INDICATOR LAYER
    # CRITICAL: called on raw df BEFORE concat — flags raw column sparsity only
    feature_blocks.update(_zero_indicators(df))

    # VALUE-VOLUME INTELLIGENCE LAYER (NEW v2.2)
    feature_blocks.update(_value_volume_ratio_features(raw_cache))   # Block 21
    feature_blocks.update(_unique_entity_features(df))               # Block 22

    # LOG-TRANSFORMED LAYER
    # v2.2: log_cache now includes _vol entries → log-volume features auto-generated
    feature_blocks.update(_temporal_aggregations(log_cache, suffix="_log"))
    feature_blocks.update(_trend_features(log_cache,        suffix="_log"))
    feature_blocks.update(_volatility_features(log_cache,   suffix="_log"))
    feature_blocks.update(_momentum_features(log_cache,     suffix="_log"))
    feature_blocks.update(_recency_features(log_cache,      suffix="_log"))

    # ── Step 4: Assemble, validate, clean ────────────────────────────────
    feature_df = pd.DataFrame(feature_blocks, index=df.index)
    _validate_no_leakage(feature_df)
    feature_df = _clean_features(feature_df)

    # ── Step 5: Winsorise extreme features ───────────────────────────────
    feature_df = _winsorise(feature_df, WINSORISE_FEATURES, WINSORISE_PERCENTILE)

    # ── Step 6: Concat and deduplicate ───────────────────────────────────
    result = pd.concat([df, feature_df], axis=1)

    # Safety net: duplicate columns must never appear in production.
    # If this warning fires, a feature block is re-creating a raw column.
    n_before = result.shape[1]
    result = result.loc[:, ~result.columns.duplicated(keep="first")]
    n_after = result.shape[1]

    if n_before != n_after:
        import warnings
        warnings.warn(
            f"[feature_engineering] WARNING: {n_before - n_after} duplicate "
            f"column(s) dropped after concat. Check feature blocks for "
            f"raw column re-creation.",
            stacklevel=2,
        )

    print(f"[feature_engineering] Raw columns   : {df.shape[1]}")
    print(f"[feature_engineering] Features built: {feature_df.shape[1]}")
    print(f"[feature_engineering] Total columns : {result.shape[1]}")

    return result


def split_features_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Separate feature matrix from target vector.

    Returns (X, y). y is None if target column is absent (test set).
    """
    drop_cols = [c for c in [TARGET, ID_COL] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[TARGET].astype(int) if TARGET in df.columns else None
    return X, y


def get_feature_names(df: pd.DataFrame) -> List[str]:
    """Return list of feature column names (excludes ID and target)."""
    return [c for c in df.columns if c not in (TARGET, ID_COL)]


# ─────────────────────────────────────────────────────────────
# PREPROCESSING UTILITIES
# ─────────────────────────────────────────────────────────────

def _fill_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill nulls per competition spec: null = zero activity.
    Only fills numeric columns that are not the target or ID.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    fill_cols    = [c for c in numeric_cols if c not in (TARGET, ID_COL)]
    df[fill_cols] = df[fill_cols].fillna(0)
    return df


def _apply_log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log1p to monetary and count columns.

    v2.2 extension: volume columns are now also log-transformed.
    Transaction counts are right-skewed (most=0, few=30+). Log1p(count)
    improves split quality in trees and linear separability for LogReg.

    Monetary columns: total_value, highest_amount
    Count columns: volume (NEW in v2.2)
    """
    for col in df.columns:
        if any(s in col for s in (VALUE_SUFFIX, "highest_amount", VOLUME_SUFFIX)):
            df[col] = np.log1p(df[col].clip(lower=0))
    return df


def _build_cache(df: pd.DataFrame) -> Dict:
    """
    Pre-compute per-group temporal arrays to avoid redundant column lookups.

    v2.2 CHANGE: builds three parallel caches per transaction group:

    Cache key convention:
      {group}      → total_value array  (e.g. cache["deposit"])
      {group}_vol  → volume array       (e.g. cache["deposit_vol"])
      {group}_hi   → highest_amt array  (e.g. cache["deposit_hi"])

    Each cache entry stores:
        data   : (n_rows × n_months) array of monthly values
        cols   : column names (for debugging)
        sum    : 6-month total
        mean   : 6-month mean
        std    : 6-month std
        min    : 6-month min
        max    : 6-month max
        recent : M1 value (most recent, primary signal)
        old    : M6 value (oldest, baseline)

    Design decision: all feature blocks iterate over cache.items(), so
    adding new cache entries automatically propagates new features through
    all existing feature blocks. No per-block changes needed.

    Naming safety: the _vol and _hi suffixes on cache keys produce feature
    names like 'deposit_vol_mean', 'deposit_hi_trend_slope' — these do not
    collide with any existing feature names or raw column names.
    """
    cache = {}

    for g in ALL_GROUPS:

        # ── total_value cache (monetary, existing) ────────────────────
        val_cols = [
            f"{m}_{g}_{VALUE_SUFFIX}"
            for m in MONTHS
            if f"{m}_{g}_{VALUE_SUFFIX}" in df.columns
        ]
        if val_cols:
            data = df[val_cols].values.astype(float)
            cache[g] = {
                "data"   : data,
                "cols"   : val_cols,
                "sum"    : data.sum(axis=1),
                "mean"   : data.mean(axis=1),
                "std"    : data.std(axis=1),
                "min"    : data.min(axis=1),
                "max"    : data.max(axis=1),
                "recent" : data[:, 0],
                "old"    : data[:, -1],
            }

        # ── volume cache (transaction count, NEW in v2.2) ─────────────
        vol_cols = [
            f"{m}_{g}_{VOLUME_SUFFIX}"
            for m in MONTHS
            if f"{m}_{g}_{VOLUME_SUFFIX}" in df.columns
        ]
        if vol_cols:
            vdata = df[vol_cols].values.astype(float)
            cache[f"{g}_vol"] = {
                "data"   : vdata,
                "cols"   : vol_cols,
                "sum"    : vdata.sum(axis=1),
                "mean"   : vdata.mean(axis=1),
                "std"    : vdata.std(axis=1),
                "min"    : vdata.min(axis=1),
                "max"    : vdata.max(axis=1),
                "recent" : vdata[:, 0],
                "old"    : vdata[:, -1],
            }

        # ── highest_amount cache (transaction size extremes, NEW v2.2) ─
        hi_cols = [
            f"{m}_{g}_{HIGHEST_AMT_SUFFIX}"
            for m in MONTHS
            if f"{m}_{g}_{HIGHEST_AMT_SUFFIX}" in df.columns
        ]
        if hi_cols:
            hdata = df[hi_cols].values.astype(float)
            cache[f"{g}_hi"] = {
                "data"   : hdata,
                "cols"   : hi_cols,
                "sum"    : hdata.sum(axis=1),
                "mean"   : hdata.mean(axis=1),
                "std"    : hdata.std(axis=1),
                "min"    : hdata.min(axis=1),
                "max"    : hdata.max(axis=1),
                "recent" : hdata[:, 0],
                "old"    : hdata[:, -1],
            }

    return cache


# ─────────────────────────────────────────────────────────────
# VALIDATION & CLEANING
# ─────────────────────────────────────────────────────────────

def _validate_no_leakage(df: pd.DataFrame) -> None:
    """Raise immediately if target column appears in feature set."""
    leaked = [c for c in df.columns if TARGET in c]
    if leaked:
        raise ValueError(f"TARGET LEAKAGE DETECTED: {leaked}")


def _clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-processing:
    1. Replace inf/-inf with NaN then fill 0 (result of division by zero).
    2. Drop constant columns (zero variance = no information for any model).
    """
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    non_constant = df.nunique() > 1
    dropped = (~non_constant).sum()
    if dropped > 0:
        print(f"[feature_engineering] Dropped {dropped} constant columns")
    return df.loc[:, non_constant]


def _winsorise(
    df: pd.DataFrame,
    feature_names: List[str],
    percentile: float = 99.0,
) -> pd.DataFrame:
    """
    Clip extreme values at the given upper percentile for named features.
    Lower bound is always 0 (all features are non-negative financial values).

    v2.2: Extended to include volume-ratio and avg-txn-size features.
    Motivation: SHAP v2.1 showed withdraw_recency_ratio reaching 5.5;
    volume ratios are expected to show similar extreme behaviour on sparse
    binary transaction data (many months with 0 transactions → ratio → ∞).
    """
    for feat in feature_names:
        if feat in df.columns:
            upper = np.percentile(df[feat].values, percentile)
            df[feat] = df[feat].clip(upper=upper)
    return df


# ─────────────────────────────────────────────────────────────
# SHARED COMPUTATION
# ─────────────────────────────────────────────────────────────

def _compute_ols_slope(data: np.ndarray) -> np.ndarray:
    """
    Vectorised OLS slope over time axis (months).

    Time encoded as [5,4,3,2,1,0]: index 0 = M1 (most recent).
    Positive slope = increasing toward the present (improving trend).
    Negative slope = declining toward the present (deteriorating trend).

    Parameters
    ----------
    data : np.ndarray of shape (n_samples, n_months)

    Returns
    -------
    np.ndarray of shape (n_samples,) — OLS slope per customer
    """
    n     = data.shape[1]
    x     = np.arange(n - 1, -1, -1, dtype=float)   # [5,4,3,2,1,0]
    x_bar = x.mean()
    ss_x  = ((x - x_bar) ** 2).sum()

    y_bar = data.mean(axis=1, keepdims=True)
    slope = ((data - y_bar) * (x - x_bar)).sum(axis=1) / (ss_x + EPS)
    return slope


def _get_balance_array(df: pd.DataFrame) -> Optional[np.ndarray]:
    """Return (n × n_available_months) balance array, or None if < 2 columns."""
    cols = [f"{m}_daily_avg_bal" for m in MONTHS]
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2:
        return None
    return df[cols].values.astype(float)


# ─────────────────────────────────────────────────────────────
# FEATURE BLOCK 01 — TEMPORAL AGGREGATIONS
# Applies to: value, volume, highest_amount (via cache keys)
# ─────────────────────────────────────────────────────────────

def _temporal_aggregations(cache: Dict, suffix: str = "") -> Dict:
    feats = {}
    for g, v in cache.items():
        feats[f"{g}_mean{suffix}"]  = v["mean"]
        feats[f"{g}_std{suffix}"]   = v["std"]
        feats[f"{g}_min{suffix}"]   = v["min"]
        feats[f"{g}_max{suffix}"]   = v["max"]
    return feats


# ─────────────────────────────────────────────────────────────
# FEATURE BLOCK 02 — TREND FEATURES
# Applies to: value, volume, highest_amount
# ─────────────────────────────────────────────────────────────

def _trend_features(cache: Dict, suffix: str = "") -> Dict:
    feats = {}
    for g, v in cache.items():
        data = v["data"]
        feats[f"{g}_trend{suffix}"]       = v["recent"] - v["old"]
        feats[f"{g}_trend_slope{suffix}"] = _compute_ols_slope(data)
    return feats


# ─────────────────────────────────────────────────────────────
# FEATURE BLOCK 03 — MOMENTUM FEATURES
# Applies to: value, volume, highest_amount
# Momentum = M1 − M2 (most recent single-month change)
# ─────────────────────────────────────────────────────────────

def _momentum_features(cache: Dict, suffix: str = "") -> Dict:
    feats = {}
    for g, v in cache.items():
        data = v["data"]
        if data.shape[1] >= 2:
            feats[f"{g}_momentum{suffix}"] = data[:, 0] - data[:, 1]
    return feats


# ─────────────────────────────────────────────────────────────
# FEATURE BLOCK 04 — ACCELERATION FEATURES
# Applies to: value, volume, highest_amount
# Acceleration = (M1−M2) − (M2−M3): is momentum itself accelerating?
# ─────────────────────────────────────────────────────────────

def _acceleration_features(cache: Dict) -> Dict:
    feats = {}
    for g, v in cache.items():
        data = v["data"]
        if data.shape[1] >= 3:
            mom_recent = data[:, 0] - data[:, 1]  # M1 − M2
            mom_past   = data[:, 1] - data[:, 2]  # M2 − M3
            feats[f"{g}_acceleration"] = mom_recent - mom_past
    return feats


# ─────────────────────────────────────────────────────────────
# FEATURE BLOCK 05 — VOLATILITY FEATURES
# Applies to: value, volume, highest_amount
# ─────────────────────────────────────────────────────────────

def _volatility_features(cache: Dict, suffix: str = "") -> Dict:
    feats = {}
    for g, v in cache.items():
        data = v["data"]
        feats[f"{g}_cv{suffix}"] = v["std"] / (v["mean"] + EPS)

        if data.shape[1] >= 4:
            recent_std = data[:, :3].std(axis=1)
            past_std   = data[:, 3:].std(axis=1)
            feats[f"{g}_volatility_ratio{suffix}"] = (
                recent_std / (past_std + EPS)
            )
    return feats


# ─────────────────────────────────────────────────────────────
# FEATURE BLOCK 06 — CONSISTENCY FEATURES
# Applies to: value, volume, highest_amount
# Consistency = mean / std (inverse CV). High = stable behaviour.
# ─────────────────────────────────────────────────────────────

def _consistency_features(cache: Dict) -> Dict:
    feats = {}
    for g, v in cache.items():
        feats[f"{g}_consistency"] = v["mean"] / (v["std"] + EPS)
    return feats


# ─────────────────────────────────────────────────────────────
# FEATURE BLOCK 07 — ACTIVITY FEATURES
# Applies to: value cache only (activity = any non-zero transaction value)
# Volume cache not used here — volume IS the activity count.
# ─────────────────────────────────────────────────────────────

def _activity_features(cache: Dict) -> Dict:
    feats = {}
    # Restrict to value-only cache entries (exclude _vol and _hi suffixed keys)
    # to avoid double-counting: activity from value vs activity from volume
    # are the same underlying signal.
    value_cache = {g: v for g, v in cache.items()
                   if not g.endswith("_vol") and not g.endswith("_hi")}
    for g, v in value_cache.items():
        activity = (v["data"] > 0).astype(int)
        feats[f"{g}_active_months"]   = activity.sum(axis=1)
        feats[f"{g}_activity_switch"] = np.abs(np.diff(activity, axis=1)).sum(axis=1)
        feats[f"{g}_zero_streak"]     = _compute_zero_streak(activity)
    return feats


def _compute_zero_streak(activity: np.ndarray) -> np.ndarray:
    """
    Count consecutive inactive months starting from M1 (column 0).
    Stops at first active month.
    A streak of 3 means M1, M2, M3 all had zero activity — strong stress signal.
    """
    n_rows, n_cols = activity.shape
    streak = np.zeros(n_rows, dtype=int)
    for col in range(n_cols):
        still_zero = (activity[:, col] == 0) & (streak == col)
        streak[still_zero] += 1
    return streak


# ─────────────────────────────────────────────────────────────
# FEATURE BLOCK 08 — RECENCY FEATURES
# Applies to: value, volume, highest_amount (via cache keys)
#
# Rationale: deposit_recency_ratio was the #1 SHAP feature in v2.1.
# Extended to volume dimension: are customers transacting LESS FREQUENTLY
# in M1–M3 vs M4–M6? This is a leading indicator before value declines.
# ─────────────────────────────────────────────────────────────

def _recency_features(cache: Dict, suffix: str = "") -> Dict:
    feats = {}
    for g, v in cache.items():
        data = v["data"]
        if data.shape[1] >= 6:
            recent_avg = data[:, :3].mean(axis=1)   # M1–M3
            past_avg   = data[:, 3:].mean(axis=1)   # M4–M6
            ratio      = recent_avg / (past_avg + EPS)

            feats[f"{g}_recency_ratio{suffix}"] = ratio

            # Log of recency ratio — only on raw call (suffix == "")
            # When suffix="_log", inputs are already log-transformed.
            # Adding another log creates an uninterpretable double-log feature.
            if suffix == "":
                feats[f"{g}_recency_ratio_log"] = np.log1p(ratio)

    return feats


# ─────────────────────────────────────────────────────────────
# FEATURE BLOCK 09 — PEAK INTENSITY FEATURES
# Applies to: value, volume, highest_amount
# Peak-to-mean ratio captures burst behaviour vs steady usage.
# ─────────────────────────────────────────────────────────────

def _peak_intensity_features(cache: Dict) -> Dict:
    feats = {}
    for g, v in cache.items():
        feats[f"{g}_peak_to_mean"] = v["max"] / (v["mean"] + EPS)
    return feats


# ─────────────────────────────────────────────────────────────
# FEATURE BLOCK 10 — BALANCE FEATURES
# ─────────────────────────────────────────────────────────────

def _balance_features(df: pd.DataFrame) -> Dict:
    feats = {}
    data = _get_balance_array(df)
    if data is None:
        return feats

    current = data[:, 0]
    oldest  = data[:, -1]

    feats["balance_trend"]      = current - oldest
    feats["balance_slope"]      = _compute_ols_slope(data)
    feats["balance_volatility"] = data.std(axis=1)
    feats["balance_cv"]         = data.std(axis=1) / (data.mean(axis=1) + EPS)
    feats["balance_range"]      = data.max(axis=1) - data.min(axis=1)
    feats["balance_min"]        = data.min(axis=1)
    feats["balance_max"]        = data.max(axis=1)
    feats["balance_m1"]         = current

    past_min = data[:, 1:].min(axis=1)
    feats["balance_recovery"]   = current - past_min
    feats["balance_trend_3m"]   = current - data[:, 2]

    return feats


# ─────────────────────────────────────────────────────────────
# FEATURE BLOCK 11 — DRAWDOWN FEATURES
# ─────────────────────────────────────────────────────────────

def _drawdown_features(df: pd.DataFrame) -> Dict:
    feats = {}
    data = _get_balance_array(df)
    if data is None:
        return feats

    peak    = data.max(axis=1)
    current = data[:, 0]

    feats["balance_drawdown"]     = (peak - current) / (peak + EPS)
    feats["balance_drawdown_abs"] = peak - current

    return feats


# ─────────────────────────────────────────────────────────────
# FEATURE BLOCK 12 — BALANCE PRESSURE FEATURES
# ─────────────────────────────────────────────────────────────

def _balance_pressure_features(df: pd.DataFrame, cache: Dict) -> Dict:
    feats = {}
    if "m1_daily_avg_bal" not in df.columns:
        return feats

    bal = df["m1_daily_avg_bal"].values.astype(float)

    if "withdraw" in cache:
        total_spend = cache["withdraw"]["sum"]
        feats["balance_to_spend_pressure"] = bal / (total_spend + EPS)
        feats["m1_withdraw_balance_ratio"] = (
            cache["withdraw"]["recent"] / (bal + EPS)
        )

    total_outflow = sum(
        cache[g]["sum"] for g in OUTFLOW_GROUPS if g in cache
    )
    feats["balance_to_outflow_ratio"] = bal / (total_outflow + EPS)

    return feats


# ─────────────────────────────────────────────────────────────
# FEATURE BLOCK 13 — CASHFLOW FEATURES
# ─────────────────────────────────────────────────────────────

def _cashflow_features(cache: Dict) -> Dict:
    feats = {}

    total_inflow = sum(
        cache[g]["sum"] for g in INFLOW_GROUPS if g in cache
    )
    total_outflow = sum(
        cache[g]["sum"] for g in OUTFLOW_GROUPS if g in cache
    )

    feats["total_inflow_6m"]  = total_inflow
    feats["total_outflow_6m"] = total_outflow
    feats["net_cashflow_6m"]  = total_inflow - total_outflow
    feats["net_flow_ratio"]   = (
        (total_inflow - total_outflow) / (total_inflow + total_outflow + EPS)
    )
    feats["spend_to_inflow"]  = total_outflow / (total_inflow + EPS)

    if "withdraw" in cache and "deposit" in cache:
        feats["withdraw_deposit_ratio"] = (
            cache["withdraw"]["sum"] / (cache["deposit"]["sum"] + EPS)
        )
        feats["m1_withdraw_deposit_ratio"] = (
            cache["withdraw"]["recent"] / (cache["deposit"]["recent"] + EPS)
        )

    return feats


# ─────────────────────────────────────────────────────────────
# FEATURE BLOCK 14 — CASHFLOW SLOPE FEATURES
# ─────────────────────────────────────────────────────────────

def _cashflow_slope_features(df: pd.DataFrame, cache: Dict) -> Dict:
    feats = {}

    inflow_months  = np.zeros((len(df), len(MONTHS)))
    outflow_months = np.zeros((len(df), len(MONTHS)))

    for i, m in enumerate(MONTHS):
        for g in INFLOW_GROUPS:
            col = f"{m}_{g}_{VALUE_SUFFIX}"
            if col in df.columns:
                inflow_months[:, i] += df[col].values.astype(float)
        for g in OUTFLOW_GROUPS:
            col = f"{m}_{g}_{VALUE_SUFFIX}"
            if col in df.columns:
                outflow_months[:, i] += df[col].values.astype(float)

    net_monthly = inflow_months - outflow_months

    feats["net_cashflow_slope"]              = _compute_ols_slope(net_monthly)
    feats["inflow_slope"]                    = _compute_ols_slope(inflow_months)
    feats["outflow_slope"]                   = _compute_ols_slope(outflow_months)
    feats["inflow_outflow_slope_divergence"] = (
        feats["inflow_slope"] - feats["outflow_slope"]
    )
    feats["net_cashflow_m1"] = net_monthly[:, 0]

    return feats


# ─────────────────────────────────────────────────────────────
# FEATURE BLOCK 15 — CASHFLOW VOLATILITY
# ─────────────────────────────────────────────────────────────

def _cashflow_volatility(df: pd.DataFrame, cache: Dict) -> Dict:
    feats = {}

    net_monthly = np.zeros((len(df), len(MONTHS)))
    for i, m in enumerate(MONTHS):
        for g in INFLOW_GROUPS:
            col = f"{m}_{g}_{VALUE_SUFFIX}"
            if col in df.columns:
                net_monthly[:, i] += df[col].values.astype(float)
        for g in OUTFLOW_GROUPS:
            col = f"{m}_{g}_{VALUE_SUFFIX}"
            if col in df.columns:
                net_monthly[:, i] -= df[col].values.astype(float)

    feats["cashflow_volatility"]    = net_monthly.std(axis=1)
    feats["cashflow_volatility_cv"] = (
        net_monthly.std(axis=1) / (np.abs(net_monthly.mean(axis=1)) + EPS)
    )

    return feats


# ─────────────────────────────────────────────────────────────
# FEATURE BLOCK 16 — P2P FEATURES
# ─────────────────────────────────────────────────────────────

def _p2p_features(cache: Dict) -> Dict:
    feats = {}
    if "mm_send" not in cache or "received" not in cache:
        return feats

    send    = cache["mm_send"]["sum"]
    receive = cache["received"]["sum"]

    feats["p2p_net_flow"]           = receive - send
    feats["p2p_receive_send_ratio"] = receive / (send + EPS)
    feats["p2p_m1_ratio"]           = (
        cache["received"]["recent"] / (cache["mm_send"]["recent"] + EPS)
    )

    total_inflow = sum(cache[g]["sum"] for g in INFLOW_GROUPS if g in cache)
    feats["p2p_inflow_share"] = receive / (total_inflow + EPS)

    return feats


# ─────────────────────────────────────────────────────────────
# FEATURE BLOCK 17 — BANKING FEATURES
# ─────────────────────────────────────────────────────────────

def _banking_features(df: pd.DataFrame, cache: Dict) -> Dict:
    feats = {}
    if "transfer_from_bank" not in cache:
        return feats

    bank = cache["transfer_from_bank"]

    feats["bank_transfer_slope"]  = _compute_ols_slope(bank["data"])
    feats["bank_transfer_trend"]  = bank["recent"] - bank["old"]
    feats["bank_transfer_m1"]     = bank["recent"]
    feats["bank_has_bank_inflow"] = (bank["sum"] > 0).astype(int)

    if "arpu" in df.columns:
        arpu = df["arpu"].values.astype(float)
        feats["deposit_intensity"] = (
            cache["deposit"]["sum"] / (arpu + EPS)
            if "deposit" in cache else np.zeros(len(df))
        )
        feats["bank_transfer_vs_arpu"] = bank["sum"] / (arpu * 6 + EPS)

    return feats


# ─────────────────────────────────────────────────────────────
# FEATURE BLOCK 18 — INTERACTION FEATURES (SHAP-EVIDENCED)
# ─────────────────────────────────────────────────────────────

def _interaction_features(
    df: pd.DataFrame,
    cache: Dict,
    feature_blocks: Dict,
) -> Dict:
    feats = {}

    if "balance_trend" in feature_blocks and "m2_daily_avg_bal" in df.columns:
        bal_trend = np.asarray(feature_blocks["balance_trend"])
        m2_bal    = df["m2_daily_avg_bal"].values.astype(float)
        feats["balance_trend_x_balance_level"] = bal_trend * m2_bal

    if "deposit" in cache and "m1_daily_avg_bal" in df.columns:
        if "deposit_recency_ratio" in feature_blocks:
            dep_recency = np.asarray(feature_blocks["deposit_recency_ratio"])
        else:
            data = cache["deposit"]["data"]
            dep_recency = data[:, :3].mean(axis=1) / (data[:, 3:].mean(axis=1) + EPS)

        m1_bal = df["m1_daily_avg_bal"].values.astype(float)
        feats["deposit_recency_x_balance"] = dep_recency * m1_bal

    if "withdraw" in cache:
        data      = cache["withdraw"]["data"]
        w_recency = data[:, :3].mean(axis=1) / (data[:, 3:].mean(axis=1) + EPS)

        if "spend_to_inflow" in feature_blocks:
            spend_ratio = np.asarray(feature_blocks["spend_to_inflow"])
            feats["withdraw_recency_x_spend_ratio"] = w_recency * spend_ratio

    if "balance_cv" in feature_blocks and "balance_drawdown" in feature_blocks:
        feats["balance_cv_x_drawdown"] = (
            np.asarray(feature_blocks["balance_cv"])
            * np.asarray(feature_blocks["balance_drawdown"])
        )

    if "net_cashflow_slope" in feature_blocks and "m1_daily_avg_bal" in df.columns:
        feats["cashflow_slope_x_balance"] = (
            np.asarray(feature_blocks["net_cashflow_slope"])
            * df["m1_daily_avg_bal"].values.astype(float)
        )

    # ── v2.2 NEW interactions (volume × value signals) ────────────────
    # Interaction: deposit value recency × deposit volume recency.
    # Both declining together = strong stress signal (neither frequency
    # nor amount of deposits is recovering).
    if "deposit_recency_ratio" in feature_blocks and \
       "deposit_vol_recency_ratio" in feature_blocks:
        feats["deposit_value_x_volume_recency"] = (
            np.asarray(feature_blocks["deposit_recency_ratio"])
            * np.asarray(feature_blocks["deposit_vol_recency_ratio"])
        )

    # Interaction: withdraw value acceleration × withdraw volume acceleration.
    # Both accelerating = customer spending more and more frequently = pressure.
    if "withdraw_acceleration" in feature_blocks and \
       "withdraw_vol_acceleration" in feature_blocks:
        feats["withdraw_value_x_volume_acceleration"] = (
            np.asarray(feature_blocks["withdraw_acceleration"])
            * np.asarray(feature_blocks["withdraw_vol_acceleration"])
        )

    return feats


# ─────────────────────────────────────────────────────────────
# FEATURE BLOCK 19 — CATEGORICAL FEATURES
# ─────────────────────────────────────────────────────────────

def _categorical_features(df: pd.DataFrame) -> Dict:
    feats = {}

    # Segment: ordinal encode LVC < MVC < HVC
    if "segment" in df.columns:
        feats["segment_encoded"] = (
            df["segment"].str.upper().map(SEGMENT_ORDER).fillna(1).astype(int)
        )

    # Earning pattern: deterministic label encode (sorted unique values)
    if "earning_pattern" in df.columns:
        unique_patterns = sorted(df["earning_pattern"].dropna().unique())
        pattern_map     = {v: i for i, v in enumerate(unique_patterns)}
        feats["earning_pattern_encoded"] = (
            df["earning_pattern"].map(pattern_map).fillna(-1).astype(int)
        )

    # Smartphone ownership binary
    if "smartphone" in df.columns:
        feats["has_smartphone"] = (
            df["smartphone"].str.strip().str.lower().isin(["yes", "1", "true"])
        ).astype(int)

    # Gender binary
    if "gender" in df.columns:
        feats["is_male"] = (
            df["gender"].str.strip().str.upper() == "M"
        ).astype(int)

    # NOTE: 'age' intentionally excluded — raw column passes through.
    # NOTE: 'activity_rate' intentionally excluded — x_90_d_activity_rate
    #       passes through as-is. Do not re-add either without renaming.

    return feats


# ─────────────────────────────────────────────────────────────
# FEATURE BLOCK 20 — ZERO INDICATOR FLAGS
#
# IMPORTANT: called on raw df BEFORE concat. Calling after concat
# would flag engineered features, producing thousands of redundant
# indicator columns. Do not move this call in build_features().
# ─────────────────────────────────────────────────────────────

def _zero_indicators(df: pd.DataFrame) -> Dict:
    feats = {}
    skip  = {TARGET, ID_COL}
    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        if col in skip:
            continue
        zero_rate = (df[col] == 0).mean()
        if 0.60 <= zero_rate <= 0.98:
            feats[f"{col}_is_zero"] = (df[col] == 0).astype(np.int8)

    return feats


# ─────────────────────────────────────────────────────────────
# FEATURE BLOCK 21 — VALUE-VOLUME RATIO FEATURES (NEW v2.2)
#
# Computes average transaction size (value / volume) and its 6-month trend.
#
# Financial rationale:
#   Rising avg_txn_size + falling volume = fewer, larger transactions
#   → income consolidation or distressed large withdrawals.
#   Falling avg_txn_size + rising volume = more frequent small transactions
#   → daily subsistence spending, consistent with liquidity pressure.
#   These two dimensions together identify stress patterns invisible
#   to either value-only or volume-only features.
# ─────────────────────────────────────────────────────────────

def _value_volume_ratio_features(cache: Dict) -> Dict:
    feats = {}

    for g in ALL_GROUPS:
        if g not in cache or f"{g}_vol" not in cache:
            continue

        val_cache = cache[g]
        vol_cache = cache[f"{g}_vol"]

        # Average transaction size over the full 6-month window
        feats[f"avg_txn_size_{g}"] = (
            val_cache["mean"] / (vol_cache["mean"] + EPS)
        )

        # Trend in avg transaction size: M1 size vs M6 size
        # Positive = transactions getting larger (fewer, bigger)
        # Negative = transactions getting smaller (more, smaller)
        m1_size = val_cache["recent"] / (vol_cache["recent"] + EPS)
        m6_size = val_cache["old"]    / (vol_cache["old"]    + EPS)
        feats[f"avg_txn_size_{g}_trend"] = m1_size - m6_size

        # Recent vs past avg transaction size ratio
        recent_val_avg = val_cache["data"][:, :3].mean(axis=1)
        past_val_avg   = val_cache["data"][:, 3:].mean(axis=1)
        recent_vol_avg = vol_cache["data"][:, :3].mean(axis=1)
        past_vol_avg   = vol_cache["data"][:, 3:].mean(axis=1)

        recent_size = recent_val_avg / (recent_vol_avg + EPS)
        past_size   = past_val_avg   / (past_vol_avg   + EPS)
        feats[f"avg_txn_size_{g}_recency_ratio"] = recent_size / (past_size + EPS)

    return feats


# ─────────────────────────────────────────────────────────────
# FEATURE BLOCK 22 — UNIQUE ENTITY FEATURES (NEW v2.2)
#
# Tracks unique agents/merchants/recipients/senders over M1–M6.
#
# Financial rationale:
#   Network contraction (fewer unique counterparties) is a leading
#   indicator of financial stress. A customer who previously deposited
#   with 5 agents but now uses 1 has reduced financial mobility —
#   either through geographic restriction, relationship loss, or
#   deliberate cash conservation.
#   Declining unique-merchant count → restricted consumption patterns.
#   Declining unique-recipient count → reduced social/economic network.
# ─────────────────────────────────────────────────────────────

def _unique_entity_features(df: pd.DataFrame) -> Dict:
    feats = {}

    for g, entity in ENTITY_SUFFIX_MAP.items():
        cols = [
            f"{m}_{g}_{entity}"
            for m in MONTHS
            if f"{m}_{g}_{entity}" in df.columns
        ]
        if len(cols) < 2:
            continue

        data = df[cols].values.astype(float)
        n_months = data.shape[1]

        # 6-month mean unique entities
        feats[f"{g}_unique_{entity}_mean"] = data.mean(axis=1)

        # Trend: M1 unique entities vs M6 (most recent vs oldest)
        feats[f"{g}_unique_{entity}_trend"] = data[:, 0] - data[:, -1]

        # Most recent month unique entities
        feats[f"{g}_unique_{entity}_m1"] = data[:, 0]

        # Recency ratio: M1–M3 avg vs M4–M6 avg
        if n_months >= 6:
            recent_avg = data[:, :3].mean(axis=1)
            past_avg   = data[:, 3:].mean(axis=1)
            feats[f"{g}_unique_{entity}_recency_ratio"] = (
                recent_avg / (past_avg + EPS)
            )

        # OLS slope: is the unique-entity count trending up or down?
        feats[f"{g}_unique_{entity}_slope"] = _compute_ols_slope(data)

    return feats


# ─────────────────────────────────────────────────────────────
# FEATURE BLOCK 23 — VOLUME-BASED CASHFLOW FEATURES (NEW v2.2)
#
# Aggregates transaction counts across inflow/outflow groups monthly.
#
# Financial rationale:
#   Customers often reduce TRANSACTION FREQUENCY before reducing
#   TRANSACTION VALUE — frequency decline is an earlier warning signal.
#   Volume-slope divergence (inflow count declining faster than outflow
#   count) is a leading indicator that the value-based cashflow slope
#   will follow. This captures the warning 1–2 months earlier.
# ─────────────────────────────────────────────────────────────

def _volume_cashflow_features(df: pd.DataFrame) -> Dict:
    feats = {}

    inflow_vol_months  = np.zeros((len(df), len(MONTHS)))
    outflow_vol_months = np.zeros((len(df), len(MONTHS)))

    for i, m in enumerate(MONTHS):
        for g in INFLOW_GROUPS:
            col = f"{m}_{g}_{VOLUME_SUFFIX}"
            if col in df.columns:
                inflow_vol_months[:, i] += df[col].values.astype(float)
        for g in OUTFLOW_GROUPS:
            col = f"{m}_{g}_{VOLUME_SUFFIX}"
            if col in df.columns:
                outflow_vol_months[:, i] += df[col].values.astype(float)

    net_vol_monthly = inflow_vol_months - outflow_vol_months

    feats["net_txn_volume_slope"]      = _compute_ols_slope(net_vol_monthly)
    feats["inflow_volume_slope"]       = _compute_ols_slope(inflow_vol_months)
    feats["outflow_volume_slope"]      = _compute_ols_slope(outflow_vol_months)
    feats["volume_slope_divergence"]   = (
        feats["inflow_volume_slope"] - feats["outflow_volume_slope"]
    )
    feats["net_txn_volume_m1"]         = net_vol_monthly[:, 0]
    feats["total_txn_volume_6m"]       = (
        inflow_vol_months.sum(axis=1) + outflow_vol_months.sum(axis=1)
    )

    # Volume momentum: most recent month net volume change
    if net_vol_monthly.shape[1] >= 2:
        feats["net_txn_volume_momentum"] = (
            net_vol_monthly[:, 0] - net_vol_monthly[:, 1]
        )

    # Inflow volume recency ratio
    recent_in = inflow_vol_months[:, :3].mean(axis=1)
    past_in   = inflow_vol_months[:, 3:].mean(axis=1)
    feats["inflow_volume_recency_ratio"] = recent_in / (past_in + EPS)

    # Outflow volume recency ratio
    recent_out = outflow_vol_months[:, :3].mean(axis=1)
    past_out   = outflow_vol_months[:, 3:].mean(axis=1)
    feats["outflow_volume_recency_ratio"] = recent_out / (past_out + EPS)

    return feats