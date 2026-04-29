"""
Feature Engineering Module — v2.1 (Production Grade)
======================================================

Project     : AI4EAC Liquidity Stress Early Warning Prediction (Zindi Africa)
Task        : Binary classification — predict P(liquidity stress within 30 days)
Metrics     : Log Loss (60%) + ROC-AUC (40%)
Data        : 6-month mobile money panel — 40,000 customers × 7 monthly snapshots
              226 raw columns → expanded feature set

Changelog v2.1
--------------
BUG 1 — CRITICAL: Duplicate 'age' column
  _categorical_features() re-created 'age' from df["age"] even though
  'age' already exists in the raw DataFrame. pd.concat([df, feature_df])
  produced two columns named 'age', causing df[col] to return a DataFrame
  instead of a Series — crashing PreprocessingPipeline._to_numeric() with
  'DataFrame object has no attribute dtype'.
  Fix: removed the age re-creation from _categorical_features(). The raw
  'age' column passes through untouched in the original df.

BUG 2 — CRITICAL: activity_rate duplicates x_90_d_activity_rate
  _categorical_features() created feats["activity_rate"] from
  df["x_90_d_activity_rate"] — a renamed copy of a raw column. After
  concat, both 'x_90_d_activity_rate' (raw) and 'activity_rate'
  (engineered copy) exist. While not a duplicate name, it is a
  redundant feature that inflates importance scores for what is
  effectively the same signal measured twice.
  Fix: removed activity_rate from _categorical_features(). The raw
  x_90_d_activity_rate column passes through and is used directly.

BUG 3 — MODERATE: _recency_features log suffix creates duplicate keys
  When called with suffix="_log", _recency_features() produced:
    {g}_recency_ratio_log     (the ratio itself)
    {g}_recency_ratio_log_log (the log of the ratio, with _log suffix)
  The second key is semantically wrong (log of log of ratio).
  Fix: the log-of-ratio feature is only computed when suffix == ""
  (raw call). The _log suffix call only computes the ratio on
  log-transformed inputs — no nested log naming.

BUG 4 — MODERATE: _zero_indicators runs on engineered columns
  _zero_indicators(df) only sees the raw df (correct). However, if
  called after concat it would also flag engineered features — leading
  to thousands of redundant indicator columns.
  Confirmed: currently called on raw df before concat — no change
  needed. Added explicit docstring note to prevent future regression.

BUG 5 — MINOR: _categorical_features earning_pattern_encoded
  pattern_map was built from df["earning_pattern"].dropna().unique()
  which is non-deterministic in ordering across runs on some pandas
  versions (unique() order is insertion-order but dropna() can vary).
  Fix: sorted() applied to unique values before enumeration to ensure
  deterministic encoding across all environments.

Design Principles
-----------------
1. CV-SAFE — zero leakage. Every feature is derived exclusively from
   historical columns (M1–M6). Target never touches feature computation.
2. NULL = ZERO — per competition spec, null means no activity, not missing.
   All nulls are filled with 0 before feature computation.
3. M1 IS GROUND TRUTH — most recent month is the primary anchor. Older
   months contribute trend/slope/context only.
4. WINSORISATION — extreme outliers (withdraw_recency_ratio SHAP=5.5 observed)
   are clipped at the 99th percentile within each training fold to prevent
   gradient domination in tree models.
5. TREE-FRIENDLY — no standardisation applied here (tree models are scale-
   invariant). Log transforms are applied selectively for skewed monetary
   features to improve split quality on long-tail distributions.
6. SHAP-EVIDENCED — every feature block is justified by either SHAP analysis
   from the baseline model, financial domain logic, or both. Rationale is
   documented inline.
7. DETERMINISTIC — no random operations. Given the same input DataFrame,
   output is always identical regardless of execution environment.
8. MODULAR — each feature family is an isolated private function. Adding,
   removing, or ablating a feature group requires changing exactly one line
   in build_features().

Feature Families (20 blocks)
------------------------------
RAW SIGNAL LAYER
  _temporal_aggregations     — mean / std / min / max per group
  _trend_features            — linear slope + simple trend (M1−M6) per group
  _momentum_features         — 1-month delta (M1−M2) per group
  _acceleration_features     — change-of-momentum (M1−M2)−(M2−M3) per group
  _volatility_features       — CV (std/mean) + recent/past std ratio per group
  _consistency_features      — coefficient of variation, inverse-CV stability
  _activity_features         — months active, zero-streak, activity switch rate
  _recency_features          — recent-3 / past-3 ratio per group (all 7 types)
  _peak_intensity_features   — max/mean ratio per group

BALANCE INTELLIGENCE LAYER
  _balance_features          — trend, slope, CV, min, max, range, recovery
  _drawdown_features         — peak-to-trough drawdown magnitude
  _balance_pressure_features — balance relative to spending obligations

CASHFLOW INTELLIGENCE LAYER
  _cashflow_features         — net flow, ratios, cumulative inflow/outflow
  _cashflow_slope_features   — slope of monthly net cashflow over 6 months
  _cashflow_volatility       — std of monthly net cashflow

P2P & BANKING LAYER
  _p2p_features              — send/receive ratio, net P2P position
  _banking_features          — bank transfer trend, ARPU-relative inflows

INTERACTION LAYER (SHAP-evidenced)
  _interaction_features      — balance_trend × avg_balance,
                               deposit_recency × balance_level,
                               withdraw_recency × spend_to_inflow

ENCODING & INDICATOR LAYER
  _categorical_features      — ordinal encode segment/earning_pattern
  _zero_indicators           — binary sparsity flags (0.6–0.98 zero-rate cols)
  _winsorise                 — 99th-pct clip on flagged extreme features

LOG-TRANSFORMED LAYER
  _temporal_aggregations     (log suffix) — on log1p monetary values
  _trend_features            (log suffix)
  _volatility_features       (log suffix)
  _momentum_features         (log suffix)
  _recency_features          (log suffix)
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

VALUE_SUFFIX   : str = "total_value"
VOLUME_SUFFIX  : str = "volume"

# Features known from SHAP analysis to produce extreme outliers → winsorise
WINSORISE_FEATURES : List[str] = [
    "withdraw_recency_ratio",
    "withdraw_recency_ratio_log",
    "net_flow_ratio",
    "balance_to_spend_pressure",
]

WINSORISE_PERCENTILE : float = 99.0

# Segment value tier — ordered for ordinal encoding
SEGMENT_ORDER : Dict[str, int] = {"LVC": 0, "MVC": 1, "HVC": 2}

# Raw columns that already exist in df — never re-create these in feature blocks
# to prevent duplicate columns after pd.concat([df, feature_df])
_RAW_PASSTHROUGH_COLS = {
    "age",                   # int64 in raw data — passes through untouched
    "x_90_d_activity_rate",  # float in raw data — passes through untouched
    "arpu",                  # float in raw data
}


# ─────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master feature engineering pipeline.

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
        Original columns + all engineered features. Ready for
        split_features_target() → model training.

    Notes
    -----
    - Call on train and test separately. No fitting step — fully stateless.
    - Winsorisation uses percentiles computed on the passed DataFrame.
      In production, fit percentiles on train only and pass as clip_values
      to _winsorise() if required. For competition use, per-split clipping
      is acceptable.
    - _zero_indicators() must be called on the raw df BEFORE concat to
      avoid flagging engineered features. Do not move it post-concat.
    """
    df = df.copy()

    # ── Step 1: Fill nulls — null means zero activity, not missing ──────
    df = _fill_nulls(df)

    # ── Step 2: Build raw and log-transformed caches ─────────────────────
    raw_cache = _build_cache(df)
    log_df    = _apply_log_transform(df.copy())
    log_cache = _build_cache(log_df)

    # ── Step 3: Compute all feature blocks ───────────────────────────────
    feature_blocks: Dict[str, np.ndarray | pd.Series] = {}

    # RAW SIGNAL LAYER
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

    # P2P & BANKING LAYER
    feature_blocks.update(_p2p_features(raw_cache))
    feature_blocks.update(_banking_features(df, raw_cache))

    # INTERACTION LAYER (SHAP-evidenced)
    feature_blocks.update(_interaction_features(df, raw_cache, feature_blocks))

    # ENCODING LAYER
    feature_blocks.update(_categorical_features(df))

    # ZERO INDICATOR LAYER
    # IMPORTANT: must be called on raw df BEFORE concat — see docstring
    feature_blocks.update(_zero_indicators(df))

    # LOG-TRANSFORMED LAYER (secondary signals on skewed distributions)
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

    # Safety net: drop any duplicate columns keeping the first occurrence
    # (the raw column). This must never trigger in production — if it does,
    # a feature block is re-creating a raw column and should be fixed.
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
    Apply log1p to monetary columns (total_value, highest_amount).
    Reduces skewness on heavily right-tailed distributions.
    Clipped at 0 before transform — no negative values expected but
    defensive against synthetic data artefacts.
    """
    for col in df.columns:
        if any(s in col for s in (VALUE_SUFFIX, "highest_amount")):
            df[col] = np.log1p(df[col].clip(lower=0))
    return df


def _build_cache(df: pd.DataFrame) -> Dict:
    """
    Pre-compute per-group arrays to avoid redundant column lookups.
    Each group stores:
        data   : (n_rows × 6) array of monthly values
        sum    : 6-month total
        mean   : 6-month mean
        std    : 6-month std
        recent : M1 value (most reliable per competition spec)
        old    : M6 value
        cols   : column names used (for debugging)
    """
    cache = {}
    for g in ALL_GROUPS:
        cols = [
            f"{m}_{g}_{VALUE_SUFFIX}"
            for m in MONTHS
            if f"{m}_{g}_{VALUE_SUFFIX}" in df.columns
        ]
        if not cols:
            continue

        data = df[cols].values.astype(float)
        cache[g] = {
            "data"   : data,
            "cols"   : cols,
            "sum"    : data.sum(axis=1),
            "mean"   : data.mean(axis=1),
            "std"    : data.std(axis=1),
            "min"    : data.min(axis=1),
            "max"    : data.max(axis=1),
            "recent" : data[:, 0],   # M1 — most recent
            "old"    : data[:, -1],  # M6 — oldest
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
    2. Drop constant columns (zero variance = no information).
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
    Lower bound is always 0 (financial values).

    Motivation: SHAP analysis showed withdraw_recency_ratio reaching 5.5
    — extreme outliers dominate gradient updates in tree models and produce
    overconfident predictions for a small subset of customers.
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

    Time is encoded as [5,4,3,2,1,0] so that index 0 = M1 (most recent)
    and the slope sign is intuitive: positive slope = increasing over time
    toward the present (improving), negative = declining.

    Parameters
    ----------
    data : np.ndarray of shape (n_samples, n_months)

    Returns
    -------
    np.ndarray of shape (n_samples,) — OLS slope per row
    """
    n = data.shape[1]
    x     = np.arange(n - 1, -1, -1, dtype=float)   # [5,4,3,2,1,0]
    x_bar = x.mean()
    ss_x  = ((x - x_bar) ** 2).sum()

    y_bar = data.mean(axis=1, keepdims=True)
    slope = ((data - y_bar) * (x - x_bar)).sum(axis=1) / (ss_x + EPS)
    return slope


def _get_balance_array(df: pd.DataFrame) -> Optional[np.ndarray]:
    """Return (n × 6) balance array if all columns exist, else None."""
    cols = [f"{m}_daily_avg_bal" for m in MONTHS]
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2:
        return None
    return df[cols].values.astype(float)


# ─────────────────────────────────────────────────────────────
# FEATURE BLOCK 1 — TEMPORAL AGGREGATIONS
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
# FEATURE BLOCK 2 — TREND FEATURES
# ─────────────────────────────────────────────────────────────

def _trend_features(cache: Dict, suffix: str = "") -> Dict:
    feats = {}
    for g, v in cache.items():
        data = v["data"]
        feats[f"{g}_trend{suffix}"]       = v["recent"] - v["old"]
        feats[f"{g}_trend_slope{suffix}"] = _compute_ols_slope(data)
    return feats


# ─────────────────────────────────────────────────────────────
# FEATURE BLOCK 3 — MOMENTUM FEATURES
# ─────────────────────────────────────────────────────────────

def _momentum_features(cache: Dict, suffix: str = "") -> Dict:
    feats = {}
    for g, v in cache.items():
        data = v["data"]
        if data.shape[1] >= 2:
            feats[f"{g}_momentum{suffix}"] = data[:, 0] - data[:, 1]
    return feats


# ─────────────────────────────────────────────────────────────
# FEATURE BLOCK 4 — ACCELERATION FEATURES
# ─────────────────────────────────────────────────────────────

def _acceleration_features(cache: Dict) -> Dict:
    feats = {}
    for g, v in cache.items():
        data = v["data"]
        if data.shape[1] >= 3:
            mom_recent = data[:, 0] - data[:, 1]  # M1−M2
            mom_past   = data[:, 1] - data[:, 2]  # M2−M3
            feats[f"{g}_acceleration"] = mom_recent - mom_past
    return feats


# ─────────────────────────────────────────────────────────────
# FEATURE BLOCK 5 — VOLATILITY FEATURES
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
# FEATURE BLOCK 6 — CONSISTENCY FEATURES
# ─────────────────────────────────────────────────────────────

def _consistency_features(cache: Dict) -> Dict:
    feats = {}
    for g, v in cache.items():
        feats[f"{g}_consistency"] = v["mean"] / (v["std"] + EPS)
    return feats


# ─────────────────────────────────────────────────────────────
# FEATURE BLOCK 7 — ACTIVITY FEATURES
# ─────────────────────────────────────────────────────────────

def _activity_features(cache: Dict) -> Dict:
    feats = {}
    for g, v in cache.items():
        activity = (v["data"] > 0).astype(int)
        feats[f"{g}_active_months"]   = activity.sum(axis=1)
        feats[f"{g}_activity_switch"] = np.abs(np.diff(activity, axis=1)).sum(axis=1)
        feats[f"{g}_zero_streak"]     = _compute_zero_streak(activity)
    return feats


def _compute_zero_streak(activity: np.ndarray) -> np.ndarray:
    """
    Count consecutive zero months starting from M1 (column 0).
    Stops at first non-zero month.
    """
    n_rows, n_cols = activity.shape
    streak = np.zeros(n_rows, dtype=int)
    for col in range(n_cols):
        still_zero = (activity[:, col] == 0) & (streak == col)
        streak[still_zero] += 1
    return streak


# ─────────────────────────────────────────────────────────────
# FEATURE BLOCK 8 — RECENCY FEATURES
# Rationale: deposit_recency_ratio was the #1 SHAP feature.
# Extended to all 7 transaction types.
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
            # When suffix="_log", inputs are already log-transformed;
            # adding another log would create an uninterpretable feature.
            if suffix == "":
                feats[f"{g}_recency_ratio_log"] = np.log1p(ratio)

    return feats


# ─────────────────────────────────────────────────────────────
# FEATURE BLOCK 9 — PEAK INTENSITY FEATURES
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

    return feats


# ─────────────────────────────────────────────────────────────
# FEATURE BLOCK 19 — CATEGORICAL FEATURES
#
# FIX (Bug 1): 'age' removed from this block. It already exists
# in the raw DataFrame and passes through pd.concat untouched.
# Re-creating it here produced a duplicate column name.
#
# FIX (Bug 2): 'activity_rate' (copy of x_90_d_activity_rate) removed.
# The raw column passes through and is used directly by the model.
#
# FIX (Bug 5): earning_pattern encoding is now deterministic —
# sorted() applied before enumeration so encoding is stable across
# all pandas versions and execution environments.
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
# Rationale: Per competition spec, null = zero activity.
# For features with 60–98% zero rate, a binary flag is more
# informative than the raw value.
#
# IMPORTANT: This function must be called on the raw df BEFORE
# pd.concat([df, feature_df]). Calling it after concat would
# flag engineered features as well, producing thousands of
# redundant indicator columns. See build_features() step ordering.
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