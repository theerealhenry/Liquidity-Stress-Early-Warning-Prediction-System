"""
Preprocessing Pipeline
======================
Project : Liquidity Stress Early Warning (AI4EAC / Zindi)
Module  : src/preprocessing/preprocessing.py

Design principles
-----------------
- CV-safe: all statistics learned in fit(), applied in transform()
- Config-aware: accepts raw YAML config dict or explicit kwargs
- Dtype-safe: quantile() results always cast to Python float before
  conditional checks — eliminates the ambiguous Series truth-value error
- Null-safe: inf/NaN replaced before any arithmetic
- Contract-enforced: feature list locked at fit() time, enforced at
  transform() time with a hard assertion
- Memory-optimised: downcasts applied after all transformations
- Debug-visible: optional print statements at every major step

Fixed bugs (vs previous version)
---------------------------------
1. CRITICAL — `if pd.notnull(low) and pd.notnull(high) and low < high`
   When a column has nullable dtype (Int8, Float64), .quantile() returns
   a pandas Series instead of a scalar. Applying `if` to a Series raises
   "The truth value of a Series is ambiguous."
   Fix: float(low) and float(high) coerce to Python scalar before check.

2. CRITICAL — PreprocessingPipeline(cfg) passed the entire config dict
   as the `feature_list` positional argument. Constructor now accepts
   an optional `config` dict and reads clip_quantiles from it.

3. MODERATE — feature_list was stored from the post-drop DataFrame inside
   fit(), but the same list was used to align columns in transform()
   before dropping TARGET/ID_COL — causing potential column mismatches.
   Fix: feature_list is always set from the clean, dropped DataFrame.

4. MINOR — optimize_memory_usage() attempted to downcast nullable Int8 /
   Float64 pandas extension dtypes, which raises TypeError in some pandas
   versions. Fix: convert to numpy-native dtypes first.

5. MINOR — No __repr__ made pipeline state invisible in logs.
   Fix: added __repr__ with key statistics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

EPS = 1e-6

# Columns always excluded from feature learning and transformation
_NON_FEATURE_COLS = {"liquidity_stress_next_30d", "ID"}


# =============================================================================
# SCHEMA VALIDATION
# =============================================================================

def validate_schema(df: pd.DataFrame, context: str = "") -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"[{context}] Input must be a pandas DataFrame, got {type(df)}")
    if df.empty:
        raise ValueError(f"[{context}] Input DataFrame is empty")


# =============================================================================
# MEMORY OPTIMISATION
# Converts to numpy-native dtypes first to avoid nullable-dtype downcast errors
# =============================================================================

def optimize_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        series = df[col]
        dtype  = series.dtype

        # Skip category columns — already handled upstream
        if isinstance(dtype, pd.CategoricalDtype):
            continue

        # Convert pandas extension types (Int8, Float64, boolean) to numpy
        if hasattr(dtype, "numpy_dtype"):
            series = series.astype(dtype.numpy_dtype)

        # Downcast float64 → float32
        if series.dtype == np.float64:
            df[col] = series.astype(np.float32)

        # Downcast int64 → smallest signed int
        elif series.dtype == np.int64:
            df[col] = pd.to_numeric(series, downcast="integer")

        else:
            df[col] = series

    return df


# =============================================================================
# PREPROCESSING PIPELINE
# =============================================================================

class PreprocessingPipeline:
    """
    Fit-transform preprocessing pipeline for tabular financial data.

    Parameters
    ----------
    config : dict, optional
        Full YAML config dict. If provided, clip_quantiles is read from
        config["preprocessing"]["clip_quantiles"].
    feature_list : list of str, optional
        Explicit feature list. If None, inferred from fit() data.
    clip_quantiles : tuple of (float, float), optional
        Lower and upper quantile bounds for clipping. Default (0.001, 0.999).
        Ignored if `config` is provided (config takes precedence).
    enable_clipping : bool
        Whether to apply quantile clipping. Default True.
    debug : bool
        Print diagnostic messages at fit/transform time. Default True.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        feature_list: Optional[List[str]] = None,
        clip_quantiles: Tuple[float, float] = (0.001, 0.999),
        enable_clipping: bool = True,
        debug: bool = True,
    ):
        # ── Config-aware initialisation ───────────────────────────────────
        if config is not None:
            preproc_cfg = config.get("preprocessing", {})
            clip_quantiles = tuple(
                preproc_cfg.get("clip_quantiles", list(clip_quantiles))
            )
            enable_clipping = preproc_cfg.get("enable_clipping", enable_clipping)
            debug = preproc_cfg.get("debug", debug)

        self.config          = config
        self.feature_list    = feature_list      # None until fit()
        self.clip_quantiles  = clip_quantiles
        self.enable_clipping = enable_clipping
        self.debug           = debug

        # ── Learned attributes (populated in fit()) ───────────────────────
        self.clip_values_:   Dict[str, Tuple[float, float]] = {}
        self.clip_cols_:     List[str] = []
        self.constant_cols_: List[str] = []
        self._is_fitted:     bool      = False

    # =========================================================================
    # FIT
    # =========================================================================

    def fit(self, df: pd.DataFrame) -> "PreprocessingPipeline":
        validate_schema(df, context="fit")
        df = df.copy()

        # ── Strip non-feature columns ─────────────────────────────────────
        df = df.drop(
            columns=[c for c in _NON_FEATURE_COLS if c in df.columns],
            errors="ignore",
        )

        # ── Convert to numpy-native numeric (handles nullable dtypes) ─────
        df = self._to_numeric(df)

        # ── Replace inf before any statistics ─────────────────────────────
        df = df.replace([np.inf, -np.inf], np.nan)

        # ── Constant column detection ──────────────────────────────────────
        nunique = df.nunique(dropna=True)
        self.constant_cols_ = nunique[nunique <= 1].index.tolist()

        # Remove constants before learning clip bounds
        df = df.drop(columns=self.constant_cols_, errors="ignore")

        # ── Lock feature list ──────────────────────────────────────────────
        # Always set from the clean, stripped DataFrame so that transform()
        # alignment is consistent
        if self.feature_list is None:
            self.feature_list = df.columns.tolist()

        # ── Quantile clipping (fit) ────────────────────────────────────────
        if self.enable_clipping and self.clip_quantiles:
            lower_q, upper_q = self.clip_quantiles

            for col in df.columns:
                series = df[col]

                if series.nunique(dropna=True) <= 10:
                    # Skip binary / low-cardinality features
                    continue

                # ── FIX 1: cast to Python float before conditional ────────
                # .quantile() on nullable-dtype columns returns a Series,
                # not a scalar. float() coerces it to a Python float so
                # `if pd.notnull(low)` is unambiguous.
                try:
                    low  = float(series.quantile(lower_q))
                    high = float(series.quantile(upper_q))
                except (TypeError, ValueError):
                    continue

                if pd.notnull(low) and pd.notnull(high) and low < high:
                    self.clip_values_[col] = (low, high)
                    self.clip_cols_.append(col)

        self._is_fitted = True

        if self.debug:
            print(f"[FIT] Features locked  : {len(self.feature_list)}")
            print(f"[FIT] Constant removed : {len(self.constant_cols_)}")
            print(f"[FIT] Clipping cols    : {len(self.clip_cols_)}")

        return self

    # =========================================================================
    # TRANSFORM
    # =========================================================================

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError(
                "PreprocessingPipeline must be fitted before transform(). "
                "Call fit() or fit_transform() first."
            )

        validate_schema(df, context="transform")
        df = df.copy()

        # ── Strip non-feature columns ─────────────────────────────────────
        df = df.drop(
            columns=[c for c in _NON_FEATURE_COLS if c in df.columns],
            errors="ignore",
        )

        # ── Align columns to fit-time contract ────────────────────────────
        df = self._align_columns(df)

        # ── Convert to numpy-native numeric ───────────────────────────────
        df = self._to_numeric(df)

        # ── Replace inf ───────────────────────────────────────────────────
        df = df.replace([np.inf, -np.inf], np.nan)

        # ── Fill NaN before clipping ──────────────────────────────────────
        df = df.fillna(0)

        # ── Apply quantile clipping ───────────────────────────────────────
        if self.enable_clipping:
            for col, (low, high) in self.clip_values_.items():
                if col in df.columns:
                    df[col] = df[col].clip(lower=low, upper=high)

        # ── Final NaN safety net ──────────────────────────────────────────
        df = df.fillna(0)

        # ── Drop constants (consistent with fit) ──────────────────────────
        df = df.drop(columns=self.constant_cols_, errors="ignore")

        # ── Enforce final feature contract ────────────────────────────────
        expected_cols = [c for c in self.feature_list if c not in self.constant_cols_]
        df = df[expected_cols]

        # ── Memory optimisation ───────────────────────────────────────────
        df = optimize_memory_usage(df)

        # ── Hard assertion: feature contract must be exact ────────────────
        actual_cols = list(df.columns)
        if actual_cols != expected_cols:
            mismatched = set(actual_cols).symmetric_difference(set(expected_cols))
            raise AssertionError(
                f"Feature contract violated after preprocessing!\n"
                f"Mismatched columns: {mismatched}"
            )

        if self.debug:
            print(f"[TRANSFORM] Output shape : {df.shape}")
            nan_count = int(df.isna().sum().sum())
            if nan_count > 0:
                print(f"[TRANSFORM] WARNING — {nan_count} NaN values remain")

        return df

    # =========================================================================
    # FIT + TRANSFORM
    # =========================================================================

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # =========================================================================
    # COLUMN ALIGNMENT (internal)
    # Aligns an unseen DataFrame to the feature list learned at fit() time.
    # Missing columns are zero-filled; extra columns are dropped.
    # =========================================================================

    def _align_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        fit_cols  = set(self.feature_list)
        data_cols = set(df.columns)

        missing_cols = list(fit_cols - data_cols)
        extra_cols   = list(data_cols - fit_cols)

        if extra_cols:
            df = df.drop(columns=extra_cols)

        if missing_cols:
            missing_df = pd.DataFrame(
                0,
                index=df.index,
                columns=missing_cols,
                dtype=np.float32,
            )
            df = pd.concat([df, missing_df], axis=1)

        # Enforce fit-time column order and defragment
        df = df[self.feature_list].copy()

        return df

    # =========================================================================
    # NUMERIC CONVERSION (internal)
    # Converts nullable pandas extension types to numpy-native before any
    # arithmetic — prevents ambiguous truth-value errors downstream.
    # =========================================================================

    @staticmethod
    def _to_numeric(df: pd.DataFrame) -> pd.DataFrame:
        result = {}
        for col in df.columns:
            s    = df[col]
            dtype = s.dtype

            # ── Category dtype: convert codes to float, NaN for unknowns ──
            if isinstance(dtype, pd.CategoricalDtype):
                result[col] = s.cat.codes.replace(-1, np.nan).astype(np.float32)
                continue

            # ── Pandas extension types (Int8, Float64, boolean) ───────────
            if hasattr(dtype, "numpy_dtype"):
                s = s.astype(dtype.numpy_dtype)

            # ── Coerce everything else to numeric ─────────────────────────
            result[col] = pd.to_numeric(s, errors="coerce")

        return pd.DataFrame(result, index=df.index)

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def save(self, path: str) -> None:
        import joblib
        joblib.dump(self, path)
        print(f"[PreprocessingPipeline] Saved to {path}")

    @staticmethod
    def load(path: str) -> "PreprocessingPipeline":
        import joblib
        pipeline = joblib.load(path)
        print(f"[PreprocessingPipeline] Loaded from {path}")
        return pipeline

    # =========================================================================
    # REPR
    # =========================================================================

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        n_features = len(self.feature_list) if self.feature_list else 0
        return (
            f"PreprocessingPipeline("
            f"status={status}, "
            f"features={n_features}, "
            f"clip_quantiles={self.clip_quantiles}, "
            f"constant_cols={len(self.constant_cols_)}, "
            f"clip_cols={len(self.clip_cols_)}"
            f")"
        )