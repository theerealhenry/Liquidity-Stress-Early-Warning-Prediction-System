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
- Scale-aware: optional StandardScaler for models that are not
  scale-invariant (LogisticRegression, TabNet). Scaler fitted AFTER
  clipping so outlier contamination does not propagate into the mean/std.
- Debug-visible: optional print statements at every major step

Changelog
---------
v1.0   Initial production version with clipping + contract enforcement.
v1.1   CRITICAL fix: float() cast on quantile() result to avoid ambiguous
       Series truth-value error on nullable dtypes.
v1.2   CRITICAL fix: config dict passed as feature_list positional arg.
       Constructor now accepts explicit `config` kwarg.
v1.3   MODERATE fix: feature_list locked from post-drop DataFrame so
       transform() alignment is consistent.
v1.4   MINOR fix: optimize_memory_usage() converts nullable extension
       dtypes before downcasting to avoid TypeError.
v1.5   MINOR fix: __repr__ added for pipeline state visibility in logs.
v2.0   NEW: scale_features flag + StandardScaler integration.
       Required for Logistic Regression (not scale-invariant) and TabNet
       (attention weights collapse without normalisation).
       Scaler is fitted after clipping so outlier values do not corrupt
       the learned mean/std statistics.
       Fully config-aware: reads scale_features from
       config["preprocessing"]["scale_features"].
       Persistence: scaler_ serialised with the pipeline via joblib.
       __repr__ updated to reflect scaling state.
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
    """Raises informative errors for non-DataFrame or empty inputs."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"[{context}] Input must be a pandas DataFrame, got {type(df)}"
        )
    if df.empty:
        raise ValueError(f"[{context}] Input DataFrame is empty")


# =============================================================================
# MEMORY OPTIMISATION
# Converts to numpy-native dtypes first to avoid nullable-dtype downcast errors
# =============================================================================

def optimize_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast float64 → float32 and int64 → smallest valid signed int.

    Converts pandas extension types (Int8, Float64, boolean) to their
    numpy-native equivalents first to avoid TypeError in pandas versions
    that do not support downcasting nullable dtypes directly.
    """
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

    The pipeline applies transformations in this strict order to preserve
    CV-safety (all statistics learned exclusively from training folds):

        fit() order
        -----------
        1. Strip non-feature columns (target, ID)
        2. Convert nullable dtypes → numpy-native
        3. Replace inf → NaN
        4. Detect and store constant columns
        5. Lock feature list from clean DataFrame
        6. Learn per-column quantile clip bounds (skipping low-cardinality)
        7. Apply clipping to the fit data (for scaler input)
        8. Fit StandardScaler on clipped data (if scale_features=True)

        transform() order
        -----------------
        1. Strip non-feature columns
        2. Align to fit-time column contract (zero-fill missing, drop extra)
        3. Convert nullable dtypes → numpy-native
        4. Replace inf → NaN
        5. Fill NaN → 0
        6. Apply learned clip bounds
        7. Fill any residual NaN → 0
        8. Drop constant columns
        9. Enforce feature contract (hard assertion)
        10. Apply StandardScaler (if scale_features=True)
        11. Downcast memory

    Parameters
    ----------
    config : dict, optional
        Full YAML config dict. If provided, preprocessing settings are read
        from config["preprocessing"]:
            clip_quantiles  : list of two floats, default [0.001, 0.999]
            enable_clipping : bool, default True
            scale_features  : bool, default False
            debug           : bool, default True
    feature_list : list of str, optional
        Explicit feature list. If None, inferred from fit() data.
    clip_quantiles : tuple of (float, float)
        Lower and upper quantile bounds. Default (0.001, 0.999).
        Ignored if `config` is provided (config takes precedence).
    enable_clipping : bool
        Whether to apply quantile clipping. Default True.
    scale_features : bool
        Whether to apply StandardScaler after clipping. Set True for
        Logistic Regression and TabNet; leave False for GBMs (tree-based
        models are scale-invariant). Default False.
    debug : bool
        Print diagnostic messages at fit/transform time. Default True.

    Attributes (populated after fit)
    ----------------------------------
    clip_values_    : dict  — {col: (low, high)} learned from training data
    clip_cols_      : list  — columns where clipping is active
    constant_cols_  : list  — columns with nunique <= 1, dropped everywhere
    scaler_         : StandardScaler or None — fitted scaler (if scale_features)
    feature_list    : list  — locked feature order from fit()
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        feature_list: Optional[List[str]] = None,
        clip_quantiles: Tuple[float, float] = (0.001, 0.999),
        enable_clipping: bool = True,
        scale_features: bool = False,
        debug: bool = True,
    ):
        # ── Config-aware initialisation ───────────────────────────────────
        if config is not None:
            preproc_cfg = config.get("preprocessing", {})
            clip_quantiles  = tuple(
                preproc_cfg.get("clip_quantiles", list(clip_quantiles))
            )
            enable_clipping = preproc_cfg.get("enable_clipping", enable_clipping)
            scale_features  = preproc_cfg.get("scale_features", scale_features)
            debug           = preproc_cfg.get("debug", debug)

        self.config          = config
        self.feature_list    = feature_list      # None until fit()
        self.clip_quantiles  = clip_quantiles
        self.enable_clipping = enable_clipping
        self.scale_features  = scale_features
        self.debug           = debug

        # ── Learned attributes (populated in fit()) ───────────────────────
        self.clip_values_:   Dict[str, Tuple[float, float]] = {}
        self.clip_cols_:     List[str] = []
        self.constant_cols_: List[str] = []
        self.scaler_:        Any       = None   # StandardScaler or None
        self._is_fitted:     bool      = False

    # =========================================================================
    # FIT
    # =========================================================================

    def fit(self, df: pd.DataFrame) -> "PreprocessingPipeline":
        """
        Learn all statistics from the training DataFrame.

        All mutations to `df` inside this method operate on a copy so the
        caller's DataFrame is never modified.
        """
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
        # alignment is consistent.
        if self.feature_list is None:
            self.feature_list = df.columns.tolist()

        # ── Fill NaN before quantile computation ──────────────────────────
        df_filled = df.fillna(0)

        # ── Quantile clipping (fit) ────────────────────────────────────────
        if self.enable_clipping and self.clip_quantiles:
            lower_q, upper_q = self.clip_quantiles

            for col in df_filled.columns:
                series = df_filled[col]

                if series.nunique(dropna=True) <= 10:
                    # Skip binary / low-cardinality features — clipping
                    # binary flags at quantiles is meaningless.
                    continue

                # ── Cast to Python float before conditional ───────────────
                # .quantile() on nullable-dtype columns may return a Series
                # rather than a scalar. float() coerces to Python scalar so
                # `if pd.notnull(low)` is unambiguous.
                try:
                    low  = float(series.quantile(lower_q))
                    high = float(series.quantile(upper_q))
                except (TypeError, ValueError):
                    continue

                if pd.notnull(low) and pd.notnull(high) and low < high:
                    self.clip_values_[col] = (low, high)
                    self.clip_cols_.append(col)

        # ── Apply clipping to fit data (scaler must see clipped values) ───
        if self.enable_clipping:
            for col, (low, high) in self.clip_values_.items():
                if col in df_filled.columns:
                    df_filled[col] = df_filled[col].clip(lower=low, upper=high)

        # ── Fit StandardScaler (after clipping, on clean data) ────────────
        if self.scale_features:
            from sklearn.preprocessing import StandardScaler
            self.scaler_ = StandardScaler()
            # Fit on the full clipped training data in feature_list order.
            # This guarantees the scaler column order matches transform().
            self.scaler_.fit(df_filled[self.feature_list].values)

            if self.debug:
                print(
                    f"[FIT] StandardScaler fitted on {len(self.feature_list)} features"
                )

        self._is_fitted = True

        if self.debug:
            print(f"[FIT] Features locked  : {len(self.feature_list)}")
            print(f"[FIT] Constant removed : {len(self.constant_cols_)}")
            print(f"[FIT] Clipping cols    : {len(self.clip_cols_)}")
            print(f"[FIT] Scale features   : {self.scale_features}")

        return self

    # =========================================================================
    # TRANSFORM
    # =========================================================================

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply learned transformations to a new DataFrame.

        Steps applied in this exact order:
          strip → align → numeric → inf→NaN → fillna → clip → fillna
          → drop constants → contract assertion → [scale] → memory opt.
        """
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

        # ── Hard assertion: feature contract must be exact ────────────────
        actual_cols = list(df.columns)
        if actual_cols != expected_cols:
            mismatched = set(actual_cols).symmetric_difference(set(expected_cols))
            raise AssertionError(
                f"Feature contract violated after preprocessing!\n"
                f"Mismatched columns: {mismatched}"
            )

        # ── StandardScaler (scale-sensitive models only) ──────────────────
        # Applied AFTER the contract assertion so we scale exactly the
        # features in the locked order — no shape mismatch possible.
        
        scale_features = getattr(self, "scale_features", False)
        scaler_fitted  = getattr(self, "scaler_", None)

        if scale_features and scaler_fitted is not None:
            scaled_values = self.scaler_.transform(df.values)
            df = pd.DataFrame(
                scaled_values,
                index=df.index,
                columns=expected_cols,
                dtype=np.float32,
            )
        else:
            # ── Memory optimisation (GBM path) ────────────────────────────
            # Skipped on the scaled path because we already cast to float32
            # above.  On the unscaled path we still want to downcast.
            df = optimize_memory_usage(df)

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
        """Convenience method: fit on df, then transform df."""
        return self.fit(df).transform(df)

    # =========================================================================
    # COLUMN ALIGNMENT (internal)
    # Aligns an unseen DataFrame to the feature list learned at fit() time.
    # Missing columns are zero-filled; extra columns are dropped.
    # =========================================================================

    def _align_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Align df columns to the feature list from fit().

        This handles two common real-world scenarios:
          1. Test set is missing a feature that appeared in training
             (e.g. a derived feature that was all-zero in one group)
          2. Test set has extra columns not seen during training
        """
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
        """
        Convert all columns to numpy-native numeric types.

        Handles:
          - CategoricalDtype  → float32 (via category codes)
          - Pandas extension types (Int8, Float64, boolean) → numpy equiv.
          - Duplicate column names → take first occurrence
          - Everything else → pd.to_numeric(errors='coerce')
        """
        result = {}
        for col in df.columns:
            s    = df[col]
            # Duplicate column names cause df[col] to return a DataFrame.
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
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
        """
        Serialise the entire fitted pipeline to disk via joblib.

        The StandardScaler (if fitted) is embedded inside `self.scaler_`
        and is saved automatically as part of the pipeline object.
        No separate scaler file is needed.
        """
        import joblib
        joblib.dump(self, path)
        print(f"[PreprocessingPipeline] Saved to {path}")

    @staticmethod
    def load(path: str) -> "PreprocessingPipeline":
        """
        Load a previously saved PreprocessingPipeline from disk.

        Returns the fully fitted pipeline ready to call transform().
        """
        import joblib
        import warnings

        from sklearn.exceptions import InconsistentVersionWarning

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=InconsistentVersionWarning
            )

            pipeline = joblib.load(path)

        print(f"[PreprocessingPipeline] Loaded from {path}")
        return pipeline

    def __setstate__(self, state):
        """
        Ensures older serialized pipeline objects remain compatible
        with newer class versions after unpickling.
        """
        self.__dict__.update(state)

        # Added in v2.0
        if not hasattr(self, "scale_features"):
            self.scale_features = False

        # Added in v2.0
        if not hasattr(self, "scaler_"):
            self.scaler_ = None

    # =========================================================================
    # REPR
    # =========================================================================

    def __repr__(self) -> str:
        status     = "fitted" if self._is_fitted else "not fitted"
        n_features = len(self.feature_list) if self.feature_list else 0
        scaler_str = (
            "StandardScaler" if (self.scale_features and self.scaler_ is not None)
            else "none"
        )
        return (
            f"PreprocessingPipeline("
            f"status={status}, "
            f"features={n_features}, "
            f"clip_quantiles={self.clip_quantiles}, "
            f"constant_cols={len(self.constant_cols_)}, "
            f"clip_cols={len(self.clip_cols_)}, "
            f"scaler={scaler_str}"
            f")"
        )