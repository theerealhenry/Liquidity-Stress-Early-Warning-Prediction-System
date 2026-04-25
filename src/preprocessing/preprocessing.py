"""
Preprocessing Pipeline (Final)
====================================================

Key Features:
- Strict feature contract enforcement
- CV-safe transformations
- Robust clipping (continuous only, unbiased)
- Stable column alignment
- Constant feature handling (fit-time only)
- Defensive assertions
- Memory optimization
- Debug visibility
"""

from typing import List, Optional
import numpy as np
import pandas as pd

EPS = 1e-6

TARGET = "liquidity_stress_next_30d"
ID_COL = "ID"


# =========================================================
# SCHEMA VALIDATION
# =========================================================

def validate_schema(df: pd.DataFrame):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    if df.empty:
        raise ValueError("Input dataframe is empty")


# =========================================================
# MEMORY OPTIMIZATION
# =========================================================

def optimize_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")

    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    return df


# =========================================================
# PREPROCESSING PIPELINE
# =========================================================

class PreprocessingPipeline:
    def __init__(
        self,
        feature_list: Optional[List[str]] = None,
        clip_quantiles: Optional[tuple] = (0.001, 0.999),
        enable_clipping: bool = True,
        debug: bool = True,
    ):
        self.feature_list = feature_list
        self.clip_quantiles = clip_quantiles
        self.enable_clipping = enable_clipping
        self.debug = debug

        self.clip_values_ = {}
        self.clip_cols_ = []
        self.constant_cols_ = []


    # =====================================================
    # FIT
    # =====================================================

    def fit(self, df: pd.DataFrame):
        validate_schema(df)
        df = df.copy()

        # Drop non-feature columns
        df = df.drop(columns=[TARGET, ID_COL], errors="ignore")

        # Store feature list
        if self.feature_list is None:
            self.feature_list = df.columns.tolist()

        # Convert to numeric safely
        df = df.apply(pd.to_numeric, errors="coerce")

        # Replace inf BEFORE stats
        df = df.replace([np.inf, -np.inf], np.nan)

        # -----------------------------------------------
        # CONSTANT COLUMN DETECTION (FIT ONLY)
        # -----------------------------------------------
        nunique = df.nunique(dropna=True)
        self.constant_cols_ = nunique[nunique <= 1].index.tolist()

        # Remove constants BEFORE learning clipping
        df = df.drop(columns=self.constant_cols_, errors="ignore")

        # -----------------------------------------------
        # CLIPPING (UNBIASED)
        # -----------------------------------------------
        if self.enable_clipping and self.clip_quantiles:
            lower_q, upper_q = self.clip_quantiles

            for col in df.columns:
                series = df[col]

                # Skip low-cardinality features
                if series.nunique(dropna=True) <= 10:
                    continue

                low = series.quantile(lower_q)
                high = series.quantile(upper_q)

                if pd.notnull(low) and pd.notnull(high) and low < high:
                    self.clip_values_[col] = (low, high)
                    self.clip_cols_.append(col)

        if self.debug:
            print(f"[FIT] Features: {len(self.feature_list)}")
            print(f"[FIT] Constant columns removed: {len(self.constant_cols_)}")
            print(f"[FIT] Clipping columns: {len(self.clip_cols_)}")

        return self


    # =====================================================
    # TRANSFORM
    # =====================================================

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        validate_schema(df)
        df = df.copy()

        # Drop non-feature columns
        df = df.drop(columns=[TARGET, ID_COL], errors="ignore")

        # Align columns FIRST
        df = self._align_columns(df)

        # Convert to numeric
        df = df.apply(pd.to_numeric, errors="coerce")

        # Replace inf
        df = df.replace([np.inf, -np.inf], np.nan)

        # Fill NaNs BEFORE clipping
        df = df.fillna(0)

        # -----------------------------------------------
        # APPLY CLIPPING
        # -----------------------------------------------
        if self.enable_clipping:
            for col, (low, high) in self.clip_values_.items():
                if col in df.columns:
                    df[col] = df[col].clip(lower=low, upper=high)

        # Final NaN safety
        df = df.fillna(0)

        # -----------------------------------------------
        # DROP CONSTANT COLUMNS (CONSISTENT WITH FIT)
        # -----------------------------------------------
        df = df.drop(columns=self.constant_cols_, errors="ignore")

        # -----------------------------------------------
        # FINAL FEATURE CONTRACT ENFORCEMENT
        # -----------------------------------------------
        expected_cols = [c for c in self.feature_list if c not in self.constant_cols_]
        df = df[expected_cols]

        # -----------------------------------------------
        # MEMORY OPTIMIZATION
        # -----------------------------------------------
        df = optimize_memory_usage(df)

        # -----------------------------------------------
        # ASSERTIONS (CRITICAL SAFETY)
        # -----------------------------------------------
        assert list(df.columns) == expected_cols, "Feature mismatch after preprocessing!"

        if self.debug:
            print(f"[TRANSFORM] Output shape: {df.shape}")

        return df


    # =====================================================
    # FIT + TRANSFORM
    # =====================================================

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


    # =====================================================
    # COLUMN ALIGNMENT
    # =====================================================

    def _align_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        # Identify missing and extra columns
        missing_cols = list(set(self.feature_list) - set(df.columns))
        extra_cols = list(set(df.columns) - set(self.feature_list))

        # Drop extra columns first
        if extra_cols:
            df = df.drop(columns=extra_cols)

        # Add missing columns in ONE operation (no fragmentation)
        if missing_cols:
            missing_df = pd.DataFrame(
                0,
                index=df.index,
                columns=missing_cols
            )
            df = pd.concat([df, missing_df], axis=1)

        # Enforce correct order
        df = df[self.feature_list]

        # Defragment memory (VERY IMPORTANT)
        df = df.copy()

        return df


    # =====================================================
    # SAVE / LOAD
    # =====================================================

    def save(self, path: str):
        import joblib
        joblib.dump(self, path)

    @staticmethod
    def load(path: str):
        import joblib
        return joblib.load(path)