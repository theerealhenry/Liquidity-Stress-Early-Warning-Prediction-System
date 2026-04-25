"""
Preprocessing Pipeline (Optimized)
====================================================

Key Improvements:
- Robust dtype enforcement
- Safe column alignment
- Smart clipping (continuous only)
- NaN handling (before + after)
- Constant feature removal
- Fully CV-safe
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
    ):
        self.feature_list = feature_list
        self.clip_quantiles = clip_quantiles

        self.clip_values_ = {}
        self.numeric_cols_ = []
        self.clip_cols_ = []


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

        # Force numeric conversion
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        self.numeric_cols_ = df.columns.tolist()

        # Handle NaNs BEFORE learning clipping
        df = df.replace([np.inf, -np.inf], np.nan)
        df[self.numeric_cols_] = df[self.numeric_cols_].fillna(0)

        # Learn clipping ONLY for continuous features
        if self.clip_quantiles:
            lower_q, upper_q = self.clip_quantiles

            for col in self.numeric_cols_:
                unique_vals = df[col].nunique()

                # Skip binary / low-cardinality features
                if unique_vals <= 10:
                    continue

                low = df[col].quantile(lower_q)
                high = df[col].quantile(upper_q)

                if pd.notnull(low) and pd.notnull(high) and low < high:
                    self.clip_values_[col] = (low, high)
                    self.clip_cols_.append(col)

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

        # Force numeric conversion
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Replace inf
        df = df.replace([np.inf, -np.inf], np.nan)

        # Fill NaNs BEFORE clipping
        df[self.numeric_cols_] = df[self.numeric_cols_].fillna(0)

        # Apply clipping safely
        for col in self.clip_cols_:
            if col in df.columns:
                low, high = self.clip_values_[col]
                df[col] = df[col].clip(lower=low, upper=high)

        # Final NaN safety
        df[self.numeric_cols_] = df[self.numeric_cols_].fillna(0)

        # Remove constant columns (important post-alignment)
        nunique = df.nunique()
        df = df.loc[:, nunique > 1]

        # Memory optimization
        df = optimize_memory_usage(df)

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
        # Add missing columns
        missing_cols = set(self.feature_list) - set(df.columns)
        for col in missing_cols:
            df[col] = 0

        # Remove extra columns
        extra_cols = set(df.columns) - set(self.feature_list)
        if extra_cols:
            df = df.drop(columns=list(extra_cols))

        # Enforce ordering
        df = df[self.feature_list]

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