"""
Preprocessing Pipeline (Production-Grade)
======================================

Responsibilities:
- Column alignment (train vs inference consistency)
- Missing value handling
- Type enforcement (safe)
- Numerical stability (inf/NaN)
- Robust outlier clipping
- Memory optimization
- CV-safe transformations

Design Principles:
- Deterministic
- Modular
- Leakage-safe
- Robust to schema drift
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
    if df.empty:
        raise ValueError("Input dataframe is empty")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")


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

        self.clip_values_ = None
        self.numeric_cols_ = None


    # =====================================================
    # FIT
    # =====================================================

    def fit(self, df: pd.DataFrame):
        validate_schema(df)

        df = df.copy()

        # Remove non-feature columns safely
        exclude_cols = [c for c in [TARGET, ID_COL] if c in df.columns]
        df = df.drop(columns=exclude_cols, errors="ignore")

        # Store feature list
        if self.feature_list is None:
            self.feature_list = df.columns.tolist()

        # Identify numeric columns
        self.numeric_cols_ = df.select_dtypes(include=np.number).columns.tolist()

        # Learn clipping thresholds
        if self.clip_quantiles:
            lower_q, upper_q = self.clip_quantiles

            self.clip_values_ = {}
            for col in self.numeric_cols_:
                low = df[col].quantile(lower_q)
                high = df[col].quantile(upper_q)

                if pd.notnull(low) and pd.notnull(high) and low < high:
                    self.clip_values_[col] = (low, high)

        return self


    # =====================================================
    # TRANSFORM
    # =====================================================

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        validate_schema(df)

        df = df.copy()

        # Remove non-feature columns
        df = df.drop(columns=[TARGET, ID_COL], errors="ignore")

        # Align columns
        df = self._align_columns(df)

        # Replace inf
        df = df.replace([np.inf, -np.inf], np.nan)

        # Clip safely
        if self.clip_values_:
            for col, (low, high) in self.clip_values_.items():
                if col in df.columns:
                    df[col] = df[col].clip(lower=low, upper=high)

        # Fill NaNs
        df[self.numeric_cols_] = df[self.numeric_cols_].fillna(0)

        # Memory optimization
        df = optimize_memory_usage(df)

        print(f"[Preprocessing] Output shape: {df.shape}")

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

        # Order columns
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