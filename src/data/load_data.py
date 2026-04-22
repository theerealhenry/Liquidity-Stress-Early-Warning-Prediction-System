"""
Data Loading & Initial Validation Module
======================================

Enhanced Version:
- Strong validation 
- Memory optimization
- Train/Test schema alignment checks
- Target diagnostics
- Missing/zero-heavy feature detection

Design Principles:
- Reproducibility
- Early failure on critical issues
- Zero leakage
- Pipeline compatibility
"""

from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
import numpy as np


# =========================================================
# CONFIGURATION
# =========================================================

REQUIRED_COLUMNS = {"ID", "liquidity_stress_next_30d"}

CATEGORICAL_COLUMNS = [
    "gender",
    "region",
    "segment",
    "earning_pattern",
    "smartphone"
]

NUMERIC_EXCLUDE_COLUMNS = ["ID", "liquidity_stress_next_30d"]

HIGH_MISSING_THRESHOLD = 0.4
HIGH_ZERO_THRESHOLD = 0.8


# =========================================================
# CORE FUNCTIONS
# =========================================================

def load_data(
    train_path: Optional[str] = None,
    test_path: Optional[str] = None,
    validate: bool = True,
    verbose: bool = True
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:

    train_df = _load_single_dataset(train_path, "TRAIN", verbose)
    test_df = _load_single_dataset(test_path, "TEST", verbose)

    if validate:
        if train_df is not None:
            _validate_dataset(train_df, "TRAIN", is_train=True)

        if test_df is not None:
            _validate_dataset(test_df, "TEST", is_train=False)

        if train_df is not None and test_df is not None:
            _check_train_test_consistency(train_df, test_df)

    return train_df, test_df


# =========================================================
# LOAD SINGLE DATASET
# =========================================================

def _load_single_dataset(path: Optional[str], dataset_name: str, verbose: bool):

    if path is None:
        return None

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"{dataset_name} file not found at: {path}")

    df = pd.read_csv(path)

    # Clean column names
    df.columns = [col.strip() for col in df.columns]

    df = _enforce_dtypes(df)
    df = _optimize_memory(df)

    if verbose:
        print(f"\n{'='*60}")
        print(f"{dataset_name} LOADED")
        print(f"{'='*60}")
        print(f"Shape: {df.shape}")
        print(f"Memory usage: {round(df.memory_usage().sum() / 1e6, 2)} MB")
        print(df.dtypes.value_counts())
        print(f"{'='*60}\n")

    return df


# =========================================================
# TYPE ENFORCEMENT
# =========================================================

def _enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    numeric_cols = [
        col for col in df.columns
        if col not in CATEGORICAL_COLUMNS + NUMERIC_EXCLUDE_COLUMNS
    ]

    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    if "liquidity_stress_next_30d" in df.columns:
        df["liquidity_stress_next_30d"] = df["liquidity_stress_next_30d"].astype("Int8")

    return df


# =========================================================
# MEMORY OPTIMIZATION
# =========================================================

def _optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast numeric columns to reduce memory footprint.
    """

    for col in df.select_dtypes(include=["int", "float"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="unsigned")

    return df


# =========================================================
# VALIDATION
# =========================================================

def _validate_dataset(df: pd.DataFrame, dataset_name: str, is_train: bool):

    print(f"🔍 Validating {dataset_name}...")

    _check_required_columns(df, dataset_name, is_train)
    _check_duplicate_ids(df, dataset_name)
    _check_basic_statistics(df, dataset_name)
    _check_missing_and_zero(df, dataset_name)

    if is_train:
        _check_target(df)

    print(f"✅ {dataset_name} validation completed.\n")


# =========================================================
# VALIDATION HELPERS
# =========================================================

def _check_required_columns(df, dataset_name, is_train):

    required = REQUIRED_COLUMNS if is_train else {"ID"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"{dataset_name} missing columns: {missing}")


def _check_duplicate_ids(df, dataset_name):

    if "ID" in df.columns:
        dup = df["ID"].duplicated().sum()

        if dup > 0:
            raise ValueError(f"{dataset_name} has {dup} duplicate IDs")
        else:
            print(f"✔ No duplicate IDs")


def _check_basic_statistics(df, dataset_name):

    numeric_cols = df.select_dtypes(include=["number"]).columns

    if len(numeric_cols) == 0:
        return

    summary = df[numeric_cols].describe().T

    extreme = summary[summary["max"] > 1e9]

    if not extreme.empty:
        print("⚠ Extreme values detected")


def _check_missing_and_zero(df, dataset_name):

    missing = df.isna().mean()
    high_missing = missing[missing > HIGH_MISSING_THRESHOLD]

    if not high_missing.empty:
        print("⚠ High missing features:")
        print(high_missing.sort_values(ascending=False).head(10))

    numeric_df = df.select_dtypes(include=["number"])

    zero_pct = (numeric_df == 0).mean()
    high_zero = zero_pct[zero_pct > HIGH_ZERO_THRESHOLD]

    if not high_zero.empty:
        print("⚠ High zero features:")
        print(high_zero.sort_values(ascending=False).head(10))


def _check_target(df):

    target = df["liquidity_stress_next_30d"]

    print("\n🎯 Target Distribution:")
    print(target.value_counts(normalize=True))

    if not set(target.dropna().unique()).issubset({0, 1}):
        raise ValueError("Target must be binary (0/1)")


def _check_train_test_consistency(train_df, test_df):

    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)

    diff = train_cols.symmetric_difference(test_cols)

    if diff:
        print("⚠ Train/Test column mismatch detected:")
        print(diff)


# =========================================================
# FEATURE SUMMARY
# =========================================================

def generate_feature_summary(df: pd.DataFrame) -> pd.DataFrame:

    summary = []

    for col in df.columns:
        col_data = df[col]

        stats = {
            "feature": col,
            "dtype": str(col_data.dtype),
            "missing_pct": round(col_data.isna().mean() * 100, 2),
        }

        if pd.api.types.is_numeric_dtype(col_data):
            stats.update({
                "zero_pct": round((col_data == 0).mean() * 100, 2),
                "mean": col_data.mean(),
                "std": col_data.std(),
                "min": col_data.min(),
                "max": col_data.max(),
                "skew": col_data.skew()
            })

        summary.append(stats)

    return pd.DataFrame(summary)