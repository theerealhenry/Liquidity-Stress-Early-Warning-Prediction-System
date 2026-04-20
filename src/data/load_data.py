"""
Data Loading & Initial Validation Module
======================================

This module provides a single source of truth for loading and performing
lightweight validation on raw datasets (train/test).

Design Principles:
- Reproducibility
- Consistency across notebooks and pipelines
- Minimal transformation (NO feature engineering)
- Early failure on critical data issues

"""

from pathlib import Path
from typing import Tuple, Optional, Dict

import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================

REQUIRED_COLUMNS = {
    "ID",
    "liquidity_stress_next_30d"
}

CATEGORICAL_COLUMNS = [
    "gender",
    "region",
    "segment",
    "earning_pattern",
    "smartphone"
]

NUMERIC_EXCLUDE_COLUMNS = [
    "ID",
    "liquidity_stress_next_30d"
]


# =========================================================
# CORE FUNCTIONS
# =========================================================

def load_data(
    train_path: Optional[str] = None,
    test_path: Optional[str] = None,
    validate: bool = True,
    verbose: bool = True
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load train and/or test datasets with standardized preprocessing.

    Parameters
    ----------
    train_path : str, optional
        Path to training dataset
    test_path : str, optional
        Path to test dataset
    validate : bool, default=True
        Whether to run validation checks
    verbose : bool, default=True
        Print dataset summaries

    Returns
    -------
    train_df : pd.DataFrame or None
    test_df : pd.DataFrame or None
    """

    train_df = _load_single_dataset(train_path, dataset_name="TRAIN", verbose=verbose)
    test_df = _load_single_dataset(test_path, dataset_name="TEST", verbose=verbose)

    if validate:
        if train_df is not None:
            _validate_dataset(train_df, dataset_name="TRAIN", is_train=True)
        if test_df is not None:
            _validate_dataset(test_df, dataset_name="TEST", is_train=False)

    return train_df, test_df


# =========================================================
# INTERNAL HELPERS
# =========================================================

def _load_single_dataset(
    path: Optional[str],
    dataset_name: str,
    verbose: bool
) -> Optional[pd.DataFrame]:
    """
    Load a single dataset and apply basic cleaning.
    """

    if path is None:
        return None

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"{dataset_name} file not found at: {path}")

    df = pd.read_csv(path)

    # Standardize column names
    df.columns = [col.strip() for col in df.columns]

    # Basic cleaning
    df = _enforce_dtypes(df)

    if verbose:
        print(f"\n{'='*50}")
        print(f"{dataset_name} DATA LOADED")
        print(f"{'='*50}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {len(df.columns)}")
        print(df.dtypes.value_counts())
        print(f"{'='*50}\n")

    return df


# =========================================================
# TYPE ENFORCEMENT
# =========================================================

def _enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce consistent data types across datasets.
    """

    df = df.copy()

    # Convert categorical columns
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Convert numeric columns
    numeric_cols = [
        col for col in df.columns
        if col not in CATEGORICAL_COLUMNS + NUMERIC_EXCLUDE_COLUMNS
    ]

    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Convert target to int (if exists)
    if "liquidity_stress_next_30d" in df.columns:
        df["liquidity_stress_next_30d"] = df["liquidity_stress_next_30d"].astype("Int64")

    return df


# =========================================================
# VALIDATION
# =========================================================

def _validate_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    is_train: bool = True
) -> None:
    """
    Run core validation checks.
    """

    print(f"🔍 Validating {dataset_name} dataset...")

    # 1. Required columns
    _check_required_columns(df, dataset_name, is_train)

    # 2. Duplicate IDs
    _check_duplicate_ids(df, dataset_name)

    # 3. Basic sanity checks
    _check_basic_statistics(df, dataset_name)

    print(f"✅ {dataset_name} validation completed.\n")


def _check_required_columns(
    df: pd.DataFrame,
    dataset_name: str,
    is_train: bool
) -> None:
    """
    Ensure required columns exist.
    """

    required = REQUIRED_COLUMNS.copy()

    if not is_train:
        required = {"ID"}  # test set does not have target

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing}"
        )


def _check_duplicate_ids(df: pd.DataFrame, dataset_name: str) -> None:
    """
    Check for duplicate IDs.
    """

    if "ID" not in df.columns:
        return

    duplicates = df["ID"].duplicated().sum()

    if duplicates > 0:
        raise ValueError(
            f"{dataset_name} contains {duplicates} duplicate IDs!"
        )
    else:
        print(f"✔ No duplicate IDs in {dataset_name}")


def _check_basic_statistics(df: pd.DataFrame, dataset_name: str) -> None:
    """
    Basic sanity checks for numerical data.
    """

    numeric_cols = df.select_dtypes(include=["number"]).columns

    if len(numeric_cols) == 0:
        return

    summary = df[numeric_cols].describe().T

    # Detect extreme anomalies (basic check)
    extreme_values = summary[summary["max"] > 1e9]

    if not extreme_values.empty:
        print(f"⚠ Warning: Extreme values detected in {dataset_name}")
        print(extreme_values[["max"]])

    # Check negative values where not expected
    # Can be extended later with domain-specific rules


# =========================================================
# FEATURE SUMMARY GENERATOR (PHASE 1 ARTIFACT)
# =========================================================

def generate_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate feature summary for validation phase.

    Returns DataFrame with:
    - dtype
    - missing %
    - zero %
    - mean, std, min, max
    """

    summary = []

    for col in df.columns:
        col_data = df[col]

        missing_pct = col_data.isna().mean() * 100
        zero_pct = ((col_data == 0).mean() * 100) if pd.api.types.is_numeric_dtype(col_data) else None

        stats = {
            "feature": col,
            "dtype": str(col_data.dtype),
            "missing_pct": round(missing_pct, 2),
            "zero_pct": round(zero_pct, 2) if zero_pct is not None else None,
        }

        if pd.api.types.is_numeric_dtype(col_data):
            stats.update({
                "mean": col_data.mean(),
                "std": col_data.std(),
                "min": col_data.min(),
                "max": col_data.max(),
                "skew": col_data.skew()
            })

        summary.append(stats)

    return pd.DataFrame(summary)