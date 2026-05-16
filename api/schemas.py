"""
API Schemas — Pydantic v2 Request / Response Contracts
=======================================================
Project  : AI4EAC Liquidity Stress Early Warning (Zindi Africa)
Module   : api/schemas.py
Author   : Henry Otsyula
Version  : 1.0.0

Overview
--------
Defines the complete Pydantic v2 data contract for the liquidity-stress
prediction API.  Every public endpoint in ``api/app.py`` uses one of the
models defined here; nothing else crosses the HTTP boundary.

Design principles
-----------------
1.  SCHEMA IS THE SINGLE SOURCE OF TRUTH FOR THE HTTP SURFACE
    All 183 raw ``Test.csv`` column names appear here exactly once, in the
    same order and with the same naming convention as the competition data.
    Any rename in the data dictionary requires a change here — nowhere else.

2.  NULL = ZERO ACTIVITY (competition spec)
    The training data specification states that null values indicate zero
    activity, not missing data.  All 168 transaction fields therefore
    default to ``0.0``.  The six balance fields and profile fields default
    to ``None`` so the API can distinguish "not provided" from "provided as
    zero" — the predictor fills balance ``None`` → ``0.0`` before feature
    engineering, matching the preprocessing pipeline behaviour.

3.  FIELD-LEVEL VALIDATION ENCODES DOMAIN KNOWLEDGE
    Constraints are derived from the competition data dictionary, the
    SHAP analysis (notebook 13), and the OOF prediction analysis:
      - Monetary amounts (total_value, highest_amount) are non-negative
      - Volume counts are non-negative
      - Unique entity counts are non-negative
      - Daily average balance is unbounded (overdraft positions are valid;
        balance_slope ±40,000 was observed in SHAP analysis)
      - x_90_d_activity_rate is a bounded rate [0.0, 1.0]
      - Categorical fields are normalised to canonical forms on ingestion

4.  CROSS-FIELD VALIDATORS CATCH IMPOSSIBLE INPUTS
    A highest_amount > total_value for the same group-month is physically
    impossible (the largest single transaction cannot exceed total volume).
    A model that receives such inputs would produce garbage silently without
    this guard.

5.  RESPONSE MODELS CARRY SHAP-INFORMED CONTEXT
    The PredictionResponse includes RiskFactor objects derived from the SHAP
    theme analysis (notebook 13):
      - Balance Deterioration (36% of signal): balance_slope, daily_avg_bal
      - Income Stability (19% of signal): inflow_slope, deposit recency
      - Spending Pressure, Network Contraction
    This allows API consumers to surface actionable explanations to end users
    without running a separate SHAP inference pass.

6.  AUDIT TRAIL ON EVERY RESPONSE
    Every response carries a ``request_id`` (UUID4), ``predicted_at``
    timestamp, and ``processing_time_ms`` so operations teams can correlate
    API logs with model artefact versions.

Column structure (183 fields in Test.csv)
------------------------------------------
  1 × ID                    customer identifier (excluded from features)
  6 × daily_avg_bal         M1–M6 daily average account balance
  8 × profile               gender, region, segment, earning_pattern,
                             smartphone, age, x_90_d_activity_rate, arpu
  168 × transaction         7 groups × 6 months × 4 suffixes:
    Groups (inflow):  deposit, received, transfer_from_bank
    Groups (outflow): withdraw, merchantpay, paybill, mm_send
    Suffixes:
      total_value       total monetary volume (KES)
      volume            transaction count
      highest_amount    largest single transaction amount (KES)
      {entity}          unique counterparty count
        deposit/withdraw  → agents
        mm_send           → recipients
        received          → senders
        merchantpay       → merchants
        paybill           → companies
        transfer_from_bank→ banks

Model ensemble (v5.1) — OOF composite scores
---------------------------------------------
  LightGBM  0.19557  weight 0.17813
  XGBoost   0.19350  weight 0.39721  ← dominant
  CatBoost  0.19430  weight 0.38777
  LogReg    0.26003  weight 0.00000  (diversity, zero contribution)
  TabNet    0.25296  weight 0.03689
  Ensemble  0.19144  optimised weighted average

Risk tier thresholds (from Phase 11 specification)
---------------------------------------------------
  low    : stress_probability < 0.25
  medium : 0.25 ≤ stress_probability < 0.60
  high   : stress_probability ≥ 0.60

Changelog
---------
1.0.0  Initial production release.  Full 183-field input schema, SHAP-
       informed response enrichment, Pydantic v2 model_validator cross-field
       checks, to_dataframe() / from_csv_row() helper methods.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# =============================================================================
# CONSTANTS
# =============================================================================

#: Competition metric: 0.6 × LogLoss + 0.4 × (1 − AUC).  Lower is better.
COMPETITION_METRIC = "0.6 × LogLoss + 0.4 × (1 − AUC)"

#: Best OOF composite score achieved by the production ensemble.
ENSEMBLE_OOF_SCORE: float = 0.19144

#: Production ensemble version.
ENSEMBLE_VERSION: str = "v5.1"

#: Risk tier probability thresholds (inclusive lower, exclusive upper).
RISK_TIER_LOW_MAX: float = 0.25
RISK_TIER_MEDIUM_MAX: float = 0.60

#: Ordered month identifiers.  M1 = most recent, M6 = oldest (6 months ago).
MONTHS: List[str] = ["m1", "m2", "m3", "m4", "m5", "m6"]

#: All seven transaction group names.
TRANSACTION_GROUPS: List[str] = [
    "deposit",
    "received",
    "transfer_from_bank",
    "withdraw",
    "merchantpay",
    "paybill",
    "mm_send",
]

#: Inflow groups (money entering the wallet).
INFLOW_GROUPS: List[str] = ["deposit", "received", "transfer_from_bank"]

#: Outflow groups (money leaving the wallet).
OUTFLOW_GROUPS: List[str] = ["withdraw", "merchantpay", "paybill", "mm_send"]

#: Maps each transaction group to its unique-counterparty column suffix.
ENTITY_SUFFIX_MAP: Dict[str, str] = {
    "deposit":             "agents",
    "withdraw":            "agents",
    "mm_send":             "recipients",
    "received":            "senders",
    "merchantpay":         "merchants",
    "paybill":             "companies",
    "transfer_from_bank":  "banks",
}

#: Per-model OOF composite scores for documentation and audit.
MODEL_OOF_SCORES: Dict[str, float] = {
    "lightgbm": 0.19557,
    "xgboost":  0.19350,
    "catboost": 0.19430,
    "logreg":   0.26003,
    "tabnet":   0.25296,
    "ensemble": ENSEMBLE_OOF_SCORE,
}

#: Production ensemble weights (from ensemble_weights.json, run_20260508_054540).
ENSEMBLE_WEIGHTS: Dict[str, float] = {
    "lightgbm": 0.17813,
    "xgboost":  0.39721,
    "catboost":  0.38777,
    "logreg":   0.00000,
    "tabnet":   0.03689,
}

#: SHAP theme importance (notebook 13, XGBoost v6 825-feature model).
SHAP_THEME_IMPORTANCE: Dict[str, float] = {
    "balance_deterioration":  0.36,
    "income_stability":       0.19,
    "spending_pressure":      0.15,
    "network_contraction":    0.12,
    "transaction_frequency":  0.10,
    "other":                  0.08,
}

#: Top SHAP features by mean |SHAP| (from shap_feature_importance.csv).
TOP_SHAP_FEATURES: List[str] = [
    "balance_slope",
    "m4_daily_avg_bal",
    "inflow_slope",
    "m3_daily_avg_bal",
    "balance_trend_3m",
    "m2_daily_avg_bal",
    "inflow_volume_recency_ratio",
    "withdraw_recency_x_spend_ratio",
    "balance_cv_x_drawdown",
    "deposit_recency_ratio",
]


# =============================================================================
# ENUMERATIONS
# =============================================================================

class RiskTier(str, Enum):
    """
    Customer liquidity risk tier based on 30-day stress probability.

    Thresholds are calibrated to the training set positive rate (15%) and
    were fixed in the Phase 11 API specification.
    """
    LOW    = "low"     #: stress_probability < 0.25
    MEDIUM = "medium"  #: 0.25 ≤ stress_probability < 0.60
    HIGH   = "high"    #: stress_probability ≥ 0.60


class HealthStatus(str, Enum):
    """Operational health of the prediction service."""
    OK       = "ok"        #: All 5 models loaded, all calibrators present.
    DEGRADED = "degraded"  #: Partial model set loaded; predictions proceed with warning.
    DOWN     = "down"      #: Critical failure; no predictions possible.


class RiskTheme(str, Enum):
    """
    SHAP-derived thematic risk drivers (notebook 13).

    Themes account for 55% of total predictive signal:
      - Balance Deterioration: 36%
      - Income Stability:      19%
    """
    BALANCE_DETERIORATION  = "balance_deterioration"
    INCOME_STABILITY       = "income_stability"
    SPENDING_PRESSURE      = "spending_pressure"
    NETWORK_CONTRACTION    = "network_contraction"
    TRANSACTION_FREQUENCY  = "transaction_frequency"


# =============================================================================
# NESTED RESPONSE COMPONENTS
# =============================================================================

class ModelInfo(BaseModel):
    """
    Ensemble model metadata embedded in every prediction response.

    Provides a complete audit trail so any prediction can be traced back to
    the exact ensemble version, run directory, and OOF performance metrics.
    """
    model_config = ConfigDict(frozen=True)

    ensemble_version: str = Field(
        default=ENSEMBLE_VERSION,
        description="Ensemble pipeline version string.",
        examples=["v5.1"],
    )
    oof_composite_score: float = Field(
        default=ENSEMBLE_OOF_SCORE,
        description=(
            f"Out-of-fold composite score ({COMPETITION_METRIC}). "
            "Lower is better.  This is the honest OOF estimate — the actual "
            "Zindi leaderboard score may differ."
        ),
        examples=[0.19144],
    )
    competition_metric: str = Field(
        default=COMPETITION_METRIC,
        description="Competition scoring formula.",
    )
    contributing_models: Dict[str, float] = Field(
        default=ENSEMBLE_WEIGHTS,
        description=(
            "Per-model ensemble weights from the optimised weighted average strategy. "
            "LogReg weight=0.000 is correct — it contributes diversity during training "
            "but zero weight in the final blend."
        ),
    )
    model_oof_scores: Dict[str, float] = Field(
        default=MODEL_OOF_SCORES,
        description="Individual model OOF composite scores before ensembling.",
    )
    run_id: str = Field(
        default="run_20260508_054540",
        description="Production ensemble run directory timestamp identifier.",
    )


class InputSummary(BaseModel):
    """
    Echo of the key input signals used by the SHAP top features.

    This is NOT feature-engineering output — it reflects the raw input values
    that feed the most predictive features (identified from notebook 13).
    API consumers can surface these to loan officers or risk analysts without
    needing to run the full feature engineering pipeline.
    """
    model_config = ConfigDict(frozen=True)

    m1_daily_avg_bal: Optional[float] = Field(
        default=None,
        description="Current month daily average balance (KES).  "
                    "Primary input to the dominant SHAP feature 'balance_slope'.",
    )
    m6_daily_avg_bal: Optional[float] = Field(
        default=None,
        description="Oldest month daily average balance (KES).  "
                    "Used with m1 to compute 6-month balance trend.",
    )
    total_deposit_6m: Optional[float] = Field(
        default=None,
        description="Total deposit value over 6 months (KES).  "
                    "Primary input to income stability features.",
    )
    total_withdraw_6m: Optional[float] = Field(
        default=None,
        description="Total withdrawal value over 6 months (KES).  "
                    "Primary input to spending pressure features.",
    )
    deposit_m1_vs_m46_ratio: Optional[float] = Field(
        default=None,
        description=(
            "Deposit recency ratio: M1–M3 avg deposit / M4–M6 avg deposit. "
            "The 10th-ranked SHAP feature (deposit_recency_ratio).  "
            "> 1.0 = increasing recent deposits; < 1.0 = declining."
        ),
    )
    recent_inflow_volume: Optional[float] = Field(
        default=None,
        description="Total inflow transaction count in M1 (most recent month).",
    )
    recent_outflow_volume: Optional[float] = Field(
        default=None,
        description="Total outflow transaction count in M1 (most recent month).",
    )
    network_breadth_m1: Optional[float] = Field(
        default=None,
        description=(
            "Total unique counterparties across all groups in M1.  "
            "Network contraction (lower recent vs past) signals financial stress."
        ),
    )
    balance_trend_direction: Optional[str] = Field(
        default=None,
        description=(
            "Qualitative balance trend: 'improving', 'stable', or 'deteriorating'.  "
            "Derived from m1_daily_avg_bal vs m6_daily_avg_bal.  "
            "Critical insight: customers with 'stable' balance but collapsing inflow "
            "represent the model's known 'balance-lag' false-negative failure mode."
        ),
        examples=["improving", "stable", "deteriorating"],
    )


class RiskFactor(BaseModel):
    """
    A single SHAP-informed risk factor contributing to the stress probability.

    Risk factors are computed from raw input signals without running the full
    825-feature SHAP pipeline.  They represent the thematic groupings identified
    in notebook 13 (shap_theme_importance.csv), not individual feature SHAP values.
    """
    model_config = ConfigDict(frozen=True)

    theme: RiskTheme = Field(
        description="The thematic risk category this factor belongs to.",
    )
    signal: str = Field(
        description="Short human-readable description of the detected pattern.",
        examples=[
            "Balance declining over 6 months",
            "Deposit frequency dropped 60% in last 3 months",
            "Withdrawal-to-income ratio exceeds 0.90",
        ],
    )
    severity: Literal["low", "moderate", "high"] = Field(
        description=(
            "Severity of this individual risk factor.  "
            "The overall stress_probability is the ensemble output — "
            "individual factor severity is indicative only."
        ),
    )
    theme_importance: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of total SHAP signal attributable to this theme across "
            "all training customers (from notebook 13).  "
            "Balance Deterioration=0.36, Income Stability=0.19."
        ),
    )
    supporting_signals: List[str] = Field(
        default_factory=list,
        description=(
            "Raw input fields or derived patterns that triggered this risk factor.  "
            "Empty list if no significant signal detected."
        ),
    )


class PredictionMetadata(BaseModel):
    """
    Operational metadata attached to every prediction for observability.
    """
    model_config = ConfigDict(frozen=True)

    request_id: str = Field(
        description="UUID4 identifier for this prediction request.  "
                    "Use for log correlation and auditing.",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )
    predicted_at: datetime = Field(
        description="UTC timestamp when the prediction was generated.",
    )
    processing_time_ms: float = Field(
        ge=0.0,
        description="Wall-clock time from request receipt to response generation (ms).",
    )
    feature_count: int = Field(
        default=825,
        description=(
            "Number of engineered features passed to the ensemble models.  "
            "Production value is 825 (184 raw → 826 total → 825 after ID drop)."
        ),
    )
    pipeline_version: str = Field(
        default="feature_engineering_v2.2.1",
        description="Feature engineering pipeline version used for this prediction.",
    )
    calibration_applied: bool = Field(
        default=True,
        description=(
            "Whether Platt calibration was applied to each model's raw output.  "
            "Should always be True in production.  Raw GBM outputs have near-unity "
            "Platt slopes; LogReg and TabNet undergo significant recalibration "
            "(slopes 5.47 and 4.37 respectively)."
        ),
    )


class ArtefactCounts(BaseModel):
    """Model artefact inventory reported by the health endpoint."""
    model_config = ConfigDict(frozen=True)

    fold_models_loaded: int = Field(
        description="Number of fold models loaded (expected: 25 = 5 models × 5 folds).",
        ge=0,
        le=25,
    )
    preprocessors_loaded: int = Field(
        description="Number of fitted PreprocessingPipeline objects (expected: 5).",
        ge=0,
        le=5,
    )
    calibrators_loaded: int = Field(
        description="Number of Platt calibrators loaded (expected: 5).",
        ge=0,
        le=5,
    )
    ensemble_weights_loaded: bool = Field(
        description="Whether ensemble_weights.json was successfully parsed.",
    )
    total_artefact_size_mb: Optional[float] = Field(
        default=None,
        description="Approximate combined size of all loaded artefacts in memory (MB).",
    )


# =============================================================================
# REQUEST SCHEMA — CustomerFeatures
# =============================================================================

class CustomerFeatures(BaseModel):
    """
    Raw feature input for a single mobile money customer snapshot.

    Represents one customer's 6-month transaction history in the same wide-format
    structure as the competition ``Test.csv`` file.  The 183 fields map directly
    to the raw columns; no feature engineering is expected from the caller.

    All transaction fields default to ``0.0`` (null = zero activity per
    competition specification).  Balance and profile fields default to ``None``
    so the predictor can apply appropriate fill strategies.

    Field naming convention
    -----------------------
    Transaction fields follow the pattern::

        {month}_{group}_{suffix}

    where:
      - ``{month}``  ∈ {m1, m2, m3, m4, m5, m6}  (m1 = most recent)
      - ``{group}``  ∈ {deposit, received, transfer_from_bank,
                         withdraw, merchantpay, paybill, mm_send}
      - ``{suffix}`` ∈ {total_value, volume, highest_amount, {entity}}
        with entity = agents | recipients | senders | merchants |
                      companies | banks

    Balance fields follow the pattern::

        {month}_daily_avg_bal

    Examples
    --------
    Minimal request (all transactions zero, single-month balance provided):

    .. code-block:: json

        {
            "ID": "CUST_001",
            "m1_daily_avg_bal": 12500.0,
            "m1_deposit_total_value": 45000.0,
            "m1_deposit_volume": 3.0,
            "segment": "MVC"
        }

    Full request with 6 months of history — see the OpenAPI schema for the
    complete field list.
    """

    model_config = ConfigDict(
        # Allow extra fields to be ignored (forward-compatibility: if the
        # competition adds columns in future, existing callers won't break).
        extra="ignore",
        # Validate on assignment so mutated objects stay consistent.
        validate_assignment=True,
        # Populate by field name (not alias) for JSON payloads.
        populate_by_name=True,
        # Enrich OpenAPI schema with examples.
        json_schema_extra={
            "title": "Customer Feature Snapshot",
            "description": (
                "6-month mobile money transaction history for one customer. "
                "Null values for transaction fields are treated as zero activity "
                "per the AI4EAC competition data specification."
            ),
            "examples": [
                {
                    "ID": "CUST_EXAMPLE_001",
                    "segment": "MVC",
                    "gender": "M",
                    "age": 34,
                    "arpu": 8500.0,
                    "m1_daily_avg_bal": 12500.0,
                    "m2_daily_avg_bal": 18200.0,
                    "m3_daily_avg_bal": 22000.0,
                    "m1_deposit_total_value": 45000.0,
                    "m1_deposit_volume": 3.0,
                    "m1_withdraw_total_value": 52000.0,
                    "m1_withdraw_volume": 8.0,
                },
            ],
        },
    )

    # ── Customer Identifier ───────────────────────────────────────────────────
    ID: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description=(
            "Unique customer identifier.  Passed through to the response as "
            "``customer_id`` for request-response correlation.  "
            "Must match the ID format in the competition dataset."
        ),
        examples=["ID_00001", "CUST_001", "MOBILE_KE_34821"],
    )

    # ── Profile / Demographic Fields ──────────────────────────────────────────
    gender: Optional[str] = Field(
        default=None,
        description=(
            "Customer gender. Accepted values: 'M' or 'F' (case-insensitive). "
            "Normalised to uppercase on ingestion.  "
            "Encoded to binary is_male=1/0 by feature engineering."
        ),
        examples=["M", "F"],
    )
    region: Optional[str] = Field(
        default=None,
        description=(
            "Geographic region of the customer.  Passed through as a categorical "
            "column; specific values depend on the operator's regional taxonomy."
        ),
        examples=["Nairobi", "Coast", "Central"],
    )
    segment: Optional[str] = Field(
        default=None,
        description=(
            "Customer value segment.  Accepted values: 'LVC', 'MVC', 'HVC' "
            "(Low/Medium/High Value Customer).  Normalised to uppercase. "
            "Ordinal encoded: LVC=0, MVC=1, HVC=2. "
            "SHAP analysis (notebook 13) shows MVC is the hardest segment to classify "
            "(mean |SHAP| = 0.0081 vs HVC = 0.0103)."
        ),
        examples=["LVC", "MVC", "HVC"],
    )
    earning_pattern: Optional[str] = Field(
        default=None,
        description=(
            "Categorical earning pattern label.  Ordinal encoded by sorted unique "
            "value order.  Specific values are operator-defined."
        ),
        examples=["Regular", "Irregular", "Seasonal"],
    )
    smartphone: Optional[str] = Field(
        default=None,
        description=(
            "Whether the customer uses a smartphone.  Accepted values: "
            "'yes', 'no', '1', '0', 'true', 'false' (case-insensitive). "
            "Encoded to binary has_smartphone=1/0."
        ),
        examples=["yes", "no"],
    )
    age: Optional[float] = Field(
        default=None,
        ge=1.0,
        le=120.0,
        description=(
            "Customer age in years.  Passed through as a raw numeric feature "
            "(listed in _RAW_PASSTHROUGH_COLS in feature engineering). "
            "Must be positive."
        ),
        examples=[28.0, 45.0],
    )
    x_90_d_activity_rate: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of days in the last 90 days the customer was active "
            "(had at least one transaction).  Bounded [0.0, 1.0].  "
            "Passed through as a raw numeric feature."
        ),
        examples=[0.72, 0.15],
    )
    arpu: Optional[float] = Field(
        default=None,
        ge=0.0,
        description=(
            "Average Revenue Per User (KES).  Used in _banking_features() "
            "to compute deposit intensity and bank transfer relative to ARPU. "
            "Must be non-negative."
        ),
        examples=[8500.0, 1200.0],
    )

    # ── Daily Average Balance — 6 Months ─────────────────────────────────────
    # Balance can be negative (overdraft).  No ge constraint applied.
    # These are the primary inputs to the dominant SHAP feature 'balance_slope'
    # (mean |SHAP| = 0.793, over 2× the second-ranked feature).
    #
    # The 'balance-lag' failure mode (false negatives): customers with positive
    # or mildly-negative balance_slope despite catastrophic inflow_slope decline.
    # Rule override: inflow_slope < -5,000 AND balance_slope > 0.
    m1_daily_avg_bal: Optional[float] = Field(
        default=None,
        description=(
            "Daily average account balance, most recent month (KES). "
            "Can be negative (overdraft).  "
            "Primary input to 'balance_slope', the dominant SHAP feature (|SHAP|=0.793). "
            "Near-vertical bifurcation at balance_slope≈0: "
            "negative slope → SHAP +1.0 to +2.5; positive → SHAP −0.5 to −1.0."
        ),
        examples=[12500.0, -3200.0],
    )
    m2_daily_avg_bal: Optional[float] = Field(
        default=None,
        description="Daily average balance, 2 months ago (KES). Can be negative.",
        examples=[18200.0],
    )
    m3_daily_avg_bal: Optional[float] = Field(
        default=None,
        description=(
            "Daily average balance, 3 months ago (KES). Can be negative.  "
            "The 4th-ranked SHAP feature (m3_daily_avg_bal, |SHAP|=0.210)."
        ),
        examples=[22000.0],
    )
    m4_daily_avg_bal: Optional[float] = Field(
        default=None,
        description=(
            "Daily average balance, 4 months ago (KES). Can be negative.  "
            "The 2nd-ranked SHAP feature (m4_daily_avg_bal, |SHAP|=0.287)."
        ),
        examples=[25000.0],
    )
    m5_daily_avg_bal: Optional[float] = Field(
        default=None,
        description="Daily average balance, 5 months ago (KES). Can be negative.",
        examples=[26500.0],
    )
    m6_daily_avg_bal: Optional[float] = Field(
        default=None,
        description="Daily average balance, oldest month (KES). Can be negative.",
        examples=[28000.0],
    )

    # =========================================================================
    # TRANSACTION FIELDS — 168 columns
    # 7 groups × 6 months × 4 suffixes (total_value, volume, highest_amount, entity)
    # All default to 0.0 (null = zero activity per competition spec).
    # All are non-negative (ge=0.0): amounts, counts, and entity counts.
    # =========================================================================

    # ── Deposit (inflow) ──────────────────────────────────────────────────────
    # M1: most recent month
    m1_deposit_total_value: Optional[float] = Field(default=0.0, ge=0.0, description="Total deposit value, most recent month (KES).", examples=[45000.0])
    m1_deposit_volume: Optional[float] = Field(default=0.0, ge=0.0, description="Number of deposit transactions, most recent month.")
    m1_deposit_highest_amount: Optional[float] = Field(default=0.0, ge=0.0, description="Largest single deposit, most recent month (KES).")
    m1_deposit_agents: Optional[float] = Field(default=0.0, ge=0.0, description="Unique agents used for deposits, most recent month. Network contraction in agent count is a leading stress indicator.")
    # M2
    m2_deposit_total_value: Optional[float] = Field(default=0.0, ge=0.0, description="Total deposit value, 2 months ago (KES).")
    m2_deposit_volume: Optional[float] = Field(default=0.0, ge=0.0, description="Number of deposit transactions, 2 months ago.")
    m2_deposit_highest_amount: Optional[float] = Field(default=0.0, ge=0.0, description="Largest single deposit, 2 months ago (KES).")
    m2_deposit_agents: Optional[float] = Field(default=0.0, ge=0.0, description="Unique agents for deposits, 2 months ago.")
    # M3
    m3_deposit_total_value: Optional[float] = Field(default=0.0, ge=0.0, description="Total deposit value, 3 months ago (KES).")
    m3_deposit_volume: Optional[float] = Field(default=0.0, ge=0.0, description="Number of deposit transactions, 3 months ago.")
    m3_deposit_highest_amount: Optional[float] = Field(default=0.0, ge=0.0, description="Largest single deposit, 3 months ago (KES).")
    m3_deposit_agents: Optional[float] = Field(default=0.0, ge=0.0, description="Unique agents for deposits, 3 months ago.")
    # M4
    m4_deposit_total_value: Optional[float] = Field(default=0.0, ge=0.0, description="Total deposit value, 4 months ago (KES).")
    m4_deposit_volume: Optional[float] = Field(default=0.0, ge=0.0, description="Number of deposit transactions, 4 months ago.")
    m4_deposit_highest_amount: Optional[float] = Field(default=0.0, ge=0.0, description="Largest single deposit, 4 months ago (KES).")
    m4_deposit_agents: Optional[float] = Field(default=0.0, ge=0.0, description="Unique agents for deposits, 4 months ago.")
    # M5
    m5_deposit_total_value: Optional[float] = Field(default=0.0, ge=0.0, description="Total deposit value, 5 months ago (KES).")
    m5_deposit_volume: Optional[float] = Field(default=0.0, ge=0.0, description="Number of deposit transactions, 5 months ago.")
    m5_deposit_highest_amount: Optional[float] = Field(default=0.0, ge=0.0, description="Largest single deposit, 5 months ago (KES).")
    m5_deposit_agents: Optional[float] = Field(default=0.0, ge=0.0, description="Unique agents for deposits, 5 months ago.")
    # M6
    m6_deposit_total_value: Optional[float] = Field(default=0.0, ge=0.0, description="Total deposit value, oldest month (KES).")
    m6_deposit_volume: Optional[float] = Field(default=0.0, ge=0.0, description="Number of deposit transactions, oldest month.")
    m6_deposit_highest_amount: Optional[float] = Field(default=0.0, ge=0.0, description="Largest single deposit, oldest month (KES).")
    m6_deposit_agents: Optional[float] = Field(default=0.0, ge=0.0, description="Unique agents for deposits, oldest month.")

    # ── Received (inflow — peer-to-peer receipts) ─────────────────────────────
    m1_received_total_value: Optional[float] = Field(default=0.0, ge=0.0, description="Total value received from peers, most recent month (KES).")
    m1_received_volume: Optional[float] = Field(default=0.0, ge=0.0, description="Number of incoming P2P transfers, most recent month.")
    m1_received_highest_amount: Optional[float] = Field(default=0.0, ge=0.0, description="Largest single incoming transfer, most recent month (KES).")
    m1_received_senders: Optional[float] = Field(default=0.0, ge=0.0, description="Unique senders, most recent month. Declining sender count signals network contraction.")
    m2_received_total_value: Optional[float] = Field(default=0.0, ge=0.0, description="Total received value, 2 months ago (KES).")
    m2_received_volume: Optional[float] = Field(default=0.0, ge=0.0, description="Incoming P2P transfers, 2 months ago.")
    m2_received_highest_amount: Optional[float] = Field(default=0.0, ge=0.0, description="Largest incoming transfer, 2 months ago (KES).")
    m2_received_senders: Optional[float] = Field(default=0.0, ge=0.0, description="Unique senders, 2 months ago.")
    m3_received_total_value: Optional[float] = Field(default=0.0, ge=0.0, description="Total received value, 3 months ago (KES).")
    m3_received_volume: Optional[float] = Field(default=0.0, ge=0.0, description="Incoming P2P transfers, 3 months ago.")
    m3_received_highest_amount: Optional[float] = Field(default=0.0, ge=0.0, description="Largest incoming transfer, 3 months ago (KES).")
    m3_received_senders: Optional[float] = Field(default=0.0, ge=0.0, description="Unique senders, 3 months ago.")
    m4_received_total_value: Optional[float] = Field(default=0.0, ge=0.0, description="Total received value, 4 months ago (KES).")
    m4_received_volume: Optional[float] = Field(default=0.0, ge=0.0, description="Incoming P2P transfers, 4 months ago.")
    m4_received_highest_amount: Optional[float] = Field(default=0.0, ge=0.0, description="Largest incoming transfer, 4 months ago (KES).")
    m4_received_senders: Optional[float] = Field(default=0.0, ge=0.0, description="Unique senders, 4 months ago.")
    m5_received_total_value: Optional[float] = Field(default=0.0, ge=0.0, description="Total received value, 5 months ago (KES).")
    m5_received_volume: Optional[float] = Field(default=0.0, ge=0.0, description="Incoming P2P transfers, 5 months ago.")
    m5_received_highest_amount: Optional[float] = Field(default=0.0, ge=0.0, description="Largest incoming transfer, 5 months ago (KES).")
    m5_received_senders: Optional[float] = Field(default=0.0, ge=0.0, description="Unique senders, 5 months ago.")
    m6_received_total_value: Optional[float] = Field(default=0.0, ge=0.0, description="Total received value, oldest month (KES).")
    m6_received_volume: Optional[float] = Field(default=0.0, ge=0.0, description="Incoming P2P transfers, oldest month.")
    m6_received_highest_amount: Optional[float] = Field(default=0.0, ge=0.0, description="Largest incoming transfer, oldest month (KES).")
    m6_received_senders: Optional[float] = Field(default=0.0, ge=0.0, description="Unique senders, oldest month.")

    # ── Transfer from bank (inflow — formal banking channel) ──────────────────
    m1_transfer_from_bank_total_value: Optional[float] = Field(default=0.0, ge=0.0, description="Total bank-to-wallet transfer value, most recent month (KES).")
    m1_transfer_from_bank_volume: Optional[float] = Field(default=0.0, ge=0.0, description="Bank-to-wallet transfer count, most recent month.")
    m1_transfer_from_bank_highest_amount: Optional[float] = Field(default=0.0, ge=0.0, description="Largest bank-to-wallet transfer, most recent month (KES).")
    m1_transfer_from_bank_banks: Optional[float] = Field(default=0.0, ge=0.0, description="Unique banks used for wallet top-up, most recent month.")
    m2_transfer_from_bank_total_value: Optional[float] = Field(default=0.0, ge=0.0, description="Bank transfer value, 2 months ago (KES).")
    m2_transfer_from_bank_volume: Optional[float] = Field(default=0.0, ge=0.0, description="Bank transfer count, 2 months ago.")
    m2_transfer_from_bank_highest_amount: Optional[float] = Field(default=0.0, ge=0.0, description="Largest bank transfer, 2 months ago (KES).")
    m2_transfer_from_bank_banks: Optional[float] = Field(default=0.0, ge=0.0, description="Unique banks, 2 months ago.")
    m3_transfer_from_bank_total_value: Optional[float] = Field(default=0.0, ge=0.0, description="Bank transfer value, 3 months ago (KES).")
    m3_transfer_from_bank_volume: Optional[float] = Field(default=0.0, ge=0.0, description="Bank transfer count, 3 months ago.")
    m3_transfer_from_bank_highest_amount: Optional[float] = Field(default=0.0, ge=0.0, description="Largest bank transfer, 3 months ago (KES).")
    m3_transfer_from_bank_banks: Optional[float] = Field(default=0.0, ge=0.0, description="Unique banks, 3 months ago.")
    m4_transfer_from_bank_total_value: Optional[float] = Field(default=0.0, ge=0.0, description="Bank transfer value, 4 months ago (KES).")
    m4_transfer_from_bank_volume: Optional[float] = Field(default=0.0, ge=0.0, description="Bank transfer count, 4 months ago.")
    m4_transfer_from_bank_highest_amount: Optional[float] = Field(default=0.0, ge=0.0, description="Largest bank transfer, 4 months ago (KES).")
    m4_transfer_from_bank_banks: Optional[float] = Field(default=0.0, ge=0.0, description="Unique banks, 4 months ago.")
    m5_transfer_from_bank_total_value: Optional[float] = Field(default=0.0, ge=0.0, description="Bank transfer value, 5 months ago (KES).")
    m5_transfer_from_bank_volume: Optional[float] = Field(default=0.0, ge=0.0, description="Bank transfer count, 5 months ago.")
    m5_transfer_from_bank_highest_amount: Optional[float] = Field(default=0.0, ge=0.0, description="Largest bank transfer, 5 months ago (KES).")
    m5_transfer_from_bank_banks: Optional[float] = Field(default=0.0, ge=0.0, description="Unique banks, 5 months ago.")
    m6_transfer_from_bank_total_value: Optional[float] = Field(default=0.0, ge=0.0, description="Bank transfer value, oldest month (KES).")
    m6_transfer_from_bank_volume: Optional[float] = Field(default=0.0, ge=0.0, description="Bank transfer count, oldest month.")
    m6_transfer_from_bank_highest_amount: Optional[float] = Field(default=0.0, ge=0.0, description="Largest bank transfer, oldest month (KES).")
    m6_transfer_from_bank_banks: Optional[float] = Field(default=0.0, ge=0.0, description="Unique banks, oldest month.")

    # ── Withdraw (outflow — cash-out at agents) ───────────────────────────────
    m1_withdraw_total_value: Optional[float] = Field(default=0.0, ge=0.0, description="Total cash withdrawal value, most recent month (KES). Key input to spending pressure features.")
    m1_withdraw_volume: Optional[float] = Field(default=0.0, ge=0.0, description="Cash withdrawal count, most recent month.")
    m1_withdraw_highest_amount: Optional[float] = Field(default=0.0, ge=0.0, description="Largest single withdrawal, most recent month (KES).")
    m1_withdraw_agents: Optional[float] = Field(default=0.0, ge=0.0, description="Unique agents used for withdrawal, most recent month.")
    m2_withdraw_total_value: Optional[float] = Field(default=0.0, ge=0.0, description="Withdrawal value, 2 months ago (KES).")
    m2_withdraw_volume: Optional[float] = Field(default=0.0, ge=0.0, description="Withdrawal count, 2 months ago.")
    m2_withdraw_highest_amount: Optional[float] = Field(default=0.0, ge=0.0, description="Largest withdrawal, 2 months ago (KES).")
    m2_withdraw_agents: Optional[float] = Field(default=0.0, ge=0.0, description="Unique agents for withdrawal, 2 months ago.")
    m3_withdraw_total_value: Optional[float] = Field(default=0.0, ge=0.0, description="Withdrawal value, 3 months ago (KES).")
    m3_withdraw_volume: Optional[float] = Field(default=0.0, ge=0.0, description="Withdrawal count, 3 months ago.")
    m3_withdraw_highest_amount: Optional[float] = Field(default=0.0, ge=0.0, description="Largest withdrawal, 3 months ago (KES).")
    m3_withdraw_agents: Optional[float] = Field(default=0.0, ge=0.0, description="Unique agents for withdrawal, 3 months ago.")
    m4_withdraw_total_value: Optional[float] = Field(default=0.0, ge=0.0, description="Withdrawal value, 4 months ago (KES).")
    m4_withdraw_volume: Optional[float] = Field(default=0.0, ge=0.0, description="Withdrawal count, 4 months ago.")
    m4_withdraw_highest_amount: Optional[float] = Field(default=0.0, ge=0.0, description="Largest withdrawal, 4 months ago (KES).")
    m4_withdraw_agents: Optional[float] = Field(default=0.0, ge=0.0, description="Unique agents for withdrawal, 4 months ago.")
    m5_withdraw_total_value: Optional[float] = Field(default=0.0, ge=0.0, description="Withdrawal value, 5 months ago (KES).")
    m5_withdraw_volume: Optional[float] = Field(default=0.0, ge=0.0, description="Withdrawal count, 5 months ago.")
    m5_withdraw_highest_amount: Optional[float] = Field(default=0.0, ge=0.0, description="Largest withdrawal, 5 months ago (KES).")
    m5_withdraw_agents: Optional[float] = Field(default=0.0, ge=0.0, description="Unique agents for withdrawal, 5 months ago.")
    m6_withdraw_total_value: Optional[float] = Field(default=0.0, ge=0.0, description="Withdrawal value, oldest month (KES).")
    m6_withdraw_volume: Optional[float] = Field(default=0.0, ge=0.0, description="Withdrawal count, oldest month.")
    m6_withdraw_highest_amount: Optional[float] = Field(default=0.0, ge=0.0, description="Largest withdrawal, oldest month (KES).")
    m6_withdraw_agents: Optional[float] = Field(default=0.0, ge=0.0, description="Unique agents for withdrawal, oldest month.")

    # ── Merchant pay (outflow — retail/service payments) ──────────────────────
    m1_merchantpay_total_value: Optional[float] = Field(default=0.0, ge=0.0, description="Total merchant payment value, most recent month (KES).")
    m1_merchantpay_volume: Optional[float] = Field(default=0.0, ge=0.0, description="Merchant payment count, most recent month.")
    m1_merchantpay_highest_amount: Optional[float] = Field(default=0.0, ge=0.0, description="Largest merchant payment, most recent month (KES).")
    m1_merchantpay_merchants: Optional[float] = Field(default=0.0, ge=0.0, description="Unique merchants paid, most recent month.")
    m2_merchantpay_total_value: Optional[float] = Field(default=0.0, ge=0.0, description="Merchant payment value, 2 months ago (KES).")
    m2_merchantpay_volume: Optional[float] = Field(default=0.0, ge=0.0, description="Merchant payment count, 2 months ago.")
    m2_merchantpay_highest_amount: Optional[float] = Field(default=0.0, ge=0.0, description="Largest merchant payment, 2 months ago (KES).")
    m2_merchantpay_merchants: Optional[float] = Field(default=0.0, ge=0.0, description="Unique merchants paid, 2 months ago.")
    m3_merchantpay_total_value: Optional[float] = Field(default=0.0, ge=0.0, description="Merchant payment value, 3 months ago (KES).")
    m3_merchantpay_volume: Optional[float] = Field(default=0.0, ge=0.0, description="Merchant payment count, 3 months ago.")
    m3_merchantpay_highest_amount: Optional[float] = Field(default=0.0, ge=0.0, description="Largest merchant payment, 3 months ago (KES).")
    m3_merchantpay_merchants: Optional[float] = Field(default=0.0, ge=0.0, description="Unique merchants paid, 3 months ago.")
    m4_merchantpay_total_value: Optional[float] = Field(default=0.0, ge=0.0, description="Merchant payment value, 4 months ago (KES).")
    m4_merchantpay_volume: Optional[float] = Field(default=0.0, ge=0.0, description="Merchant payment count, 4 months ago.")
    m4_merchantpay_highest_amount: Optional[float] = Field(default=0.0, ge=0.0, description="Largest merchant payment, 4 months ago (KES).")
    m4_merchantpay_merchants: Optional[float] = Field(default=0.0, ge=0.0, description="Unique merchants paid, 4 months ago.")
    m5_merchantpay_total_value: Optional[float] = Field(default=0.0, ge=0.0, description="Merchant payment value, 5 months ago (KES).")
    m5_merchantpay_volume: Optional[float] = Field(default=0.0, ge=0.0, description="Merchant payment count, 5 months ago.")
    m5_merchantpay_highest_amount: Optional[float] = Field(default=0.0, ge=0.0, description="Largest merchant payment, 5 months ago (KES).")
    m5_merchantpay_merchants: Optional[float] = Field(default=0.0, ge=0.0, description="Unique merchants paid, 5 months ago.")
    m6_merchantpay_total_value: Optional[float] = Field(default=0.0, ge=0.0, description="Merchant payment value, oldest month (KES).")
    m6_merchantpay_volume: Optional[float] = Field(default=0.0, ge=0.0, description="Merchant payment count, oldest month.")
    m6_merchantpay_highest_amount: Optional[float] = Field(default=0.0, ge=0.0, description="Largest merchant payment, oldest month (KES).")
    m6_merchantpay_merchants: Optional[float] = Field(default=0.0, ge=0.0, description="Unique merchants paid, oldest month.")

    # ── Paybill (outflow — utility / bill payments) ───────────────────────────
    m1_paybill_total_value: Optional[float] = Field(default=0.0, ge=0.0, description="Total paybill value, most recent month (KES). Includes utilities, rent, insurance.")
    m1_paybill_volume: Optional[float] = Field(default=0.0, ge=0.0, description="Paybill payment count, most recent month.")
    m1_paybill_highest_amount: Optional[float] = Field(default=0.0, ge=0.0, description="Largest paybill payment, most recent month (KES).")
    m1_paybill_companies: Optional[float] = Field(default=0.0, ge=0.0, description="Unique companies paid via paybill, most recent month.")
    m2_paybill_total_value: Optional[float] = Field(default=0.0, ge=0.0, description="Paybill value, 2 months ago (KES).")
    m2_paybill_volume: Optional[float] = Field(default=0.0, ge=0.0, description="Paybill count, 2 months ago.")
    m2_paybill_highest_amount: Optional[float] = Field(default=0.0, ge=0.0, description="Largest paybill, 2 months ago (KES).")
    m2_paybill_companies: Optional[float] = Field(default=0.0, ge=0.0, description="Unique companies, 2 months ago.")
    m3_paybill_total_value: Optional[float] = Field(default=0.0, ge=0.0, description="Paybill value, 3 months ago (KES).")
    m3_paybill_volume: Optional[float] = Field(default=0.0, ge=0.0, description="Paybill count, 3 months ago.")
    m3_paybill_highest_amount: Optional[float] = Field(default=0.0, ge=0.0, description="Largest paybill, 3 months ago (KES).")
    m3_paybill_companies: Optional[float] = Field(default=0.0, ge=0.0, description="Unique companies, 3 months ago.")
    m4_paybill_total_value: Optional[float] = Field(default=0.0, ge=0.0, description="Paybill value, 4 months ago (KES).")
    m4_paybill_volume: Optional[float] = Field(default=0.0, ge=0.0, description="Paybill count, 4 months ago.")
    m4_paybill_highest_amount: Optional[float] = Field(default=0.0, ge=0.0, description="Largest paybill, 4 months ago (KES).")
    m4_paybill_companies: Optional[float] = Field(default=0.0, ge=0.0, description="Unique companies, 4 months ago.")
    m5_paybill_total_value: Optional[float] = Field(default=0.0, ge=0.0, description="Paybill value, 5 months ago (KES).")
    m5_paybill_volume: Optional[float] = Field(default=0.0, ge=0.0, description="Paybill count, 5 months ago.")
    m5_paybill_highest_amount: Optional[float] = Field(default=0.0, ge=0.0, description="Largest paybill, 5 months ago (KES).")
    m5_paybill_companies: Optional[float] = Field(default=0.0, ge=0.0, description="Unique companies, 5 months ago.")
    m6_paybill_total_value: Optional[float] = Field(default=0.0, ge=0.0, description="Paybill value, oldest month (KES).")
    m6_paybill_volume: Optional[float] = Field(default=0.0, ge=0.0, description="Paybill count, oldest month.")
    m6_paybill_highest_amount: Optional[float] = Field(default=0.0, ge=0.0, description="Largest paybill, oldest month (KES).")
    m6_paybill_companies: Optional[float] = Field(default=0.0, ge=0.0, description="Unique companies, oldest month.")

    # ── Mobile money send (outflow — P2P sends) ───────────────────────────────
    m1_mm_send_total_value: Optional[float] = Field(default=0.0, ge=0.0, description="Total P2P send value, most recent month (KES). Key input to p2p_receive_send_ratio.")
    m1_mm_send_volume: Optional[float] = Field(default=0.0, ge=0.0, description="P2P send count, most recent month.")
    m1_mm_send_highest_amount: Optional[float] = Field(default=0.0, ge=0.0, description="Largest single P2P send, most recent month (KES).")
    m1_mm_send_recipients: Optional[float] = Field(default=0.0, ge=0.0, description="Unique recipients of P2P sends, most recent month.")
    m2_mm_send_total_value: Optional[float] = Field(default=0.0, ge=0.0, description="P2P send value, 2 months ago (KES).")
    m2_mm_send_volume: Optional[float] = Field(default=0.0, ge=0.0, description="P2P send count, 2 months ago.")
    m2_mm_send_highest_amount: Optional[float] = Field(default=0.0, ge=0.0, description="Largest P2P send, 2 months ago (KES).")
    m2_mm_send_recipients: Optional[float] = Field(default=0.0, ge=0.0, description="Unique recipients, 2 months ago.")
    m3_mm_send_total_value: Optional[float] = Field(default=0.0, ge=0.0, description="P2P send value, 3 months ago (KES).")
    m3_mm_send_volume: Optional[float] = Field(default=0.0, ge=0.0, description="P2P send count, 3 months ago.")
    m3_mm_send_highest_amount: Optional[float] = Field(default=0.0, ge=0.0, description="Largest P2P send, 3 months ago (KES).")
    m3_mm_send_recipients: Optional[float] = Field(default=0.0, ge=0.0, description="Unique recipients, 3 months ago.")
    m4_mm_send_total_value: Optional[float] = Field(default=0.0, ge=0.0, description="P2P send value, 4 months ago (KES).")
    m4_mm_send_volume: Optional[float] = Field(default=0.0, ge=0.0, description="P2P send count, 4 months ago.")
    m4_mm_send_highest_amount: Optional[float] = Field(default=0.0, ge=0.0, description="Largest P2P send, 4 months ago (KES).")
    m4_mm_send_recipients: Optional[float] = Field(default=0.0, ge=0.0, description="Unique recipients, 4 months ago.")
    m5_mm_send_total_value: Optional[float] = Field(default=0.0, ge=0.0, description="P2P send value, 5 months ago (KES).")
    m5_mm_send_volume: Optional[float] = Field(default=0.0, ge=0.0, description="P2P send count, 5 months ago.")
    m5_mm_send_highest_amount: Optional[float] = Field(default=0.0, ge=0.0, description="Largest P2P send, 5 months ago (KES).")
    m5_mm_send_recipients: Optional[float] = Field(default=0.0, ge=0.0, description="Unique recipients, 5 months ago.")
    m6_mm_send_total_value: Optional[float] = Field(default=0.0, ge=0.0, description="P2P send value, oldest month (KES).")
    m6_mm_send_volume: Optional[float] = Field(default=0.0, ge=0.0, description="P2P send count, oldest month.")
    m6_mm_send_highest_amount: Optional[float] = Field(default=0.0, ge=0.0, description="Largest P2P send, oldest month (KES).")
    m6_mm_send_recipients: Optional[float] = Field(default=0.0, ge=0.0, description="Unique recipients, oldest month.")

    # =========================================================================
    # FIELD-LEVEL VALIDATORS
    # =========================================================================

    @field_validator("gender", mode="before")
    @classmethod
    def normalise_gender(cls, v: Optional[str]) -> Optional[str]:
        """
        Normalise gender to uppercase canonical form.

        Accepted inputs: 'M', 'F', 'm', 'f', 'Male', 'Female'.
        Anything that does not reduce to 'M' or 'F' after stripping and
        upcasing is left as-is (the feature engineering encodes it as
        is_male = (gender.strip().upper() == 'M'), so unknown values
        silently map to is_male=0, matching training behaviour).
        """
        if v is None:
            return None
        normalised = v.strip().upper()
        # Accept 'MALE' → 'M', 'FEMALE' → 'F'
        if normalised in ("MALE",):
            return "M"
        if normalised in ("FEMALE",):
            return "F"
        return normalised

    @field_validator("segment", mode="before")
    @classmethod
    def normalise_segment(cls, v: Optional[str]) -> Optional[str]:
        """
        Normalise segment to uppercase.  Valid values: 'LVC', 'MVC', 'HVC'.

        The ordinal encoding (LVC=0, MVC=1, HVC=2) is applied in
        feature_engineering._categorical_features().  An unrecognised
        value maps to 1 (MVC) via the .fillna(1) fallback in that function,
        matching the training-time behaviour exactly.
        """
        if v is None:
            return None
        return v.strip().upper()

    @field_validator("smartphone", mode="before")
    @classmethod
    def normalise_smartphone(cls, v: Optional[str]) -> Optional[str]:
        """
        Normalise smartphone flag to lowercase.

        The feature engineering accepts: 'yes', '1', 'true' → has_smartphone=1.
        All others → has_smartphone=0.
        """
        if v is None:
            return None
        return v.strip().lower()

    @field_validator("earning_pattern", mode="before")
    @classmethod
    def normalise_earning_pattern(cls, v: Optional[str]) -> Optional[str]:
        """Strip whitespace from earning_pattern.  Value is passed through as-is."""
        if v is None:
            return None
        return v.strip()

    @field_validator("region", mode="before")
    @classmethod
    def normalise_region(cls, v: Optional[str]) -> Optional[str]:
        """Strip whitespace from region.  Value is passed through as-is."""
        if v is None:
            return None
        return v.strip()

    # =========================================================================
    # CROSS-FIELD VALIDATORS (model_validator)
    # =========================================================================

    @model_validator(mode="after")
    def validate_highest_amount_leq_total_value(self) -> "CustomerFeatures":
        """
        Enforce the invariant: highest_amount ≤ total_value for each group-month.

        A highest_amount > total_value is physically impossible: the largest
        single transaction cannot exceed the aggregate volume.  This condition
        indicates either a data entry error or a misaligned field mapping from
        the caller.

        Only checked when both fields are non-zero (zero defaults are valid).
        Raises ValueError with a specific field reference to aid debugging.
        """
        eps = 1e-2  # Tolerance for floating-point rounding in KES amounts.

        for g in TRANSACTION_GROUPS:
            for m in MONTHS:
                tv_field  = f"{m}_{g}_total_value"
                hi_field  = f"{m}_{g}_highest_amount"
                tv_val    = getattr(self, tv_field, None) or 0.0
                hi_val    = getattr(self, hi_field, None) or 0.0

                if hi_val > 0.0 and tv_val > 0.0 and hi_val > tv_val + eps:
                    raise ValueError(
                        f"Field constraint violated: {hi_field} ({hi_val:.2f}) "
                        f"exceeds {tv_field} ({tv_val:.2f}). "
                        f"The highest single transaction cannot exceed the total "
                        f"transaction value for the same group and month.  "
                        f"Verify that {m.upper()} {g} fields are correctly mapped."
                    )
        return self

    @model_validator(mode="after")
    def warn_balance_lag_risk_pattern(self) -> "CustomerFeatures":
        """
        Detect the known 'balance-lag' false-negative failure mode.

        SHAP analysis (notebook 13) identified that the ensemble systematically
        misclassifies customers whose balance remains stable or slightly positive
        while their inflow is collapsing.  This validator injects a warning flag
        into the model instance (accessible post-construction) so that predictor.py
        can append the RiskFactor for SPENDING_PRESSURE / INCOME_STABILITY.

        Heuristic rule (from notebook 13 §8):
          - m1_daily_avg_bal ≥ m6_daily_avg_bal - 1000  (balance appears stable)
          - m1 inflow < m6 inflow × 0.5                 (inflow halved or worse)
        """
        m1_bal = self.m1_daily_avg_bal or 0.0
        m6_bal = self.m6_daily_avg_bal or 0.0

        m1_inflow = (
            (self.m1_deposit_total_value or 0.0)
            + (self.m1_received_total_value or 0.0)
            + (self.m1_transfer_from_bank_total_value or 0.0)
        )
        m6_inflow = (
            (self.m6_deposit_total_value or 0.0)
            + (self.m6_received_total_value or 0.0)
            + (self.m6_transfer_from_bank_total_value or 0.0)
        )

        balance_appears_stable = m1_bal >= (m6_bal - 1_000.0)
        inflow_collapsed = (m6_inflow > 0.0) and (m1_inflow < m6_inflow * 0.5)

        # Store as instance attribute — read by predictor.build_risk_factors()
        # Use object.__setattr__ because validate_assignment=True would
        # trigger recursion if we tried self._balance_lag_flag = ...
        object.__setattr__(self, "_balance_lag_detected", balance_appears_stable and inflow_collapsed)
        return self

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert this customer snapshot to a 1-row pandas DataFrame.

        The resulting DataFrame has the same column names and dtypes as
        ``Test.csv``, ready to pass directly to
        ``src.features.feature_engineering.build_features()``.

        Key conventions
        ---------------
        - ``None`` values for numeric fields are left as ``NaN`` in the
          DataFrame.  ``build_features._fill_nulls()`` will replace them
          with 0.0, matching the competition null = zero activity spec.
        - Categorical fields (gender, region, segment, earning_pattern,
          smartphone) are cast to ``category`` dtype, matching
          ``load_data._enforce_dtypes()`` behaviour.
        - The ``ID`` column is included so downstream code can use
          ``split_features_target()`` which drops it cleanly.

        Returns
        -------
        pd.DataFrame
            Shape (1, 183).  Column order matches ``Test.csv``.

        Raises
        ------
        RuntimeError
            If serialisation to dict unexpectedly fails Pydantic validation.
        """
        raw_dict = self.model_dump(mode="python")

        _CATEGORICAL_COLS = {
            "gender", "region", "segment", "earning_pattern", "smartphone"
        }

        row: Dict[str, Any] = {}
        for col, val in raw_dict.items():
            if col.startswith("_"):
                # Skip private validator-injected attributes.
                continue
            if col in _CATEGORICAL_COLS:
                # Category dtype matches load_data.py enforce_dtypes().
                row[col] = pd.Categorical([val] if val is not None else [None])
            else:
                row[col] = [val]

        df = pd.DataFrame(row)
        # Ensure numeric columns are float64 (not object) even when None.
        numeric_cols = [
            c for c in df.columns
            if c not in _CATEGORICAL_COLS and c != "ID"
        ]
        df[numeric_cols] = df[numeric_cols].astype(float)
        return df

    @classmethod
    def from_csv_row(cls, row: Dict[str, Any]) -> "CustomerFeatures":
        """
        Construct a ``CustomerFeatures`` from a raw CSV row dictionary.

        Handles type coercion from string-typed CSV values (pandas
        ``read_csv`` with ``dtype=str`` produces all string columns).
        Float-typed fields are converted via ``float()`` with NaN/empty
        string mapping to ``None``.

        Parameters
        ----------
        row : dict
            A single row from ``pd.read_csv(...).to_dict(orient='records')[i]``.
            Keys must match the ``Test.csv`` column names.

        Returns
        -------
        CustomerFeatures
            Fully validated instance with normalised categoricals.

        Raises
        ------
        ValueError
            If required field ``ID`` is missing or empty.
        pydantic.ValidationError
            If any field fails type coercion or constraint validation.

        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.read_csv("data/raw/Test.csv", dtype=str)
        >>> customers = [CustomerFeatures.from_csv_row(r)
        ...              for r in df.to_dict(orient="records")]
        """
        _CATEGORICAL_COLS = {
            "gender", "region", "segment", "earning_pattern", "smartphone"
        }
        _STRING_COLS = {"ID"}

        coerced: Dict[str, Any] = {}
        for k, v in row.items():
            if k in _STRING_COLS or k in _CATEGORICAL_COLS:
                coerced[k] = v if (v is not None and str(v).strip() != "") else None
            else:
                # Numeric field: coerce to float; treat NaN/empty/nan string as None.
                if v is None or str(v).strip() in ("", "nan", "NaN", "None", "null"):
                    coerced[k] = None
                else:
                    try:
                        coerced[k] = float(v)
                    except (ValueError, TypeError):
                        coerced[k] = None  # Silently drop unparseable values.

        return cls(**coerced)

    @property
    def balance_lag_detected(self) -> bool:
        """
        True if the balance-lag false-negative pattern was detected.

        This property exposes the result of the ``warn_balance_lag_risk_pattern``
        cross-field validator.  When True, the predictor should append a
        SPENDING_PRESSURE risk factor to alert downstream consumers that the
        ensemble may be underestimating risk for this customer.
        """
        return getattr(self, "_balance_lag_detected", False)

    @property
    def computed_input_summary(self) -> InputSummary:
        """
        Compute key input signals for embedding in PredictionResponse.

        Derived directly from raw input values — no feature engineering.
        Used by ``predictor.py`` to populate the ``input_summary`` field
        without a second pass through the 825-feature pipeline.
        """
        m1_inflow = (
            (self.m1_deposit_total_value or 0.0)
            + (self.m1_received_total_value or 0.0)
            + (self.m1_transfer_from_bank_total_value or 0.0)
        )
        m4_m6_inflow_avg = (
            (self.m4_deposit_total_value or 0.0) + (self.m4_received_total_value or 0.0) + (self.m4_transfer_from_bank_total_value or 0.0)
            + (self.m5_deposit_total_value or 0.0) + (self.m5_received_total_value or 0.0) + (self.m5_transfer_from_bank_total_value or 0.0)
            + (self.m6_deposit_total_value or 0.0) + (self.m6_received_total_value or 0.0) + (self.m6_transfer_from_bank_total_value or 0.0)
        ) / 3.0

        m1_m3_dep_avg = (
            (self.m1_deposit_total_value or 0.0)
            + (self.m2_deposit_total_value or 0.0)
            + (self.m3_deposit_total_value or 0.0)
        ) / 3.0
        m4_m6_dep_avg = (
            (self.m4_deposit_total_value or 0.0)
            + (self.m5_deposit_total_value or 0.0)
            + (self.m6_deposit_total_value or 0.0)
        ) / 3.0
        dep_recency_ratio = (m1_m3_dep_avg / (m4_m6_dep_avg + 1e-6)) if m4_m6_dep_avg >= 0 else None

        total_dep_6m = sum(
            getattr(self, f"{m}_deposit_total_value", None) or 0.0
            for m in MONTHS
        )
        total_wd_6m = sum(
            getattr(self, f"{m}_withdraw_total_value", None) or 0.0
            for m in MONTHS
        )

        m1_outflow_vol = (
            (self.m1_withdraw_volume or 0.0)
            + (self.m1_merchantpay_volume or 0.0)
            + (self.m1_paybill_volume or 0.0)
            + (self.m1_mm_send_volume or 0.0)
        )
        m1_inflow_vol = (
            (self.m1_deposit_volume or 0.0)
            + (self.m1_received_volume or 0.0)
            + (self.m1_transfer_from_bank_volume or 0.0)
        )
        m1_network = (
            (self.m1_deposit_agents or 0.0)
            + (self.m1_withdraw_agents or 0.0)
            + (self.m1_received_senders or 0.0)
            + (self.m1_mm_send_recipients or 0.0)
            + (self.m1_merchantpay_merchants or 0.0)
            + (self.m1_paybill_companies or 0.0)
            + (self.m1_transfer_from_bank_banks or 0.0)
        )

        m1_bal = self.m1_daily_avg_bal
        m6_bal = self.m6_daily_avg_bal
        if m1_bal is not None and m6_bal is not None:
            diff = m1_bal - m6_bal
            if diff > 500.0:
                direction = "improving"
            elif diff < -500.0:
                direction = "deteriorating"
            else:
                direction = "stable"
        else:
            direction = None

        return InputSummary(
            m1_daily_avg_bal=m1_bal,
            m6_daily_avg_bal=m6_bal,
            total_deposit_6m=round(total_dep_6m, 2),
            total_withdraw_6m=round(total_wd_6m, 2),
            deposit_m1_vs_m46_ratio=round(dep_recency_ratio, 4) if dep_recency_ratio is not None else None,
            recent_inflow_volume=m1_inflow_vol,
            recent_outflow_volume=m1_outflow_vol,
            network_breadth_m1=m1_network,
            balance_trend_direction=direction,
        )


# =============================================================================
# RESPONSE SCHEMAS
# =============================================================================

class PredictionResponse(BaseModel):
    """
    Complete prediction response for a single customer.

    Contains the stress probability, risk tier, SHAP-informed risk factors,
    key input signal summary, full ensemble metadata, and operational metadata.

    The ``risk_tier`` thresholds are fixed in the API specification:
      - ``low``    : stress_probability < 0.25
      - ``medium`` : 0.25 ≤ stress_probability < 0.60
      - ``high``   : stress_probability ≥ 0.60

    Note on calibration
    --------------------
    The ``stress_probability`` is produced by the optimised weighted average
    of Platt-calibrated per-model predictions.  The ensemble was calibrated
    against a 15% positive rate on 40,000 OOF samples.  Scores from this
    endpoint represent estimated posterior probabilities of 30-day liquidity
    stress and are suitable for ranking customers by risk.  For downstream
    decision thresholds (e.g. intervention triggers), recalibrate on live
    deployment population data.
    """

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "title": "Liquidity Stress Prediction",
            "description": (
                "30-day liquidity stress probability for a single mobile money customer."
            ),
            "examples": [
                {
                    "customer_id": "CUST_001",
                    "stress_probability": 0.342,
                    "risk_tier": "medium",
                    "risk_tier_thresholds": {"low": [0.0, 0.25], "medium": [0.25, 0.60], "high": [0.60, 1.0]},
                    "confidence_context": "Calibrated against 15% positive rate, 40,000 OOF samples.",
                }
            ],
        },
    )

    customer_id: str = Field(
        description="Customer identifier, echoed from the request ID field.",
    )
    stress_probability: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Estimated probability that this customer will experience liquidity stress "
            "within the next 30 days.  Produced by the v5.1 optimised weighted average "
            "ensemble (OOF composite score = 0.19144).  "
            "Clipped to [1e-6, 1−1e-6] for log-loss numerical stability."
        ),
        examples=[0.342, 0.078, 0.731],
    )
    risk_tier: RiskTier = Field(
        description=(
            "Categorical risk tier derived from stress_probability.  "
            "low < 0.25 ≤ medium < 0.60 ≤ high."
        ),
    )
    risk_tier_thresholds: Dict[str, List[float]] = Field(
        default={
            "low":    [0.0,  RISK_TIER_LOW_MAX],
            "medium": [RISK_TIER_LOW_MAX, RISK_TIER_MEDIUM_MAX],
            "high":   [RISK_TIER_MEDIUM_MAX, 1.0],
        },
        description="Probability bounds for each risk tier (inclusive lower, exclusive upper).",
    )
    input_summary: InputSummary = Field(
        description=(
            "Key raw input signals echoed back for audit and downstream display.  "
            "Derived without running the full 825-feature engineering pipeline."
        ),
    )
    risk_factors: List[RiskFactor] = Field(
        default_factory=list,
        description=(
            "SHAP-informed risk factors contributing to this prediction.  "
            "Derived from the thematic analysis in notebook 13.  "
            "Empty list if no significant risk patterns detected.  "
            "These are thematic indicators — not individual feature SHAP values."
        ),
    )
    model_info: ModelInfo = Field(
        default_factory=ModelInfo,
        description="Ensemble model metadata for audit and reproducibility.",
    )
    prediction_metadata: PredictionMetadata = Field(
        description="Operational metadata: request_id, timestamp, latency.",
    )
    confidence_context: str = Field(
        default=(
            f"Calibrated against {15}% positive rate on 40,000 OOF samples. "
            f"Ensemble OOF composite score = {ENSEMBLE_OOF_SCORE:.5f} "
            f"({COMPETITION_METRIC}, lower is better). "
            "Platt calibration applied to all 5 base models. "
            "Recalibrate thresholds on live deployment population before production intervention use."
        ),
        description=(
            "Plain-language note on calibration context and appropriate use.  "
            "Included to prevent misinterpretation of raw probability outputs."
        ),
    )

    @classmethod
    def build(
        cls,
        customer_id: str,
        stress_probability: float,
        input_summary: InputSummary,
        risk_factors: List[RiskFactor],
        request_id: str,
        processing_time_ms: float,
    ) -> "PredictionResponse":
        """
        Factory method used by ``predictor.py`` to construct a complete response.

        Derives the risk tier from the probability and populates all
        metadata fields.  This is the only way responses should be created
        outside of test fixtures — it guarantees the tier derivation logic
        is applied consistently.

        Parameters
        ----------
        customer_id : str
            Echoed from the request ``ID`` field.
        stress_probability : float
            Clipped ensemble output in [1e-6, 1−1e-6].
        input_summary : InputSummary
            Computed from ``CustomerFeatures.computed_input_summary``.
        risk_factors : list[RiskFactor]
            From ``predictor.build_risk_factors()``.
        request_id : str
            UUID4 from the request context.
        processing_time_ms : float
            Wall-clock latency.

        Returns
        -------
        PredictionResponse
        """
        if stress_probability < RISK_TIER_LOW_MAX:
            tier = RiskTier.LOW
        elif stress_probability < RISK_TIER_MEDIUM_MAX:
            tier = RiskTier.MEDIUM
        else:
            tier = RiskTier.HIGH

        return cls(
            customer_id=customer_id,
            stress_probability=round(stress_probability, 6),
            risk_tier=tier,
            input_summary=input_summary,
            risk_factors=risk_factors,
            prediction_metadata=PredictionMetadata(
                request_id=request_id,
                predicted_at=datetime.now(tz=timezone.utc),
                processing_time_ms=round(processing_time_ms, 2),
            ),
        )


class BatchPredictionRequest(BaseModel):
    """
    Batch prediction request — list of customer snapshots.

    Maximum batch size is enforced here to protect the synchronous CPU-bound
    inference from unbounded request sizes.  For batch sizes > 500, use the
    offline batch scoring script (``src/inference/predict.py``) directly.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "title": "Batch Prediction Request",
            "description": "Up to 500 customer snapshots for batch prediction.",
        }
    )

    customers: List[CustomerFeatures] = Field(
        min_length=1,
        max_length=500,
        description=(
            "List of customer snapshots to score.  Each element has the same "
            "schema as a single /predict request body.  "
            "Maximum 500 per request; for larger batches use the offline pipeline."
        ),
    )
    return_risk_factors: bool = Field(
        default=True,
        description=(
            "Whether to include SHAP-informed risk factors in each response. "
            "Set False to reduce response payload for high-throughput scenarios."
        ),
    )
    return_input_summary: bool = Field(
        default=True,
        description=(
            "Whether to echo key input signals back in each response. "
            "Set False to reduce response payload."
        ),
    )


class BatchPredictionResponse(BaseModel):
    """
    Batch prediction response — per-customer results plus aggregate statistics.

    The aggregate statistics (high_risk_count, stress_rate_estimate) are
    provided to allow callers to quickly assess the batch risk profile without
    iterating over individual predictions.
    """

    model_config = ConfigDict(frozen=True)

    predictions: List[PredictionResponse] = Field(
        description="Per-customer prediction results, in the same order as the request.",
    )
    batch_id: str = Field(
        description="UUID4 identifier for this batch request.",
        examples=["a3b9c1d2-e4f5-6789-abcd-ef0123456789"],
    )
    batch_size: int = Field(
        ge=1,
        description="Number of customers scored in this batch.",
    )
    processing_time_ms: float = Field(
        ge=0.0,
        description="Total wall-clock time for the entire batch (ms).",
    )
    high_risk_count: int = Field(
        ge=0,
        description=(
            f"Number of customers with stress_probability ≥ {RISK_TIER_MEDIUM_MAX} "
            "(risk_tier='high')."
        ),
    )
    medium_risk_count: int = Field(
        ge=0,
        description=(
            f"Number of customers with {RISK_TIER_LOW_MAX} ≤ stress_probability < "
            f"{RISK_TIER_MEDIUM_MAX} (risk_tier='medium')."
        ),
    )
    low_risk_count: int = Field(
        ge=0,
        description=f"Number of customers with stress_probability < {RISK_TIER_LOW_MAX} (risk_tier='low').",
    )
    stress_rate_estimate: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of batch with stress_probability > 0.5.  "
            "Expected ≈ 0.15 for a representative customer population "
            "(matching the 15% training set positive rate).  "
            "Significant deviation may indicate population shift."
        ),
    )
    population_drift_warning: Optional[str] = Field(
        default=None,
        description=(
            "Warning message if stress_rate_estimate deviates significantly "
            "from the expected 15% positive rate.  "
            "Threshold: |rate − 0.15| > 0.10.  "
            "Possible causes: non-representative batch, feature drift, "
            "or pipeline misconfiguration."
        ),
    )

    @classmethod
    def build(
        cls,
        predictions: List[PredictionResponse],
        batch_id: str,
        processing_time_ms: float,
    ) -> "BatchPredictionResponse":
        """
        Factory method for constructing a BatchPredictionResponse.

        Computes aggregate statistics from the individual prediction results.
        Appends a population drift warning if the batch stress rate deviates
        more than 10 percentage points from the expected 15%.
        """
        n = len(predictions)
        high   = sum(1 for p in predictions if p.risk_tier == RiskTier.HIGH)
        medium = sum(1 for p in predictions if p.risk_tier == RiskTier.MEDIUM)
        low    = sum(1 for p in predictions if p.risk_tier == RiskTier.LOW)
        stress_rate = sum(1 for p in predictions if p.stress_probability > 0.5) / n

        drift_warning: Optional[str] = None
        if abs(stress_rate - 0.15) > 0.10:
            drift_warning = (
                f"Batch stress rate ({stress_rate:.1%}) deviates from expected 15% "
                f"by {abs(stress_rate - 0.15):.1%}.  "
                "Verify batch is representative of the target customer population.  "
                "Consider recalibrating thresholds if this is a systematic shift."
            )

        return cls(
            predictions=predictions,
            batch_id=batch_id,
            batch_size=n,
            processing_time_ms=round(processing_time_ms, 2),
            high_risk_count=high,
            medium_risk_count=medium,
            low_risk_count=low,
            stress_rate_estimate=round(stress_rate, 4),
            population_drift_warning=drift_warning,
        )


class HealthResponse(BaseModel):
    """
    Service health status returned by ``GET /health``.

    Reports whether all model artefacts loaded successfully and provides
    the key performance indicator (OOF composite score) for at-a-glance
    confirmation that the production model is active.
    """

    model_config = ConfigDict(frozen=True)

    status: HealthStatus = Field(
        description=(
            "Service operational status.  'ok' = all 25 fold models, 5 preprocessors, "
            "and 5 calibrators loaded.  'degraded' = partial load.  "
            "'down' = no predictions possible."
        ),
    )
    model_version: str = Field(
        description="Production ensemble version string.",
        examples=[ENSEMBLE_VERSION],
    )
    ensemble_oof_score: float = Field(
        description=(
            f"OOF composite score ({COMPETITION_METRIC}).  "
            "Lower is better.  Expected value for production ensemble: 0.19144."
        ),
        examples=[ENSEMBLE_OOF_SCORE],
    )
    models_loaded: Dict[str, bool] = Field(
        description=(
            "Per-model load status.  All five should be True in healthy state: "
            "lightgbm, xgboost, catboost, logreg, tabnet."
        ),
        examples=[{
            "lightgbm": True, "xgboost": True, "catboost": True,
            "logreg": True, "tabnet": True,
        }],
    )
    artefact_counts: ArtefactCounts = Field(
        description="Detailed counts of loaded artefacts.",
    )
    uptime_seconds: float = Field(
        ge=0.0,
        description="Seconds since the service started and models were loaded.",
    )
    checked_at: datetime = Field(
        description="UTC timestamp of this health check.",
    )


class ErrorResponse(BaseModel):
    """
    Structured error payload for all 4xx and 5xx responses.

    FastAPI's default validation errors are augmented with this schema in
    ``app.py`` exception handlers so clients receive consistent error objects
    regardless of error origin (validation, inference failure, missing artefact).
    """

    model_config = ConfigDict(frozen=True)

    error_code: str = Field(
        description=(
            "Machine-readable error code.  "
            "Examples: 'VALIDATION_ERROR', 'INFERENCE_FAILURE', "
            "'ARTEFACT_NOT_FOUND', 'BATCH_TOO_LARGE', 'SERVICE_DEGRADED'."
        ),
        examples=["VALIDATION_ERROR"],
    )
    message: str = Field(
        description="Human-readable error summary.",
        examples=["Field 'm1_deposit_highest_amount' exceeds 'm1_deposit_total_value'."],
    )
    detail: Optional[Any] = Field(
        default=None,
        description=(
            "Additional structured detail.  For validation errors, this is a list "
            "of Pydantic error dicts.  For inference failures, the exception message."
        ),
    )
    request_id: str = Field(
        description="UUID4 request identifier for log correlation.",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        description="UTC timestamp when the error occurred.",
    )

    @classmethod
    def from_exception(
        cls,
        error_code: str,
        message: str,
        request_id: Optional[str] = None,
        detail: Optional[Any] = None,
    ) -> "ErrorResponse":
        """
        Convenience factory for constructing error responses in exception handlers.

        Parameters
        ----------
        error_code : str
            Machine-readable code, e.g. 'INFERENCE_FAILURE'.
        message : str
            Human-readable summary.
        request_id : str, optional
            UUID4 from request context.  Generated if not provided.
        detail : Any, optional
            Additional structured detail.
        """
        return cls(
            error_code=error_code,
            message=message,
            detail=detail,
            request_id=request_id or str(uuid.uuid4()),
        )


# =============================================================================
# SCHEMA UTILITIES
# =============================================================================

def derive_risk_tier(probability: float) -> RiskTier:
    """
    Pure function mapping a probability to a risk tier.

    Extracted as a standalone function so it can be imported and tested
    independently of the full Pydantic model machinery.

    Parameters
    ----------
    probability : float
        Stress probability in [0, 1].

    Returns
    -------
    RiskTier
    """
    if probability < RISK_TIER_LOW_MAX:
        return RiskTier.LOW
    if probability < RISK_TIER_MEDIUM_MAX:
        return RiskTier.MEDIUM
    return RiskTier.HIGH


def get_raw_column_names() -> List[str]:
    """
    Return the complete ordered list of all 183 ``Test.csv`` column names.

    This is the canonical source of truth for column ordering used by
    ``CustomerFeatures.to_dataframe()`` and the feature engineering pipeline.
    The list matches the field declaration order in ``CustomerFeatures``
    exactly.

    Returns
    -------
    list[str]
        183 column names starting with 'ID'.
    """
    return list(CustomerFeatures.model_fields.keys())


def get_transaction_field_names() -> List[str]:
    """
    Return only the 168 transaction field names (excludes ID, balance, profile).

    Useful for validation scripts that need to iterate specifically over
    the wide-format transaction columns.
    """
    all_fields = get_raw_column_names()
    _NON_TRANSACTION = {
        "ID", "gender", "region", "segment", "earning_pattern", "smartphone",
        "age", "x_90_d_activity_rate", "arpu",
        "m1_daily_avg_bal", "m2_daily_avg_bal", "m3_daily_avg_bal",
        "m4_daily_avg_bal", "m5_daily_avg_bal", "m6_daily_avg_bal",
    }
    return [f for f in all_fields if f not in _NON_TRANSACTION]


def assert_schema_integrity() -> None:
    """
    Verify schema field counts match competition data specification.

    Run this at import time (or in tests) to catch any field additions or
    deletions that break the 183-column contract.

    Raises
    ------
    AssertionError
        If field counts do not match expected values.
    """
    all_fields = get_raw_column_names()
    tx_fields  = get_transaction_field_names()

    if len(all_fields) != 183:
        raise AssertionError(
            f"CustomerFeatures has {len(all_fields)} fields; expected 183. "
            "The schema has drifted from the competition data specification."
        )
    if len(tx_fields) != 168:
        raise AssertionError(
            f"Transaction field count is {len(tx_fields)}; expected 168 "
            "(7 groups × 6 months × 4 suffixes)."
        )


# Run integrity check at import time.
# This is a hard guard — if a developer accidentally deletes or renames a field,
# the service will fail to start with a clear message rather than silently
# producing wrong predictions.
assert_schema_integrity()