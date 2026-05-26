from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd
from pandas import CategoricalDtype
from pandas.core.dtypes.common import is_numeric_dtype, is_string_dtype

DEFAULT_TARGET = "annual_medical_cost"

DEFAULT_LEAKAGE_OR_OUTCOME_COLUMNS = [
    "person_id",
    "annual_medical_cost",
    "log1p_annual_medical_cost",
    "total_claims_paid",
    "avg_claim_amount",
    "claims_count",
    "annual_premium",
    "monthly_premium",
    "high_cost_90",
    "high_cost_95",
    "high_cost_99",
]

RAW_COLUMNS_REPLACED_BY_GROUPS = [
    "hospitalizations_last_3yrs",
    "chronic_count",
    "days_hospitalized_last_3yrs",
]

GROUPED_HEALTH_COLUMNS = [
    "hospitalizations_grouped",
    "chronic_count_grouped",
    "days_hospitalized_grouped",
]


def get_excluded_columns(
        target_col: str = DEFAULT_TARGET,
        use_grouped_health_features: bool = True,
        extra_drop_cols: Optional[Sequence[str]] = None,
) -> set[str]:
    excluded_columns = set(DEFAULT_LEAKAGE_OR_OUTCOME_COLUMNS)

    excluded_columns.add(target_col)
    excluded_columns.add(f"log1p_{target_col}")

    if use_grouped_health_features:
        excluded_columns.update(RAW_COLUMNS_REPLACED_BY_GROUPS)
    else:
        excluded_columns.update(GROUPED_HEALTH_COLUMNS)

    if extra_drop_cols is not None:
        excluded_columns.update(extra_drop_cols)

    return excluded_columns


def get_model_feature_columns(
        df: pd.DataFrame,
        target_col: str = DEFAULT_TARGET,
        use_grouped_health_features: bool = True,
        extra_drop_cols: Optional[Sequence[str]] = None,
) -> list[str]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' was not found in the dataframe.")

    excluded_columns = get_excluded_columns(
        target_col=target_col,
        use_grouped_health_features=use_grouped_health_features,
        extra_drop_cols=extra_drop_cols,
    )

    return [
        col for col in df.columns
        if col not in excluded_columns
    ]


def get_grouped_model_feature_columns(
        df: pd.DataFrame,
        target_col: str = DEFAULT_TARGET,
        extra_drop_cols: Optional[Sequence[str]] = None,
) -> list[str]:
    return get_model_feature_columns(
        df=df,
        target_col=target_col,
        use_grouped_health_features=True,
        extra_drop_cols=extra_drop_cols,
    )


def get_raw_model_feature_columns(
        df: pd.DataFrame,
        target_col: str = DEFAULT_TARGET,
        extra_drop_cols: Optional[Sequence[str]] = None,
) -> list[str]:
    return get_model_feature_columns(
        df=df,
        target_col=target_col,
        use_grouped_health_features=False,
        extra_drop_cols=extra_drop_cols,
    )


def split_feature_types(
        df: pd.DataFrame,
        feature_cols: Sequence[str],
) -> tuple[list[str], list[str]]:
    numeric_features = [
        col for col in feature_cols
        if is_numeric_dtype(df[col])
    ]

    categorical_features = [
        col for col in feature_cols
        if is_string_dtype(df[col]) or isinstance(df[col].dtype, CategoricalDtype)
    ]

    return numeric_features, categorical_features
