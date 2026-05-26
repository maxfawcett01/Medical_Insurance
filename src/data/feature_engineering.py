from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

DEFAULT_TARGET = "annual_medical_cost"


def add_log_target(
        df: pd.DataFrame,
        target_col: str = DEFAULT_TARGET,
        output_col: Optional[str] = None,
) -> pd.DataFrame:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' was not found in the dataframe.")

    output_col = output_col or f"log1p_{target_col}"

    data = df.copy()

    if (data[target_col] < 0).any():
        raise ValueError("log1p transformation requires non-negative target values.")

    data[output_col] = np.log1p(data[target_col])

    return data


def add_hospitalization_group(
        df: pd.DataFrame,
        source_col: str = "hospitalizations_last_3yrs",
        output_col: str = "hospitalizations_grouped",
) -> pd.DataFrame:
    if source_col not in df.columns:
        raise ValueError(f"Column '{source_col}' was not found in the dataframe.")

    data = df.copy()

    data[output_col] = data[source_col].replace({
        2: "2+",
        3: "2+",
    }).astype(str)

    data[output_col] = pd.Categorical(
        data[output_col],
        categories=["0", "1", "2+"],
        ordered=True,
    )

    return data


def add_chronic_count_group(
        df: pd.DataFrame,
        source_col: str = "chronic_count",
        output_col: str = "chronic_count_grouped",
) -> pd.DataFrame:
    if source_col not in df.columns:
        raise ValueError(f"Column '{source_col}' was not found in the dataframe.")

    data = df.copy()

    data[output_col] = data[source_col].apply(
        lambda x: str(x) if x <= 3 else "4+"
    )

    data[output_col] = pd.Categorical(
        data[output_col],
        categories=["0", "1", "2", "3", "4+"],
        ordered=True,
    )

    return data


def add_days_hospitalized_group(
        df: pd.DataFrame,
        source_col: str = "days_hospitalized_last_3yrs",
        output_col: str = "days_hospitalized_grouped",
) -> pd.DataFrame:
    if source_col not in df.columns:
        raise ValueError(f"Column '{source_col}' was not found in the dataframe.")

    data = df.copy()

    data[output_col] = data[source_col].apply(
        lambda x: "0" if x == 0 else "1-3" if x <= 3 else "4+"
    )

    data[output_col] = pd.Categorical(
        data[output_col],
        categories=["0", "1-3", "4+"],
        ordered=True,
    )

    return data


def add_grouped_health_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data = add_hospitalization_group(data)
    data = add_chronic_count_group(data)
    data = add_days_hospitalized_group(data)

    return data


def add_high_cost_flags(
        df: pd.DataFrame,
        target_col: str = DEFAULT_TARGET,
        quantiles: Iterable[float] = (0.90, 0.95, 0.99),
) -> pd.DataFrame:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' was not found in the dataframe.")

    data = df.copy()

    for q in quantiles:
        percentile = int(round(q * 100))
        threshold = data[target_col].quantile(q)
        data[f"high_cost_{percentile}"] = (data[target_col] >= threshold).astype(int)

    return data


def prepare_model_data(
        df: pd.DataFrame,
        target_col: str = DEFAULT_TARGET,
        add_log: bool = True,
        add_groups: bool = True,
        add_tail_flags: bool = False,
) -> pd.DataFrame:
    data = df.copy()

    if add_log:
        data = add_log_target(data, target_col=target_col)

    if add_groups:
        data = add_grouped_health_features(data)

    if add_tail_flags:
        data = add_high_cost_flags(data, target_col=target_col)

    return data
