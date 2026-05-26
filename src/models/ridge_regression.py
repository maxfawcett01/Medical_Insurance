from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class RidgeRegressionResult:
    model: Pipeline
    target_col: str
    feature_cols: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    y_train_transformed: np.ndarray
    y_test_transformed: np.ndarray
    target_transform: str
    selected_alpha: Optional[float]


def _make_one_hot_encoder(
        handle_unknown: str = "ignore",
        sparse_output: bool = False,
) -> OneHotEncoder:
    try:
        return OneHotEncoder(
            handle_unknown=handle_unknown,
            sparse_output=sparse_output,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown=handle_unknown,
            sparse=sparse_output,
        )


def infer_feature_columns(
        df: pd.DataFrame,
        target_col: str,
        drop_cols: Optional[Sequence[str]] = None,
) -> list[str]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' was not found in the dataframe.")

    drop_set = set(drop_cols or [])
    drop_set.add(target_col)

    return [col for col in df.columns if col not in drop_set]


def split_feature_types(
        df: pd.DataFrame,
        feature_cols: Sequence[str],
) -> tuple[list[str], list[str]]:
    x = df[list(feature_cols)]

    numeric_cols = x.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [col for col in feature_cols if col not in numeric_cols]

    return numeric_cols, categorical_cols


def transform_target(
        y: pd.Series | np.ndarray,
        target_transform: str = "none",
) -> np.ndarray:
    y_array = np.asarray(y)

    if target_transform == "none":
        return y_array

    if target_transform == "log1p":
        if np.any(y_array < 0):
            raise ValueError("target_transform='log1p' requires non-negative target values.")
        return np.log1p(y_array)

    raise ValueError(
        "Unsupported target_transform. Use one of: 'none', 'log1p'."
    )


def inverse_transform_target(
        y_pred: np.ndarray,
        target_transform: str = "none",
        clip_min: Optional[float] = None,
) -> np.ndarray:
    if target_transform == "none":
        y_output = np.asarray(y_pred)

    elif target_transform == "log1p":
        y_output = np.expm1(y_pred)

    else:
        raise ValueError(
            "Unsupported target_transform. Use one of: 'none', 'log1p'."
        )

    if clip_min is not None:
        y_output = np.maximum(y_output, clip_min)

    return y_output


def build_ridge_pipeline(
        numeric_cols: Sequence[str],
        categorical_cols: Sequence[str],
        use_cv: bool = True,
        alpha: float = 1.0,
        alphas: Optional[Iterable[float]] = None,
        ridge_params: Optional[dict] = None,
        ridge_cv_params: Optional[dict] = None,
        numeric_impute_strategy: str = "median",
        categorical_impute_strategy: str = "most_frequent",
        scale_numeric: bool = True,
        one_hot_encode: bool = True,
        one_hot_handle_unknown: str = "ignore",
) -> Pipeline:
    ridge_params = ridge_params or {}
    ridge_cv_params = ridge_cv_params or {}

    if alphas is None:
        alphas = np.logspace(-3, 3, 25)

    transformers = []

    if numeric_cols:
        numeric_steps: list[tuple[str, BaseEstimator]] = [
            ("imputer", SimpleImputer(strategy=numeric_impute_strategy)),
        ]

        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))

        numeric_pipeline = Pipeline(steps=numeric_steps)

        transformers.append(
            ("numeric", numeric_pipeline, list(numeric_cols))
        )

    if categorical_cols:
        categorical_steps: list[tuple[str, BaseEstimator]] = [
            ("imputer", SimpleImputer(strategy=categorical_impute_strategy)),
        ]

        if one_hot_encode:
            categorical_steps.append(
                (
                    "one_hot_encoder",
                    _make_one_hot_encoder(handle_unknown=one_hot_handle_unknown),
                )
            )

        categorical_pipeline = Pipeline(steps=categorical_steps)

        transformers.append(
            ("categorical", categorical_pipeline, list(categorical_cols))
        )

    if not transformers:
        raise ValueError("No usable numeric or categorical feature columns were found.")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    if use_cv:
        ridge_model = RidgeCV(
            alphas=np.array(list(alphas)),
            **ridge_cv_params,
        )
    else:
        ridge_model = Ridge(
            alpha=alpha,
            **ridge_params,
        )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", ridge_model),
        ]
    )


def train_ridge_regression(
        df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[Sequence[str]] = None,
        drop_cols: Optional[Sequence[str]] = None,
        test_size: float = 0.20,
        random_state: int = 42,
        shuffle: bool = True,
        target_transform: str = "none",
        use_cv: bool = True,
        alpha: float = 1.0,
        alphas: Optional[Iterable[float]] = None,
        ridge_params: Optional[dict] = None,
        ridge_cv_params: Optional[dict] = None,
        numeric_impute_strategy: str = "median",
        categorical_impute_strategy: str = "most_frequent",
        scale_numeric: bool = True,
        one_hot_encode: bool = True,
        one_hot_handle_unknown: str = "ignore",
) -> RidgeRegressionResult:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' was not found in the dataframe.")

    data = df.copy()
    data = data.dropna(subset=[target_col])

    if feature_cols is None:
        feature_cols = infer_feature_columns(
            data,
            target_col=target_col,
            drop_cols=drop_cols,
        )
    else:
        feature_cols = list(feature_cols)

    x = data[list(feature_cols)].copy()
    y = data[target_col].copy()

    numeric_cols, categorical_cols = split_feature_types(data, feature_cols)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )

    y_train_transformed = transform_target(
        y_train,
        target_transform=target_transform,
    )

    y_test_transformed = transform_target(
        y_test,
        target_transform=target_transform,
    )

    model = build_ridge_pipeline(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        use_cv=use_cv,
        alpha=alpha,
        alphas=alphas,
        ridge_params=ridge_params,
        ridge_cv_params=ridge_cv_params,
        numeric_impute_strategy=numeric_impute_strategy,
        categorical_impute_strategy=categorical_impute_strategy,
        scale_numeric=scale_numeric,
        one_hot_encode=one_hot_encode,
        one_hot_handle_unknown=one_hot_handle_unknown,
    )

    model.fit(x_train, y_train_transformed)

    ridge_step = model.named_steps["model"]

    if hasattr(ridge_step, "alpha_"):
        selected_alpha = float(ridge_step.alpha_)
    elif hasattr(ridge_step, "alpha"):
        selected_alpha = float(ridge_step.alpha)
    else:
        selected_alpha = None

    return RidgeRegressionResult(
        model=model,
        target_col=target_col,
        feature_cols=list(feature_cols),
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        y_train_transformed=y_train_transformed,
        y_test_transformed=y_test_transformed,
        target_transform=target_transform,
        selected_alpha=selected_alpha,
    )


def predict_ridge_regression(
        model: Pipeline,
        df: pd.DataFrame,
        target_transform: str = "none",
        clip_min: Optional[float] = None,
) -> np.ndarray:
    y_pred_transformed = model.predict(df)

    return inverse_transform_target(
        y_pred_transformed,
        target_transform=target_transform,
        clip_min=clip_min,
    )


def get_transformed_predictions(
        model: Pipeline,
        df: pd.DataFrame,
) -> np.ndarray:
    return model.predict(df)


def get_feature_coefficients(
        result: RidgeRegressionResult,
) -> pd.DataFrame:
    pipeline = result.model
    preprocessor = pipeline.named_steps["preprocessor"]
    ridge_model = pipeline.named_steps["model"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except (AttributeError, ValueError, NotImplementedError):
        feature_names = np.array(result.feature_cols)

    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": ridge_model.coef_,
            "abs_coefficient": np.abs(ridge_model.coef_),
        }
    )

    return coef_df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def save_ridge_model(
        result: RidgeRegressionResult,
        file_path: str | Path,
) -> None:
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, file_path)


def load_ridge_model(
        file_path: str | Path,
) -> RidgeRegressionResult:
    return joblib.load(file_path)
