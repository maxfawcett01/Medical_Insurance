from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from shap.maskers import Independent
from sklearn.pipeline import Pipeline


@dataclass
class ShapAnalysisResult:
    explainer: Any
    shap_values: Any
    values: np.ndarray
    base_values: np.ndarray | float
    data: pd.DataFrame
    feature_names: list[str]
    importance_df: pd.DataFrame
    model_type: str
    used_transformed_features: bool


def sample_dataframe(
        df: pd.DataFrame,
        sample_size: Optional[int] = None,
        random_state: int = 42,
) -> pd.DataFrame:
    if sample_size is None or sample_size >= len(df):
        return df.copy()

    return df.sample(sample_size, random_state=random_state).copy()


def _is_pipeline(model: Any) -> bool:
    return isinstance(model, Pipeline)


def _get_pipeline_preprocessor_and_estimator(
        model: Pipeline,
        preprocessor_step_name: str = "preprocessor",
        estimator_step_name: str = "model",
) -> tuple[Any, Any]:
    if preprocessor_step_name not in model.named_steps:
        raise ValueError(
            f"Pipeline does not contain a '{preprocessor_step_name}' step."
        )

    if estimator_step_name not in model.named_steps:
        raise ValueError(
            f"Pipeline does not contain a '{estimator_step_name}' step."
        )

    return model.named_steps[preprocessor_step_name], model.named_steps[estimator_step_name]


def _get_feature_names(
        preprocessor: Any,
        fallback_feature_names: list[str],
) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except (AttributeError, ValueError, NotImplementedError):
        return fallback_feature_names


def _to_dense_array(values: Any) -> np.ndarray:
    if hasattr(values, "toarray"):
        return values.toarray()

    return np.asarray(values)


def prepare_shap_input(
        model: Any,
        x_background: pd.DataFrame,
        x_explain: pd.DataFrame,
        use_transformed_features: bool = True,
        preprocessor_step_name: str = "preprocessor",
        estimator_step_name: str = "model",
) -> tuple[Any, pd.DataFrame, pd.DataFrame, list[str], bool]:
    if _is_pipeline(model) and use_transformed_features:
        preprocessor, estimator = _get_pipeline_preprocessor_and_estimator(
            model,
            preprocessor_step_name=preprocessor_step_name,
            estimator_step_name=estimator_step_name,
        )

        background_transformed = _to_dense_array(
            preprocessor.transform(x_background)
        )

        explain_transformed = _to_dense_array(
            preprocessor.transform(x_explain)
        )

        feature_names = _get_feature_names(
            preprocessor,
            fallback_feature_names=list(x_background.columns),
        )

        background_df = pd.DataFrame(
            background_transformed,
            columns=feature_names,
            index=x_background.index,
        )

        explain_df = pd.DataFrame(
            explain_transformed,
            columns=feature_names,
            index=x_explain.index,
        )

        return estimator, background_df, explain_df, feature_names, True

    feature_names = list(x_background.columns)

    return model, x_background.copy(), x_explain.copy(), feature_names, False


def infer_model_type(model: Any) -> str:
    model_name = model.__class__.__name__.lower()

    linear_keywords = [
        "linear",
        "ridge",
        "lasso",
        "elasticnet",
        "logisticregression",
    ]

    tree_keywords = [
        "randomforest",
        "gradientboosting",
        "histgradientboosting",
        "decisiontree",
        "extratrees",
        "xgb",
        "xgboost",
        "lgbm",
        "lightgbm",
        "catboost",
    ]

    if any(keyword in model_name for keyword in linear_keywords):
        return "linear"

    if any(keyword in model_name for keyword in tree_keywords):
        return "tree"

    return "generic"


def _extract_shap_values_array(shap_values: Any) -> np.ndarray:
    if hasattr(shap_values, "values"):
        values = shap_values.values
    else:
        values = shap_values

    values = np.asarray(values)

    if values.ndim == 3:
        values = values[:, :, 0]

    return values


def _extract_base_values(shap_values: Any) -> np.ndarray | float:
    if hasattr(shap_values, "base_values"):
        return shap_values.base_values

    return np.nan


def calculate_shap_importance(
        values: np.ndarray,
        feature_names: list[str],
) -> pd.DataFrame:
    mean_abs_shap = np.mean(np.abs(values), axis=0)
    mean_shap = np.mean(values, axis=0)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
        "mean_shap": mean_shap,
    })

    return (
        importance_df
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def run_shap_analysis(
        model: Any,
        x_background: pd.DataFrame,
        x_explain: pd.DataFrame,
        model_type: str = "auto",
        use_transformed_features: bool = True,
        background_size: Optional[int] = 500,
        explain_size: Optional[int] = 2000,
        masker_max_samples: Optional[int] = None,
        random_state: int = 42,
        preprocessor_step_name: str = "preprocessor",
        estimator_step_name: str = "model",
) -> ShapAnalysisResult:
    x_background_sample = sample_dataframe(
        x_background,
        sample_size=background_size,
        random_state=random_state,
    )

    x_explain_sample = sample_dataframe(
        x_explain,
        sample_size=explain_size,
        random_state=random_state,
    )

    shap_model, background_prepared, explain_prepared, feature_names, used_transformed = prepare_shap_input(
        model=model,
        x_background=x_background_sample,
        x_explain=x_explain_sample,
        use_transformed_features=use_transformed_features,
        preprocessor_step_name=preprocessor_step_name,
        estimator_step_name=estimator_step_name,
    )

    if model_type == "auto":
        model_type = infer_model_type(shap_model)

    if model_type == "linear":
        if masker_max_samples is None:
            masker_max_samples = len(background_prepared)

        masker = Independent(
            background_prepared,
            max_samples=masker_max_samples,
        )

        explainer = shap.LinearExplainer(
            shap_model,
            masker,
        )
        shap_values = explainer(explain_prepared)

    elif model_type == "tree":
        explainer = shap.TreeExplainer(
            shap_model,
        )
        shap_values = explainer(explain_prepared)

    elif model_type == "generic":
        if masker_max_samples is None:
            masker_max_samples = len(background_prepared)

        masker = Independent(
            background_prepared,
            max_samples=masker_max_samples,
        )

        explainer = shap.Explainer(
            shap_model.predict,
            masker,
        )
        shap_values = explainer(explain_prepared)

    else:
        raise ValueError("model_type must be one of: 'auto', 'linear', 'tree', 'generic'.")

    values = _extract_shap_values_array(shap_values)
    base_values = _extract_base_values(shap_values)

    importance_df = calculate_shap_importance(
        values=values,
        feature_names=feature_names,
    )

    return ShapAnalysisResult(
        explainer=explainer,
        shap_values=shap_values,
        values=values,
        base_values=base_values,
        data=explain_prepared,
        feature_names=feature_names,
        importance_df=importance_df,
        model_type=model_type,
        used_transformed_features=used_transformed,
    )


def plot_shap_bar(
        shap_result: ShapAnalysisResult,
        max_display: int = 25,
        show: bool = True,
):
    shap.plots.bar(
        shap_result.shap_values,
        max_display=max_display,
        show=show,
    )


def plot_shap_beeswarm(
        shap_result: ShapAnalysisResult,
        max_display: int = 25,
        text_colour: str = "#d9d9d9",
):
    shap.plots.beeswarm(
        shap_result.shap_values,
        max_display=max_display,
        show=False
    )

    fig = plt.gcf()

    for axis in fig.axes:
        axis.tick_params(colors=text_colour)

        axis.xaxis.label.set_color(text_colour)
        axis.yaxis.label.set_color(text_colour)
        axis.title.set_color(text_colour)

        for spine in axis.spines.values():
            spine.set_color(text_colour)

    return fig


def plot_shap_waterfall(
        shap_result: ShapAnalysisResult,
        row_number: int = 0,
        max_display: int = 20,
        show: bool = True,
):
    shap.plots.waterfall(
        shap_result.shap_values[row_number],
        max_display=max_display,
        show=show,
    )


def plot_shap_importance_dataframe(
        shap_result: ShapAnalysisResult,
        max_display: int = 25,
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_data = (
        shap_result.importance_df
        .head(max_display)
        .sort_values("mean_abs_shap")
    )

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    sns.barplot(
        data=plot_data,
        x="mean_abs_shap",
        y="feature",
        ax=ax,
    )

    ax.set_title("Mean Absolute SHAP Feature Importance")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_ylabel("Feature")

    fig.tight_layout()

    return fig, ax


def save_shap_outputs(
        shap_result: ShapAnalysisResult,
        output_dir: str | Path,
        prefix: str = "shap",
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    importance_path = output_dir / f"{prefix}_importance.csv"
    values_path = output_dir / f"{prefix}_values.npy"
    data_path = output_dir / f"{prefix}_explained_data.csv"
    result_path = output_dir / f"{prefix}_result.joblib"

    shap_result.importance_df.to_csv(importance_path, index=False)
    np.save(values_path, shap_result.values)
    shap_result.data.to_csv(data_path, index=False)
    joblib.dump(shap_result, result_path)

    return {
        "importance": importance_path,
        "values": values_path,
        "explained_data": data_path,
        "result": result_path,
    }


def load_shap_result(file_path: str | Path) -> ShapAnalysisResult:
    return joblib.load(file_path)
