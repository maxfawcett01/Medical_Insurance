from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score


def create_prediction_frame(
        x_test: Optional[pd.DataFrame],
        y_true: pd.Series | np.ndarray,
        y_pred: pd.Series | np.ndarray,
        y_true_transformed: Optional[pd.Series | np.ndarray] = None,
        y_pred_transformed: Optional[pd.Series | np.ndarray] = None,
        actual_col: str = "actual",
        predicted_col: str = "predicted",
        transformed_actual_col: str = "actual_transformed",
        transformed_predicted_col: str = "predicted_transformed",
        residual_col: str = "residual",
        absolute_error_col: str = "absolute_error",
) -> pd.DataFrame:
    if x_test is None:
        output = pd.DataFrame()
    else:
        output = x_test.copy()

    output[actual_col] = np.asarray(y_true)
    output[predicted_col] = np.asarray(y_pred)
    output[residual_col] = output[actual_col] - output[predicted_col]
    output[absolute_error_col] = output[residual_col].abs()

    if y_true_transformed is not None:
        output[transformed_actual_col] = np.asarray(y_true_transformed)

    if y_pred_transformed is not None:
        output[transformed_predicted_col] = np.asarray(y_pred_transformed)

    return output


def _pearson_correlation(
        y_true: np.ndarray,
        y_pred: np.ndarray,
) -> float:
    if len(y_true) < 2:
        return np.nan

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan

    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _standard_deviation_error_prediction(
        errors: np.ndarray,
) -> float:
    if len(errors) < 2:
        return np.nan

    return float(np.std(errors, ddof=1))


def _base_regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        scale_name: str,
) -> dict[str, float]:
    errors = y_true - y_pred

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    return {
        f"rmse_{scale_name}": float(rmse),
        f"r2_{scale_name}": float(r2_score(y_true, y_pred)),
        f"pearson_correlation_{scale_name}": _pearson_correlation(y_true, y_pred),
        f"sdep_{scale_name}": _standard_deviation_error_prediction(errors),
        f"bias_{scale_name}": float(np.mean(errors)),
        f"mae_{scale_name}": float(mae),
    }


def regression_metrics(
        y_true: pd.Series | np.ndarray,
        y_pred: pd.Series | np.ndarray,
        y_true_transformed: Optional[pd.Series | np.ndarray] = None,
        y_pred_transformed: Optional[pd.Series | np.ndarray] = None,
        original_scale_name: str = "target",
        transformed_scale_name: str = "transformed",
        include_tail_metrics: bool = True,
        tail_quantiles: Sequence[float] = (0.90, 0.95, 0.99),
) -> pd.DataFrame:
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)

    metrics = _base_regression_metrics(
        y_true=y_true_array,
        y_pred=y_pred_array,
        scale_name=original_scale_name,
    )

    metrics.update({
        f"mean_actual_{original_scale_name}": float(np.mean(y_true_array)),
        f"mean_predicted_{original_scale_name}": float(np.mean(y_pred_array)),
        f"median_actual_{original_scale_name}": float(np.median(y_true_array)),
        f"median_predicted_{original_scale_name}": float(np.median(y_pred_array)),
    })

    if include_tail_metrics:
        for q in tail_quantiles:
            q_label = int(round(q * 100))
            actual_q = float(np.quantile(y_true_array, q))
            predicted_q = float(np.quantile(y_pred_array, q))

            metrics[f"actual_p{q_label}_{original_scale_name}"] = actual_q
            metrics[f"predicted_p{q_label}_{original_scale_name}"] = predicted_q
            metrics[f"p{q_label}_underprediction_{original_scale_name}"] = actual_q - predicted_q

    if y_true_transformed is not None and y_pred_transformed is not None:
        y_true_transformed_array = np.asarray(y_true_transformed)
        y_pred_transformed_array = np.asarray(y_pred_transformed)

        transformed_metrics = _base_regression_metrics(
            y_true=y_true_transformed_array,
            y_pred=y_pred_transformed_array,
            scale_name=transformed_scale_name,
        )

        metrics.update(transformed_metrics)

    return (
        pd.DataFrame.from_dict(metrics, orient="index", columns=["value"])
        .round(4)
    )


def tail_summary(
        y_true: pd.Series | np.ndarray,
        y_pred: pd.Series | np.ndarray,
        quantiles: Sequence[float] = (0.50, 0.90, 0.95, 0.99),
        include_mean: bool = True,
        include_max: bool = True,
        actual_col: str = "actual",
        predicted_col: str = "predicted",
) -> pd.DataFrame:
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)

    rows = []

    if include_mean:
        rows.append({
            "statistic": "mean",
            actual_col: np.mean(y_true_array),
            predicted_col: np.mean(y_pred_array),
        })

    for q in quantiles:
        q_label = int(round(q * 100))
        rows.append({
            "statistic": f"p{q_label}",
            actual_col: np.quantile(y_true_array, q),
            predicted_col: np.quantile(y_pred_array, q),
        })

    if include_max:
        rows.append({
            "statistic": "max",
            actual_col: np.max(y_true_array),
            predicted_col: np.max(y_pred_array),
        })

    output = pd.DataFrame(rows)
    output["difference"] = output[actual_col] - output[predicted_col]
    output["relative_difference"] = output["difference"] / output[actual_col]

    return output


def grouped_prediction_summary(
        data: pd.DataFrame,
        group_col: str,
        actual_col: str = "actual",
        predicted_col: str = "predicted",
        absolute_error_col: str = "absolute_error",
        quantiles: Sequence[float] = (0.90, 0.95, 0.99),
        round_digits: int = 2,
) -> pd.DataFrame:
    if group_col not in data.columns:
        raise ValueError(f"Group column '{group_col}' was not found in the dataframe.")

    required_cols = [actual_col, predicted_col, absolute_error_col]
    missing_cols = [col for col in required_cols if col not in data.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    grouped = data.groupby(group_col, observed=True)

    summary = grouped.agg(
        n=(actual_col, "size"),
        actual_mean=(actual_col, "mean"),
        predicted_mean=(predicted_col, "mean"),
        actual_median=(actual_col, "median"),
        predicted_median=(predicted_col, "median"),
        mae=(absolute_error_col, "mean"),
    )

    for q in quantiles:
        q_label = int(round(q * 100))
        summary[f"actual_p{q_label}"] = grouped[actual_col].quantile(q)
        summary[f"predicted_p{q_label}"] = grouped[predicted_col].quantile(q)
        summary[f"p{q_label}_difference"] = (
                summary[f"actual_p{q_label}"] - summary[f"predicted_p{q_label}"]
        )

    return summary.round(round_digits)


def error_by_actual_quantile(
        data: pd.DataFrame,
        actual_col: str = "actual",
        predicted_col: str = "predicted",
        absolute_error_col: str = "absolute_error",
        bins: int = 10,
) -> pd.DataFrame:
    required_cols = [actual_col, predicted_col, absolute_error_col]
    missing_cols = [col for col in required_cols if col not in data.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    temp = data[[actual_col, predicted_col, absolute_error_col]].copy()

    temp["actual_quantile_band"] = pd.qcut(
        temp[actual_col],
        q=bins,
        duplicates="drop",
    )

    return (
        temp.groupby("actual_quantile_band", observed=True)
        .agg(
            n=(actual_col, "size"),
            actual_mean=(actual_col, "mean"),
            predicted_mean=(predicted_col, "mean"),
            mae=(absolute_error_col, "mean"),
            actual_min=(actual_col, "min"),
            actual_max=(actual_col, "max"),
        )
        .reset_index()
    )


def plot_actual_vs_predicted(
        data: pd.DataFrame,
        actual_col: str = "actual",
        predicted_col: str = "predicted",
        sample_size: int = 5000,
        random_state: int = 42,
        title: str = "Actual vs Predicted",
        xlabel: str = "Actual",
        ylabel: str = "Predicted",
):
    if len(data) > sample_size:
        plot_data = data.sample(sample_size, random_state=random_state)
    else:
        plot_data = data.copy()

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    sns.scatterplot(
        data=plot_data,
        x=actual_col,
        y=predicted_col,
        alpha=0.35,
        edgecolor=None,
        ax=ax,
    )

    actual_values = pd.to_numeric(plot_data[actual_col], errors="coerce")
    predicted_values = pd.to_numeric(plot_data[predicted_col], errors="coerce")

    min_val = float(np.nanmin([actual_values.min(), predicted_values.min()]))
    max_val = float(np.nanmax([actual_values.max(), predicted_values.max()]))

    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    fig.tight_layout()

    return fig, ax


def plot_residual_distribution(
        data: pd.DataFrame,
        residual_col: str = "residual",
        bins: int = 80,
        title: str = "Residual Distribution",
        xlabel: str = "Residual",
):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    sns.histplot(
        data=data,
        x=residual_col,
        bins=bins,
        edgecolor="white",
        ax=ax,
    )

    ax.axvline(0, linestyle="--", linewidth=2)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")

    fig.tight_layout()

    return fig, ax


def plot_tail_comparison(
        tail_df: pd.DataFrame,
        statistics: Sequence[str] = ("p90", "p95", "p99"),
        actual_col: str = "actual",
        predicted_col: str = "predicted",
        title: str = "Actual vs Predicted Tail Percentiles",
        ylabel: str = "Target",
):
    plot_data = tail_df[tail_df["statistic"].isin(statistics)].copy()

    plot_data_long = plot_data.melt(
        id_vars="statistic",
        value_vars=[actual_col, predicted_col],
        var_name="series",
        value_name="value",
    )

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    sns.barplot(
        data=plot_data_long,
        x="statistic",
        y="value",
        hue="series",
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel("Statistic")
    ax.set_ylabel(ylabel)

    fig.tight_layout()

    return fig, ax


def plot_grouped_actual_vs_predicted(
        grouped_summary: pd.DataFrame,
        group_col: str,
        actual_col: str = "actual_mean",
        predicted_col: str = "predicted_mean",
        title: str = "Actual vs Predicted by Group",
        xlabel: str = "Group",
        ylabel: str = "Target",
):
    plot_data = grouped_summary.reset_index().melt(
        id_vars=group_col,
        value_vars=[actual_col, predicted_col],
        var_name="series",
        value_name="value",
    )

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    sns.barplot(
        data=plot_data,
        x=group_col,
        y="value",
        hue="series",
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    fig.tight_layout()

    return fig, ax


def run_regression_evaluation(
        x_test: Optional[pd.DataFrame],
        y_true: pd.Series | np.ndarray,
        y_pred: pd.Series | np.ndarray,
        y_true_transformed: Optional[pd.Series | np.ndarray] = None,
        y_pred_transformed: Optional[pd.Series | np.ndarray] = None,
        actual_col: str = "actual",
        predicted_col: str = "predicted",
        transformed_actual_col: str = "actual_transformed",
        transformed_predicted_col: str = "predicted_transformed",
        group_cols: Optional[Sequence[str]] = None,
        original_scale_name: str = "target",
        transformed_scale_name: str = "transformed",
) -> dict[str, pd.DataFrame]:
    predictions = create_prediction_frame(
        x_test=x_test,
        y_true=y_true,
        y_pred=y_pred,
        y_true_transformed=y_true_transformed,
        y_pred_transformed=y_pred_transformed,
        actual_col=actual_col,
        predicted_col=predicted_col,
        transformed_actual_col=transformed_actual_col,
        transformed_predicted_col=transformed_predicted_col,
    )

    metrics = regression_metrics(
        y_true=predictions[actual_col],
        y_pred=predictions[predicted_col],
        y_true_transformed=(
            predictions[transformed_actual_col]
            if transformed_actual_col in predictions.columns
            else None
        ),
        y_pred_transformed=(
            predictions[transformed_predicted_col]
            if transformed_predicted_col in predictions.columns
            else None
        ),
        original_scale_name=original_scale_name,
        transformed_scale_name=transformed_scale_name,
    )

    tails = tail_summary(
        y_true=predictions[actual_col],
        y_pred=predictions[predicted_col],
        actual_col=actual_col,
        predicted_col=predicted_col,
    )

    error_quantiles = error_by_actual_quantile(
        data=predictions,
        actual_col=actual_col,
        predicted_col=predicted_col,
        absolute_error_col="absolute_error",
    )

    results = {
        "predictions": predictions,
        "metrics": metrics,
        "tail_summary": tails,
        "error_by_actual_quantile": error_quantiles,
    }

    if group_cols is not None:
        for group_col in group_cols:
            results[f"group_summary_{group_col}"] = grouped_prediction_summary(
                data=predictions,
                group_col=group_col,
                actual_col=actual_col,
                predicted_col=predicted_col,
                absolute_error_col="absolute_error",
            )

    return results
