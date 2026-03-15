import pandas as pd


def data_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_count": df.isna().sum(),
        "missing_pct": df.isna().mean() * 100,
        "n_unique": df.nunique()
    })
    return summary.sort_values(["missing_pct", "n_unique"], ascending=[False, False])


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]).T
