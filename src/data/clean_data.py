import pandas as pd


def standardise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^\w]", "", regex=True)
    )
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().copy()


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = standardise_column_names(df)
    df = remove_duplicates(df)
    return df
