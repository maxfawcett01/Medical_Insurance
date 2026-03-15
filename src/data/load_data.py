import pandas as pd

from src.config import RAW_MEDICAL_INSURANCE_FILE


def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_MEDICAL_INSURANCE_FILE)
    return df
