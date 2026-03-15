from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data"
RAW_DATA_DIR = DATA_DIR / "Raw"
PROCESSED_DATA_DIR = DATA_DIR / "Processed"

RAW_MEDICAL_INSURANCE_FILE = RAW_DATA_DIR / "medical_insurance.csv"
