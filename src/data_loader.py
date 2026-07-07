"""Utilities for downloading and preparing the German Credit dataset."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict
from urllib.error import URLError
from urllib.request import urlretrieve

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "german.data"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "german_credit_processed.csv"
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

COLUMN_NAMES = [
    "checking_account_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_account",
    "employment_duration",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "present_residence_years",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "people_liable",
    "telephone",
    "foreign_worker",
    "credit_risk",
]

TARGET_MAPPING = {1: "good", 2: "bad"}


def download_dataset(url: str = DATASET_URL, destination_path: Path = RAW_DATA_PATH) -> Path:
    """Download the German Credit dataset if it is not already available locally."""
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    if destination_path.exists():
        logger.info("Dataset already exists at %s", destination_path)
        return destination_path

    try:
        urlretrieve(url, destination_path)
    except (URLError, OSError) as exc:
        raise RuntimeError(f"Failed to download dataset from {url}: {exc}") from exc

    logger.info("Dataset downloaded successfully to %s", destination_path)
    return destination_path


def load_raw_dataset(file_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw German Credit dataset from disk."""
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    try:
        df = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")
    except Exception as exc:
        raise RuntimeError(f"Unable to read dataset file: {exc}") from exc

    logger.info("Raw data loaded successfully from %s", file_path)
    return df


def add_column_names(df: pd.DataFrame, column_names: list[str] | None = None) -> pd.DataFrame:
    """Assign column names to the raw dataset and validate the expected shape."""
    names = column_names or COLUMN_NAMES
    if df.shape[1] != len(names):
        raise ValueError(f"Expected {len(names)} columns but found {df.shape[1]}")

    df = df.copy()
    df.columns = names
    return df


def save_processed_dataset(df: pd.DataFrame, output_path: Path = PROCESSED_DATA_PATH) -> Path:
    """Save the processed dataset as a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Processed data saved to %s", output_path)
    return output_path


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Return a beginner-friendly summary of the dataset."""
    return {
        "shape": df.shape,
        "first_rows": df.head().to_dict(orient="records"),
        "target_distribution": df["credit_risk"].value_counts().to_dict(),
    }


def main() -> None:
    """Run the data loading workflow and print a short dataset preview."""
    try:
        download_dataset()
        raw_df = load_raw_dataset()
        processed_df = add_column_names(raw_df)
        processed_df["credit_risk"] = processed_df["credit_risk"].map(TARGET_MAPPING)
        save_processed_dataset(processed_df)

        summary = get_dataset_summary(processed_df)
        logger.info("Dataset shape: %s", summary["shape"])

        print("\nDataset shape:")
        print(summary["shape"])
        print("\nFirst 5 rows:")
        print(processed_df.head())
        print("\nTarget distribution:")
        print(processed_df["credit_risk"].value_counts())
    except Exception as exc:
        logger.error("An error occurred: %s", exc)
        raise


if __name__ == "__main__":
    main()
