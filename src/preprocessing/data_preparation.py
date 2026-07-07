"""Helpers for preparing model-ready training and test data.

This module keeps the data preparation workflow separate from model training.
It performs the leakage-safe split and feature engineering sequence required by
future modeling phases.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from .feature_config import TARGET_COLUMN
from .feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "german_credit_processed.csv"


def load_processed_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load the processed German Credit dataset from disk.

    Args:
        path: Optional path to the processed dataset CSV file.

    Returns:
        pd.DataFrame: The loaded dataframe.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the required columns are missing.
    """
    data_path = Path(path) if path is not None else DEFAULT_DATA_PATH
    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    required_columns = {TARGET_COLUMN, "credit_amount", "duration_months", "age"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    logger.info("Loaded processed dataset from %s", data_path)
    return df


def encode_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Encode the target column into binary form without mutating the input.

    Args:
        df: Dataframe containing the target column.

    Returns:
        tuple[pd.DataFrame, pd.Series]: A dataframe with the original columns and
            the encoded target series.
    """
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' is missing from the dataframe.")

    encoded_df = df.copy(deep=True)
    target_mapping = {"good": 1, "bad": 0}
    target_series = encoded_df[TARGET_COLUMN].map(target_mapping)

    if target_series.isna().any():
        raise ValueError("Target values must be either 'good' or 'bad' before encoding.")

    encoded_df = encoded_df.drop(columns=[TARGET_COLUMN])
    logger.info("Encoded target column to binary values.")
    return encoded_df, target_series


def create_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a stratified train/test split.

    Args:
        X: Feature dataframe.
        y: Target series.
        test_size: Fraction of data to retain for testing.
        random_state: Random seed for reproducibility.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Train and test
            feature/target splits.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    logger.info(
        "Created train/test split with %d train rows and %d test rows.",
        len(X_train),
        len(X_test),
    )
    return X_train, X_test, y_train, y_test


def prepare_model_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, FeatureEngineer]:
    """Prepare leakage-safe train/test data with engineered features.

    The correct sequence is:
    1. Split the raw data into train and test sets.
    2. Fit the feature engineering transformer on the training set only.
    3. Transform both train and test sets using the same fitted transformer.

    Args:
        df: Dataframe containing the processed dataset.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, FeatureEngineer]:
            Engineered train/test features and target splits plus the fitted
            feature engineering transformer.
    """
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' is missing from the dataframe.")

    X, y = encode_target(df)
    X_train, X_test, y_train, y_test = create_train_test_split(X, y)

    feature_engineer = FeatureEngineer()
    X_train_fe = feature_engineer.fit_transform(X_train)
    X_test_fe = feature_engineer.transform(X_test)

    logger.info("Prepared model-ready train/test data with engineered features.")
    return X_train_fe, X_test_fe, y_train, y_test, feature_engineer
