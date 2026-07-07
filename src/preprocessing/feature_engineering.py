"""Reusable feature engineering helpers for the German Credit project.

This module implements leakage-safe feature engineering for the initial EDA-based
features requested for the project. The logic is designed to be reused later by
training notebooks, evaluation scripts, and deployment pipelines.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_REQUIRED_COLUMNS = {"credit_amount", "duration_months", "age"}
_ENGINEERED_FEATURE_NAMES = [
    "credit_amount_per_month",
    "high_credit_amount_flag",
    "long_duration_flag",
    "age_group",
    "credit_duration_interaction",
]


class FeatureEngineer:
    """Fit and transform domain-inspired features without leaking training statistics.

    The class stores statistics learned during ``fit`` so that later calls to
    ``transform`` use only training-derived values. This prevents leakage when
    the same logic is reused during train/test workflows.
    """

    def __init__(self) -> None:
        """Initialize the transformer with no fitted state."""
        self.credit_amount_75th_percentile: float | None = None

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        """Learn statistics from the supplied dataset.

        Args:
            df: Training data used to compute the threshold for the high-credit
                flag.

        Returns:
            FeatureEngineer: The fitted transformer instance.
        """
        self._validate_required_columns(df)
        self.credit_amount_75th_percentile = float(df["credit_amount"].quantile(0.75))
        logger.info(
            "FeatureEngineer fitted with credit_amount 75th percentile: %.2f",
            self.credit_amount_75th_percentile,
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features for a dataframe.

        Args:
            df: Dataframe to transform.

        Returns:
            pd.DataFrame: A copy of the dataframe with engineered features added.
        """
        self._validate_required_columns(df)
        if self.credit_amount_75th_percentile is None:
            raise RuntimeError("FeatureEngineer must be fitted before transform().")

        transformed_df = df.copy(deep=True)

        transformed_df["credit_amount_per_month"] = np.where(
            transformed_df["duration_months"] != 0,
            transformed_df["credit_amount"] / transformed_df["duration_months"],
            np.nan,
        )
        transformed_df["high_credit_amount_flag"] = (
            transformed_df["credit_amount"] > self.credit_amount_75th_percentile
        ).astype(int)
        transformed_df["long_duration_flag"] = (
            transformed_df["duration_months"] > 24
        ).astype(int)
        transformed_df["age_group"] = pd.cut(
            transformed_df["age"],
            bins=[0, 25, 35, 50, np.inf],
            labels=["young", "adult", "middle_aged", "senior"],
            include_lowest=True,
            right=True,
        )
        transformed_df["credit_duration_interaction"] = (
            transformed_df["credit_amount"] * transformed_df["duration_months"]
        )

        logger.info("Engineered features added to dataframe.")
        return transformed_df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit the transformer and return transformed features in one step.

        Args:
            df: Dataframe used to learn statistics and to generate engineered
                features.

        Returns:
            pd.DataFrame: Engineered dataframe.
        """
        return self.fit(df).transform(df)

    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        """Ensure the dataframe contains the base columns required for engineering."""
        missing_columns = _REQUIRED_COLUMNS.difference(df.columns)
        if missing_columns:
            raise KeyError(
                f"Missing required columns for feature engineering: {sorted(missing_columns)}"
            )


def create_engineered_features(
    df: pd.DataFrame, fitted_engineer: FeatureEngineer | None = None
) -> pd.DataFrame:
    """Create engineered features using a fitted transformer when available.

    Args:
        df: Dataframe to transform.
        fitted_engineer: Optional pre-fitted transformer. When omitted, a new
            transformer is created and fitted on the supplied data.

    Returns:
        pd.DataFrame: Dataframe with engineered features appended.
    """
    engineer = fitted_engineer or FeatureEngineer()
    if fitted_engineer is None:
        engineer.fit(df)
    return engineer.transform(df)


def validate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the dataframe before and after feature engineering.

    The function ensures the base columns required for engineering are present
    and that the engineered output columns are available once transformation has
    been applied.

    Args:
        df: Dataframe to validate.

    Returns:
        pd.DataFrame: The validated dataframe.
    """
    missing_required_columns = _REQUIRED_COLUMNS.difference(df.columns)
    if missing_required_columns:
        raise KeyError(
            f"Missing required columns for feature engineering: {sorted(missing_required_columns)}"
        )

    engineered_columns_present = all(column in df.columns for column in _ENGINEERED_FEATURE_NAMES)
    if not engineered_columns_present:
        raise ValueError(
            "Engineered features are not present. Run feature engineering before validation."
        )

    logger.info("Feature validation passed for dataframe with engineered columns.")
    return df.copy(deep=True)


def get_engineered_feature_names() -> list[str]:
    """Return the full list of engineered feature names.

    Returns:
        list[str]: Names of the engineered features added by this module.
    """
    return list(_ENGINEERED_FEATURE_NAMES)
