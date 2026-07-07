"""Central feature definitions and validation helpers for the German Credit pipeline.

This module defines the canonical feature groups used by the project. Keeping
these definitions in one place prevents the training notebooks, model scripts,
and future deployment services from drifting out of sync.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

logger = logging.getLogger(__name__)

TARGET_COLUMN = "credit_risk"

# Numerical features represent quantitative measurements with meaningful
# magnitude. They are suitable for scaling because models such as logistic
# regression, SVM, and KNN are sensitive to feature magnitude.
NUMERICAL_FEATURES = (
    "duration_months",  # Number of months for the credit term; its scale matters.
    "credit_amount",  # Loan amount; larger values should be treated distinctly.
    "installment_rate",  # Monthly installment as a percentage of disposable income.
    "age",  # Customer age; numeric and continuous enough for scaling.
    "present_residence_years",  # Years at current residence; a count-like measure.
    "existing_credits",  # Number of existing credit lines; a count variable.
    "people_liable",  # Number of people financially liable; another count measure.
)

# Ordinal features are categorical variables with a meaningful order. The order
# is explicitly defined in the encoder module so that it never depends on the
# values encountered during training or inference.
ORDINAL_FEATURES = (
    "checking_account_status",  # Ordered from less favorable to more favorable status.
    "employment_duration",  # Ordered from shorter to longer tenure.
    "savings_account",  # Ordered from lower to higher savings capacity.
)

# Nominal features are categorical and do not carry a natural ranking. They are
# encoded as one-hot features so the model can learn category-specific effects
# without implying an artificial order.
NOMINAL_FEATURES = (
    "credit_history",  # Past credit behavior category with no natural rank.
    "purpose",  # Borrower purpose of credit; categories are distinct labels.
    "personal_status_sex",  # Social/demographic category without ranking.
    "other_debtors",  # Category of co-applicant or guarantor arrangement.
    "property",  # Asset ownership category without ordinal meaning.
    "other_installment_plans",  # Payment plan category without rank.
    "housing",  # Housing arrangement category with no intrinsic order.
    "job",  # Employment role category without natural ordering.
    "telephone",  # Binary contact availability feature from a coding perspective.
    "foreign_worker",  # Binary worker status with no ranking semantics.
)


def get_feature_summary() -> dict[str, Any]:
    """Return a summary of the canonical feature configuration.

    Returns:
        dict[str, Any]: A dictionary describing the target column and the
            predictor groups used by the preprocessing architecture.
    """
    return {
        "target_column": TARGET_COLUMN,
        "numerical_features": list(NUMERICAL_FEATURES),
        "ordinal_features": list(ORDINAL_FEATURES),
        "nominal_features": list(NOMINAL_FEATURES),
        "total_predictors": len(NUMERICAL_FEATURES) + len(ORDINAL_FEATURES) + len(NOMINAL_FEATURES),
    }


def validate_feature_lists(columns: Iterable[str]) -> None:
    """Validate that the configured feature groups cover the dataset schema exactly.

    The validation checks that:
    - every dataset column appears exactly once in the combined predictor set,
    - there are no duplicate definitions across feature groups,
    - the target column is excluded from predictor definitions.

    Args:
        columns: The dataset columns from the training data.

    Raises:
        ValueError: If the feature lists are inconsistent with the dataset schema.
    """
    column_list = list(columns)
    column_set = set(column_list)

    if TARGET_COLUMN not in column_set:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing from the dataset schema.")

    predictor_names = list(NUMERICAL_FEATURES) + list(ORDINAL_FEATURES) + list(NOMINAL_FEATURES)
    predictor_set = set(predictor_names)

    if TARGET_COLUMN in predictor_set:
        raise ValueError("The target column must not appear in predictor feature lists.")

    if len(predictor_names) != len(set(predictor_names)):
        raise ValueError("Feature lists contain duplicates. Each predictor should appear once.")

    expected_predictors = column_set - {TARGET_COLUMN}
    if predictor_set != expected_predictors:
        raise ValueError(
            "Feature groups do not exactly match the dataset schema. "
            f"Expected predictors: {sorted(expected_predictors)}, "
            f"Configured predictors: {sorted(predictor_set)}"
        )

    logger.info("Feature configuration validated successfully for %d columns.", len(column_list))
