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

# Engineered numerical features are derived from the original financial fields.
# They are kept numeric because they represent quantities or binary signals that
# can be consumed directly by most models.
ENGINEERED_NUMERICAL_FEATURES = (
    "credit_amount_per_month",  # Credit amount normalized by loan duration.
    "high_credit_amount_flag",  # Binary indicator for large-loan applicants.
    "long_duration_flag",  # Binary indicator for longer-term credits.
    "credit_duration_interaction",  # Interaction term between amount and duration.
)

# The age-group feature is treated as ordinal because the bins are naturally
# ordered from younger to older. This preserves the meaningful progression
# while still keeping the encoding explicit and reusable.
ENGINEERED_CATEGORICAL_FEATURES = (
    "age_group",
)

ALL_NUMERICAL_FEATURES = tuple(NUMERICAL_FEATURES) + tuple(ENGINEERED_NUMERICAL_FEATURES)
ALL_ORDINAL_FEATURES = tuple(ORDINAL_FEATURES) + tuple(ENGINEERED_CATEGORICAL_FEATURES)
ALL_NOMINAL_FEATURES = tuple(NOMINAL_FEATURES)
ALL_PREDICTOR_FEATURES = (
    tuple(ALL_NUMERICAL_FEATURES) + tuple(ALL_ORDINAL_FEATURES) + tuple(ALL_NOMINAL_FEATURES)
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
        "engineered_numerical_features": list(ENGINEERED_NUMERICAL_FEATURES),
        "engineered_categorical_features": list(ENGINEERED_CATEGORICAL_FEATURES),
        "all_numerical_features": list(ALL_NUMERICAL_FEATURES),
        "all_ordinal_features": list(ALL_ORDINAL_FEATURES),
        "all_nominal_features": list(ALL_NOMINAL_FEATURES),
        "all_predictor_features": list(ALL_PREDICTOR_FEATURES),
        "total_predictors": len(ALL_PREDICTOR_FEATURES),
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

    raw_predictor_names = list(NUMERICAL_FEATURES) + list(ORDINAL_FEATURES) + list(NOMINAL_FEATURES)
    raw_predictor_set = set(raw_predictor_names)
    engineered_predictor_names = list(ALL_PREDICTOR_FEATURES)
    engineered_predictor_set = set(engineered_predictor_names)

    if TARGET_COLUMN in raw_predictor_set or TARGET_COLUMN in engineered_predictor_set:
        raise ValueError("The target column must not appear in predictor feature lists.")

    if len(raw_predictor_names) != len(set(raw_predictor_names)):
        raise ValueError("Feature lists contain duplicates. Each predictor should appear once.")

    if len(engineered_predictor_names) != len(set(engineered_predictor_names)):
        raise ValueError("Engineered feature lists contain duplicates. Each predictor should appear once.")

    expected_predictors = column_set - {TARGET_COLUMN}
    if expected_predictors != raw_predictor_set and expected_predictors != engineered_predictor_set:
        raise ValueError(
            "Feature groups do not exactly match the dataset schema. "
            f"Expected predictors: {sorted(expected_predictors)}, "
            f"Configured raw predictors: {sorted(raw_predictor_set)}, "
            f"Configured engineered predictors: {sorted(engineered_predictor_set)}"
        )

    logger.info("Feature configuration validated successfully for %d columns.", len(column_list))
