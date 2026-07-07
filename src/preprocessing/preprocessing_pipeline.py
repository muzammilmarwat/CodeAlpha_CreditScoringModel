"""Reusable preprocessing pipeline builders for the credit scoring project.

This module defines two independent sklearn Pipeline objects that can be reused
by training notebooks, model evaluation scripts, and future deployment services.
The builders intentionally do not fit, transform, split, or train anything.
"""

from __future__ import annotations

import logging

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .encoders import build_ordinal_encoder
from .feature_config import ALL_NOMINAL_FEATURES, ALL_NUMERICAL_FEATURES, ALL_ORDINAL_FEATURES
from .scaler import build_standard_scaler

logger = logging.getLogger(__name__)


def build_scaled_preprocessor() -> Pipeline:
    """Build a preprocessing pipeline for scale-sensitive models.

    The resulting pipeline is intended for algorithms such as logistic
    regression, SVM, and KNN, which benefit from standardized numeric inputs.
    It is designed to be reusable across training, evaluation, hyperparameter
    tuning, and inference without modifying the underlying data.

    Returns:
        Pipeline: A scikit-learn pipeline that is ready to be fitted later.
    """
    logger.info("Building scaled preprocessing pipeline for linear and distance-based models.")

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", build_standard_scaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("ordinal", build_ordinal_encoder()),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, list(ALL_NUMERICAL_FEATURES)),
            ("ordinal", build_ordinal_encoder(), list(ALL_ORDINAL_FEATURES)),
            ("nominal", OneHotEncoder(handle_unknown="ignore"), list(ALL_NOMINAL_FEATURES)),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
        ]
    )


def build_tree_preprocessor() -> Pipeline:
    """Build a preprocessing pipeline for tree-based models.

    Tree-based models do not require standardized numerical features. This
    builder therefore passes numeric features through unchanged while still
    encoding categorical values into model-friendly representations.

    Returns:
        Pipeline: A scikit-learn pipeline that is ready to be fitted later.
    """
    logger.info("Building tree-based preprocessing pipeline without scaling.")

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", list(ALL_NUMERICAL_FEATURES)),
            ("ordinal", build_ordinal_encoder(), list(ALL_ORDINAL_FEATURES)),
            ("nominal", OneHotEncoder(handle_unknown="ignore"), list(ALL_NOMINAL_FEATURES)),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
        ]
    )
