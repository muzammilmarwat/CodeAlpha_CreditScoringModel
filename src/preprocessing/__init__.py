"""Reusable preprocessing architecture for the credit scoring project."""

from .encoders import build_ordinal_encoder
from .feature_config import (
    NOMINAL_FEATURES,
    NUMERICAL_FEATURES,
    ORDINAL_FEATURES,
    TARGET_COLUMN,
    get_feature_summary,
    validate_feature_lists,
)
from .feature_engineering import (
    FeatureEngineer,
    create_engineered_features,
    get_engineered_feature_names,
    validate_features,
)
from .preprocessing_pipeline import build_scaled_preprocessor, build_tree_preprocessor
from .scaler import build_robust_scaler, build_standard_scaler

__all__ = [
    "build_ordinal_encoder",
    "build_scaled_preprocessor",
    "build_standard_scaler",
    "build_robust_scaler",
    "build_tree_preprocessor",
    "create_engineered_features",
    "validate_features",
    "TARGET_COLUMN",
    "NUMERICAL_FEATURES",
    "ORDINAL_FEATURES",
    "NOMINAL_FEATURES",
    "get_feature_summary",
    "validate_feature_lists",
]
