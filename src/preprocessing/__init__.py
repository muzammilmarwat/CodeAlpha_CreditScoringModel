"""Reusable preprocessing architecture for the credit scoring project."""

from .encoders import build_ordinal_encoder
from .feature_config import (
    ALL_NOMINAL_FEATURES,
    ALL_NUMERICAL_FEATURES,
    ALL_ORDINAL_FEATURES,
    ALL_PREDICTOR_FEATURES,
    ENGINEERED_CATEGORICAL_FEATURES,
    ENGINEERED_NUMERICAL_FEATURES,
    NOMINAL_FEATURES,
    NUMERICAL_FEATURES,
    ORDINAL_FEATURES,
    TARGET_COLUMN,
    get_feature_summary,
    validate_feature_lists,
)
from .artifact_manager import load_artifact, save_artifact, save_dataset, save_prepared_data, save_series
from .data_preparation import create_train_test_split, encode_target, load_processed_dataset, prepare_model_data
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
    "load_processed_dataset",
    "encode_target",
    "create_train_test_split",
    "prepare_model_data",
    "save_dataset",
    "save_series",
    "save_artifact",
    "load_artifact",
    "save_prepared_data",
    "build_standard_scaler",
    "build_robust_scaler",
    "build_tree_preprocessor",
    "create_engineered_features",
    "validate_features",
    "TARGET_COLUMN",
    "NUMERICAL_FEATURES",
    "ORDINAL_FEATURES",
    "NOMINAL_FEATURES",
    "ENGINEERED_NUMERICAL_FEATURES",
    "ENGINEERED_CATEGORICAL_FEATURES",
    "ALL_NUMERICAL_FEATURES",
    "ALL_ORDINAL_FEATURES",
    "ALL_NOMINAL_FEATURES",
    "ALL_PREDICTOR_FEATURES",
    "get_feature_summary",
    "validate_feature_lists",
]
