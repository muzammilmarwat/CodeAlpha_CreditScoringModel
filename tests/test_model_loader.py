"""Tests for model and preprocessing artifact loading."""

from __future__ import annotations

from app import config
from app.services.model_loader import (
    get_available_models,
    load_feature_engineer,
    load_feature_importance,
    load_model,
)


def test_available_models_include_deployment_choices() -> None:
    """Configured deployment models should be exposed by the loader service."""
    available_models = get_available_models()

    assert config.PRIMARY_MODEL_NAME in available_models
    assert config.ALTERNATIVE_MODEL_NAME in available_models


def test_model_loading_works() -> None:
    """The primary saved model artifact should load and expose sklearn methods."""
    model = load_model(config.PRIMARY_MODEL_NAME)

    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_feature_engineer_loading_works() -> None:
    """The saved fitted feature engineer should load for inference transforms."""
    feature_engineer = load_feature_engineer()

    assert hasattr(feature_engineer, "transform")


def test_feature_importance_loading_works() -> None:
    """The saved feature importance CSV should load with expected columns."""
    feature_importance = load_feature_importance()

    assert {"feature", "importance"}.issubset(feature_importance.columns)
    assert not feature_importance.empty
