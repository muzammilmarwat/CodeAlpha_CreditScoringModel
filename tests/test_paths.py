"""Tests for deployment artifact path helpers."""

from __future__ import annotations

from app import config
from app.utils.paths import (
    get_feature_engineer_path,
    get_feature_importance_path,
    get_model_card_path,
    get_model_path,
    validate_required_artifacts,
)


def test_required_artifact_paths_exist() -> None:
    """All required deployment artifacts should exist in the repository."""
    expected_paths = [
        get_feature_engineer_path(),
        get_feature_importance_path(),
        get_model_card_path(),
        get_model_path(config.PRIMARY_MODEL_NAME),
        get_model_path(config.ALTERNATIVE_MODEL_NAME),
    ]

    for path in expected_paths:
        assert path.exists(), f"Missing required artifact: {path}"


def test_validate_required_artifacts_passes() -> None:
    """Path-level artifact validation should pass for the saved deployment bundle."""
    validate_required_artifacts()
