"""Artifact loading services for backend inference."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from app import config
from app.utils.exceptions import ArtifactNotFoundError, ModelLoadingError
from app.utils.paths import (
    get_feature_engineer_path,
    get_feature_importance_path,
    get_model_path,
    validate_required_artifacts,
)


def get_available_models() -> list[str]:
    """Return supported model choices for deployment inference."""
    return sorted(config.MODEL_ARTIFACT_PATHS)


def validate_model_artifacts() -> None:
    """Validate required model and preprocessing artifacts."""
    validate_required_artifacts()


@lru_cache(maxsize=1)
def load_feature_engineer() -> Any:
    """Load the saved fitted feature-engineering artifact.

    Returns:
        Any: Fitted feature engineering object.

    Raises:
        ArtifactNotFoundError: If the artifact is missing.
        ModelLoadingError: If joblib loading fails.
    """
    path = get_feature_engineer_path()
    if not path.exists():
        raise ArtifactNotFoundError(f"Feature engineering artifact not found: {path}")
    try:
        import joblib

        return joblib.load(path)
    except Exception as exc:
        raise ModelLoadingError(f"Could not load feature engineer from {path}: {exc}") from exc


@lru_cache(maxsize=None)
def load_model(model_name: str = config.DEFAULT_MODEL_CHOICE) -> Any:
    """Load a supported saved sklearn model pipeline.

    Args:
        model_name: Supported model name.

    Returns:
        Any: Loaded sklearn pipeline.

    Raises:
        ArtifactNotFoundError: If the model is unsupported or missing.
        ModelLoadingError: If joblib loading fails.
    """
    path = get_model_path(model_name)
    if not path.exists():
        raise ArtifactNotFoundError(f"Model artifact not found for '{model_name}': {path}")
    try:
        import joblib

        return joblib.load(path)
    except Exception as exc:
        raise ModelLoadingError(f"Could not load model '{model_name}' from {path}: {exc}") from exc


@lru_cache(maxsize=1)
def load_feature_importance() -> Any:
    """Load the saved Random Forest feature-importance report."""
    path = get_feature_importance_path()
    if not path.exists():
        raise ArtifactNotFoundError(f"Feature importance report not found: {path}")
    try:
        import pandas as pd

        return pd.read_csv(path)
    except Exception as exc:
        raise ModelLoadingError(f"Could not load feature importance from {path}: {exc}") from exc
