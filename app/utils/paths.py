"""Path helpers and artifact validation for the deployment layer."""

from __future__ import annotations

from pathlib import Path

from app import config
from app.utils.exceptions import ArtifactNotFoundError


def get_project_root() -> Path:
    """Return the repository root path.

    Returns:
        Path: Project root directory.
    """
    return Path(__file__).resolve().parents[2]


def get_model_path(model_name: str) -> Path:
    """Return the saved artifact path for a supported model.

    Args:
        model_name: Supported model name.

    Returns:
        Path: Model artifact path.

    Raises:
        ArtifactNotFoundError: If the model name is unsupported.
    """
    try:
        return config.MODEL_ARTIFACT_PATHS[model_name]
    except KeyError as exc:
        supported = ", ".join(sorted(config.MODEL_ARTIFACT_PATHS))
        raise ArtifactNotFoundError(
            f"Unsupported model '{model_name}'. Supported models: {supported}."
        ) from exc


def get_feature_engineer_path() -> Path:
    """Return the saved feature-engineering artifact path."""
    return config.FEATURE_ENGINEER_ARTIFACT_PATH


def get_feature_importance_path() -> Path:
    """Return the saved feature-importance report path."""
    return config.FEATURE_IMPORTANCE_PATH


def get_model_card_path() -> Path:
    """Return the model card path."""
    return config.MODEL_CARD_PATH


def validate_required_artifacts() -> None:
    """Validate that all required deployment artifacts exist.

    Raises:
        ArtifactNotFoundError: If one or more required artifacts are missing.
    """
    required_paths = {
        "feature_engineer": get_feature_engineer_path(),
        "feature_importance": get_feature_importance_path(),
        "model_card": get_model_card_path(),
        **{model_name: path for model_name, path in config.MODEL_ARTIFACT_PATHS.items()},
    }
    missing = [f"{name}: {path}" for name, path in required_paths.items() if not path.exists()]
    if missing:
        raise ArtifactNotFoundError(
            "Required deployment artifact(s) missing: " + "; ".join(missing)
        )
