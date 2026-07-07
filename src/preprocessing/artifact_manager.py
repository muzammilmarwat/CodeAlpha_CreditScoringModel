"""Utilities for saving preprocessing artifacts and prepared datasets.

This module centralizes the file-system operations required to persist prepared
training data and reusable preprocessing objects for later phases.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "model_ready"
ARTIFACT_OUTPUT_DIR = PROJECT_ROOT / "models" / "preprocessing"


def ensure_directory(path: str | Path) -> Path:
    """Create a directory and its parents if they do not already exist.

    Args:
        path: Directory path to create.

    Returns:
        Path: The created directory path.
    """
    directory_path = Path(path)
    directory_path.mkdir(parents=True, exist_ok=True)
    return directory_path


def save_dataset(df: pd.DataFrame, path: str | Path) -> Path:
    """Persist a dataframe as a CSV file.

    Args:
        df: Dataframe to save.
        path: Output path for the CSV file.

    Returns:
        Path: The saved file path.
    """
    output_path = Path(path)
    ensure_directory(output_path.parent)
    df.to_csv(output_path, index=False)
    logger.info("Saved dataset to %s", output_path)
    return output_path


def save_series(series: pd.Series, path: str | Path) -> Path:
    """Persist a pandas Series as a CSV file.

    Args:
        series: Series to save.
        path: Output path for the CSV file.

    Returns:
        Path: The saved file path.
    """
    output_path = Path(path)
    ensure_directory(output_path.parent)
    series.to_frame().to_csv(output_path, index=False, header=True)
    logger.info("Saved series to %s", output_path)
    return output_path


def save_artifact(obj: Any, path: str | Path) -> Path:
    """Persist an object to disk using joblib.

    Args:
        obj: Object to save.
        path: Output path for the joblib artifact.

    Returns:
        Path: The saved file path.
    """
    output_path = Path(path)
    ensure_directory(output_path.parent)
    joblib.dump(obj, output_path)
    logger.info("Saved artifact to %s", output_path)
    return output_path


def load_artifact(path: str | Path) -> Any:
    """Load an artifact from disk.

    Args:
        path: Path to the artifact.

    Returns:
        Any: The loaded object.
    """
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")
    return joblib.load(artifact_path)


def save_prepared_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_engineer: Any,
    output_dir: str | Path | None = None,
    artifact_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Save prepared train/test data and the fitted feature engineering artifact.

    Args:
        X_train: Engineered training features.
        X_test: Engineered test features.
        y_train: Encoded training targets.
        y_test: Encoded test targets.
        feature_engineer: Fitted feature engineering transformer.
        output_dir: Directory for processed datasets.
        artifact_dir: Directory for preprocessing artifacts.

    Returns:
        dict[str, Path]: Paths of created files.
    """
    prepared_dir = Path(output_dir) if output_dir is not None else PROCESSED_OUTPUT_DIR
    artifact_dir_path = Path(artifact_dir) if artifact_dir is not None else ARTIFACT_OUTPUT_DIR

    ensure_directory(prepared_dir)
    ensure_directory(artifact_dir_path)

    x_train_path = save_dataset(X_train, prepared_dir / "X_train.csv")
    x_test_path = save_dataset(X_test, prepared_dir / "X_test.csv")
    y_train_path = save_series(y_train, prepared_dir / "y_train.csv")
    y_test_path = save_series(y_test, prepared_dir / "y_test.csv")
    feature_engineer_path = save_artifact(feature_engineer, artifact_dir_path / "feature_engineer.joblib")

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "number_of_features": int(X_train.shape[1]),
        "target_mapping": {"good": 1, "bad": 0},
        "engineered_features": list(X_train.columns),
        "random_state": 42,
        "test_size": 0.2,
    }

    metadata_path = artifact_dir_path / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file_handle:
        json.dump(metadata, file_handle, indent=2)

    logger.info("Saved preprocessing metadata to %s", metadata_path)
    return {
        "X_train": x_train_path,
        "X_test": x_test_path,
        "y_train": y_train_path,
        "y_test": y_test_path,
        "feature_engineer": feature_engineer_path,
        "metadata": metadata_path,
    }
