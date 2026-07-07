"""Train baseline credit scoring models with reusable preprocessing pipelines.

The training logic keeps preprocessing inside each sklearn pipeline so that the
preprocessor is fitted only on the training data during model training.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ..preprocessing.preprocessing_pipeline import build_scaled_preprocessor, build_tree_preprocessor
from .model_config import MODEL_NAMES, RANDOM_STATE
from .model_registry import get_baseline_models

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "model_ready"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models" / "baseline"


def load_model_ready_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load model-ready train and test datasets from disk.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Train/test
            features and targets.
    """
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").iloc[:, 0]
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").iloc[:, 0]

    logger.info("Loaded model-ready data from %s", DATA_DIR)
    return X_train, X_test, y_train, y_test


def build_model_pipeline(model_name: str, model: Any) -> Any:
    """Build a sklearn pipeline that includes preprocessing and a classifier.

    Args:
        model_name: Name of the model.
        model: Unfitted sklearn-compatible classifier.

    Returns:
        Any: Fitted pipeline built from a preprocessing step and classifier.
    """
    if model_name in {"logistic_regression", "svm", "knn"}:
        preprocessor = build_scaled_preprocessor()
    elif model_name in {"decision_tree", "random_forest"}:
        preprocessor = build_tree_preprocessor()
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return (preprocessor, model)


def train_single_model(model_name: str, model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """Train a single baseline model pipeline.

    Args:
        model_name: Name of the model.
        model: Unfitted sklearn classifier.
        X_train: Training features.
        y_train: Training targets.

    Returns:
        Any: Fitted sklearn pipeline.
    """
    preprocessor, classifier = build_model_pipeline(model_name, model)
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])
    pipeline.fit(X_train, y_train)
    logger.info("Trained baseline model: %s", model_name)
    return pipeline


def train_all_baseline_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, Any]:
    """Train all baseline models on the supplied training data.

    Args:
        X_train: Training features.
        y_train: Training targets.

    Returns:
        dict[str, Any]: Dictionary of fitted sklearn pipelines keyed by model
            name.
    """
    baseline_models = get_baseline_models()
    trained_models: dict[str, Any] = {}

    for model_name in MODEL_NAMES:
        trained_models[model_name] = train_single_model(model_name, baseline_models[model_name], X_train, y_train)

    return trained_models


def save_trained_models(trained_models: dict[str, Any]) -> dict[str, Path]:
    """Serialize trained model pipelines to disk.

    Args:
        trained_models: Dictionary of fitted sklearn pipelines.

    Returns:
        dict[str, Path]: Paths for the saved model artifacts.
    """
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[str, Path] = {}
    for model_name, model in trained_models.items():
        output_path = MODEL_OUTPUT_DIR / f"{model_name}_baseline.joblib"
        joblib.dump(model, output_path)
        saved_paths[model_name] = output_path

    logger.info("Saved baseline model artifacts to %s", MODEL_OUTPUT_DIR)
    return saved_paths


def main() -> None:
    """Run the baseline training workflow end to end."""
    X_train, _, y_train, _ = load_model_ready_data()
    trained_models = train_all_baseline_models(X_train, y_train)
    save_trained_models(trained_models)


if __name__ == "__main__":
    main()
