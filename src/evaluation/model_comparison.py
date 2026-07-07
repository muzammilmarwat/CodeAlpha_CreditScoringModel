"""Compare baseline credit scoring models and save comparison artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ..models.model_config import MODEL_NAMES
from ..models.train_baseline_models import load_model_ready_data, save_trained_models, train_all_baseline_models
from .evaluation_metrics import evaluate_classification_model, get_classification_report_dict, get_confusion_matrix

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "model_comparison"
CLASSIFICATION_REPORT_DIR = REPORT_OUTPUT_DIR / "classification_reports"
CONFUSION_MATRIX_DIR = REPORT_OUTPUT_DIR / "confusion_matrices"


def compare_models(trained_models: dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Evaluate each trained model and return a comparison dataframe.

    Args:
        trained_models: Dictionary of fitted model pipelines.
        X_test: Test features.
        y_test: True test labels.

    Returns:
        pd.DataFrame: Comparison results sorted by macro F1 descending.
    """
    rows: list[dict[str, Any]] = []
    for model_name in MODEL_NAMES:
        metrics = evaluate_classification_model(trained_models[model_name], X_test, y_test)
        metrics["model_name"] = model_name
        rows.append(metrics)

    comparison_df = pd.DataFrame(rows)
    comparison_df = comparison_df.sort_values(by="macro_f1", ascending=False).reset_index(drop=True)
    logger.info("Generated model comparison dataframe.")
    return comparison_df


def save_comparison_results(results_df: pd.DataFrame) -> Path:
    """Save the model comparison results dataframe to disk.

    Args:
        results_df: Comparison dataframe.

    Returns:
        Path: Output CSV path.
    """
    REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_OUTPUT_DIR / "baseline_model_comparison.csv"
    results_df.to_csv(output_path, index=False)
    logger.info("Saved model comparison results to %s", output_path)
    return output_path


def save_classification_reports(trained_models: dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Path]:
    """Save classification reports for each trained model as JSON files.

    Args:
        trained_models: Dictionary of fitted pipelines.
        X_test: Test features.
        y_test: True test labels.

    Returns:
        dict[str, Path]: Paths of saved classification reports.
    """
    CLASSIFICATION_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths: dict[str, Path] = {}

    for model_name in MODEL_NAMES:
        report = get_classification_report_dict(trained_models[model_name], X_test, y_test)
        output_path = CLASSIFICATION_REPORT_DIR / f"{model_name}_classification_report.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        saved_paths[model_name] = output_path

    logger.info("Saved classification reports to %s", CLASSIFICATION_REPORT_DIR)
    return saved_paths


def save_confusion_matrices(trained_models: dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Path]:
    """Save confusion matrices for each trained model as CSV files.

    Args:
        trained_models: Dictionary of fitted pipelines.
        X_test: Test features.
        y_test: True test labels.

    Returns:
        dict[str, Path]: Paths of saved confusion matrix files.
    """
    CONFUSION_MATRIX_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths: dict[str, Path] = {}

    for model_name in MODEL_NAMES:
        matrix_df = get_confusion_matrix(trained_models[model_name], X_test, y_test)
        output_path = CONFUSION_MATRIX_DIR / f"{model_name}_confusion_matrix.csv"
        matrix_df.to_csv(output_path, index=True)
        saved_paths[model_name] = output_path

    logger.info("Saved confusion matrices to %s", CONFUSION_MATRIX_DIR)
    return saved_paths


def main() -> pd.DataFrame:
    """Load model-ready data, train baseline models if needed, evaluate them, and save outputs."""
    X_train, X_test, y_train, y_test = load_model_ready_data()

    try:
        from joblib import load

        model_paths = {
            "logistic_regression": PROJECT_ROOT / "models" / "baseline" / "logistic_regression_baseline.joblib",
            "decision_tree": PROJECT_ROOT / "models" / "baseline" / "decision_tree_baseline.joblib",
            "random_forest": PROJECT_ROOT / "models" / "baseline" / "random_forest_baseline.joblib",
            "svm": PROJECT_ROOT / "models" / "baseline" / "svm_baseline.joblib",
            "knn": PROJECT_ROOT / "models" / "baseline" / "knn_baseline.joblib",
        }
        trained_models = {name: load(path) for name, path in model_paths.items() if path.exists()}
        if len(trained_models) != len(MODEL_NAMES):
            trained_models = {**train_all_baseline_models(X_train, y_train), **trained_models}
    except Exception:
        trained_models = train_all_baseline_models(X_train, y_train)

    if not trained_models:
        raise RuntimeError("No trained models were available for comparison.")

    results_df = compare_models(trained_models, X_test, y_test)
    save_comparison_results(results_df)
    save_classification_reports(trained_models, X_test, y_test)
    save_confusion_matrices(trained_models, X_test, y_test)
    return results_df


if __name__ == "__main__":
    main()
