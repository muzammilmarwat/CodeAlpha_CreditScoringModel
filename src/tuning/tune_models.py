"""Tune Random Forest and SVM credit scoring pipelines with cross-validation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from ..evaluation.evaluation_metrics import evaluate_classification_model
from ..models.model_config import MODEL_RANDOM_FOREST, MODEL_SVM
from ..models.model_registry import get_baseline_models
from ..preprocessing.preprocessing_pipeline import build_scaled_preprocessor, build_tree_preprocessor
from .hyperparameter_grids import get_hyperparameter_grids

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "model_ready"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models" / "tuned"
REPORT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "tuning"


def load_model_ready_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load model-ready train and test datasets from disk.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Train/test
            feature matrices and target vectors.
    """
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").iloc[:, 0]
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").iloc[:, 0]

    logger.info("Loaded model-ready data from %s", DATA_DIR)
    return X_train, X_test, y_train, y_test


def build_tuning_candidates() -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    """Build pipeline candidates for the models that require tuning.

    Returns:
        dict[str, tuple[Pipeline, dict[str, list[Any]]]]: Pipeline and parameter
            grid pairs for each tuning candidate.
    """
    baseline_models = get_baseline_models()
    hyperparameter_grids = get_hyperparameter_grids()

    random_forest_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_tree_preprocessor()),
            ("classifier", baseline_models[MODEL_RANDOM_FOREST]),
        ]
    )
    svm_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_scaled_preprocessor()),
            ("classifier", baseline_models[MODEL_SVM]),
        ]
    )

    return {
        MODEL_RANDOM_FOREST: (random_forest_pipeline, hyperparameter_grids[MODEL_RANDOM_FOREST]),
        MODEL_SVM: (svm_pipeline, hyperparameter_grids[MODEL_SVM]),
    }


def tune_single_model(
    model_name: str,
    pipeline: Pipeline,
    param_grid: dict[str, list[Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> GridSearchCV:
    """Tune a single pipeline using cross-validated grid search.

    Args:
        model_name: Name of the model being tuned.
        pipeline: Full sklearn pipeline with preprocessing and classifier.
        param_grid: Parameter grid for the classifier step.
        X_train: Training features.
        y_train: Training target.

    Returns:
        GridSearchCV: Fitted grid search object.
    """
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    search.model_name = model_name
    search.fit(X_train, y_train)
    logger.info("Completed tuning for %s with best macro F1 %.4f", model_name, search.best_score_)
    return search


def tune_all_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, GridSearchCV]:
    """Tune the Random Forest and SVM pipelines on the supplied training data.

    Args:
        X_train: Training features.
        y_train: Training target.

    Returns:
        dict[str, GridSearchCV]: Fitted search objects keyed by model name.
    """
    search_results: dict[str, GridSearchCV] = {}
    for model_name, (pipeline, param_grid) in build_tuning_candidates().items():
        search_results[model_name] = tune_single_model(model_name, pipeline, param_grid, X_train, y_train)

    return search_results


def evaluate_tuned_model(search: GridSearchCV, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Evaluate the best tuned estimator on the hold-out test set.

    Args:
        search: Fitted grid search object containing the best estimator.
        X_test: Test features.
        y_test: Test target.

    Returns:
        dict[str, float]: Dictionary of classification metrics.
    """
    metrics = evaluate_classification_model(search.best_estimator_, X_test, y_test)
    metrics["model_name"] = getattr(search, "model_name", "unknown")
    metrics["best_params"] = json.dumps(search.best_params_, sort_keys=True)
    return metrics


def save_tuned_models(search_results: dict[str, GridSearchCV]) -> dict[str, Path]:
    """Serialize the best tuned estimators to disk.

    Args:
        search_results: Dictionary of fitted grid search objects.

    Returns:
        dict[str, Path]: Saved file paths keyed by model name.
    """
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths: dict[str, Path] = {}

    for model_name, search in search_results.items():
        output_path = MODEL_OUTPUT_DIR / f"{model_name}_tuned.joblib"
        joblib.dump(search.best_estimator_, output_path)
        saved_paths[model_name] = output_path

    logger.info("Saved tuned model artifacts to %s", MODEL_OUTPUT_DIR)
    return saved_paths


def save_tuning_results(
    search_results: dict[str, GridSearchCV],
    evaluation_results: dict[str, dict[str, Any]],
) -> dict[str, Path]:
    """Save tuning summaries and evaluation metrics for the tuned models.

    Args:
        search_results: Dictionary of fitted grid search objects.
        evaluation_results: Metrics generated from test-set evaluation.

    Returns:
        dict[str, Path]: Saved output paths.
    """
    REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    saved_paths: dict[str, Path] = {}

    for model_name, search in search_results.items():
        cv_results_df = pd.DataFrame(search.cv_results_)
        cv_output_path = REPORT_OUTPUT_DIR / f"{model_name}_cv_results.csv"
        cv_results_df.to_csv(cv_output_path, index=False)
        saved_paths[f"{model_name}_cv_results"] = cv_output_path

        metrics = evaluation_results[model_name]
        summary_rows.append(
            {
                "model_name": model_name,
                "best_params": json.dumps(search.best_params_, sort_keys=True),
                "best_cv_macro_f1": search.best_score_,
                "test_accuracy": metrics.get("accuracy", np.nan),
                "precision_bad": metrics.get("precision_bad", np.nan),
                "recall_bad": metrics.get("recall_bad", np.nan),
                "f1_bad": metrics.get("f1_bad", np.nan),
                "macro_f1": metrics.get("macro_f1", np.nan),
                "weighted_f1": metrics.get("weighted_f1", np.nan),
                "roc_auc": metrics.get("roc_auc", np.nan),
            }
        )

    tuning_summary_df = pd.DataFrame(summary_rows)
    summary_path = REPORT_OUTPUT_DIR / "tuning_summary.csv"
    tuning_summary_df.to_csv(summary_path, index=False)
    saved_paths["tuning_summary"] = summary_path

    metrics_df = pd.DataFrame([evaluation_results[model_name] for model_name in search_results])
    metrics_path = REPORT_OUTPUT_DIR / "tuned_model_test_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    saved_paths["tuned_model_test_metrics"] = metrics_path

    logger.info("Saved tuning reports to %s", REPORT_OUTPUT_DIR)
    return saved_paths


def main() -> tuple[dict[str, GridSearchCV], dict[str, dict[str, Any]]]:
    """Run the complete tuning workflow end to end."""
    X_train, X_test, y_train, y_test = load_model_ready_data()
    search_results = tune_all_models(X_train, y_train)
    evaluation_results = {
        model_name: evaluate_tuned_model(search, X_test, y_test)
        for model_name, search in search_results.items()
    }
    save_tuned_models(search_results)
    save_tuning_results(search_results, evaluation_results)
    return search_results, evaluation_results


if __name__ == "__main__":
    main()
