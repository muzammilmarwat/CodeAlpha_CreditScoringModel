"""Hyperparameter grids for tuning the selected credit scoring models."""

from __future__ import annotations

from typing import Any


def get_random_forest_param_grid() -> dict[str, list[Any]]:
    """Return the parameter grid for the Random Forest pipeline."""
    return {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [None, 5, 10, 20],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2"],
    }


def get_svm_param_grid() -> dict[str, list[Any]]:
    """Return the parameter grid for the SVM pipeline."""
    return {
        "classifier__C": [0.1, 1, 10],
        "classifier__kernel": ["linear", "rbf"],
        "classifier__gamma": ["scale", "auto"],
    }


def get_hyperparameter_grids() -> dict[str, dict[str, list[Any]]]:
    """Return all tuning grids keyed by model name."""
    return {
        "random_forest": get_random_forest_param_grid(),
        "svm": get_svm_param_grid(),
    }
