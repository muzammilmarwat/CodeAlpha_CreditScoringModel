"""Reusable evaluation metrics for baseline classification models."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from ..models.model_config import NEGATIVE_LABEL, POSITIVE_LABEL

logger = logging.getLogger(__name__)


def evaluate_classification_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Evaluate a trained classification model on the test set.

    Args:
        model: Fitted sklearn-compatible model or pipeline.
        X_test: Test features.
        y_test: True test labels.

    Returns:
        dict[str, float]: Dictionary of evaluation metrics.
    """
    predictions = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        positive_probabilities = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, positive_probabilities)
    else:
        roc_auc = float("nan")

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision_bad": precision_score(y_test, predictions, pos_label=NEGATIVE_LABEL, zero_division=0),
        "recall_bad": recall_score(y_test, predictions, pos_label=NEGATIVE_LABEL, zero_division=0),
        "f1_bad": f1_score(y_test, predictions, pos_label=NEGATIVE_LABEL, zero_division=0),
        "precision_good": precision_score(y_test, predictions, pos_label=POSITIVE_LABEL, zero_division=0),
        "recall_good": recall_score(y_test, predictions, pos_label=POSITIVE_LABEL, zero_division=0),
        "f1_good": f1_score(y_test, predictions, pos_label=POSITIVE_LABEL, zero_division=0),
        "macro_precision": precision_score(y_test, predictions, average="macro", zero_division=0),
        "macro_recall": recall_score(y_test, predictions, average="macro", zero_division=0),
        "macro_f1": f1_score(y_test, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, predictions, average="weighted", zero_division=0),
        "roc_auc": roc_auc,
    }

    logger.info("Evaluated model with accuracy %.4f", metrics["accuracy"])
    return metrics


def get_confusion_matrix(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Return the confusion matrix for a trained model.

    Args:
        model: Fitted sklearn-compatible model or pipeline.
        X_test: Test features.
        y_test: True test labels.

    Returns:
        pd.DataFrame: Confusion matrix as a dataframe.
    """
    predictions = model.predict(X_test)
    matrix = confusion_matrix(y_test, predictions)
    return pd.DataFrame(matrix, index=[0, 1], columns=[0, 1])


def get_classification_report_dict(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """Return a classification report as a dictionary.

    Args:
        model: Fitted sklearn-compatible model or pipeline.
        X_test: Test features.
        y_test: True test labels.

    Returns:
        dict[str, Any]: Classification report dictionary.
    """
    from sklearn.metrics import classification_report

    return classification_report(
        y_test,
        model.predict(X_test),
        output_dict=True,
        zero_division=0,
    )
