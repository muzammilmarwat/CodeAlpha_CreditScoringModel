"""Configuration constants for baseline model training and evaluation."""

from __future__ import annotations

RANDOM_STATE = 42
POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0

MODEL_LOGISTIC_REGRESSION = "logistic_regression"
MODEL_DECISION_TREE = "decision_tree"
MODEL_RANDOM_FOREST = "random_forest"
MODEL_SVM = "svm"
MODEL_KNN = "knn"

MODEL_NAMES = [
    MODEL_LOGISTIC_REGRESSION,
    MODEL_DECISION_TREE,
    MODEL_RANDOM_FOREST,
    MODEL_SVM,
    MODEL_KNN,
]

CLASS_LABELS = {
    "bad": NEGATIVE_LABEL,
    "good": POSITIVE_LABEL,
}
