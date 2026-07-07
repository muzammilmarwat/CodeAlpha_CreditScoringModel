"""Registry of baseline sklearn models for the credit scoring project."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .model_config import RANDOM_STATE


def get_baseline_models() -> dict[str, object]:
    """Return baseline sklearn models without fitting them.

    Returns:
        dict[str, object]: A dictionary of unfitted model objects keyed by
            model name.
    """
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "decision_tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "svm": SVC(
            probability=True,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "knn": KNeighborsClassifier(n_neighbors=5),
    }
