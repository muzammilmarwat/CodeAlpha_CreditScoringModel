"""Application configuration for backend credit-risk inference."""

from __future__ import annotations

from pathlib import Path

APP_NAME = "CodeAlpha Credit Scoring Model"
APP_VERSION = "0.1.0"

PRIMARY_MODEL_NAME = "random_forest_baseline"
ALTERNATIVE_MODEL_NAME = "svm_baseline"
DEFAULT_MODEL_CHOICE = PRIMARY_MODEL_NAME

TARGET_MAPPING = {"good": 1, "bad": 0}
RISK_LABELS = {1: "good", 0: "bad"}

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_ARTIFACT_PATHS = {
    PRIMARY_MODEL_NAME: PROJECT_ROOT / "models" / "baseline" / "random_forest_baseline.joblib",
    ALTERNATIVE_MODEL_NAME: PROJECT_ROOT / "models" / "baseline" / "svm_baseline.joblib",
}

FEATURE_ENGINEER_ARTIFACT_PATH = PROJECT_ROOT / "models" / "preprocessing" / "feature_engineer.joblib"
FEATURE_IMPORTANCE_PATH = PROJECT_ROOT / "reports" / "explainability" / "random_forest_feature_importance.csv"
MODEL_CARD_PATH = PROJECT_ROOT / "reports" / "model_card.md"
