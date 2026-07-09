"""Feature-importance reporting for the selected Random Forest baseline model.

The saved baseline Random Forest artifact is a sklearn pipeline. This module
loads that artifact, extracts the fitted preprocessing feature names when
available, falls back to stable synthetic names when needed, and saves a
portfolio-ready feature-importance report without retraining the model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RANDOM_FOREST_MODEL_PATH = PROJECT_ROOT / "models" / "baseline" / "random_forest_baseline.joblib"
EXPLAINABILITY_REPORT_DIR = PROJECT_ROOT / "reports" / "explainability"
EXPLAINABILITY_IMAGE_DIR = PROJECT_ROOT / "images" / "explainability"
FEATURE_IMPORTANCE_REPORT_PATH = EXPLAINABILITY_REPORT_DIR / "random_forest_feature_importance.csv"
FEATURE_IMPORTANCE_PLOT_PATH = EXPLAINABILITY_IMAGE_DIR / "random_forest_feature_importance.png"


def load_random_forest_model(model_path: str | Path | None = None) -> Any:
    """Load the saved baseline Random Forest pipeline.

    Args:
        model_path: Optional path to a saved Random Forest pipeline.

    Returns:
        Any: Loaded sklearn pipeline.

    Raises:
        FileNotFoundError: If the model artifact does not exist.
    """
    path = Path(model_path) if model_path is not None else RANDOM_FOREST_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Random Forest model artifact not found: {path}")

    logger.info("Loading baseline Random Forest model from %s", path)
    return joblib.load(path)


def _clean_feature_name(feature_name: str) -> str:
    """Remove sklearn transformer prefixes from a feature name."""
    cleaned_name = feature_name
    for prefix in ("numeric__", "ordinal__", "nominal__", "remainder__"):
        cleaned_name = cleaned_name.replace(prefix, "")
    return cleaned_name


def _get_column_transformer(model: Any) -> Any | None:
    """Return the fitted ColumnTransformer from the saved model pipeline."""
    try:
        preprocessor_step = model.named_steps["preprocessor"]
    except (AttributeError, KeyError):
        logger.warning("Model does not expose a named 'preprocessor' step.")
        return None

    if hasattr(preprocessor_step, "named_steps") and "preprocessor" in preprocessor_step.named_steps:
        return preprocessor_step.named_steps["preprocessor"]

    return preprocessor_step


def extract_feature_names(model: Any, n_features: int) -> list[str]:
    """Extract post-preprocessing feature names from a fitted pipeline.

    Args:
        model: Fitted sklearn pipeline.
        n_features: Expected number of features after preprocessing.

    Returns:
        list[str]: Feature names aligned to model feature importances.

    Notes:
        The preferred path uses sklearn's fitted ``get_feature_names_out``. If
        that is unavailable or returns a mismatched length, the function falls
        back to deterministic names such as ``feature_000``. This keeps the
        report reproducible while clearly signaling limited name traceability.
    """
    column_transformer = _get_column_transformer(model)
    if column_transformer is not None and hasattr(column_transformer, "get_feature_names_out"):
        try:
            feature_names = [_clean_feature_name(str(name)) for name in column_transformer.get_feature_names_out()]
            if len(feature_names) == n_features:
                return feature_names
            logger.warning(
                "Extracted %d feature names but model has %d importances. Falling back to synthetic names.",
                len(feature_names),
                n_features,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback for sklearn version drift.
            logger.warning("Could not extract transformed feature names: %s", exc)

    return [f"feature_{index:03d}" for index in range(n_features)]


def compute_random_forest_feature_importance(model: Any) -> pd.DataFrame:
    """Compute feature importances from a fitted Random Forest pipeline.

    Args:
        model: Fitted sklearn pipeline with a Random Forest classifier step.

    Returns:
        pd.DataFrame: Feature importance table sorted descending.

    Raises:
        AttributeError: If the classifier does not expose feature importances.
    """
    classifier = model.named_steps["classifier"]
    if not hasattr(classifier, "feature_importances_"):
        raise AttributeError("The classifier does not expose feature_importances_.")

    importances = classifier.feature_importances_
    feature_names = extract_feature_names(model, len(importances))

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )
    importance_df["rank"] = importance_df["importance"].rank(method="first", ascending=False).astype(int)
    return importance_df.sort_values("importance", ascending=False).reset_index(drop=True)


def save_feature_importance(importance_df: pd.DataFrame) -> Path:
    """Save feature importances as a CSV report.

    Args:
        importance_df: Feature importance dataframe.

    Returns:
        Path: Saved CSV path.
    """
    EXPLAINABILITY_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(FEATURE_IMPORTANCE_REPORT_PATH, index=False)
    logger.info("Saved feature importance report to %s", FEATURE_IMPORTANCE_REPORT_PATH)
    return FEATURE_IMPORTANCE_REPORT_PATH


def save_top_feature_importance_plot(importance_df: pd.DataFrame, top_n: int = 20) -> Path:
    """Save a horizontal bar plot for the top Random Forest features.

    Args:
        importance_df: Feature importance dataframe.
        top_n: Number of top features to plot.

    Returns:
        Path: Saved PNG path.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    EXPLAINABILITY_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    plot_df = importance_df.head(top_n).sort_values("importance", ascending=True)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 7))
    axis = sns.barplot(data=plot_df, x="importance", y="feature", color="#4C78A8")
    axis.set_title(f"Top {top_n} Random Forest Feature Importances")
    axis.set_xlabel("Importance")
    axis.set_ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Saved feature importance plot to %s", FEATURE_IMPORTANCE_PLOT_PATH)
    return FEATURE_IMPORTANCE_PLOT_PATH


def generate_random_forest_feature_importance() -> tuple[pd.DataFrame, Path, Path]:
    """Generate Random Forest feature-importance reports from saved artifacts.

    Returns:
        tuple[pd.DataFrame, Path, Path]: Importance table, CSV path, and plot path.
    """
    model = load_random_forest_model()
    importance_df = compute_random_forest_feature_importance(model)
    csv_path = save_feature_importance(importance_df)
    plot_path = save_top_feature_importance_plot(importance_df)
    return importance_df, csv_path, plot_path


def main() -> None:
    """Generate feature importance CSV and plot for the baseline Random Forest."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    importance_df, csv_path, plot_path = generate_random_forest_feature_importance()

    print(f"Saved feature importance report: {csv_path}")
    print(f"Saved feature importance plot: {plot_path}")
    print("Top 10 features:")
    print(importance_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
