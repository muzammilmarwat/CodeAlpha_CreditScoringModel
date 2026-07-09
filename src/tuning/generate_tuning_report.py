"""Generate tuning comparison reports from saved metrics.

This module intentionally does not run hyperparameter search. It reads the
existing baseline and tuned model reports, creates a baseline-vs-tuned
comparison table, and saves the comparison plots used by Notebook 04.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_REPORT_PATH = PROJECT_ROOT / "reports" / "model_comparison" / "baseline_model_comparison.csv"
TUNED_METRICS_PATH = PROJECT_ROOT / "reports" / "tuning" / "tuned_model_test_metrics.csv"
TUNING_SUMMARY_PATH = PROJECT_ROOT / "reports" / "tuning" / "tuning_summary.csv"
TUNING_REPORT_DIR = PROJECT_ROOT / "reports" / "tuning"
TUNING_IMAGE_DIR = PROJECT_ROOT / "images" / "tuning"
COMPARISON_REPORT_PATH = TUNING_REPORT_DIR / "baseline_vs_tuned_comparison.csv"

TUNED_MODEL_NAMES = ("random_forest", "svm")
METRIC_COLUMNS = [
    "accuracy",
    "precision_bad",
    "recall_bad",
    "f1_bad",
    "precision_good",
    "recall_good",
    "f1_good",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "roc_auc",
]
PLOT_METRICS = {
    "macro_f1": "Macro F1",
    "recall_bad": "Recall for Bad Credit",
    "roc_auc": "ROC-AUC",
}


def _read_required_csv(path: Path) -> pd.DataFrame:
    """Read a required CSV file and raise a clear error if it is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Required report not found: {path}")
    return pd.read_csv(path)


def load_saved_reports() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load baseline, tuned test metrics, and tuning summary reports."""
    baseline_df = _read_required_csv(BASELINE_REPORT_PATH)
    tuned_metrics_df = _read_required_csv(TUNED_METRICS_PATH)
    tuning_summary_df = _read_required_csv(TUNING_SUMMARY_PATH)
    return baseline_df, tuned_metrics_df, tuning_summary_df


def build_baseline_vs_tuned_comparison(
    baseline_df: pd.DataFrame,
    tuned_metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a comparison table from saved baseline and tuned metrics."""
    baseline_subset = baseline_df[baseline_df["model_name"].isin(TUNED_MODEL_NAMES)].copy()
    baseline_subset["model_variant"] = baseline_subset["model_name"] + "_baseline"
    baseline_subset["source"] = "baseline"

    tuned_subset = tuned_metrics_df[tuned_metrics_df["model_name"].isin(TUNED_MODEL_NAMES)].copy()
    tuned_subset["model_variant"] = tuned_subset["model_name"] + "_tuned"
    tuned_subset["source"] = "tuned"

    selected_columns = ["model_variant", "model_name", "source", *METRIC_COLUMNS]
    comparison_df = pd.concat(
        [baseline_subset[selected_columns], tuned_subset[selected_columns]],
        ignore_index=True,
    )
    return comparison_df.sort_values("macro_f1", ascending=False).reset_index(drop=True)


def save_comparison_report(comparison_df: pd.DataFrame) -> Path:
    """Save the baseline-vs-tuned comparison CSV."""
    TUNING_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(COMPARISON_REPORT_PATH, index=False)
    return COMPARISON_REPORT_PATH


def save_metric_plots(comparison_df: pd.DataFrame) -> dict[str, Path]:
    """Save comparison bar charts for the key tuning metrics."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    TUNING_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    saved_paths: dict[str, Path] = {}
    for metric_name, metric_label in PLOT_METRICS.items():
        plt.figure(figsize=(9, 4.5))
        axis = sns.barplot(
            data=comparison_df,
            x="model_variant",
            y=metric_name,
            hue="source",
            dodge=False,
            palette={"baseline": "#4C78A8", "tuned": "#F58518"},
        )
        axis.set_title(f"Baseline vs Tuned {metric_label}")
        axis.set_xlabel("Model")
        axis.set_ylabel(metric_label)
        axis.set_ylim(0, max(1.0, float(comparison_df[metric_name].max()) * 1.1))
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()

        output_path = TUNING_IMAGE_DIR / f"baseline_vs_tuned_{metric_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        saved_paths[metric_name] = output_path

    return saved_paths


def generate_tuning_report() -> tuple[pd.DataFrame, dict[str, Path]]:
    """Generate all saved tuning comparison outputs without rerunning tuning."""
    baseline_df, tuned_metrics_df, _ = load_saved_reports()
    comparison_df = build_baseline_vs_tuned_comparison(baseline_df, tuned_metrics_df)
    save_comparison_report(comparison_df)
    plot_paths = save_metric_plots(comparison_df)
    return comparison_df, plot_paths


def main() -> None:
    """Generate the tuning comparison report and print the best model."""
    comparison_df, plot_paths = generate_tuning_report()
    best_row = comparison_df.loc[comparison_df["macro_f1"].idxmax()]

    print(f"Saved comparison report: {COMPARISON_REPORT_PATH}")
    for metric_name, path in plot_paths.items():
        print(f"Saved {metric_name} plot: {path}")
    print(
        "Best model by macro F1: "
        f"{best_row['model_variant']} ({best_row['macro_f1']:.4f})"
    )


if __name__ == "__main__":
    main()
