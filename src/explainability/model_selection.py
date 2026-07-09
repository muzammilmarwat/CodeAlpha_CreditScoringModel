"""Final model selection report generation for the credit scoring project."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_COMPARISON_PATH = PROJECT_ROOT / "reports" / "model_comparison" / "baseline_model_comparison.csv"
BASELINE_VS_TUNED_PATH = PROJECT_ROOT / "reports" / "tuning" / "baseline_vs_tuned_comparison.csv"
TUNED_METRICS_PATH = PROJECT_ROOT / "reports" / "tuning" / "tuned_model_test_metrics.csv"
FINAL_SELECTION_DIR = PROJECT_ROOT / "reports" / "final_model_selection"
FINAL_REPORT_PATH = FINAL_SELECTION_DIR / "final_model_selection_report.md"
FINAL_SUMMARY_PATH = FINAL_SELECTION_DIR / "final_model_selection_summary.csv"

PRIMARY_FINAL_MODEL = "random_forest_baseline"
RISK_FOCUSED_ALTERNATIVE = "svm_baseline"


def _read_required_csv(path: Path) -> pd.DataFrame:
    """Read a required CSV file and raise a clear error if it is missing.

    Args:
        path: CSV file path.

    Returns:
        pd.DataFrame: Loaded dataframe.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Required report not found: {path}")
    return pd.read_csv(path)


def load_model_selection_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load saved baseline, baseline-vs-tuned, and tuned metrics reports."""
    baseline_df = _read_required_csv(BASELINE_COMPARISON_PATH)
    comparison_df = _read_required_csv(BASELINE_VS_TUNED_PATH)
    tuned_metrics_df = _read_required_csv(TUNED_METRICS_PATH)
    return baseline_df, comparison_df, tuned_metrics_df


def build_final_selection_summary(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Build the concise final model selection summary table.

    Args:
        comparison_df: Baseline-vs-tuned comparison dataframe.

    Returns:
        pd.DataFrame: Summary table with decision roles.
    """
    best_macro_f1 = comparison_df.loc[comparison_df["macro_f1"].idxmax()]
    best_recall_bad = comparison_df.loc[comparison_df["recall_bad"].idxmax()]

    required_variants = comparison_df.set_index("model_variant")
    primary = required_variants.loc[PRIMARY_FINAL_MODEL]
    alternative = required_variants.loc[RISK_FOCUSED_ALTERNATIVE]

    rows = [
        {
            "decision_role": "best_overall_by_macro_f1",
            "model_variant": best_macro_f1["model_variant"],
            "accuracy": best_macro_f1["accuracy"],
            "macro_f1": best_macro_f1["macro_f1"],
            "recall_bad": best_macro_f1["recall_bad"],
            "roc_auc": best_macro_f1["roc_auc"],
        },
        {
            "decision_role": "best_risk_detection_by_recall_bad",
            "model_variant": best_recall_bad["model_variant"],
            "accuracy": best_recall_bad["accuracy"],
            "macro_f1": best_recall_bad["macro_f1"],
            "recall_bad": best_recall_bad["recall_bad"],
            "roc_auc": best_recall_bad["roc_auc"],
        },
        {
            "decision_role": "primary_final_model",
            "model_variant": PRIMARY_FINAL_MODEL,
            "accuracy": primary["accuracy"],
            "macro_f1": primary["macro_f1"],
            "recall_bad": primary["recall_bad"],
            "roc_auc": primary["roc_auc"],
        },
        {
            "decision_role": "risk_focused_alternative",
            "model_variant": RISK_FOCUSED_ALTERNATIVE,
            "accuracy": alternative["accuracy"],
            "macro_f1": alternative["macro_f1"],
            "recall_bad": alternative["recall_bad"],
            "roc_auc": alternative["roc_auc"],
        },
    ]
    return pd.DataFrame(rows)


def _metric_table_markdown(df: pd.DataFrame) -> str:
    """Convert key model metrics to a compact markdown table."""
    columns = ["model_variant", "source", "accuracy", "macro_f1", "recall_bad", "f1_bad", "roc_auc"]
    table_df = df[columns].copy()
    for metric in ["accuracy", "macro_f1", "recall_bad", "f1_bad", "roc_auc"]:
        table_df[metric] = table_df[metric].map(lambda value: f"{value:.4f}")

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in table_df.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def build_final_model_selection_report(
    baseline_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    tuned_metrics_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> str:
    """Build the final model selection report markdown.

    Args:
        baseline_df: Full baseline model comparison dataframe.
        comparison_df: Baseline-vs-tuned comparison dataframe.
        tuned_metrics_df: Tuned model test metrics dataframe.
        summary_df: Final decision summary dataframe.

    Returns:
        str: Markdown report content.
    """
    best_overall = comparison_df.loc[comparison_df["macro_f1"].idxmax()]
    best_risk = comparison_df.loc[comparison_df["recall_bad"].idxmax()]
    primary = comparison_df.set_index("model_variant").loc[PRIMARY_FINAL_MODEL]
    alternative = comparison_df.set_index("model_variant").loc[RISK_FOCUSED_ALTERNATIVE]

    tuned_best_macro = tuned_metrics_df.loc[tuned_metrics_df["macro_f1"].idxmax()]
    baseline_best_macro = baseline_df.loc[baseline_df["macro_f1"].idxmax()]

    return f"""# Final Model Selection Report

## Executive Summary

The recommended primary model for deployment readiness is `{PRIMARY_FINAL_MODEL}`. It has the strongest overall balance among the reviewed candidates, with macro F1 of {primary["macro_f1"]:.4f}, accuracy of {primary["accuracy"]:.4f}, bad-class recall of {primary["recall_bad"]:.4f}, and ROC-AUC of {primary["roc_auc"]:.4f}.

The recommended risk-focused alternative is `{RISK_FOCUSED_ALTERNATIVE}`. It has the highest bad-class recall among the final candidates at {alternative["recall_bad"]:.4f}, which makes it useful when the business objective prioritizes catching more risky borrowers.

## Models Reviewed

This phase reviewed the saved results for:

- `random_forest_baseline`
- `svm_baseline`
- `random_forest_tuned`
- `svm_tuned`

No model was retrained, and hyperparameter search was not rerun.

## Baseline vs Tuned Results

{_metric_table_markdown(comparison_df)}

## Best Overall Model by Macro F1

The best overall candidate by macro F1 is `{best_overall["model_variant"]}` with macro F1 of {best_overall["macro_f1"]:.4f}. Macro F1 is appropriate here because the dataset is class-imbalanced and both good-credit and bad-credit performance matter.

## Best Risk-Detection Model by Bad-Class Recall

The best risk-detection candidate by bad-class recall is `{best_risk["model_variant"]}` with bad-class recall of {best_risk["recall_bad"]:.4f}. In a credit-risk setting, bad-class recall reflects the model's ability to identify applicants who are more likely to default or represent elevated credit risk.

## Why Tuned Models Were Not Selected

The tuned models did not outperform their baseline counterparts on the saved hold-out test metrics. The best tuned model by macro F1 is `{tuned_best_macro["model_name"]}_tuned` with macro F1 of {tuned_best_macro["macro_f1"]:.4f}, while the best baseline model is `{baseline_best_macro["model_name"]}_baseline` with macro F1 of {baseline_best_macro["macro_f1"]:.4f}.

Because tuning reduced the observed hold-out macro F1 and did not create a clearly superior risk-recall profile, the tuned models are not recommended as the primary final model.

## Final Recommended Model

Primary final model: `{PRIMARY_FINAL_MODEL}`

Rationale:

- It has the highest macro F1 among the reviewed final candidates.
- It also has the highest accuracy among the reviewed final candidates.
- It provides a stronger balanced-performance profile than the tuned Random Forest and tuned SVM.
- It remains interpretable enough for portfolio reporting through Random Forest feature importances.

## Business Tradeoff

The primary Random Forest baseline is the best balanced model. It is a good default choice when the project values both classes and wants stable overall classification performance.

The SVM baseline is the risk-focused alternative. It catches more bad-credit cases, but it gives up some overall accuracy and macro F1. In a stricter lending environment, this may be preferable because missing risky borrowers can be more costly than incorrectly flagging some good borrowers.

## Limitations

- The dataset is relatively small, with 1,000 rows.
- The model uses historical German Credit dataset categories, so modern credit-policy deployment would require domain review.
- ROC-AUC is reported from saved model outputs and should be interpreted alongside class-specific recall and precision.
- Feature importance from Random Forest is model-specific and should not be treated as causal explanation.
- No threshold optimization has been performed yet.
- Fairness, bias, and regulatory validation are not complete.

## Next Deployment Recommendation

Package `{PRIMARY_FINAL_MODEL}` as the default inference model and keep `{RISK_FOCUSED_ALTERNATIVE}` documented as a policy alternative for risk-sensitive review. The next engineering phase should build a single inference workflow that loads the saved feature engineering artifact, applies the selected model, validates inputs, and reports model outputs with clear probability and risk interpretation.
"""


def save_final_selection_outputs(summary_df: pd.DataFrame, report_markdown: str) -> tuple[Path, Path]:
    """Save final model selection summary and markdown report.

    Args:
        summary_df: Final model decision summary dataframe.
        report_markdown: Markdown report content.

    Returns:
        tuple[Path, Path]: Summary CSV path and markdown report path.
    """
    FINAL_SELECTION_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(FINAL_SUMMARY_PATH, index=False)
    FINAL_REPORT_PATH.write_text(report_markdown, encoding="utf-8")
    logger.info("Saved final model selection summary to %s", FINAL_SUMMARY_PATH)
    logger.info("Saved final model selection report to %s", FINAL_REPORT_PATH)
    return FINAL_SUMMARY_PATH, FINAL_REPORT_PATH


def generate_final_model_selection_report() -> tuple[pd.DataFrame, Path, Path]:
    """Generate final model selection outputs from saved report CSVs.

    Returns:
        tuple[pd.DataFrame, Path, Path]: Summary dataframe, summary CSV path,
            and markdown report path.
    """
    baseline_df, comparison_df, tuned_metrics_df = load_model_selection_inputs()
    summary_df = build_final_selection_summary(comparison_df)
    report_markdown = build_final_model_selection_report(
        baseline_df=baseline_df,
        comparison_df=comparison_df,
        tuned_metrics_df=tuned_metrics_df,
        summary_df=summary_df,
    )
    summary_path, report_path = save_final_selection_outputs(summary_df, report_markdown)
    return summary_df, summary_path, report_path


def main() -> None:
    """Generate final model selection CSV and markdown report."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    summary_df, summary_path, report_path = generate_final_model_selection_report()

    print(f"Saved final model selection summary: {summary_path}")
    print(f"Saved final model selection report: {report_path}")
    print("Final model selection:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
