"""Reusable Streamlit UI components for the credit-risk app."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from app import config
from app.services.model_loader import load_feature_importance
from app.ui.form_fields import (
    SAMPLE_APPLICANTS,
    apply_applicant_to_session,
    reset_form_state,
)

FEATURE_IMPORTANCE_IMAGE_PATH = (
    config.PROJECT_ROOT / "images" / "explainability" / "random_forest_feature_importance.png"
)
FINAL_MODEL_SELECTION_REPORT_PATH = (
    config.PROJECT_ROOT / "reports" / "final_model_selection" / "final_model_selection_report.md"
)
TOP_GLOBAL_FACTORS = [
    "Checking Account Status",
    "Credit Amount per Month",
    "Credit Duration Interaction",
    "Credit Amount",
    "Age",
    "Savings Account",
]


def render_header() -> None:
    """Render the application header."""
    st.markdown(
        """
        <div class="hero-panel">
            <div class="brand-row">
                <div class="brand-logo">🏦</div>
                <div>
                    <div class="main-title">Credit Risk Assessment System</div>
                    <div class="subtitle">
                        Production-style Machine Learning application for German Credit Risk Classification
                    </div>
                </div>
            </div>
            <div class="info-note">
                Educational & Internship Portfolio Project Only. Not intended for real lending decisions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    """Render sidebar metadata and navigation.

    Returns:
        str: Selected navigation page.
    """
    with st.sidebar:
        st.title("🏦 Credit Risk Assessment")
        st.caption(f"Version {config.APP_VERSION}")
        st.divider()
        st.success(f"Default model: `{config.PRIMARY_MODEL_NAME}`")
        st.info(f"Risk-focused alternative: `{config.ALTERNATIVE_MODEL_NAME}`")
        st.write("Model card: `reports/model_card.md`")
        st.warning("Not approved for automated real-world credit decisions.")
        st.divider()
        st.subheader("Project Status")
        st.markdown(
            """
            <div class="status-list">
                <strong>Release Candidate RC-2</strong><br>
                <span class="status-check">✓</span> Exploratory Data Analysis<br>
                <span class="status-check">✓</span> Feature Engineering<br>
                <span class="status-check">✓</span> Baseline Models<br>
                <span class="status-check">✓</span> Hyperparameter Tuning<br>
                <span class="status-check">✓</span> Explainability<br>
                <span class="status-check">✓</span> Final Model Selection<br>
                <span class="status-check">✓</span> Backend Inference<br>
                <span class="status-check">✓</span> Streamlit Deployment<br>
                <span class="status-check">✓</span> Documentation<br>
                <span class="status-check">✓</span> Testing
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()
        page = st.radio(
            "Navigation",
            ["Prediction", "Model Information", "About Project"],
            index=0,
        )
        st.divider()
        st.write("Example applicants")
        selected_sample_name = st.selectbox("Load example profile", list(SAMPLE_APPLICANTS))
        if st.button("Apply example applicant", use_container_width=True):
            apply_applicant_to_session(SAMPLE_APPLICANTS[selected_sample_name])
            st.rerun()
        if st.button("Reset form", use_container_width=True):
            reset_form_state()
            st.rerun()
    return page


def _risk_css_class(risk_level: str) -> str:
    """Return CSS class name for a risk level."""
    if risk_level == "Low Risk":
        return "risk-low"
    if risk_level == "High Risk":
        return "risk-high"
    return "risk-medium"


def _risk_icon(risk_level: str) -> str:
    """Return an icon for a risk level."""
    if risk_level == "Low Risk":
        return "✓"
    if risk_level == "High Risk":
        return "✕"
    return "⚠"


def _badge_css_class(risk_level: str) -> str:
    """Return CSS class name for a risk badge."""
    if risk_level == "Low Risk":
        return "badge-low"
    if risk_level == "High Risk":
        return "badge-high"
    return "badge-medium"


def _business_recommendation(risk_level: str) -> str:
    """Return a short business recommendation for a risk level."""
    if risk_level == "Low Risk":
        return "Approve application with standard verification."
    if risk_level == "High Risk":
        return "Reject or perform detailed financial review."
    return "Approve with manual review."


def _prediction_confidence(prediction: dict[str, Any]) -> float:
    """Return confidence for the predicted class."""
    if prediction["predicted_class"] == "good":
        return float(prediction["probability_good"])
    return float(prediction["probability_bad"])


def build_prediction_summary(prediction: dict[str, Any], file_format: str = "md") -> str:
    """Build a downloadable prediction summary.

    Args:
        prediction: Prediction dictionary returned by the backend service.
        file_format: Either ``md`` or ``txt``.

    Returns:
        str: Prediction summary content.
    """
    recommendation = _business_recommendation(str(prediction["risk_level"]))
    confidence = _prediction_confidence(prediction)
    if file_format == "txt":
        return (
            "Credit Risk Assessment\n"
            f"Model: {prediction['model_name']}\n"
            f"Prediction: {prediction['predicted_class'].title()} Credit\n"
            f"Confidence: {confidence:.2%}\n"
            f"Probability good: {prediction['probability_good']:.2%}\n"
            f"Probability bad: {prediction['probability_bad']:.2%}\n"
            f"Risk level: {prediction['risk_level']}\n"
            f"Business recommendation: {recommendation}\n"
            f"Summary: {prediction['summary_message']}\n"
            f"Disclaimer: {prediction['disclaimer']}\n"
        )

    return (
        "# Credit Risk Assessment\n\n"
        f"- Model: `{prediction['model_name']}`\n"
        f"- Prediction: **{prediction['predicted_class'].title()} Credit**\n"
        f"- Confidence: **{confidence:.2%}**\n"
        f"- Probability good: {prediction['probability_good']:.2%}\n"
        f"- Probability bad: {prediction['probability_bad']:.2%}\n"
        f"- Risk level: **{prediction['risk_level']}**\n"
        f"- Business recommendation: {recommendation}\n\n"
        f"## Summary\n\n{prediction['summary_message']}\n\n"
        f"## Disclaimer\n\n{prediction['disclaimer']}\n"
    )


def render_prediction_result(prediction: dict[str, Any]) -> None:
    """Render a prediction result card.

    Args:
        prediction: Prediction dictionary returned by the backend service.
    """
    css_class = _risk_css_class(str(prediction["risk_level"]))
    badge_class = _badge_css_class(str(prediction["risk_level"]))
    confidence = _prediction_confidence(prediction)
    recommendation = _business_recommendation(str(prediction["risk_level"]))
    risk_level_short = str(prediction["risk_level"]).replace(" Risk", "")
    good_probability = float(prediction["probability_good"])
    bad_probability = float(prediction["probability_bad"])
    st.markdown(
        f"""
        <div class="risk-card {css_class}">
            <h3>📈 Credit Risk Assessment</h3>
            <div class="result-grid">
                <div class="result-item">
                    <div class="result-label">Prediction</div>
                    <div class="result-value">
                        <span class="result-badge {badge_class}">
                            {_risk_icon(str(prediction["risk_level"]))} {prediction["predicted_class"].title()} Credit
                        </span>
                    </div>
                </div>
                <div class="result-item">
                    <div class="result-label">Confidence</div>
                    <div class="result-value">{confidence:.2%}</div>
                </div>
                <div class="result-item">
                    <div class="result-label">Risk Level</div>
                    <div class="result-value">
                        <span class="result-badge {badge_class}">{risk_level_short}</span>
                    </div>
                </div>
                <div class="result-item">
                    <div class="result-label">Selected Model</div>
                    <div class="result-value">Random Forest Baseline</div>
                </div>
            </div>
            <p><strong>Business Recommendation:</strong> {recommendation}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="probability-panel">
            <h4>Probability Comparison</h4>
            <div class="probability-row">
                <div class="probability-label">
                    <span>Probability of Good Credit</span>
                    <span>{good_probability:.2%}</span>
                </div>
                <div class="bar-track">
                    <div class="bar-fill-good" style="width: {good_probability * 100:.2f}%"></div>
                </div>
            </div>
            <div class="probability-row">
                <div class="probability-label">
                    <span>Probability of Bad Credit</span>
                    <span>{bad_probability:.2%}</span>
                </div>
                <div class="bar-track">
                    <div class="bar-fill-bad" style="width: {bad_probability * 100:.2f}%"></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Good Credit Probability", f"{good_probability:.2%}")
    col2.metric("Bad Credit Probability", f"{bad_probability:.2%}")
    col3.metric("Engineered Features", prediction["engineered_feature_count"])
    st.caption(prediction["disclaimer"])

    render_download_center(prediction)

    st.subheader("📈 Prediction Summary")
    st.markdown(
        f"""
        <div class="explanation-grid">
            <div class="explanation-card">
                <h4>Why this prediction?</h4>
                <p>{prediction["summary_message"]}</p>
            </div>
            <div class="explanation-card">
                <h4>Top Global Drivers</h4>
                <p>Checking account status, credit amount, duration interactions, age, and savings behavior
                are among the strongest global Random Forest drivers.</p>
            </div>
            <div class="explanation-card">
                <h4>Important Disclaimer</h4>
                <p>Global feature importance is not an explanation for this individual applicant. It summarizes
                overall model behavior across the training data.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Top global factor badges", expanded=True):
        render_top_global_factors()


def render_top_global_factors() -> None:
    """Render top global factors as compact chips."""
    st.write("Top global factors")
    factor_html = "".join(
        f'<span class="factor-pill">+ {factor}</span>' for factor in TOP_GLOBAL_FACTORS
    )
    st.markdown(factor_html, unsafe_allow_html=True)


def _read_text_file(path: Path) -> str | None:
    """Read a text file if it exists."""
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def render_download_center(prediction: dict[str, Any] | None = None) -> None:
    """Render report and prediction summary download buttons."""
    with st.expander("⬇ Download Center", expanded=prediction is not None):
        if prediction is not None:
            col1, col2 = st.columns(2)
            col1.download_button(
                "Download Prediction Summary (Markdown)",
                data=build_prediction_summary(prediction, "md"),
                file_name="credit_risk_prediction_summary.md",
                mime="text/markdown",
                use_container_width=True,
            )
            col2.download_button(
                "Download Prediction Summary (TXT)",
                data=build_prediction_summary(prediction, "txt"),
                file_name="credit_risk_prediction_summary.txt",
                mime="text/plain",
                use_container_width=True,
            )

        model_card = _read_text_file(config.MODEL_CARD_PATH)
        final_report = _read_text_file(FINAL_MODEL_SELECTION_REPORT_PATH)
        col1, col2 = st.columns(2)
        if model_card is not None:
            col1.download_button(
                "Download Model Card",
                data=model_card,
                file_name="model_card.md",
                mime="text/markdown",
                use_container_width=True,
            )
        else:
            col1.warning("Model card not found.")

        if final_report is not None:
            col2.download_button(
                "Download Final Selection Report",
                data=final_report,
                file_name="final_model_selection_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
        else:
            col2.warning("Final model selection report not found.")


def render_feature_importance() -> None:
    """Render global Random Forest feature importance."""
    st.subheader("📊 Global Feature Importance")
    try:
        feature_importance_df = load_feature_importance().head(10).copy()
    except Exception as exc:
        st.error(f"Feature importance could not be loaded: {exc}")
        return

    render_top_global_factors()
    st.dataframe(feature_importance_df[["feature", "importance"]], use_container_width=True)

    plot_df = feature_importance_df.sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    sns.barplot(data=plot_df, x="importance", y="feature", color="#2563eb", ax=ax)
    ax.set_title("Top Random Forest Feature Importances", fontsize=14, weight="bold", pad=12)
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(axis="x", alpha=0.22)
    sns.despine(left=True, bottom=False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    image_path = Path(FEATURE_IMPORTANCE_IMAGE_PATH)
    if image_path.exists():
        st.image(str(image_path), caption="Saved Random Forest feature importance plot")
    st.caption(
        "Global importance is not a causal explanation for a specific applicant. "
        "It summarizes the Random Forest model's overall split-importance patterns."
    )


def render_model_info() -> None:
    """Render model information and selection rationale."""
    st.header("⚙ Model Information")
    st.write(f"Selected model: `{config.PRIMARY_MODEL_NAME}`")
    st.write(f"Business-risk alternative: `{config.ALTERNATIVE_MODEL_NAME}`")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", "0.7550")
    col2.metric("Macro F1", "0.7173")
    col3.metric("Bad recall", "0.6500")
    col4.metric("ROC-AUC", "0.7907")

    st.write(
        "The Random Forest baseline was selected because it provides the strongest balanced "
        "performance among the reviewed candidates and remains explainable through global "
        "feature importance."
    )
    st.info(
        "The SVM baseline is retained as a business-risk alternative because it has higher "
        "bad-class recall at 0.7833."
    )
    render_feature_importance()
    render_download_center()


def render_about_page() -> None:
    """Render project background and limitations."""
    st.header("ℹ About Project")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dataset", "German Credit")
    col2.metric("Selected Model", "Random Forest")
    col3.metric("Alternative", "SVM")
    col4.metric("Accuracy", "0.7550")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Macro F1", "0.7173")
    col2.metric("ROC-AUC", "0.7907")
    col3.metric("Author", "Syed Muzammil Shah")
    col4.metric("Repository", "CodeAlpha")

    st.subheader("ML Pipeline")
    _render_pipeline_diagram()

    with st.expander("About This App", expanded=True):
        st.write(
            "This application demonstrates a production-oriented machine learning workflow "
            "for binary credit-risk classification."
        )
        st.warning(
            "Educational disclaimer: this model is for portfolio and internship demonstration "
            "only. It is not suitable for real lending decisions."
        )
    with st.expander("Completed phases"):
        st.markdown(
            """
            - Exploratory data analysis
            - Reusable preprocessing architecture
            - Feature engineering
            - Baseline model training and comparison
            - Hyperparameter tuning review
            - Model explainability and final selection
            - Backend inference services
            - Streamlit prediction UI
            - Release-candidate UI/documentation polish
            """
        )
    with st.expander("Known limitations"):
        st.markdown(
            """
            - Dataset is small and historical.
            - No fairness audit or compliance review has been completed.
            - No probability calibration or threshold optimization has been performed.
            - Global feature importance is not an individual causal explanation.
            - This app is not suitable for real lending decisions.
            """
        )
    render_download_center()


def render_footer() -> None:
    """Render a simple footer."""
    st.markdown(
        '<div class="footer"><strong>CodeAlpha Credit Scoring Model</strong><br>'
        "Release Candidate RC-2<br>"
        "Python • Scikit-learn • Streamlit • Joblib<br>"
        "Educational Machine Learning Portfolio</div>",
        unsafe_allow_html=True,
    )


def _render_pipeline_diagram() -> None:
    """Render a lightweight project workflow diagram."""
    steps = [
        "Raw Dataset",
        "Exploratory Data Analysis",
        "Preprocessing",
        "Feature Engineering",
        "Baseline Models",
        "Hyperparameter Tuning",
        "Model Evaluation",
        "Final Model Selection",
        "Deployment",
        "Prediction",
    ]
    step_html = "".join(
        f'<div class="pipeline-step">{step}<div class="pipeline-arrow">↓</div></div>'
        for step in steps
    )
    st.markdown(f'<div class="pipeline">{step_html}</div>', unsafe_allow_html=True)
