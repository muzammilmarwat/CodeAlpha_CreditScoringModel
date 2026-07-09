"""Reusable Streamlit UI components for the credit-risk app."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
            <div class="main-title">Credit Risk Assessment System</div>
            <div class="subtitle">
                Production-style ML application for German Credit risk classification,
                model interpretation, and portfolio deployment review.
            </div>
            <div class="info-note">
                Educational and internship project only. This application is not intended
                for real lending decisions.
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
        st.title(config.APP_NAME)
        st.caption(f"Version {config.APP_VERSION}")
        st.divider()
        st.success(f"Default model: `{config.PRIMARY_MODEL_NAME}`")
        st.info(f"Risk-focused alternative: `{config.ALTERNATIVE_MODEL_NAME}`")
        st.write("Model card: `reports/model_card.md`")
        st.warning("Not approved for automated real-world credit decisions.")
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
        return "[OK]"
    if risk_level == "High Risk":
        return "[!]"
    return "[Review]"


def _business_recommendation(risk_level: str) -> str:
    """Return a short business recommendation for a risk level."""
    if risk_level == "Low Risk":
        return "Approve with normal review."
    if risk_level == "High Risk":
        return "High risk. Perform additional verification."
    return "Manual review recommended."


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
    confidence = _prediction_confidence(prediction)
    recommendation = _business_recommendation(str(prediction["risk_level"]))
    risk_level_short = str(prediction["risk_level"]).replace(" Risk", "")
    st.markdown(
        f"""
        <div class="risk-card {css_class}">
            <h3>{_risk_icon(str(prediction["risk_level"]))} Credit Risk Assessment</h3>
            <div class="result-grid">
                <div class="result-item">
                    <div class="result-label">Prediction</div>
                    <div class="result-value">{prediction["predicted_class"].title()} Credit</div>
                </div>
                <div class="result-item">
                    <div class="result-label">Confidence</div>
                    <div class="result-value">{confidence:.2%}</div>
                </div>
                <div class="result-item">
                    <div class="result-label">Risk Level</div>
                    <div class="result-value">{risk_level_short}</div>
                </div>
                <div class="result-item">
                    <div class="result-label">Selected Model</div>
                    <div class="result-value">Random Forest Baseline</div>
                </div>
            </div>
            <p><strong>Business recommendation:</strong> {recommendation}</p>
            <p>{prediction["summary_message"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(min(max(float(prediction["probability_bad"]), 0.0), 1.0), text="Estimated bad-credit probability")

    col1, col2, col3 = st.columns(3)
    col1.metric("Probability good", f"{prediction['probability_good']:.2%}")
    col2.metric("Probability bad", f"{prediction['probability_bad']:.2%}")
    col3.metric("Engineered features", prediction["engineered_feature_count"])
    st.caption(prediction["disclaimer"])

    render_download_center(prediction)

    with st.expander("Plain-language explanation", expanded=True):
        st.write(
            "This prediction is based on historical patterns learned from the German Credit dataset."
        )
        st.write(
            "Important global drivers include checking account status, credit amount per month, "
            "credit duration interaction, credit amount, and age."
        )
        st.write(
            "Global importance does not causally explain this individual prediction; it summarizes "
            "which features were most influential for the Random Forest model overall."
        )
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
    with st.expander("Download Center", expanded=prediction is not None):
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
    st.subheader("Global Feature Importance")
    try:
        feature_importance_df = load_feature_importance().head(10)
    except Exception as exc:
        st.error(f"Feature importance could not be loaded: {exc}")
        return

    render_top_global_factors()
    st.dataframe(feature_importance_df[["feature", "importance"]], use_container_width=True)
    st.bar_chart(feature_importance_df.set_index("feature")["importance"])

    image_path = Path(FEATURE_IMPORTANCE_IMAGE_PATH)
    if image_path.exists():
        st.image(str(image_path), caption="Saved Random Forest feature importance plot")
    st.caption(
        "Global importance is not a causal explanation for a specific applicant. "
        "It summarizes the Random Forest model's overall split-importance patterns."
    )


def render_model_info() -> None:
    """Render model information and selection rationale."""
    st.header("Model Information")
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
    st.header("About Project")
    st.markdown(
        """
        <div class="section-card">
            <strong>Repository:</strong> CodeAlpha_CreditScoringModel<br>
            <strong>Author:</strong> Syed Muzammil Shah<br>
            <strong>Dataset:</strong> German Credit Dataset<br>
            <strong>Model:</strong> Random Forest Baseline<br>
            <strong>Alternative:</strong> SVM Baseline<br>
            <strong>Accuracy:</strong> 0.755 &nbsp; <strong>Macro F1:</strong> 0.7173
        </div>
        """,
        unsafe_allow_html=True,
    )
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
        '<div class="footer">CodeAlpha Credit Scoring Model | Release Candidate RC-1 | '
        "Educational ML deployment project</div>",
        unsafe_allow_html=True,
    )
