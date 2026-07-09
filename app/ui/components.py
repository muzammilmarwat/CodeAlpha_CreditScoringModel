"""Reusable Streamlit UI components for the credit-risk app."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from app import config
from app.services.model_loader import load_feature_importance
from app.ui.form_fields import SAMPLE_LOW_RISK_APPLICANT, SAMPLE_MEDIUM_RISK_APPLICANT

FEATURE_IMPORTANCE_IMAGE_PATH = (
    config.PROJECT_ROOT / "images" / "explainability" / "random_forest_feature_importance.png"
)


def render_header() -> None:
    """Render the application header."""
    st.markdown('<div class="main-title">Credit Risk Assessment System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">A portfolio ML application for German Credit risk classification.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="info-note">Educational and internship project only. '
        "This application is not intended for real lending decisions.</div>",
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
        st.write(f"Selected model: `{config.PRIMARY_MODEL_NAME}`")
        st.write(f"Risk-focused alternative: `{config.ALTERNATIVE_MODEL_NAME}`")
        st.write("Model card: `reports/model_card.md`")
        st.warning("Not approved for automated real-world credit decisions.")
        st.divider()
        page = st.radio(
            "Navigation",
            ["Prediction", "Model Information", "About Project"],
            index=0,
        )
        st.divider()
        st.write("Sample input")
        if st.button("Load medium-risk sample", use_container_width=True):
            st.session_state["sample_applicant"] = SAMPLE_MEDIUM_RISK_APPLICANT.copy()
            st.rerun()
        if st.button("Load low-risk sample", use_container_width=True):
            st.session_state["sample_applicant"] = SAMPLE_LOW_RISK_APPLICANT.copy()
            st.rerun()
    return page


def _risk_css_class(risk_level: str) -> str:
    """Return CSS class name for a risk level."""
    if risk_level == "Low Risk":
        return "risk-low"
    if risk_level == "High Risk":
        return "risk-high"
    return "risk-medium"


def render_prediction_result(prediction: dict[str, Any]) -> None:
    """Render a prediction result card.

    Args:
        prediction: Prediction dictionary returned by the backend service.
    """
    css_class = _risk_css_class(str(prediction["risk_level"]))
    st.markdown(
        f"""
        <div class="risk-card {css_class}">
            <h3>Prediction: {prediction["predicted_class"].title()} Credit Risk</h3>
            <p><strong>Risk level:</strong> {prediction["risk_level"]}</p>
            <p>{prediction["summary_message"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Probability good", f"{prediction['probability_good']:.2%}")
    col2.metric("Probability bad", f"{prediction['probability_bad']:.2%}")
    col3.metric("Engineered features", prediction["engineered_feature_count"])
    st.caption(prediction["disclaimer"])

    with st.expander("Plain-language explanation"):
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


def render_feature_importance() -> None:
    """Render global Random Forest feature importance."""
    st.subheader("Global Feature Importance")
    try:
        feature_importance_df = load_feature_importance().head(10)
    except Exception as exc:
        st.error(f"Feature importance could not be loaded: {exc}")
        return

    st.dataframe(feature_importance_df[["feature", "importance"]], use_container_width=True)
    st.bar_chart(feature_importance_df.set_index("feature")["importance"])

    image_path = Path(FEATURE_IMPORTANCE_IMAGE_PATH)
    if image_path.exists():
        st.image(str(image_path), caption="Saved Random Forest feature importance plot")


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


def render_about_page() -> None:
    """Render project background and limitations."""
    st.header("About Project")
    st.write("Project: CodeAlpha Credit Scoring Model")
    st.write(
        "This application uses the German Credit dataset to demonstrate a production-oriented "
        "machine learning workflow for binary credit-risk classification."
    )
    st.subheader("Completed phases")
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
        """
    )
    st.subheader("Limitations")
    st.markdown(
        """
        - Dataset is small and historical.
        - No fairness audit or compliance review has been completed.
        - No probability calibration or threshold optimization has been performed.
        - This app is not suitable for real lending decisions.
        """
    )
    st.warning(
        "Ethical disclaimer: this model should support learning and review only. It should not "
        "replace qualified decision-makers or be used for automated credit approval."
    )
