"""Streamlit application for credit-risk prediction and explainability."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app import config
from app.services.prediction_service import predict_credit_risk
from app.ui.components import (
    render_about_page,
    render_footer,
    render_header,
    render_model_info,
    render_prediction_result,
    render_sidebar,
)
from app.ui.form_fields import render_prediction_form
from app.ui.style import get_custom_css
from app.utils.exceptions import (
    ArtifactNotFoundError,
    ModelLoadingError,
    PredictionInputError,
    PredictionServiceError,
)
from app.utils.paths import validate_required_artifacts


def configure_page() -> None:
    """Configure Streamlit page metadata and layout."""
    st.set_page_config(
        page_title="Credit Risk Assessment System",
        page_icon="CR",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(get_custom_css(), unsafe_allow_html=True)


def render_prediction_page() -> None:
    """Render the main prediction workflow."""
    st.header("Applicant Risk Prediction")
    st.write(
        "Enter one applicant profile using the same German Credit category codes used during training."
    )

    col1, col2, col3 = st.columns(3)
    col1.info("Default model: Random Forest baseline")
    col2.info("Alternative: SVM baseline for bad-class recall")
    col3.warning("Educational use only")

    submitted, input_data = render_prediction_form()
    if not submitted:
        if "latest_prediction" in st.session_state:
            render_prediction_result(st.session_state["latest_prediction"])
        return

    try:
        validate_required_artifacts()
        with st.spinner("Generating credit risk assessment..."):
            prediction = predict_credit_risk(input_data, model_name=config.DEFAULT_MODEL_CHOICE)
    except PredictionInputError as exc:
        st.error(f"Input validation failed: {exc}")
        return
    except ArtifactNotFoundError as exc:
        st.error(f"Required artifact missing: {exc}")
        st.info("Check that saved model, preprocessing, feature importance, and model card artifacts exist.")
        return
    except (ModelLoadingError, PredictionServiceError) as exc:
        st.error(f"Prediction service error: {exc}")
        return
    except Exception as exc:
        st.error(f"Unexpected application error: {exc}")
        return

    st.session_state["latest_input"] = input_data
    st.session_state["latest_prediction"] = prediction
    render_prediction_result(prediction)


def main() -> None:
    """Run the Streamlit app."""
    configure_page()
    render_header()
    selected_page = render_sidebar()

    if selected_page == "Prediction":
        render_prediction_page()
    elif selected_page == "Model Information":
        render_model_info()
    else:
        render_about_page()
    render_footer()


if __name__ == "__main__":
    main()
