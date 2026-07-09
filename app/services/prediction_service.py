"""Backend prediction service for single-applicant credit-risk inference."""

from __future__ import annotations

from typing import Any

from app import config
from app.schemas.input_schema import RAW_INPUT_FIELDS
from app.services.risk_interpreter import interpret_prediction
from app.services.validation_service import validate_applicant_input
from app.utils.exceptions import PredictionServiceError


def prepare_input_dataframe(input_data: dict[str, Any]) -> Any:
    """Convert cleaned applicant input into a single-row dataframe.

    Args:
        input_data: Cleaned applicant input dictionary.

    Returns:
        pd.DataFrame: Single-row dataframe in raw training column order.
    """
    import pandas as pd

    return pd.DataFrame([{field: input_data[field] for field in RAW_INPUT_FIELDS}])


def _extract_class_probability(model: Any, probabilities: list[float], class_label: int) -> float:
    """Extract a probability by sklearn class label."""
    try:
        class_index = list(model.classes_).index(class_label)
    except (AttributeError, ValueError) as exc:
        raise PredictionServiceError(
            f"Loaded model does not expose expected class label {class_label}."
        ) from exc
    return float(probabilities[class_index])


def predict_credit_risk(
    input_data: dict[str, Any],
    model_name: str = config.DEFAULT_MODEL_CHOICE,
) -> dict[str, Any]:
    """Predict credit risk for one applicant using saved artifacts.

    Args:
        input_data: Raw applicant input dictionary.
        model_name: Supported model artifact name.

    Returns:
        dict[str, Any]: Complete prediction result and interpretation.

    Raises:
        PredictionServiceError: If the loaded model cannot produce probabilities
            or prediction fails after validation.
    """
    cleaned_input = validate_applicant_input(input_data)
    raw_df = prepare_input_dataframe(cleaned_input)

    try:
        from app.services.model_loader import load_feature_engineer, load_model

        feature_engineer = load_feature_engineer()
        engineered_df = feature_engineer.transform(raw_df)
        model = load_model(model_name)

        prediction = model.predict(engineered_df)
        if not hasattr(model, "predict_proba"):
            raise PredictionServiceError(
                f"Model '{model_name}' does not support probability predictions."
            )
        probabilities = model.predict_proba(engineered_df)[0]
    except PredictionServiceError:
        raise
    except Exception as exc:
        raise PredictionServiceError(f"Prediction failed for model '{model_name}': {exc}") from exc

    probability_good = _extract_class_probability(model, probabilities, config.TARGET_MAPPING["good"])
    probability_bad = _extract_class_probability(model, probabilities, config.TARGET_MAPPING["bad"])
    interpretation = interpret_prediction(
        predicted_label=int(prediction[0]),
        probability_good=probability_good,
        probability_bad=probability_bad,
    )

    return {
        "model_name": model_name,
        **interpretation,
        "engineered_feature_count": int(engineered_df.shape[1]),
        "input_validation_status": "passed",
    }
