"""Input validation service for credit-risk predictions."""

from __future__ import annotations

from typing import Any

from app.schemas.input_schema import ALLOWED_CATEGORIES, NUMERIC_RULES, RAW_INPUT_FIELDS
from app.utils.exceptions import PredictionInputError


def _validate_field_set(input_data: dict[str, Any]) -> None:
    """Validate required and unexpected fields."""
    received = set(input_data)
    expected = set(RAW_INPUT_FIELDS)
    missing = sorted(expected - received)
    unexpected = sorted(received - expected)

    if missing:
        raise PredictionInputError(f"Missing required input fields: {missing}")
    if unexpected:
        raise PredictionInputError(f"Unexpected input fields: {unexpected}")


def _clean_numeric_field(field_name: str, value: Any) -> int:
    """Validate and normalize a numeric field."""
    if isinstance(value, bool):
        raise PredictionInputError(f"Field '{field_name}' must be numeric, not boolean.")
    try:
        cleaned_value = int(value)
    except (TypeError, ValueError) as exc:
        raise PredictionInputError(f"Field '{field_name}' must be an integer.") from exc

    rule = NUMERIC_RULES[field_name]
    allowed = rule.get("allowed")
    if allowed is not None and cleaned_value not in allowed:
        raise PredictionInputError(
            f"Field '{field_name}' must be one of {sorted(allowed)}."
        )

    minimum = rule.get("min")
    maximum = rule.get("max")
    if minimum is not None and cleaned_value < minimum:
        raise PredictionInputError(f"Field '{field_name}' must be >= {minimum}.")
    if maximum is not None and cleaned_value > maximum:
        raise PredictionInputError(f"Field '{field_name}' must be <= {maximum}.")

    return cleaned_value


def _clean_categorical_field(field_name: str, value: Any) -> str:
    """Validate and normalize a categorical field."""
    cleaned_value = str(value).strip()
    allowed_values = ALLOWED_CATEGORIES[field_name]
    if cleaned_value not in allowed_values:
        raise PredictionInputError(
            f"Field '{field_name}' must be one of {sorted(allowed_values)}."
        )
    return cleaned_value


def validate_applicant_input(input_data: dict[str, Any]) -> dict[str, Any]:
    """Validate one applicant input payload.

    Args:
        input_data: Raw applicant input dictionary.

    Returns:
        dict[str, Any]: Cleaned input dictionary in model-training column order.

    Raises:
        PredictionInputError: If input fields, values, or ranges are invalid.
    """
    if not isinstance(input_data, dict):
        raise PredictionInputError("Applicant input must be provided as a dictionary.")

    _validate_field_set(input_data)
    cleaned: dict[str, Any] = {}
    for field_name in RAW_INPUT_FIELDS:
        if field_name in NUMERIC_RULES:
            cleaned[field_name] = _clean_numeric_field(field_name, input_data[field_name])
        elif field_name in ALLOWED_CATEGORIES:
            cleaned[field_name] = _clean_categorical_field(field_name, input_data[field_name])
        else:
            raise PredictionInputError(f"No validation rule configured for '{field_name}'.")

    return cleaned
