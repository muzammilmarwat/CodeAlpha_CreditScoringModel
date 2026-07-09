"""Tests for applicant input validation."""

from __future__ import annotations

import pytest

from app.schemas.input_schema import RAW_INPUT_FIELDS
from app.services.validation_service import validate_applicant_input
from app.utils.exceptions import PredictionInputError
from tests import VALID_APPLICANT


def test_valid_applicant_input_passes_validation() -> None:
    """A complete valid applicant payload should validate in model input order."""
    cleaned = validate_applicant_input(VALID_APPLICANT)

    assert list(cleaned) == list(RAW_INPUT_FIELDS)
    assert cleaned["duration_months"] == 24
    assert cleaned["checking_account_status"] == "A12"


def test_missing_field_raises_prediction_input_error() -> None:
    """Validation should fail when a required raw field is absent."""
    invalid_input = VALID_APPLICANT.copy()
    invalid_input.pop("credit_amount")

    with pytest.raises(PredictionInputError, match="Missing required input fields"):
        validate_applicant_input(invalid_input)


def test_invalid_category_raises_prediction_input_error() -> None:
    """Validation should fail when a categorical code is outside the schema."""
    invalid_input = VALID_APPLICANT.copy()
    invalid_input["checking_account_status"] = "INVALID"

    with pytest.raises(PredictionInputError, match="checking_account_status"):
        validate_applicant_input(invalid_input)


def test_invalid_numeric_range_raises_prediction_input_error() -> None:
    """Validation should fail when a numeric value is outside its allowed range."""
    invalid_input = VALID_APPLICANT.copy()
    invalid_input["age"] = 17

    with pytest.raises(PredictionInputError, match="age"):
        validate_applicant_input(invalid_input)
