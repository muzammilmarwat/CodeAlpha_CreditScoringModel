"""Tests for saved-artifact prediction orchestration."""

from __future__ import annotations

import pytest

from app.services.prediction_service import predict_credit_risk
from tests import VALID_APPLICANT


def test_prediction_service_returns_required_fields() -> None:
    """Prediction service should return the public result contract."""
    prediction = predict_credit_risk(VALID_APPLICANT)

    required_fields = {
        "predicted_class",
        "probability_good",
        "probability_bad",
        "risk_level",
        "summary_message",
    }

    assert required_fields.issubset(prediction)
    assert prediction["predicted_class"] in {"good", "bad"}
    assert prediction["risk_level"] in {"Low Risk", "Medium Risk", "High Risk"}
    assert isinstance(prediction["summary_message"], str)
    assert prediction["summary_message"]


def test_prediction_probabilities_are_valid() -> None:
    """Class probabilities should be bounded and sum to approximately one."""
    prediction = predict_credit_risk(VALID_APPLICANT)
    probability_good = prediction["probability_good"]
    probability_bad = prediction["probability_bad"]

    assert 0.0 <= probability_good <= 1.0
    assert 0.0 <= probability_bad <= 1.0
    assert probability_good + probability_bad == pytest.approx(1.0, abs=1e-6)
