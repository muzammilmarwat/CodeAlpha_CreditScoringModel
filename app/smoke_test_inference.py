"""Smoke test for backend credit-risk inference services."""

from __future__ import annotations

from app.services.prediction_service import predict_credit_risk
from app.services.validation_service import validate_applicant_input
from app.utils.logging_config import configure_app_logging
from app.utils.paths import validate_required_artifacts

SAMPLE_APPLICANT = {
    "checking_account_status": "A12",
    "duration_months": 24,
    "credit_history": "A32",
    "purpose": "A43",
    "credit_amount": 2500,
    "savings_account": "A61",
    "employment_duration": "A73",
    "installment_rate": 2,
    "personal_status_sex": "A93",
    "other_debtors": "A101",
    "present_residence_years": 2,
    "property": "A121",
    "age": 35,
    "other_installment_plans": "A143",
    "housing": "A152",
    "existing_credits": 1,
    "job": "A173",
    "people_liable": 1,
    "telephone": "A191",
    "foreign_worker": "A201",
}


def main() -> None:
    """Run a single-applicant backend inference smoke test."""
    configure_app_logging()
    validate_required_artifacts()
    validate_applicant_input(SAMPLE_APPLICANT)
    prediction = predict_credit_risk(SAMPLE_APPLICANT)

    print("Artifacts found: yes")
    print("Input validated: yes")
    print(f"Predicted class: {prediction['predicted_class']}")
    print(f"Probability good: {prediction['probability_good']:.4f}")
    print(f"Probability bad: {prediction['probability_bad']:.4f}")
    print(f"Risk level: {prediction['risk_level']}")
    print(f"Summary: {prediction['summary_message']}")


if __name__ == "__main__":
    main()
