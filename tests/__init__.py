"""Shared test fixtures for deployment-layer tests."""

from __future__ import annotations

from typing import Any


VALID_APPLICANT: dict[str, Any] = {
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
