"""Applicant input schema and validation constants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

RAW_INPUT_FIELDS = (
    "checking_account_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_account",
    "employment_duration",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "present_residence_years",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "people_liable",
    "telephone",
    "foreign_worker",
)

NUMERIC_RULES = {
    "duration_months": {"min": 1, "max": 120},
    "credit_amount": {"min": 1, "max": 100000},
    "installment_rate": {"min": 1, "max": 4},
    "present_residence_years": {"min": 1, "max": 4},
    "age": {"min": 18, "max": 100},
    "existing_credits": {"min": 1, "max": 4},
    "people_liable": {"allowed": {1, 2}},
}

ALLOWED_CATEGORIES = {
    "checking_account_status": {"A11", "A12", "A13", "A14"},
    "credit_history": {"A30", "A31", "A32", "A33", "A34"},
    "purpose": {"A40", "A41", "A42", "A43", "A44", "A45", "A46", "A48", "A49", "A410"},
    "savings_account": {"A61", "A62", "A63", "A64", "A65"},
    "employment_duration": {"A71", "A72", "A73", "A74", "A75"},
    "personal_status_sex": {"A91", "A92", "A93", "A94", "A95"},
    "other_debtors": {"A101", "A102", "A103"},
    "property": {"A121", "A122", "A123", "A124"},
    "other_installment_plans": {"A141", "A142", "A143"},
    "housing": {"A151", "A152", "A153"},
    "job": {"A171", "A172", "A173", "A174"},
    "telephone": {"A191", "A192"},
    "foreign_worker": {"A201", "A202"},
}


@dataclass(frozen=True)
class ApplicantInput:
    """Raw applicant input before feature engineering."""

    checking_account_status: str
    duration_months: int
    credit_history: str
    purpose: str
    credit_amount: int
    savings_account: str
    employment_duration: str
    installment_rate: int
    personal_status_sex: str
    other_debtors: str
    present_residence_years: int
    property: str
    age: int
    other_installment_plans: str
    housing: str
    existing_credits: int
    job: str
    people_liable: int
    telephone: str
    foreign_worker: str

    def to_dict(self) -> dict[str, Any]:
        """Convert applicant input to a dictionary."""
        return self.__dict__.copy()
