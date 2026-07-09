"""Form field labels, sample applicants, and Streamlit form rendering."""

from __future__ import annotations

from typing import Any

import streamlit as st

from app.schemas.input_schema import RAW_INPUT_FIELDS

CATEGORY_LABELS = {
    "checking_account_status": {
        "A11": "A11 - checking account < 0 DM",
        "A12": "A12 - checking account 0 to 200 DM",
        "A13": "A13 - checking account >= 200 DM",
        "A14": "A14 - no checking account",
    },
    "credit_history": {
        "A30": "A30 - no credits taken / all paid back",
        "A31": "A31 - all credits at this bank paid back",
        "A32": "A32 - existing credits paid back duly",
        "A33": "A33 - delay in paying off in the past",
        "A34": "A34 - critical account / other credits",
    },
    "purpose": {
        "A40": "A40 - car (new)",
        "A41": "A41 - car (used)",
        "A42": "A42 - furniture/equipment",
        "A43": "A43 - radio/television",
        "A44": "A44 - domestic appliances",
        "A45": "A45 - repairs",
        "A46": "A46 - education",
        "A48": "A48 - retraining",
        "A49": "A49 - business",
        "A410": "A410 - others",
    },
    "savings_account": {
        "A61": "A61 - savings < 100 DM",
        "A62": "A62 - savings 100 to 500 DM",
        "A63": "A63 - savings 500 to 1000 DM",
        "A64": "A64 - savings >= 1000 DM",
        "A65": "A65 - unknown / no savings account",
    },
    "employment_duration": {
        "A71": "A71 - unemployed",
        "A72": "A72 - employed < 1 year",
        "A73": "A73 - employed 1 to 4 years",
        "A74": "A74 - employed 4 to 7 years",
        "A75": "A75 - employed >= 7 years",
    },
    "personal_status_sex": {
        "A91": "A91 - male divorced/separated",
        "A92": "A92 - female divorced/separated/married",
        "A93": "A93 - male single",
        "A94": "A94 - male married/widowed",
        "A95": "A95 - female single",
    },
    "other_debtors": {
        "A101": "A101 - none",
        "A102": "A102 - co-applicant",
        "A103": "A103 - guarantor",
    },
    "property": {
        "A121": "A121 - real estate",
        "A122": "A122 - building society / life insurance",
        "A123": "A123 - car or other property",
        "A124": "A124 - unknown / no property",
    },
    "other_installment_plans": {
        "A141": "A141 - bank",
        "A142": "A142 - stores",
        "A143": "A143 - none",
    },
    "housing": {
        "A151": "A151 - rent",
        "A152": "A152 - own",
        "A153": "A153 - free",
    },
    "job": {
        "A171": "A171 - unemployed / unskilled non-resident",
        "A172": "A172 - unskilled resident",
        "A173": "A173 - skilled employee / official",
        "A174": "A174 - management / self-employed",
    },
    "telephone": {
        "A191": "A191 - no telephone",
        "A192": "A192 - telephone registered",
    },
    "foreign_worker": {
        "A201": "A201 - foreign worker",
        "A202": "A202 - not foreign worker",
    },
}

SAMPLE_MEDIUM_RISK_APPLICANT = {
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

SAMPLE_LOW_RISK_APPLICANT = {
    "checking_account_status": "A14",
    "duration_months": 12,
    "credit_history": "A34",
    "purpose": "A43",
    "credit_amount": 1200,
    "savings_account": "A65",
    "employment_duration": "A75",
    "installment_rate": 2,
    "personal_status_sex": "A93",
    "other_debtors": "A101",
    "present_residence_years": 4,
    "property": "A121",
    "age": 45,
    "other_installment_plans": "A143",
    "housing": "A152",
    "existing_credits": 2,
    "job": "A173",
    "people_liable": 1,
    "telephone": "A192",
    "foreign_worker": "A201",
}

SAMPLE_HIGH_RISK_APPLICANT = {
    "checking_account_status": "A11",
    "duration_months": 48,
    "credit_history": "A30",
    "purpose": "A40",
    "credit_amount": 7500,
    "savings_account": "A61",
    "employment_duration": "A72",
    "installment_rate": 4,
    "personal_status_sex": "A92",
    "other_debtors": "A101",
    "present_residence_years": 1,
    "property": "A124",
    "age": 24,
    "other_installment_plans": "A141",
    "housing": "A151",
    "existing_credits": 1,
    "job": "A172",
    "people_liable": 2,
    "telephone": "A191",
    "foreign_worker": "A201",
}

SAMPLE_APPLICANTS = {
    "Low Risk Applicant": SAMPLE_LOW_RISK_APPLICANT,
    "Medium Risk Applicant": SAMPLE_MEDIUM_RISK_APPLICANT,
    "High Risk Applicant": SAMPLE_HIGH_RISK_APPLICANT,
}


def _label_for_code(field_name: str, code: str) -> str:
    """Return display label for a category code."""
    return CATEGORY_LABELS[field_name][code]


def _code_from_label(label: str) -> str:
    """Extract the raw category code from a friendly UI label."""
    return label.split(" - ", maxsplit=1)[0]


def _session_value_for_field(field_name: str, default_code: str) -> Any:
    """Return session-backed widget value for a field."""
    key = f"field_{field_name}"
    if key in st.session_state:
        return st.session_state[key]
    if field_name in CATEGORY_LABELS:
        return _label_for_code(field_name, default_code)
    return default_code


def _select_code(field_name: str, label: str, default_code: str) -> str:
    """Render a selectbox and return the selected raw code."""
    labels = list(CATEGORY_LABELS[field_name].values())
    default_label = _session_value_for_field(field_name, default_code)
    selected_label = st.selectbox(
        label,
        labels,
        index=labels.index(default_label),
        key=f"field_{field_name}",
    )
    return _code_from_label(selected_label)


def _number_input(field_name: str, label: str, default_value: int, **kwargs: Any) -> int:
    """Render an integer number input."""
    return int(
        st.number_input(
            label,
            value=int(_session_value_for_field(field_name, default_value)),
            step=1,
            key=f"field_{field_name}",
            **kwargs,
        )
    )


def get_default_applicant() -> dict[str, Any]:
    """Return the default applicant payload for the prediction form."""
    return st.session_state.get("sample_applicant", SAMPLE_MEDIUM_RISK_APPLICANT).copy()


def apply_applicant_to_session(applicant: dict[str, Any]) -> None:
    """Populate Streamlit widget session state from an applicant payload.

    Args:
        applicant: Valid raw applicant dictionary.
    """
    st.session_state["sample_applicant"] = applicant.copy()
    for field_name, value in applicant.items():
        if field_name in CATEGORY_LABELS:
            st.session_state[f"field_{field_name}"] = _label_for_code(field_name, str(value))
        else:
            st.session_state[f"field_{field_name}"] = int(value)


def reset_form_state() -> None:
    """Reset the form to the default medium-risk sample."""
    apply_applicant_to_session(SAMPLE_MEDIUM_RISK_APPLICANT)
    st.session_state.pop("latest_prediction", None)
    st.session_state.pop("latest_input", None)


def render_prediction_form() -> tuple[bool, dict[str, Any]]:
    """Render the applicant prediction form.

    Returns:
        tuple[bool, dict[str, Any]]: Submit state and raw model input payload.
    """
    defaults = get_default_applicant()

    with st.form("credit_risk_prediction_form"):
        st.subheader("A. Account & Credit History")
        col1, col2 = st.columns(2)
        with col1:
            checking_account_status = _select_code(
                "checking_account_status",
                "Checking account status",
                defaults["checking_account_status"],
            )
            savings_account = _select_code(
                "savings_account",
                "Savings account",
                defaults["savings_account"],
            )
        with col2:
            credit_history = _select_code("credit_history", "Credit history", defaults["credit_history"])
            employment_duration = _select_code(
                "employment_duration",
                "Employment duration",
                defaults["employment_duration"],
            )

        st.subheader("B. Loan Details")
        col1, col2 = st.columns(2)
        with col1:
            duration_months = _number_input(
                "duration_months",
                "Duration in months",
                defaults["duration_months"],
                min_value=1,
                max_value=120,
            )
            purpose = _select_code("purpose", "Purpose", defaults["purpose"])
        with col2:
            credit_amount = _number_input(
                "credit_amount",
                "Credit amount",
                defaults["credit_amount"],
                min_value=1,
                max_value=100000,
            )
            installment_rate = _number_input(
                "installment_rate",
                "Installment rate",
                defaults["installment_rate"],
                min_value=1,
                max_value=4,
            )

        st.subheader("C. Personal & Residence Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            personal_status_sex = _select_code(
                "personal_status_sex",
                "Personal status / sex",
                defaults["personal_status_sex"],
            )
            present_residence_years = _number_input(
                "present_residence_years",
                "Present residence years",
                defaults["present_residence_years"],
                min_value=1,
                max_value=4,
            )
        with col2:
            age = _number_input("age", "Age", defaults["age"], min_value=18, max_value=100)
            housing = _select_code("housing", "Housing", defaults["housing"])
        with col3:
            property_value = _select_code("property", "Property", defaults["property"])
            job = _select_code("job", "Job", defaults["job"])

        st.subheader("D. Additional Obligations")
        col1, col2, col3 = st.columns(3)
        with col1:
            other_debtors = _select_code("other_debtors", "Other debtors", defaults["other_debtors"])
            existing_credits = _number_input(
                "existing_credits",
                "Existing credits",
                defaults["existing_credits"],
                min_value=1,
                max_value=4,
            )
        with col2:
            other_installment_plans = _select_code(
                "other_installment_plans",
                "Other installment plans",
                defaults["other_installment_plans"],
            )
            people_liable = _number_input(
                "people_liable",
                "People liable",
                defaults["people_liable"],
                min_value=1,
                max_value=2,
            )
        with col3:
            telephone = _select_code("telephone", "Telephone", defaults["telephone"])
            foreign_worker = _select_code("foreign_worker", "Foreign worker", defaults["foreign_worker"])

        submitted = st.form_submit_button("Assess Credit Risk", use_container_width=True)

    input_data = {
        "checking_account_status": checking_account_status,
        "duration_months": duration_months,
        "credit_history": credit_history,
        "purpose": purpose,
        "credit_amount": credit_amount,
        "savings_account": savings_account,
        "employment_duration": employment_duration,
        "installment_rate": installment_rate,
        "personal_status_sex": personal_status_sex,
        "other_debtors": other_debtors,
        "present_residence_years": present_residence_years,
        "property": property_value,
        "age": age,
        "other_installment_plans": other_installment_plans,
        "housing": housing,
        "existing_credits": existing_credits,
        "job": job,
        "people_liable": people_liable,
        "telephone": telephone,
        "foreign_worker": foreign_worker,
    }
    return submitted, {field: input_data[field] for field in RAW_INPUT_FIELDS}
