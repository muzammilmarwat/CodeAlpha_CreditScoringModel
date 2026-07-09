"""Risk interpretation helpers for model predictions."""

from __future__ import annotations

from app import config

DISCLAIMER = (
    "This prediction is generated for educational and portfolio purposes only. "
    "It is not approved for automated real-world credit decisions."
)


def assign_risk_level(probability_bad: float) -> str:
    """Assign a risk level from the bad-credit probability.

    Args:
        probability_bad: Model probability for the bad-credit class.

    Returns:
        str: Low Risk, Medium Risk, or High Risk.
    """
    if probability_bad < 0.30:
        return "Low Risk"
    if probability_bad < 0.60:
        return "Medium Risk"
    return "High Risk"


def generate_plain_language_summary(
    predicted_class: str,
    probability_good: float,
    probability_bad: float,
    risk_level: str,
) -> str:
    """Generate a user-facing prediction summary.

    Args:
        predicted_class: Predicted target class label.
        probability_good: Probability assigned to good credit risk.
        probability_bad: Probability assigned to bad credit risk.
        risk_level: Assigned risk level.

    Returns:
        str: Plain-language prediction summary.
    """
    return (
        f"The model predicts '{predicted_class}' credit risk. "
        f"Estimated probability of good credit risk is {probability_good:.2%}; "
        f"estimated probability of bad credit risk is {probability_bad:.2%}. "
        f"Overall risk level: {risk_level}."
    )


def interpret_prediction(
    predicted_label: int,
    probability_good: float,
    probability_bad: float,
) -> dict[str, object]:
    """Build a complete interpretation dictionary for a prediction.

    Args:
        predicted_label: Numeric model prediction, where good=1 and bad=0.
        probability_good: Probability for class 1.
        probability_bad: Probability for class 0.

    Returns:
        dict[str, object]: Prediction interpretation fields.
    """
    predicted_class = config.RISK_LABELS.get(int(predicted_label), "unknown")
    risk_level = assign_risk_level(probability_bad)
    summary_message = generate_plain_language_summary(
        predicted_class=predicted_class,
        probability_good=probability_good,
        probability_bad=probability_bad,
        risk_level=risk_level,
    )
    return {
        "predicted_label": int(predicted_label),
        "predicted_class": predicted_class,
        "probability_good": float(probability_good),
        "probability_bad": float(probability_bad),
        "risk_level": risk_level,
        "summary_message": summary_message,
        "disclaimer": DISCLAIMER,
    }
