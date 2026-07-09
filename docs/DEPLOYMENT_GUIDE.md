# Deployment Guide

This guide describes how to run the Streamlit credit-risk demo using the saved repository artifacts.

## Deployment Scope

The app is a portfolio and internship demonstration. It is not approved for automated lending decisions, production underwriting, or customer-facing financial advice.

## Required Artifacts

Confirm these files exist before deployment:

- `models/preprocessing/feature_engineer.joblib`
- `models/baseline/random_forest_baseline.joblib`
- `models/baseline/svm_baseline.joblib`
- `reports/model_card.md`
- `reports/final_model_selection/final_model_selection_report.md`
- `reports/explainability/random_forest_feature_importance.csv`

## Local Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Pre-Deployment Verification

Run:

```bash
python -m compileall app src
python -m app.smoke_test_inference
```

Both commands should complete without retraining, tuning, or modifying model artifacts.

## Streamlit Community Cloud Notes

Use these settings if deploying to Streamlit Community Cloud:

- Main file path: `app/app.py`
- Python dependencies: `requirements.txt`
- Repository root: `CodeAlpha_CreditScoringModel`

Keep model artifacts committed only if they are appropriate for the target hosting platform and file-size limits.

## Operational Limitations

- No live monitoring is implemented.
- No fairness or compliance audit is implemented.
- No probability calibration is implemented.
- No drift detection is implemented.
- No user authentication is implemented.
- Saved artifacts are loaded from local repository paths.

## Recommended Next Production Steps

1. Add automated tests for validation and inference services.
2. Add probability calibration and threshold review.
3. Add fairness and bias analysis.
4. Add model monitoring and prediction logging for a controlled environment.
5. Add CI checks for compile, smoke tests, and notebook execution.
