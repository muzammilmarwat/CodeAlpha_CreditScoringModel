# Model Card: Credit Scoring Model

## 1. Model Overview

- Model name: Random Forest Baseline
- Project: CodeAlpha_CreditScoringModel
- Task type: Binary Classification
- Primary model: `random_forest_baseline`
- Business-risk alternative: `svm_baseline`
- Author: Syed Muzammil Shah
- Status: Phase 7 complete

This model card documents the selected credit-risk classification model after exploratory analysis, preprocessing, baseline model comparison, hyperparameter tuning, model explainability, and final model selection.

## 2. Intended Use

The model predicts whether a credit applicant is likely to be a good credit risk or a bad credit risk based on applicant and loan-related features.

This project is intended for educational, portfolio, and internship demonstration purposes. It is not intended for real banking deployment, automated credit approval, or production financial decision-making without substantial additional validation, governance, compliance review, and fairness assessment.

## 3. Dataset

- Dataset: German Credit Dataset
- Total samples: 1,000
- Training samples: 800
- Testing samples: 200
- Number of predictors after feature engineering: 25
- Target: `credit_risk`
- Target encoding: `good = 1`, `bad = 0`

The project uses a train/test split with saved preprocessing and model-ready artifacts. Feature engineering is performed before model evaluation through reusable project code and saved artifacts.

## 4. Model Selection Rationale

The primary final model is `random_forest_baseline` because it has the strongest balanced performance among the reviewed final candidates:

- Highest macro F1 among the final reviewed candidates.
- Highest accuracy among the final reviewed candidates.
- Strong balance between good-credit and bad-credit classification performance.
- Interpretable through Random Forest feature importance.

The business-risk alternative is `svm_baseline` because it has the highest bad-class recall. This makes it useful in scenarios where minimizing missed risky applicants is more important than maximizing overall accuracy.

Tuned Random Forest and tuned SVM models were reviewed, but they were not selected because they did not outperform their baseline counterparts on the saved hold-out test metrics.

## 5. Evaluation Metrics

| Model | Accuracy | Macro F1 | Recall Bad | F1 Bad | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| `random_forest_baseline` | 0.7550 | 0.7173 | 0.6500 | 0.6142 | 0.7907 |
| `svm_baseline` | 0.7300 | 0.7104 | 0.7833 | 0.6351 | 0.7933 |
| `svm_tuned` | 0.7050 | 0.6845 | 0.7500 | 0.6040 | 0.7754 |
| `random_forest_tuned` | 0.7150 | 0.6790 | 0.6333 | 0.5714 | 0.7887 |

These metrics are taken from the saved final model selection and tuning comparison reports. No metrics were fabricated or recomputed through retraining.

## 6. Top Feature Importance Signals

Top 10 Random Forest feature-importance signals:

| Feature | Importance |
| --- | ---: |
| `checking_account_status` | 0.105778 |
| `credit_amount_per_month` | 0.088823 |
| `credit_duration_interaction` | 0.074288 |
| `credit_amount` | 0.072269 |
| `age` | 0.063716 |
| `duration_months` | 0.053098 |
| `savings_account` | 0.037987 |
| `employment_duration` | 0.033420 |
| `present_residence_years` | 0.027772 |
| `installment_rate` | 0.026665 |

Feature importance values are model-specific signals from the saved Random Forest baseline. They describe how the trained Random Forest used transformed inputs, but they should not be interpreted as causal effects.

## 7. Business Interpretation

A good-credit prediction indicates the model estimates the applicant is more likely to repay responsibly. A bad-credit prediction indicates elevated credit risk based on the learned historical patterns.

In credit-risk review:

- A false positive can mean a good applicant is incorrectly flagged as risky, potentially leading to unnecessary review or rejection.
- A false negative can mean a risky applicant is incorrectly classified as good, potentially increasing credit losses.

Bad-class recall matters because missed risky applicants can be costly for lenders. This is why `svm_baseline` is documented as a risk-focused alternative, even though `random_forest_baseline` is the primary balanced-performance model.

## 8. Limitations

- The dataset is small, with 1,000 total samples.
- The dataset is historical and may not reflect modern banking behavior or policy.
- The model is not modern banking-grade.
- No fairness audit has been completed.
- No threshold optimization has been completed.
- No probability calibration has been completed.
- Random Forest feature importance is not causal.
- The model is not suitable for automated real-world credit decisions.

## 9. Ethical Considerations

Credit-risk models can encode historical bias and may produce uneven impacts across demographic or socioeconomic groups. Any real-world use would require fairness testing, compliance review, explainability review, and human oversight.

This model should support human decision-making in an educational setting. It should not replace qualified decision-makers or be used for automated credit approval or denial.

## 10. Future Improvements

- Perform threshold optimization for business-specific risk tolerance.
- Calibrate predicted probabilities.
- Add SHAP or permutation-importance explanations.
- Conduct fairness and bias analysis.
- Evaluate on a larger, modern dataset.
- Add deployment monitoring.
- Build the Streamlit inference app.
- Add CI and automated tests.
