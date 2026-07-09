# CodeAlpha Credit Scoring Model v1.0.0

## Highlights

- Production-style credit-risk classification workflow using the German Credit Dataset.
- Final model decision documented with Random Forest Baseline as the primary model.
- Streamlit deployment app completed through RC-2 UI polish.
- Automated tests and GitHub Actions CI workflow added.
- Model card, final model selection report, deployment guide, testing checklist, and repository audits prepared.

## Features

- Dataset inspection and exploratory data analysis.
- Reusable preprocessing architecture.
- Feature engineering with saved fitted artifact.
- Baseline model comparison across five classifiers.
- Hyperparameter tuning review for Random Forest and SVM.
- Saved model artifacts for baseline and tuned models.
- Random Forest global feature importance.
- Interactive Streamlit app for saved-artifact prediction.
- Example applicants, risk interpretation, probability display, and downloadable summaries.
- Pytest suite for deployment-layer behavior.

## ML Pipeline Summary

The project follows a clean ML engineering flow:

1. German Credit Dataset ingestion.
2. Exploratory data analysis.
3. Preprocessing pipeline design.
4. Feature engineering.
5. Train/test split.
6. Baseline model training and comparison.
7. Hyperparameter tuning.
8. Final model selection.
9. Saved deployment artifacts.
10. Streamlit prediction app.

## Streamlit Deployment Summary

The app uses saved artifacts only. It does not retrain models, rerun GridSearchCV, or overwrite `.joblib` files during normal execution.

Deployment entrypoint:

```bash
streamlit run app/app.py
```

## Testing and CI Summary

Local verification commands:

```bash
python -m compileall app src
python -m app.smoke_test_inference
pytest
```

Current automated test result:

- 12 tests passed.

GitHub Actions CI runs compile checks, saved-artifact smoke inference, and pytest.

## Final Selected Model

- Primary final model: `random_forest_baseline`
- Business-risk alternative: `svm_baseline`

Random Forest Baseline was selected for the best balanced macro F1. SVM Baseline is retained as the risk-focused alternative because it achieved the strongest bad-class recall.

## Key Metrics

| Model Variant | Accuracy | Macro F1 | Recall Bad | ROC-AUC |
|---|---:|---:|---:|---:|
| `random_forest_baseline` | 0.7550 | 0.7173 | 0.6500 | 0.7907 |
| `svm_baseline` | 0.7300 | 0.7104 | 0.7833 | 0.7933 |
| `svm_tuned` | 0.7050 | 0.6845 | 0.7500 | 0.7754 |
| `random_forest_tuned` | 0.7150 | 0.6790 | 0.6333 | 0.7887 |

## Known Limitations

- Dataset is small and historical.
- No fairness or compliance audit has been completed.
- No probability calibration has been completed.
- No decision-threshold optimization has been completed.
- Global feature importance is not an individual causal explanation.
- The project is educational and not production lending software.

## Installation

```bash
git clone https://github.com/<your-username>/CodeAlpha_CreditScoringModel.git
cd CodeAlpha_CreditScoringModel
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py
```

## Future Work

- Fairness and bias analysis.
- Probability calibration.
- Decision-threshold optimization.
- Local explanations for individual predictions.
- Monitoring and drift checks.
- Cloud deployment.
- Evaluation on newer credit-risk datasets.
