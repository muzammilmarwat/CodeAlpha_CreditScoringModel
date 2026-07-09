# CodeAlpha Credit Scoring Model

## Problem Statement
Credit scoring is the process of evaluating an individual's creditworthiness using past financial and credit-related behavior. This project builds a production-oriented machine learning system that predicts whether an applicant is likely to be a good or risky borrower.

## Objectives
- Explore and understand a credit dataset
- Build reusable preprocessing and feature engineering code
- Train, tune, and compare classification models
- Generate model explainability and final selection reports
- Create a deployment-ready Streamlit app for saved-artifact inference

## Tech Stack
- Python
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib
- streamlit

## Completed Workflow
1. Collect and inspect the dataset
2. Perform exploratory data analysis
3. Build reusable preprocessing and feature engineering code
4. Train baseline machine learning models
5. Tune selected models and compare against baselines
6. Generate explainability and final model selection reports
7. Build a Streamlit app for saved-artifact inference

## Model Documentation
- Final model card: `reports/model_card.md`
- Final model selection report: `reports/final_model_selection/final_model_selection_report.md`
- Testing checklist: `docs/TESTING_CHECKLIST.md`
- Deployment guide: `docs/DEPLOYMENT_GUIDE.md`

## Streamlit App Features
- Applicant credit-risk prediction using saved model artifacts
- Low, medium, and high example applicant profiles
- Professional result card with class probabilities and risk level
- Downloadable prediction summary, model card, and final selection report
- Global Random Forest feature importance view
- Clear educational disclaimer and known limitations

## Repository Structure
```text
CodeAlpha_CreditScoringModel/
|-- app/
|-- data/
|-- docs/
|-- images/
|-- models/
|-- notebooks/
|-- reports/
|-- src/
|-- README.md
|-- requirements.txt
|-- .gitignore
`-- LICENSE
```

## How to Run the Project
1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebooks folder for analysis:
   ```bash
   jupyter notebook
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app/app.py
   ```
5. Run verification checks:
   ```bash
   python -m compileall app src
   python -m app.smoke_test_inference
   ```

## Screenshots
Screenshot placeholders and capture guidance are available in `docs/screenshots/README.md`.

Recommended portfolio screenshots:
- Home page
- Prediction form
- Prediction result
- Model information and explainability
- About project

## Deployment
See `docs/DEPLOYMENT_GUIDE.md` for local and Streamlit deployment guidance.

## Author
Syed Muzammil Shah
