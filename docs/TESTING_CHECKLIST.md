# Testing Checklist

Use this checklist before tagging a release candidate or publishing the project portfolio.

## Environment

- [ ] Create and activate the virtual environment.
- [ ] Install dependencies with `pip install -r requirements.txt`.
- [ ] Confirm Python can import the app and source packages.

## Static Checks

```bash
python -m compileall app src
```

- [ ] `app/` compiles successfully.
- [ ] `src/` compiles successfully.

## Inference Smoke Test

```bash
python -m app.smoke_test_inference
```

- [ ] Saved preprocessing artifact loads.
- [ ] Saved baseline Random Forest model loads.
- [ ] Prediction service returns class probabilities.
- [ ] No model training or tuning is triggered.

## Streamlit App

```bash
streamlit run app/app.py
```

- [ ] App starts without import errors.
- [ ] Prediction page renders.
- [ ] Low, medium, and high example applicants can be applied.
- [ ] Reset button restores the default form state.
- [ ] Prediction result card renders after submission.
- [ ] Download buttons generate prediction summaries.
- [ ] Model Information page renders feature importance.
- [ ] About Project page renders limitations and report downloads.

## Artifact Safety

- [ ] Baseline model artifacts are not overwritten.
- [ ] Tuned model artifacts are not overwritten.
- [ ] GridSearchCV is not rerun by notebooks or app code.
- [ ] Streamlit app uses saved artifacts only.

## Documentation

- [ ] README includes app usage instructions.
- [ ] Model card is present at `reports/model_card.md`.
- [ ] Final model selection report is present.
- [ ] Screenshots are captured or placeholders are documented.
