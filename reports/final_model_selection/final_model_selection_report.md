# Final Model Selection Report

## Executive Summary

The recommended primary model for deployment readiness is `random_forest_baseline`. It has the strongest overall balance among the reviewed candidates, with macro F1 of 0.7173, accuracy of 0.7550, bad-class recall of 0.6500, and ROC-AUC of 0.7907.

The recommended risk-focused alternative is `svm_baseline`. It has the highest bad-class recall among the final candidates at 0.7833, which makes it useful when the business objective prioritizes catching more risky borrowers.

## Models Reviewed

This phase reviewed the saved results for:

- `random_forest_baseline`
- `svm_baseline`
- `random_forest_tuned`
- `svm_tuned`

No model was retrained, and hyperparameter search was not rerun.

## Baseline vs Tuned Results

| model_variant | source | accuracy | macro_f1 | recall_bad | f1_bad | roc_auc |
| --- | --- | --- | --- | --- | --- | --- |
| random_forest_baseline | baseline | 0.7550 | 0.7173 | 0.6500 | 0.6142 | 0.7907 |
| svm_baseline | baseline | 0.7300 | 0.7104 | 0.7833 | 0.6351 | 0.7933 |
| svm_tuned | tuned | 0.7050 | 0.6845 | 0.7500 | 0.6040 | 0.7754 |
| random_forest_tuned | tuned | 0.7150 | 0.6790 | 0.6333 | 0.5714 | 0.7887 |

## Best Overall Model by Macro F1

The best overall candidate by macro F1 is `random_forest_baseline` with macro F1 of 0.7173. Macro F1 is appropriate here because the dataset is class-imbalanced and both good-credit and bad-credit performance matter.

## Best Risk-Detection Model by Bad-Class Recall

The best risk-detection candidate by bad-class recall is `svm_baseline` with bad-class recall of 0.7833. In a credit-risk setting, bad-class recall reflects the model's ability to identify applicants who are more likely to default or represent elevated credit risk.

## Why Tuned Models Were Not Selected

The tuned models did not outperform their baseline counterparts on the saved hold-out test metrics. The best tuned model by macro F1 is `svm_tuned` with macro F1 of 0.6845, while the best baseline model is `random_forest_baseline` with macro F1 of 0.7173.

Because tuning reduced the observed hold-out macro F1 and did not create a clearly superior risk-recall profile, the tuned models are not recommended as the primary final model.

## Final Recommended Model

Primary final model: `random_forest_baseline`

Rationale:

- It has the highest macro F1 among the reviewed final candidates.
- It also has the highest accuracy among the reviewed final candidates.
- It provides a stronger balanced-performance profile than the tuned Random Forest and tuned SVM.
- It remains interpretable enough for portfolio reporting through Random Forest feature importances.

## Business Tradeoff

The primary Random Forest baseline is the best balanced model. It is a good default choice when the project values both classes and wants stable overall classification performance.

The SVM baseline is the risk-focused alternative. It catches more bad-credit cases, but it gives up some overall accuracy and macro F1. In a stricter lending environment, this may be preferable because missing risky borrowers can be more costly than incorrectly flagging some good borrowers.

## Limitations

- The dataset is relatively small, with 1,000 rows.
- The model uses historical German Credit dataset categories, so modern credit-policy deployment would require domain review.
- ROC-AUC is reported from saved model outputs and should be interpreted alongside class-specific recall and precision.
- Feature importance from Random Forest is model-specific and should not be treated as causal explanation.
- No threshold optimization has been performed yet.
- Fairness, bias, and regulatory validation are not complete.

## Next Deployment Recommendation

Package `random_forest_baseline` as the default inference model and keep `svm_baseline` documented as a policy alternative for risk-sensitive review. The next engineering phase should build a single inference workflow that loads the saved feature engineering artifact, applies the selected model, validates inputs, and reports model outputs with clear probability and risk interpretation.

## Deployment Recommendation

Default deployment model: `random_forest_baseline`

Alternative deployment policy: `svm_baseline`

The Random Forest baseline provides the strongest balanced performance among the final reviewed candidates. It is the recommended default model when the deployment objective is to maintain the best overall balance between good-credit and bad-credit classification performance.

The SVM baseline should be considered when minimizing missed high-risk applicants is more important than maximizing overall accuracy. It has stronger bad-class recall, which can be valuable in stricter risk-review settings where false negatives are especially costly.

The deployment pipeline should load:

- `feature_engineer.joblib`
- the selected baseline model artifact
- preprocessing steps inside the saved sklearn pipeline

Predictions should include:

- predicted class
- probability score
- plain-language risk interpretation
- model limitation warning
