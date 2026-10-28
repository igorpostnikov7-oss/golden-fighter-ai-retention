# Evaluation Report: Golden Fighter Churn Model

_Produced by the Data Scientist crew. Numbers come from `evaluation.json` (seed 42, scikit-learn 1.9.1)._

## Summary

- **Selected model: Logistic Regression.** Rule: highest mean 5-fold CV roc_auc on the training set (tie-break: CV PR-AUC); the test set was not used for selection.
- On the held-out test set (800 members never seen in training) it reaches **ROC-AUC 0.957** and catches **90% of churners** (191 of 212), with precision 74%.
- The majority baseline scores ROC-AUC 0.50 and catches 0% of churners, so the model adds real value.

## Setup

- Data: `clean_data.csv` validated against `dataset_contract.json` v1.1 (the validation gate ran before features were built and again before training).
- Split: stratified 3200 train / 800 test; churn rate 26.5% train vs 26.5% test.
- Validation: 5-fold stratified cross-validation on the **training set only**. Scaling and feature engineering sit inside the model pipeline, so they are fitted on each training fold, never on validation or test rows.
- Decision threshold: 0.5 on the predicted churn probability (classes are weighted, so the model is tuned to find churners rather than to maximise accuracy).
- Model inputs (11): `near_location`, `partner`, `promo_friends`, `phone`, `contract_period`, `group_visits`, `age`, `avg_additional_charges_total`, `month_to_end_contract`, `lifetime`, `avg_class_frequency_total`.
- Engineered inside the pipeline: `contract_elapsed_share`, `is_new_member`, `social_ties`.
- Never used as inputs: `churn`, `member_id`, `gender` (target, identifier, protected attribute) and `avg_class_frequency_current_month` (not known at prediction time).

## Model comparison: cross-validation (used for selection)

| Model | ROC-AUC | PR-AUC | Recall | Precision | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (majority) | 0.500 ± 0.000 | 0.265 | 0.000 | 0.000 | 0.000 | 0.735 |
| Logistic Regression **(selected)** | 0.957 ± 0.007 | 0.888 | 0.912 | 0.720 | 0.805 | 0.882 |
| Random Forest | 0.954 ± 0.008 | 0.888 | 0.897 | 0.743 | 0.813 | 0.890 |

## Model comparison: held-out test set (reported after selection)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (majority) | 0.735 | 0.000 | 0.000 | 0.000 | 0.500 | 0.265 |
| Logistic Regression | 0.891 | 0.743 | 0.901 | 0.815 | 0.957 | 0.900 |
| Random Forest | 0.889 | 0.747 | 0.877 | 0.807 | 0.948 | 0.883 |

### Why this model

Logistic Regression had the best mean cross-validation ROC-AUC. The margin over Random Forest is 0.002, smaller than the fold-to-fold variation (± 0.007), so the two models perform about equally well; the simpler, more explainable model wins the tie in practice. The test set confirms the choice without having driven it.

## Confusion matrix: Logistic Regression (test set, threshold 0.5)

| | Predicted: stays | Predicted: churns |
|---|---:|---:|
| **Actually stayed** | 522 (true negative) | 66 (false alarm) |
| **Actually churned** | 21 (missed) | 191 (caught) |

Out of 212 members who actually churned, the model flagged 191 and missed 21. It also flagged 66 members who stayed (false alarms); for a retention call or a discount offer, a false alarm costs far less than a missed churner.

## Leakage ablation (not deployable)

To show why the contract excludes `avg_class_frequency_current_month`, the same Logistic Regression was retrained **with** it:

| Variant | CV ROC-AUC | Test ROC-AUC | Test recall |
|---|---:|---:|---:|
| Primary model (contract features) | 0.957 | 0.957 | 0.901 |
| + current-month visits (leaky) | 0.976 | 0.977 | 0.910 |

The leaky variant looks better (+0.020 ROC-AUC) only because current-month visits are recorded during the month the member is already leaving. At the start of the month, when the retention team must act, that value does not exist yet. This is exactly the kind of silently inflated score the validation gate is designed to prevent.

## What drives the predictions (permutation importance on the test set)

| Input column | ROC-AUC drop when shuffled |
|---|---:|
| `lifetime` | 0.210 ± 0.018 |
| `contract_period` | 0.155 ± 0.010 |
| `month_to_end_contract` | 0.045 ± 0.006 |
| `age` | 0.036 ± 0.004 |
| `avg_class_frequency_total` | 0.012 ± 0.003 |
| `avg_additional_charges_total` | 0.008 ± 0.002 |
| `group_visits` | 0.005 ± 0.001 |
| `promo_friends` | 0.002 ± 0.001 |
| `phone` | 0.000 ± 0.000 |
| `partner` | 0.000 ± 0.000 |

## Fairness check (test set)

Gender is not a model input; it is used here only to check that the model treats groups alike.

**By gender**

| Group | Members | Actual churn | Flagged as at-risk | Recall | Precision |
|---|---:|---:|---:|---:|---:|
| 0 | 389 | 26.0% | 31.9% | 0.891 | 0.726 |
| 1 | 411 | 27.0% | 32.4% | 0.910 | 0.759 |

**By age band**

| Group | Members | Actual churn | Flagged as at-risk | Recall | Precision |
|---|---:|---:|---:|---:|---:|
| under 25 | 59 | 67.8% | 71.2% | 0.975 | 0.929 |
| 25-29 | 377 | 34.5% | 41.4% | 0.923 | 0.769 |
| 30-34 | 328 | 12.8% | 18.0% | 0.762 | 0.542 |
| 35+ | 36 | 0.0% | 0.0% | n/a | n/a |

## Sanity checks

- No model reached a suspiciously perfect ROC-AUC (> 0.99) in cross-validation or on the test set.
- Train and test churn rates match (stratified split).
