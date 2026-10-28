# Model Card: Golden Fighter Churn Predictor (Logistic Regression)

<!-- agent-summary:start -->
## Summary for the retention team (written by the Model Card agent)

- **What it predicts:** The model scores each member's likelihood of churning, so the retention team can decide who to offer support to. It is not a verdict on any member.
- **How well it finds churners:** The selected logistic regression model catches 90.1% of actual churners on the test set (recall), and 74.3% of the members it flags do churn (precision). [source: train_and_compare_models]
- **Fairness result:** Recall differs by group. For gender it is 89.1% for group 0 and 91.0% for group 1. For age bands it ranges from 76.2% (30-34) to 97.5% (under 25), and the 35+ band has no churners in the evaluation, so recall is not reported for it. Review these gaps before acting on scores for any group. [source: write_model_reports]
- **Main limitation:** The higher-scoring model that uses avg_class_frequency_current_month is leakage and was rejected, because that feature partly records the churn itself. Its scores must not be quoted as performance. [source: train_and_compare_models]
- **Allowed use:** Use scores only to prioritise retention offers and extra help. Never use them to penalise, restrict, or lower the service level of any member.
<!-- agent-summary:end -->

## Model details

- **Type:** scikit-learn pipeline: row-wise feature engineering -> standard scaling -> logistic regression (balanced class weights).
- **Artifact:** `model.joblib` (bundle with the pipeline, input list, threshold 0.5 and the input ranges from the contract). Load with `golden_fighter.modeling.predict.load_model`.
- **Version inputs:** dataset contract v1.1, clean data SHA-256 `e792eb7a4f35a5c9...`, seed 42, scikit-learn 1.9.1.
- **Owner:** Golden Fighter AI course project (Data Scientist crew).

## Intended use

- Rank current gym members by their probability of leaving at the start of a month, so the retention team can prioritise personal calls, onboarding help or contract-upgrade offers.
- Decision support for humans. A person decides what to offer.

## Out of scope (do not use for)

- Penalising members, raising prices, or refusing service based on a churn score.
- Any decision about individuals based on gender or age.
- Other clubs, countries or time periods without re-validating on their data.

## Training data

- Public Kaggle dataset "Gym customers features and churn": 4,000 members, one row each, churn rate 26.5%. No names, contact details or other direct identifiers.
- Cleaned and documented by the Data Analyst crew; accepted only after the validation gate passed.
- Stratified split: 3,200 training / 800 test members.

## Features

Inputs (all known at the start of the month): `near_location`, `partner`, `promo_friends`, `phone`, `contract_period`, `group_visits`, `age`, `avg_additional_charges_total`, `month_to_end_contract`, `lifetime`, `avg_class_frequency_total`.

Engineered inside the model:

- `contract_elapsed_share`: Share of the current contract already used: (contract_period - months left) / contract_period. 0 = just signed, close to 1 = about to expire.
- `is_new_member`: 1 if the member joined 0-1 months ago (the highest-churn period in the EDA).
- `social_ties`: Number of social links to the club: partner company + friend referral + group classes (0-3).

Deliberately excluded: `churn` (the answer), `member_id` (no meaning), `gender` (protected attribute) and `avg_class_frequency_current_month` (measured during the churn month: leakage).

## Performance

| Metric | Cross-validation (train) | Test (held out) |
|---|---:|---:|
| Accuracy | 0.882 | 0.891 |
| Precision | 0.720 | 0.743 |
| Recall | 0.912 | 0.901 |
| F1 | 0.805 | 0.815 |
| ROC-AUC | 0.957 | 0.957 |
| PR-AUC | 0.888 | 0.900 |

On the test set the model catches 191 of 212 churners and raises 66 false alarms among 588 members who stayed. See `evaluation_report.md` for the full comparison with the baseline and the Random Forest.

## Fairness

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

Recall differs by only 0.019 between the two gender codes (within the 0.05 tolerance), so churners are found equally well in both groups. Across age bands, the share of members flagged follows the actual churn rate of each band: younger members churn far more often in this data, so they are flagged more often. Use the score to offer help, never to treat any age group worse.

## Limitations

- **A single snapshot:** one chain, one period of unknown date, 4,000 members. Behaviour may differ elsewhere or later.
- **Associations, not causes:** a long contract goes with low churn, but forcing long contracts will not necessarily keep people. Test retention actions (A/B) before rolling them out.
- **Tenure dominates:** `lifetime` and `contract_period` carry most of the signal; the model is weakest at telling apart established members, where churn is rare.
- **Threshold:** 0.5 favours catching churners (high recall) over precision. A different budget for retention offers may call for a different threshold.
- **Assumed data timing:** `avg_class_frequency_total` is assumed to exclude the churn month. If it does not, it would also need to be excluded.
- **Units:** money is in the source's currency units; the contract and the gate reject rescaled data.

## Ethical considerations

- Built only from a public, anonymised dataset; not connected to any production CRM.
- Gender is excluded from the inputs and used only for the fairness check above.
- Age is an input because it is strongly related to churn here. Its use is limited to offering support, and its effect should be reviewed before any real deployment.
- Scores should be explained to staff as probabilities with uncertainty, not as facts about a person.
- Retrain and re-check fairness whenever the dataset contract changes.
