# Golden Fighter: Membership Churn Insights

_Produced by the Data Analyst crew from `clean_data.csv`. All numbers are computed in code._

<!-- agent-summary:start -->
## Analyst summary (written by the EDA & Insights agent)

- **Overall churn:** 26.5% of members churned (1061 of 4000 members).
- **Strongest drivers:** Short contracts and short tenure churn most. Members on 1-month contracts churn at 42.3%, versus 12.5% on 6-month and 2.4% on 12-month contracts. Members in their first 0-1 months churn at 61.4%, and members with 7+ months churn at 0.3%. Under-25s churn at 66.2%.
- **High-risk segment:** 1-month contract holders who joined in the last 0-1 months and do not attend group classes make up 16.1% of members (644 people) and churn at 80.3%. This is the first group to target with onboarding and contract-upgrade offers.
- **Possible levers:** Group-class attendance, friend referral, and partner-company membership are each linked to lower churn (17.3%, 15.8%, and 19.4% when present, versus 33.0%, 31.3%, and 33.3% when absent). These are associations, not proven causes, so test them before investing heavily.
- **Caveat on leakage risk:** avg_class_frequency_current_month is rated high leakage risk (univariate AUC 0.7749). It is measured in the same month as the churn label, so it partly records the churn itself. It was excluded from the primary model and should not be used to predict churn or to set retention targets.
<!-- agent-summary:end -->

## Executive summary

- **1,061 of 4,000 members churned (26.5%).** About one member in four leaves, so retention is a real revenue lever.
- **Contract length is the clearest dividing line.** Churn is 42.3% on 1-month contracts, 12.5% on 6-month and 2.4% on 12-month contracts.
- **The first months decide.** Members in their first 0-1 months churn at 61.4%; after 7+ months it falls to 0.3%.
- **Community keeps people.** Churn is 17.3% for group-class members vs 33.0% for others, and 15.8% for friend referrals vs 31.3%.
- **High-risk segment:** 1-month contract, joined in the last 0-1 months, no group classes. It covers 644 members (16.1%) with a churn rate of 80.3%.

## Churn by segment

| Segment | Members | Churn rate |
|---|---:|---:|
| Contract: 1 month(s) | 2,207 | 42.3% |
| Contract: 6 month(s) | 833 | 12.5% |
| Contract: 12 month(s) | 960 | 2.4% |
| Membership age: 0-1 months | 1,330 | 61.4% |
| Membership age: 2-3 months | 1,100 | 18.8% |
| Membership age: 4-6 months | 876 | 4.0% |
| Membership age: 7+ months | 694 | 0.3% |
| Age: under 25 | 305 | 66.2% |
| Age: 25-29 | 1,824 | 36.0% |
| Age: 30-34 | 1,682 | 11.8% |
| Age: 35+ | 189 | 2.1% |

| Member attribute | Churn if no | Churn if yes |
|---|---:|---:|
| Lives/works near the club | 39.7% | 24.1% |
| Partner-company employee | 33.3% | 19.4% |
| Joined via friend referral | 31.3% | 15.8% |
| Attends group classes | 33.0% | 17.3% |
| Gave a phone number | 26.7% | 26.5% |

## Who churns: average profile

| Measure | Stayed | Churned |
|---|---:|---:|
| `age` | 29.977 | 26.99 |
| `lifetime` | 4.712 | 0.991 |
| `contract_period` | 5.747 | 1.729 |
| `month_to_end_contract` | 5.283 | 1.663 |
| `avg_additional_charges_total` | 158.446 | 115.083 |
| `avg_class_frequency_total` | 2.025 | 1.475 |
| `avg_class_frequency_current_month` | 2.028 | 1.045 |

## Retention levers (recommendations)

1. **Upgrade offers for 1-month members.** Offer a discounted move to a 6- or 12-month contract during the first month, when churn risk is highest.
2. **First-90-days onboarding.** A welcome program (trainer check-in, first group class booked) for new members, since tenure is the strongest early protector.
3. **Push group classes and referrals.** Both are associated with much lower churn, so invite new members to a free group class and reward 'bring a friend'.
4. **Watch for falling attendance.** Churners visit less in the current month than their own average (-0.43 visits/week vs +0.00 for members who stay). An attendance drop is an early-warning trigger for a personal call.

## Caveats for the modeling crew

- `avg_class_frequency_current_month` is measured in the churn month itself, so it partly *records* the churn instead of predicting it. It is excluded from the primary model (see `dataset_contract.json`).
- `gender` shows no relation to churn and is a protected attribute; it is not a model input.
- These are associations, not proven causes. Retention offers should be tested (A/B) before roll-out.

## Strongest legitimate signals (single-column AUC)

- `lifetime`: 0.878
- `age`: 0.761
- `contract_period`: 0.737
- `month_to_end_contract`: 0.736
- `avg_class_frequency_total`: 0.665
