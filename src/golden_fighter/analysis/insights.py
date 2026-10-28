"""Business facts about churn (deterministic numbers) and a baseline insights.md built from them.

In M4 the EDA & Insights agent receives `compute_insight_facts()` through a tool and writes the
business narrative; the numbers always come from here, never from the LLM.
"""
import pandas as pd

from golden_fighter.analysis.leakage import scan_leakage
from golden_fighter.analysis.schema_spec import TARGET

FLAG_LABELS = {
    "near_location": "Lives/works near the club",
    "partner": "Partner-company employee",
    "promo_friends": "Joined via friend referral",
    "group_visits": "Attends group classes",
    "phone": "Gave a phone number",
}
LIFETIME_BINS = [-1, 1, 3, 6, 1000]
LIFETIME_LABELS = ["0-1 months", "2-3 months", "4-6 months", "7+ months"]
AGE_BINS = [0, 24, 29, 34, 200]
AGE_LABELS = ["under 25", "25-29", "30-34", "35+"]
COMPARE_COLUMNS = ["age", "lifetime", "contract_period", "month_to_end_contract", "avg_additional_charges_total",
                   "avg_class_frequency_total", "avg_class_frequency_current_month"]


def _rate_table(df: pd.DataFrame, by: pd.Series) -> list[dict]:
    grouped = df.groupby(by, observed=True)[TARGET].agg(["count", "mean"])
    return [{"group": str(idx), "members": int(row["count"]), "churn_rate": round(float(row["mean"]), 4)}
            for idx, row in grouped.iterrows()]


def compute_insight_facts(df: pd.DataFrame) -> dict:
    churned = df[df[TARGET] == 1]
    stayed = df[df[TARGET] == 0]
    high_risk = (df["contract_period"] == 1) & (df["lifetime"] <= 1) & (df["group_visits"] == 0)
    findings = scan_leakage(df)
    return {
        "members": len(df),
        "churned": len(churned),
        "churn_rate": round(float(df[TARGET].mean()), 4),
        "churn_by_contract_period": _rate_table(df, df["contract_period"]),
        "churn_by_lifetime": _rate_table(df, pd.cut(df["lifetime"], LIFETIME_BINS, labels=LIFETIME_LABELS)),
        "churn_by_age_band": _rate_table(df, pd.cut(df["age"], AGE_BINS, labels=AGE_LABELS)),
        "churn_by_flag": {
            col: {"label": label,
                  "rate_if_no": round(float(df.loc[df[col] == 0, TARGET].mean()), 4),
                  "rate_if_yes": round(float(df.loc[df[col] == 1, TARGET].mean()), 4)}
            for col, label in FLAG_LABELS.items()
        },
        "mean_stayed_vs_churned": {
            col: {"stayed": round(float(stayed[col].mean()), 3), "churned": round(float(churned[col].mean()), 3)}
            for col in COMPARE_COLUMNS
        },
        "visit_change_current_vs_total": {
            "stayed": round(float((stayed["avg_class_frequency_current_month"]
                                   - stayed["avg_class_frequency_total"]).mean()), 3),
            "churned": round(float((churned["avg_class_frequency_current_month"]
                                    - churned["avg_class_frequency_total"]).mean()), 3),
        },
        "high_risk_segment": {
            "definition": "1-month contract, joined in the last 0-1 months, no group classes",
            "members": int(high_risk.sum()),
            "share_of_members": round(float(high_risk.mean()), 4),
            "churn_rate": round(float(df.loc[high_risk, TARGET].mean()), 4) if high_risk.any() else None,
        },
        "leakage_findings": findings,
        "strongest_safe_signals": [
            {"column": f["column"], "univariate_auc": f["univariate_auc"]}
            for f in sorted(findings, key=lambda f: -f["univariate_auc"])
            if f["risk"] != "high" and f["column"] != "gender"
        ][:5],
    }


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def render_insights_md(facts: dict) -> str:
    by_contract = {r["group"]: r for r in facts["churn_by_contract_period"]}
    lifetime = facts["churn_by_lifetime"]
    flags = facts["churn_by_flag"]
    means = facts["mean_stayed_vs_churned"]
    visits = facts["visit_change_current_vs_total"]
    seg = facts["high_risk_segment"]

    lines = [
        "# Golden Fighter: Membership Churn Insights",
        "",
        "_Produced by the Data Analyst crew from `clean_data.csv`. All numbers are computed in code._",
        "",
        "## Executive summary",
        "",
        f"- **{facts['churned']:,} of {facts['members']:,} members churned ({_pct(facts['churn_rate'])}).** "
        "About one member in four leaves, so retention is a real revenue lever.",
        f"- **Contract length is the clearest dividing line.** Churn is {_pct(by_contract['1']['churn_rate'])} on "
        f"1-month contracts, {_pct(by_contract['6']['churn_rate'])} on 6-month and "
        f"{_pct(by_contract['12']['churn_rate'])} on 12-month contracts.",
        f"- **The first months decide.** Members in their first 0-1 months churn at "
        f"{_pct(lifetime[0]['churn_rate'])}; after 7+ months it falls to {_pct(lifetime[-1]['churn_rate'])}.",
        f"- **Community keeps people.** Churn is {_pct(flags['group_visits']['rate_if_yes'])} for group-class "
        f"members vs {_pct(flags['group_visits']['rate_if_no'])} for others, and "
        f"{_pct(flags['promo_friends']['rate_if_yes'])} for friend referrals vs "
        f"{_pct(flags['promo_friends']['rate_if_no'])}.",
        f"- **High-risk segment:** {seg['definition']}. It covers {seg['members']:,} members "
        f"({_pct(seg['share_of_members'])}) with a churn rate of {_pct(seg['churn_rate'])}.",
        "",
        "## Churn by segment",
        "",
        "| Segment | Members | Churn rate |",
        "|---|---:|---:|",
    ]
    for r in facts["churn_by_contract_period"]:
        lines.append(f"| Contract: {r['group']} month(s) | {r['members']:,} | {_pct(r['churn_rate'])} |")
    for r in lifetime:
        lines.append(f"| Membership age: {r['group']} | {r['members']:,} | {_pct(r['churn_rate'])} |")
    for r in facts["churn_by_age_band"]:
        lines.append(f"| Age: {r['group']} | {r['members']:,} | {_pct(r['churn_rate'])} |")

    lines += ["", "| Member attribute | Churn if no | Churn if yes |", "|---|---:|---:|"]
    for f in flags.values():
        lines.append(f"| {f['label']} | {_pct(f['rate_if_no'])} | {_pct(f['rate_if_yes'])} |")

    lines += ["", "## Who churns: average profile", "", "| Measure | Stayed | Churned |", "|---|---:|---:|"]
    for col, m in means.items():
        lines.append(f"| `{col}` | {m['stayed']:g} | {m['churned']:g} |")

    lines += [
        "",
        "## Retention levers (recommendations)",
        "",
        "1. **Upgrade offers for 1-month members.** Offer a discounted move to a 6- or 12-month contract "
        "during the first month, when churn risk is highest.",
        "2. **First-90-days onboarding.** A welcome program (trainer check-in, first group class booked) for new "
        "members, since tenure is the strongest early protector.",
        "3. **Push group classes and referrals.** Both are associated with much lower churn, so invite new members "
        "to a free group class and reward 'bring a friend'.",
        "4. **Watch for falling attendance.** Churners visit less in the current month than their own average "
        f"({visits['churned']:+.2f} visits/week vs {visits['stayed']:+.2f} for members who stay). An attendance "
        "drop is an early-warning trigger for a personal call.",
        "",
        "## Caveats for the modeling crew",
        "",
        "- `avg_class_frequency_current_month` is measured in the churn month itself, so it partly *records* the "
        "churn instead of predicting it. It is excluded from the primary model (see `dataset_contract.json`).",
        "- `gender` shows no relation to churn and is a protected attribute; it is not a model input.",
        "- These are associations, not proven causes. Retention offers should be tested (A/B) before roll-out.",
        "",
        "## Strongest legitimate signals (single-column AUC)",
        "",
    ]
    for s in facts["strongest_safe_signals"]:
        lines.append(f"- `{s['column']}`: {s['univariate_auc']:.3f}")
    return "\n".join(lines) + "\n"
