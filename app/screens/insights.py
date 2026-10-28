"""Churn insights: who leaves, from the deterministic EDA facts (Crew 1)."""
import streamlit as st

from golden_fighter.dashboard import data, ui

ui.header("Churn insights", "Who leaves the club? All numbers below were computed by the Data Analyst pipeline "
          "(`insight_facts.json`). They show **associations, not proven causes**.")


def render():
    facts = ui.load("facts")
    segment = facts["high_risk_segment"]
    cols = st.columns(4)
    cols[0].metric("Members", f"{facts['members']:,}")
    cols[1].metric("Left the club", f"{facts['churned']:,}", help="Members with churn = 1.")
    cols[2].metric("Churn rate", ui.pct(facts["churn_rate"]))
    cols[3].metric("High-risk group churn", ui.pct(segment["churn_rate"]),
                   help=f"{segment['definition']}: {segment['members']:,} members ({ui.pct(segment['share_of_members'])}).")
    st.info(f"**Highest-risk group:** {segment['definition']}. It holds {segment['members']:,} members "
            f"({ui.pct(segment['share_of_members'])} of all) and {ui.pct(segment['churn_rate'])} of them left.",
            icon=":material/priority_high:")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("By contract length")
        df = data.churn_by(facts, "churn_by_contract_period")
        df["group"] = df["group"] + " month(s)"
        st.altair_chart(ui.bar_by_group(df, "Contract length"), width="stretch")
        st.subheader("By age")
        st.altair_chart(ui.bar_by_group(data.churn_by(facts, "churn_by_age_band"), "Age band"), width="stretch")
    with right:
        st.subheader("By time since joining")
        st.altair_chart(ui.bar_by_group(data.churn_by(facts, "churn_by_lifetime"), "Months since joining"),
                        width="stretch")
        st.subheader("Yes / no factors")
        st.altair_chart(ui.bar_yes_no(data.churn_by_flags(facts)), width="stretch")

    st.subheader("A column we deliberately do not use")
    leaky = [f for f in facts["leakage_findings"] if f["risk"] == "high"]
    for finding in leaky:
        st.warning(f"**`{finding['column']}`** (single-column ROC-AUC {finding['univariate_auc']:.3f}): "
                   f"{finding['reason']}", icon=":material/visibility_off:")

    with st.expander("Full EDA report (charts and statistics)"):
        report = ui.paths().analyst_dir / "eda_report.html"
        if report.exists():
            st.iframe(report, height=900)  # written by our own deterministic EDA code, not user or AI input
        else:
            st.info("eda_report.html is not available.")
    with st.expander("insights.md (the analyst's written insights)"):
        st.markdown(data.document(ui.paths().analyst_dir / "insights.md"))


ui.guarded(render)
