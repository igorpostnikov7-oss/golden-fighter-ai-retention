"""Churn insights: who leaves, from the deterministic EDA facts (Crew 1)."""
import streamlit as st

from golden_fighter.dashboard import data, hebrew, ui

ui.header("תובנות על עזיבה", "מי עוזב את המועדון? כל המספרים כאן חושבו על ידי תהליך האנליסט "
          "(`insight_facts.json`). הם מראים **קשר סטטיסטי, לא סיבה מוכחת**.")


def render():
    facts = ui.load("facts")
    segment = facts["high_risk_segment"]
    definition = hebrew.text(hebrew.SEGMENTS, segment["definition"])
    cols = st.columns(4)
    cols[0].metric("מתאמנים", f"{facts['members']:,}")
    cols[1].metric("עזבו את המועדון", f"{facts['churned']:,}", help="מתאמנים שעבורם churn = 1.")
    cols[2].metric("שיעור עזיבה", ui.pct(facts["churn_rate"]))
    cols[3].metric("עזיבה בקבוצת הסיכון הגבוה", ui.pct(segment["churn_rate"]),
                   help=f"{definition}: {segment['members']:,} מתאמנים ({ui.pct(segment['share_of_members'])}).")
    st.info(f"**קבוצת הסיכון הגבוה ביותר:** {definition}. בקבוצה {segment['members']:,} מתאמנים "
            f"({ui.pct(segment['share_of_members'])} מכלל המתאמנים), ו-{ui.pct(segment['churn_rate'])} מהם עזבו.",
            icon=":material/priority_high:")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("לפי תקופת המנוי")
        df = data.churn_by(facts, "churn_by_contract_period", hebrew.CONTRACT_GROUPS)
        st.altair_chart(ui.bar_by_group(df, "תקופת המנוי"), width="stretch")
        st.subheader("לפי גיל")
        st.altair_chart(ui.bar_by_group(data.churn_by(facts, "churn_by_age_band"), "קבוצת גיל"), width="stretch")
    with right:
        st.subheader("לפי הזמן מאז ההצטרפות")
        st.altair_chart(ui.bar_by_group(data.churn_by(facts, "churn_by_lifetime"), "חודשים מאז ההצטרפות"),
                        width="stretch")
        st.subheader("גורמי כן / לא")
        st.altair_chart(ui.bar_yes_no(data.churn_by_flags(facts)), width="stretch")

    st.subheader("עמודה שאנחנו בכוונה לא משתמשים בה")
    leaky = [f for f in facts["leakage_findings"] if f["risk"] == "high"]
    for finding in leaky:
        reason = hebrew.LEAKAGE_REASONS.get(finding["column"], finding["reason"])
        st.warning(f"**`{finding['column']}`** (ROC-AUC של העמודה לבדה: {finding['univariate_auc']:.3f}): "
                   f"{reason}", icon=":material/visibility_off:")

    with st.expander("דוח ה-EDA המלא (גרפים וסטטיסטיקה, באנגלית)"):
        report = ui.paths().analyst_dir / "eda_report.html"
        if report.exists():
            st.iframe(report, height=900)  # written by our own deterministic EDA code, not user or AI input
        else:
            st.info("הקובץ `eda_report.html` לא זמין.")
    with st.expander("insights.md (התובנות שהאנליסט כתב, באנגלית)"):
        ui.english_document("insights_md", data.document(ui.paths().analyst_dir / "insights.md"))


ui.guarded(render)
