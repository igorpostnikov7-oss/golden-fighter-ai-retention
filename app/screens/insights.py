"""Churn insights: who leaves, from the deterministic EDA facts (Crew 1), plus the Hebrew view of the EDA report."""
import streamlit as st

from golden_fighter.dashboard import data, hebrew, hebrew_reports, ui

ui.header("תובנות על עזיבה", "מי עוזב את המועדון? כל המספרים כאן חושבו על ידי תהליך האנליסט (צוות 1). הם מראים "
          "**קשר סטטיסטי, לא סיבה מוכחת**.")
st.info(ui.DATA_NOTE, icon=":material/database:")

DISTRIBUTION_COLUMNS = ["lifetime", "age", "avg_additional_charges_total", "avg_class_frequency_total",
                        "contract_period", "month_to_end_contract"]


def render():
    facts = ui.load("facts")
    segment = facts["high_risk_segment"]
    definition = hebrew.text(hebrew.SEGMENTS, segment["definition"])
    cols = st.columns(4)
    cols[0].metric("מתאמנים", f"{facts['members']:,}", border=True)
    cols[1].metric("עזבו את המועדון", f"{facts['churned']:,}", border=True)
    cols[2].metric("שיעור עזיבה", ui.pct(facts["churn_rate"]), border=True)
    cols[3].metric("עזיבה בקבוצת הסיכון הגבוה", ui.pct(segment["churn_rate"]), border=True,
                   help=f"{definition}: {segment['members']:,} מתאמנים.")
    st.warning(f"**קבוצת הסיכון הגבוה ביותר:** {definition}. בקבוצה {segment['members']:,} מתאמנים "
               f"({ui.pct(segment['share_of_members'])} מכלל המתאמנים), ו-**{ui.pct(segment['churn_rate'])} מהם "
               "עזבו**. זו הקבוצה הראשונה לטפל בה: ליווי בהתחלה והצעה למנוי ארוך יותר.",
               icon=":material/priority_high:")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("לפי אורך המנוי")
        df = data.churn_by(facts, "churn_by_contract_period", hebrew.CONTRACT_GROUPS)
        st.altair_chart(ui.bar_by_group(df, "אורך המנוי (חודשים)"), width="stretch")
        st.subheader("לפי גיל")
        st.altair_chart(ui.bar_by_group(data.churn_by(facts, "churn_by_age_band"), "קבוצת גיל"), width="stretch")
    with right:
        st.subheader("לפי ותק במועדון")
        st.altair_chart(ui.bar_by_group(data.churn_by(facts, "churn_by_lifetime"), "ותק במועדון"),
                        width="stretch")
        st.subheader("מאפיינים של כן / לא")
        st.altair_chart(ui.bar_yes_no(data.churn_by_flags(facts)), width="stretch")

    st.subheader("עמודה שאנחנו בכוונה לא משתמשים בה")
    for finding in (f for f in facts["leakage_findings"] if f["risk"] == "high"):
        with st.container(border=True):
            st.markdown(f"##### ⚠️ {hebrew.column(finding['column'])}")
            st.markdown(hebrew.LEAKAGE_REASONS.get(finding["column"], finding["reason"]))
            st.caption(f"שם העמודה בקוד: `{finding['column']}` · AUC של העמודה לבדה: "
                       f"{finding['univariate_auc']:.3f}. הסיבה להוצאה היא **התזמון** (מתי הערך נמדד), לא גובה "
                       "ה-AUC.")

    st.subheader("דוח ה-EDA המלא")
    st.caption("גרסה עברית של `eda_report.html`, מחולקת ללשוניות. המספרים מחושבים באותן פונקציות ומאותם קבצים "
               "כמו הדוח המקורי באנגלית, שנמצא בלשונית האחרונה.")
    tabs = st.tabs(["איכות הנתונים", "סטטיסטיקה תיאורית", "התפלגויות", "מתאמים", "סריקת דליפה",
                    "מסמך התובנות", "המקור באנגלית"])
    # Each tab is guarded on its own: on a fresh clone clean_data.csv (member-level rows) does not exist yet.
    with tabs[0]:
        ui.guarded(lambda: _quality(ui.load("cleaning")))
    with tabs[1]:
        ui.guarded(_statistics)
    with tabs[2]:
        ui.guarded(lambda: _distributions(facts))
    with tabs[3]:
        ui.guarded(_correlations)
    with tabs[4]:
        st.altair_chart(ui.auc_chart(data.leakage_table(facts)), width="stretch")
        st.caption("כמה טוב כל עמודה, לבדה, מבדילה בין עוזבים לנשארים. אף עמודה לא מנבאת באופן מושלם וחשוד. "
                   "העמודה באדום סומנה בגלל התזמון שלה (נמדדת בחודש העזיבה), לא בגלל ה-AUC.")
    with tabs[5]:
        st.markdown(hebrew_reports.insights_document(facts))
        summary = data.agent_section(ui.paths().analyst_dir / "insights.md")
        if summary:
            with st.expander("הטקסט המקורי שכתב סוכן ה-AI (באנגלית)"):
                ui.english_document("insights_agent", summary)
    with tabs[6]:
        report = ui.paths().analyst_dir / "eda_report.html"
        if report.exists():
            st.caption("הדוח המקורי באנגלית, כפי שצוות 1 כתב אותו (תוצר רשמי של הפרויקט).")
            st.iframe(report, height=700)  # written by our own deterministic EDA code, not user or AI input
        else:
            st.info("הקובץ `eda_report.html` לא זמין.")
        with st.expander("insights.md המקורי (באנגלית)"):
            ui.english_document("insights_md", data.document(ui.paths().analyst_dir / "insights.md"))


def _statistics() -> None:
    ui.table(data.describe_table(ui.load("clean_data")))
    st.caption("לכל עמודה: ממוצע, סטיית תקן, מינימום, חציון ומקסימום (מעוגל ל-2 ספרות).")


def _distributions(facts: dict) -> None:
    choice = st.selectbox("איזו עמודה להציג?", DISTRIBUTION_COLUMNS, format_func=hebrew.column)
    st.altair_chart(ui.distribution_chart(ui.load("clean_data"), choice), width="stretch")
    st.markdown("**הממוצע אצל מי שנשאר לעומת מי שעזב**")
    ui.table(data.mean_profile(facts), {"ממוצע אצל מי שנשאר": "{:g}", "ממוצע אצל מי שעזב": "{:g}"})
    st.caption("עוזבים חדשים יותר, צעירים יותר ומוציאים פחות על שירותים נוספים.")


def _correlations() -> None:
    st.altair_chart(ui.correlation_heatmap(data.correlation_long(ui.load("clean_data"))), width="stretch")
    st.caption("אדום = שתי העמודות עולות יחד; כחול = כשאחת עולה השנייה יורדת; קרוב ל-0 = אין קשר. אורך המנוי "
               "וחודשים עד סוף המנוי קשורים מאוד (הם מתארים את אותו מנוי), וכך גם שתי עמודות הביקורים.")


def _quality(report: dict) -> None:
    dropped = (report["dropped_non_numeric_or_null"] + report["dropped_duplicates"]
               + sum(report["dropped_by_rule"].values()))
    cols = st.columns(3)
    cols[0].metric("שורות בקובץ הגולמי", f"{report['rows_in']:,}", border=True)
    cols[1].metric("שורות אחרי הניקוי", f"{report['rows_out']:,}", border=True)
    cols[2].metric("שורות שהוסרו", f"{dropped:,}", border=True)
    rules = ", ".join(f"{hebrew.column(k)} ({v})" for k, v in report["dropped_by_rule"].items()) or "אין"
    st.markdown(f"- ערכים חסרים או לא מספריים שהוסרו: {report['dropped_non_numeric_or_null']:,}\n"
                f"- שורות כפולות שהוסרו: {report['dropped_duplicates']:,}\n"
                f"- הפרות של כללי התחום שהוסרו: {rules}\n"
                "- שמות העמודות הומרו לפורמט אחיד, ונוספה עמודת מזהה (`member_id`) לפי מספר השורה, למעקב.")


ui.guarded(render)
