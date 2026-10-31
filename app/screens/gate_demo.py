"""Break-the-data demo: damage a COPY of Crew 1's output and watch the gate stop the Flow."""
from pathlib import Path

import streamlit as st

from golden_fighter import config
from golden_fighter.dashboard import data, ui
from golden_fighter.flow.artifacts import sha256

ui.header("הדגמת שבירת נתונים", "תקלה אמיתית: מישהו מעביר את עמודת הכסף מדולרים לסנטים. כל מספר גדל פי 100, "
          "אבל שום דבר לא קורס. בלי בדיקה, המודל היה לומד בשקט נתונים שגויים. כאן **שער הבדיקה** (קוד רגיל, בלי "
          "AI) משווה את הנתונים לחוזה שלהם ועוצר את ה-Flow לפני שמאמנים מודל כלשהו.")


def _fingerprints() -> dict:
    files = sorted(config.ANALYST_DIR.glob("*")) + [config.RAW_DATA_PATH]
    return {f.name: sha256(f) for f in files if f.is_file()}


def render():
    st.subheader("שלב 1: הנתונים האמיתיים היום")
    result = data.gate_result(ui.paths())
    if result.passed:
        st.success(f"הנתונים הנוכחיים עוברים את השער: {result.checks_run} בדיקות, 0 שגיאות.",
                   icon=":material/verified:")
    else:
        st.error(f"הנתונים הנוכחיים נכשלים בשער ({len(result.errors)} שגיאות).", icon=":material/block:")
        ui.explain_gate_issues([str(issue) for issue in result.errors])

    st.subheader("שלב 2: שוברים עותק ומריצים את ה-Flow")
    scenarios = data.break_scenarios()
    scenario = st.selectbox("מה משתבש?", list(scenarios), format_func=lambda s: scenarios[s][0],
                            index=list(scenarios).index("unit_x100"))
    st.markdown(f"**מה קורה בתרחיש הזה:** {scenarios[scenario][1]}")
    st.caption("הקבצים המקוריים אף פעם לא משתנים: הנזק נעשה בעותק בתיקייה `artifacts/_broken/`. ההדגמה לא "
               "משתמשת ב-AI ולא עולה כסף.")
    if st.button("שברו עותק והריצו את ה-Flow", type="primary", icon=":material/bolt:"):
        before = _fingerprints()
        with st.spinner("מעתיקים, שוברים את העותק ומריצים את ה-Flow..."):
            state = data.run_break_demo(scenario)
        after = _fingerprints()
        st.session_state["demo"] = (scenario, state.model_dump(mode="json"), before == after,
                                    before.get(config.RAW_DATA_PATH.name, ""), after.get(config.RAW_DATA_PATH.name, ""))

    if "demo" in st.session_state:
        scenario, summary, untouched, raw_before, raw_after = st.session_state["demo"]
        st.subheader(f"שלב 3: התוצאה עבור \"{scenarios.get(scenario, (scenario,))[0]}\"")
        stopped = summary["status"] != "succeeded"
        cols = st.columns(3)
        cols[0].metric("צוות 2 (מדען נתונים)", "לא התחיל" if stopped else "רץ", border=True,
                       help="צוות 2 מתחיל רק אם השער עבר. לכן לא נבנו פיצ'רים ולא אומן מודל.")
        cols[1].metric("קריאות API בתשלום", summary["totals"].get("model_api_calls", 0), border=True,
                       help="ההדגמה משתמשת בקוד בלבד, בלי סוכני AI, ולכן לא עלתה כסף.")
        cols[2].metric("הנתונים המקוריים", "לא השתנו" if untouched else "השתנו!", border=True,
                       help="טביעת אצבע (SHA-256) של הקובץ הגולמי ושל קבצי צוות 1, לפני ההדגמה ואחריה.")
        ui.show_flow_result(summary)
        if raw_before:
            st.caption(f"טביעת האצבע של קובץ הנתונים הגולמי: לפני `{raw_before[:16]}…` · אחרי `{raw_after[:16]}…`"
                       f" · {'זהות' if raw_before == raw_after else 'שונות!'}")
        report = summary.get("paths", {}).get("failure_report")
        if report:
            path = Path(report) if Path(report).is_absolute() else config.ROOT / report
            with st.expander(f"דוח הכשל המקורי שה-Flow כתב (באנגלית): {Path(report).name}"):
                ui.english_document("failure_report", path.read_text(encoding="utf-8") if path.exists()
                                    else "The report file was removed.")


ui.guarded(render)
