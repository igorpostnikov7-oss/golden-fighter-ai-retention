"""Break-the-data demo: damage a COPY of Crew 1's output and watch the gate stop the Flow."""
from pathlib import Path

import streamlit as st

from golden_fighter import config
from golden_fighter.dashboard import data, ui
from golden_fighter.flow.artifacts import sha256

ui.header("הדגמת שבירת נתונים", "תקלה אמיתית: מישהו מעביר את עמודת הכסף מדולרים לסנטים. כל מספר גדל פי 100, "
          "אבל שום דבר לא קורס. בלי בדיקה, המודל היה לומד בשקט שטויות. כאן **שער הבדיקה** (Python רגיל, בלי AI) "
          "משווה את הנתונים לחוזה שלהם ועוצר את ה-Flow לפני שמאמנים מודל כלשהו.")


def _fingerprints() -> dict:
    files = sorted(config.ANALYST_DIR.glob("*")) + [config.RAW_DATA_PATH]
    return {f.name: sha256(f) for f in files if f.is_file()}


def render():
    st.subheader("1. הנתונים האמיתיים היום")
    result = data.gate_result(ui.paths())
    if result.passed:
        st.success(f"הנתונים הנוכחיים עוברים את השער: {result.checks_run} בדיקות, 0 שגיאות.",
                   icon=":material/verified:")
    else:
        st.error(f"הנתונים הנוכחיים נכשלים בשער ({len(result.errors)} שגיאות).", icon=":material/block:")
        for issue in result.errors:
            st.markdown(f"- `{issue}`")

    st.subheader("2. שוברים עותק ומריצים את ה-Flow")
    scenarios = data.break_scenarios()
    scenario = st.selectbox("מה משתבש?", list(scenarios), format_func=lambda s: f"{scenarios[s]} ({s})",
                            index=list(scenarios).index("unit_x100"))
    st.caption("הקבצים המקוריים אף פעם לא משתנים: הנזק נעשה בעותק בתיקייה `artifacts/_broken/`. "
               "ההדגמה לא משתמשת ב-AI ולא עולה כסף.")
    if st.button("שברו עותק והריצו את ה-Flow", type="primary"):
        before = _fingerprints()
        with st.spinner("מעתיקים, שוברים את העותק ומריצים את ה-Flow..."):
            state = data.run_break_demo(scenario)
        st.session_state["demo"] = (scenario, state.model_dump(mode="json"), before == _fingerprints())

    if "demo" in st.session_state:
        scenario, summary, untouched = st.session_state["demo"]
        st.subheader(f"3. התוצאה עבור: {scenarios.get(scenario, scenario)}")
        ui.show_flow_result(summary)
        cols = st.columns(3)
        cols[0].metric("קריאות API למודל", summary["totals"].get("model_api_calls", 0))
        cols[1].metric("אומן מודל", "לא" if summary["status"] != "succeeded" else "כן")
        cols[2].metric("הנתונים המקוריים לא השתנו", "כן" if untouched else "לא!")
        report = summary.get("paths", {}).get("failure_report")
        if report:
            path = Path(report) if Path(report).is_absolute() else config.ROOT / report
            with st.expander(f"דוח הכשל שה-Flow כתב (באנגלית): {Path(report).name}"):
                ui.english_document("failure_report", path.read_text(encoding="utf-8") if path.exists()
                                    else "The report file was removed.")


ui.guarded(render)
