"""Flow runs & costs: what the CrewAI Flow did in each run, the agents, model API calls, tokens and cost."""
import pandas as pd
import streamlit as st

from golden_fighter.dashboard import data, hebrew, ui

ui.header("ריצות ועלויות", "ה-Flow (תהליך העבודה של CrewAI) מחבר את שני הצוותים דרך שער הבדיקה. כל ריצה שומרת "
          "קובץ סיכום (`run_summary.json`), והמסך הזה קורא אותם.")

AGENT_COLUMNS = ["סוכן", "מטרה", "כלים", "שם ב-CrewAI"]


def render():
    st.subheader("מי עושה מה")
    agents = data.agents_overview()
    crew1, crew2 = agents[agents["צוות"] == hebrew.CREWS["analyst_crew"]], agents[
        agents["צוות"] == hebrew.CREWS["scientist_crew"]]
    with st.container(border=True):
        st.markdown(f"##### 1. {hebrew.CREWS['analyst_crew']} (3 סוכני AI)")
        ui.table(crew1[AGENT_COLUMNS])
    with st.container(border=True):
        st.markdown("##### 2. שער הבדיקה (בלי AI)")
        st.markdown("קוד Python רגיל, לא סוכן. הוא משווה את הקבצים של צוות 1 לחוזה הנתונים (עמודות, סוגי נתונים, "
                    "ערכים חסרים, טווחים, יחידות מידה ועוד). **אם אפילו בדיקה אחת נכשלת, ה-Flow נעצר וצוות 2 לא "
                    "מתחיל.** אף AI לא יכול לשנות את ההחלטה הזו.")
    with st.container(border=True):
        st.markdown(f"##### 3. {hebrew.CREWS['scientist_crew']} (3 סוכני AI)")
        ui.table(crew2[AGENT_COLUMNS])
    st.caption("הסוכנים מחליטים איזה כלי להפעיל וכותבים את ההסברים. הכלים הם קוד Python נבדק, ולכן המספרים אף "
               "פעם לא תלויים ב-AI.")

    st.subheader("מה עולה כסף ומה חינם")
    paid, free = st.columns(2, gap="large")
    with paid.container(border=True):
        st.markdown("##### 💳 קריאות API בתשלום")
        st.markdown("בקשות ששולחים ל-Claude (מודל השפה). הן צורכות טוקנים ועולות כסף. קורות **רק** בריצה מלאה "
                    "עם AI, שמופעלת מהטרמינל.")
    with free.container(border=True):
        st.markdown("##### 🖥️ הפעלות כלים מקומיות (חינם)")
        st.markdown("הפעמים שסוכן הפעיל כלי Python של הפרויקט על המחשב (ניקוי, EDA, חוזה, אימון). לא עולות "
                    "כסף, ותמיד נותנות אותה תוצאה. ריצות בלי AI והדגמות שבירה משתמשות רק בקוד, בלי סוכנים.")

    p = ui.paths()
    history = data.run_history(p)
    st.subheader("היסטוריית ריצות")
    if history.empty:
        st.info("עדיין לא נרשמו ריצות. אפשר להריץ את ה-Flow בחינם עם הכפתור למטה.")
    else:
        full = history[history[data.KIND] == data.FULL_RUNS]
        cols = st.columns(4)
        cols[0].metric("ריצות מלאות עם AI", len(full), border=True)
        cols[1].metric("קריאות API בתשלום (סה\"כ)", int(full[data.API_CALLS].sum()), border=True)
        cols[2].metric("עלות API משוערת (סה\"כ)", f"${full[data.COST].sum():.4f}", border=True)
        cols[3].metric("ריצות חינמיות והדגמות", len(history) - len(full), border=True)
        shown = history.drop(columns=[data.RUN])
        ui.table(shown, {data.COST: "${:.4f}", data.TOKENS: "{:,}", "שניות": "{:.1f}"})
        if len(full) > 0:
            st.markdown("**עלות כל ריצה מלאה עם AI**")
            st.altair_chart(ui.cost_bars(full), width="stretch")

        st.subheader("פרטי ריצה")
        summaries = {s["run_id"]: (kind, s) for kind, s in data.run_summaries(p)}
        when = dict(zip(history[data.RUN], history[data.STARTED]))
        run_id = st.selectbox("בחרו ריצה", list(summaries), format_func=lambda r: (
            f"{when.get(r, '')} · {summaries[r][0]} · {data.status_label(summaries[r][1].get('status'))}"))
        _show_run(*summaries[run_id])

    st.subheader("הריצו את ה-Flow עכשיו")
    st.markdown("**ריצה חינמית (בלי AI):** אותו Flow, כשהקוד הנבדק עושה את העבודה של הצוותים. היא קוראת את "
                "הנתונים הגולמיים וכותבת לתיקייה נפרדת (`artifacts/_no_llm/`), כך שהתוצאות המאומתות לא מוחלפות. "
                "לוקחת בערך 10-30 שניות.")
    if st.button("הריצו את ה-Flow בלי AI (חינם)", type="primary", icon=":material/play_arrow:"):
        with st.spinner("מריצים בדיקות מקדימות, צוות 1, את השער ואת צוות 2..."):
            st.session_state["free_run"] = data.run_free_flow().model_dump(mode="json")
    if "free_run" in st.session_state:
        st.markdown("**התוצאה של הריצה החינמית שלכם** (היא תופיע גם בהיסטוריה אחרי רענון):")
        ui.show_flow_result(st.session_state["free_run"])
    st.info("**ריצה מלאה עם סוכני AI עולה כסף**, ולכן האפליקציה לא יכולה להפעיל אותה. מפעילים אותה מהטרמינל, "
            "בכוונה:\n\n`.\\.venv\\Scripts\\python.exe -m golden_fighter.flow.main --no-retries`\n\n"
            "עם Claude Haiku 5.5 ריצה מלאה עולה בערך $0.007.", icon=":material/payments:")


def _show_run(kind: str, summary: dict) -> None:
    tabs = st.tabs(["תוצאה", "שימוש ועלות", "קבצים"])
    with tabs[0]:
        st.caption(f"{kind} · מזהה ריצה: `{summary['run_id']}`")
        ui.show_flow_result(summary)
        if summary.get("metrics"):
            m = summary["metrics"]
            st.markdown(f"**המודל מהריצה הזו:** {data.MODEL_LABELS.get(m['selected_model'])} · ROC-AUC במבחן "
                        f"{m['test']['roc_auc']:.3f} · Recall {ui.pct(m['test']['recall'])} · Precision "
                        f"{ui.pct(m['test']['precision'])}")
    with tabs[1]:
        rows = [{"צוות": hebrew.text(hebrew.CREW_KEYS, name), data.API_CALLS: c.get("model_api_calls", 0),
                 data.TOOL_RUNS: c.get("agent_tool_executions", 0),
                 "דחיות בבדיקת המקורות": c.get("source_check_rejections", 0),
                 data.TOKENS: c.get("total_tokens", 0), data.COST: c.get("estimated_cost_usd") or 0.0}
                for name, c in summary.get("crews", {}).items()]
        if rows:
            ui.table(pd.DataFrame(rows), {data.COST: "${:.4f}", data.TOKENS: "{:,}"})
            st.caption("דחייה בבדיקת המקורות = הקוד החזיר לסוכן תשובה שבה מספר יוחס למקור הלא נכון, כדי שיתקן.")
        else:
            st.caption("אף צוות של סוכני AI לא רץ בריצה הזו (ריצה חינמית או הדגמה), ולכן אין קריאות API ואין עלות.")
    with tabs[2]:
        files = [{"צוות": hebrew.text(hebrew.CREW_KEYS, group), "קובץ": a["file"],
                  "נוצר בריצה הזו": "כן" if a["produced_by_this_run"]
                  else ("לא - נשאר מריצה קודמת" if a["exists"] else "לא - חסר")}
                 for group, entries in summary.get("artifacts", {}).items() for a in entries]
        ui.table(pd.DataFrame(files))
        st.caption("קובץ נחשב תוצר של ריצה רק אם הריצה הזו כתבה אותו מחדש; קבצים ישנים לא נספרים כהצלחה.")


ui.guarded(render)
