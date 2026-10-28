"""Flow runs & costs: what the CrewAI Flow did in each run, the agents, model API calls, tokens and cost."""
import pandas as pd
import streamlit as st

from golden_fighter.dashboard import data, hebrew, ui

ui.header("ריצות Flow ועלויות", "ה-CrewAI Flow מחבר את שני הצוותים דרך שער הבדיקה. כל ריצה משאירה קובץ "
          "`run_summary.json`, והמסך הזה קורא אותם.")


def render():
    st.subheader("ששת סוכני ה-AI והכלים שלהם")
    ui.table(data.agents_overview())
    st.caption("הסוכנים מחליטים איזה כלי להפעיל וכותבים את ההסברים. הכלים הם קוד Python נבדק, ולכן המספרים אף "
               "פעם לא תלויים ב-AI.")

    p = ui.paths()
    history = data.run_history(p)
    st.subheader("היסטוריית ריצות")
    if history.empty:
        st.info("עדיין לא נרשמו ריצות. אפשר להריץ את ה-Flow בחינם עם הכפתור למטה.")
    else:
        full = history[history[data.KIND] == data.FULL_RUNS]
        cols = st.columns(4)
        cols[0].metric("ריצות מלאות עם סוכני AI", len(full))
        cols[1].metric("קריאות API למודל (כל הריצות המלאות)", int(full[data.API_CALLS].sum()))
        cols[2].metric("טוקנים (כל הריצות המלאות)", f"{int(full[data.TOKENS].sum()):,}")
        cols[3].metric("עלות API משוערת (כל הריצות המלאות)", f"${full[data.COST].sum():.4f}")
        ui.table(history, {data.COST: "${:.4f}", data.TOKENS: "{:,}", "שניות": "{:.1f}"})
        with st.expander("קריאות API למודל לעומת הפעלות כלים של סוכנים - מה ההבדל?"):
            for name, text in {"קריאות API למודל": "בקשות בתשלום ל-Claude API. הן צורכות טוקנים ועולות כסף.",
                               "הפעלות כלים של סוכנים": "הפעמים שסוכן הפעיל אחד מכלי ה-Python של הפרויקט "
                               "(ניקוי, EDA, חוזה, אימון...). חינמי ודטרמיניסטי.",
                               "ריצות חינמיות והדגמות שבירה": "לא משתמשות ב-AI בכלל, ולכן שני המונים הם אפס."}.items():
                st.markdown(f"- **{name}:** {text}")
        if len(full) > 0:
            st.altair_chart(ui.cost_bars(full), width="stretch")

        st.subheader("פרטי ריצה")
        summaries = dict((s["run_id"], (kind, s)) for kind, s in data.run_summaries(p))
        run_id = st.selectbox("בחרו ריצה", list(summaries), format_func=lambda r: f"{r} - {summaries[r][0]}")
        _show_run(*summaries[run_id])

    st.subheader("הריצו את ה-Flow עכשיו")
    st.markdown("**ריצה חינמית (בלי AI):** אותו Flow, כשהתהליכים הנבדקים עושים את העבודה של הצוותים. היא קוראת "
                "את הנתונים הגולמיים וכותבת ל-`artifacts/_no_llm/`, כך שהתוצאות המאומתות שלמעלה לא מוחלפות. "
                "לוקחת בערך 10-30 שניות.")
    if st.button("הריצו את ה-Flow בלי AI (חינם)", type="primary"):
        with st.spinner("מריצים בדיקות מקדימות, צוות 1, את השער ואת צוות 2..."):
            st.session_state["free_run"] = data.run_free_flow().model_dump(mode="json")
    if "free_run" in st.session_state:
        st.markdown("**התוצאה של הריצה החינמית שלכם** (היא תופיע גם בהיסטוריית הריצות אחרי רענון):")
        ui.show_flow_result(st.session_state["free_run"])
    st.info("**ריצות עם סוכני ה-AI עולות כסף**, ולכן האפליקציה לא יכולה להפעיל אותן. מפעילים אותן מהטרמינל, "
            "בכוונה:\n\n`.\\.venv\\Scripts\\python.exe -m golden_fighter.flow.main --no-retries`\n\n"
            "עם Claude Haiku 5.5 ריצה מלאה עולה בערך $0.007.", icon=":material/payments:")


def _show_run(kind: str, summary: dict) -> None:
    ui.show_flow_result(summary)
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**שימוש לפי צוות**")
        rows = [{"צוות": hebrew.text(hebrew.CREW_KEYS, name), data.API_CALLS: c.get("model_api_calls", 0),
                 data.TOOL_RUNS: c.get("agent_tool_executions", 0),
                 "דחיות בבדיקת המקורות": c.get("source_check_rejections", 0),
                 data.TOKENS: c.get("total_tokens", 0), data.COST: c.get("estimated_cost_usd") or 0.0}
                for name, c in summary.get("crews", {}).items()]
        if rows:
            ui.table(pd.DataFrame(rows), {data.COST: "${:.4f}", data.TOKENS: "{:,}"})
        else:
            st.caption("אף צוות לא רץ בריצה הזו.")
        if summary.get("metrics"):
            m = summary["metrics"]
            st.markdown(f"**המודל מהריצה הזו:** {data.MODEL_LABELS.get(m['selected_model'])}, ROC-AUC במבחן "
                        f"{m['test']['roc_auc']:.3f}, Recall {ui.pct(m['test']['recall'])}, Precision "
                        f"{ui.pct(m['test']['precision'])}.")
    with right:
        st.markdown("**קבצים, והאם הריצה הזו יצרה אותם**")
        files = [{"צוות": hebrew.text(hebrew.CREW_KEYS, group), "קובץ": a["file"],
                  "נוצר בריצה הזו": "כן" if a["produced_by_this_run"]
                  else ("לא - נשאר מריצה קודמת" if a["exists"] else "לא - חסר")}
                 for group, entries in summary.get("artifacts", {}).items() for a in entries]
        ui.table(pd.DataFrame(files))


ui.guarded(render)
