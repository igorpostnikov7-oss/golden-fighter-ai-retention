"""Overview: the business problem, the headline results and how the system is built."""
import streamlit as st

from golden_fighter.dashboard import data, ui

ui.header("Golden Fighter AI: חיזוי נטישת מתאמנים",
          "בערך מתאמן אחד מכל ארבעה עוזב את המועדון. המערכת מזהה **מי צפוי לעזוב, כבר בתחילת החודש**, כדי שצוות "
          "השימור יוכל להציע עזרה בזמן, ומסבירה למה.")


def render():
    facts, evaluation = ui.load("facts"), ui.load("evaluation")
    test = evaluation["results"][evaluation["selected_model"]]["test"]
    cols = st.columns(5)
    cols[0].metric("מתאמנים שנותחו", f"{facts['members']:,}")
    cols[1].metric("עזבו את המועדון", ui.pct(facts["churn_rate"]), help="שיעור המתאמנים שעזבו (churn).")
    cols[2].metric("עוזבים שזוהו", ui.pct(test["recall"]),
                   help="Recall: איזה חלק מהעוזבים האמיתיים המודל מסמן, על מתאמנים שהוא לא ראה באימון.")
    cols[3].metric("אזהרות נכונות", ui.pct(test["precision"]),
                   help="Precision: איזה חלק מהמתאמנים שסומנו באמת עוזבים (נתוני המבחן).")
    cols[4].metric("ROC-AUC", f"{test['roc_auc']:.3f}",
                   help="1.0 = דירוג מושלם של העוזבים מעל הנשארים; 0.5 = הטלת מטבע.")

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("מה סוכני ה-AI מצאו")
        summary = data.agent_section(ui.paths().analyst_dir / "insights.md")
        if summary:
            st.caption("הסיכום נכתב באנגלית על ידי סוכן ה-EDA והתובנות, ומוצג כפי שנכתב. לפני השמירה, הקוד בדק "
                       "שכל מספר בו מופיע בעובדות שחושבו.")
            ui.english_document("agent_summary", summary)
        else:
            st.info("עדיין אין סיכום שסוכן כתב (הריצה החינמית כותבת דוחות בלי טקסט של AI).")
    with right:
        st.subheader("הריצה המלאה האחרונה")
        history = data.run_history(ui.paths())
        full = history[history[data.KIND] == data.FULL_RUNS] if not history.empty else history
        if full.empty:
            st.info("עדיין לא נרשמה ריצה מלאה עם סוכני ה-AI.")
        else:
            run = full.iloc[0]
            st.markdown(f"**{run['תוצאה']}** - {run['התחלה (UTC)']} UTC  \n"
                        f"מודל שפה: `{run['מודל שפה']}`  \n"
                        f"{run[data.API_CALLS]} קריאות API למודל, {run[data.TOOL_RUNS]} הפעלות כלים של סוכנים, "
                        f"{run[data.TOKENS]:,} טוקנים  \n"
                        f"עלות משוערת: **${run[data.COST]:.4f}**")
        st.subheader("עיקרון התכנון")
        st.markdown("- **המספרים מגיעים מקוד Python נבדק.** סוכני ה-AI מפעילים את הקוד הזה דרך כלים ומסבירים "
                    "את התוצאות.\n"
                    "- **שער הבדיקה הוא Python רגיל.** שום AI לא יכול לשכנע אותו לעבור.\n"
                    "- **צוות 2 רואה רק את הקבצים שצוות 1 פרסם ואת חוזה הנתונים**, אף פעם לא את הנתונים הגולמיים.")

    st.subheader("איך המערכת עובדת")
    st.graphviz_chart("""
digraph {
  rankdir=RL; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#FFF8E1", color="#B8860B", fontname="Arial",
        fontsize=16, margin="0.25,0.12"];
  edge [color="#666666", fontname="Arial", fontsize=14];
  raw [label="נתוני חדר כושר ציבוריים\\n(4,000 מתאמנים)"];
  c1 [label="צוות 1: אנליסט נתונים\\n3 סוכני AI\\nניקוי, חקירה, כתיבת חוזה"];
  gate [label="שער הבדיקה\\nPython רגיל, 88 בדיקות", shape=diamond, fillcolor="#FFFFFF"];
  c2 [label="צוות 2: מדען נתונים\\n3 סוכני AI\\nפיצ'רים, מודלים, כרטיס מודל"];
  out [label="מודל + דוחות\\nrun_summary.json"];
  fail [label="עצירה\\nfailure_report.md\\nאף מודל לא אומן", fillcolor="#FDECEA", color="#C62828"];
  raw -> c1 -> gate; gate -> c2 [label="עבר"]; c2 -> out; gate -> fail [label="נכשל"];
}""", width="stretch")
    st.caption("נסו בעצמכם: **הדגמת שבירת נתונים** משנה עותק של הנתונים ומראה איך השער עוצר את ה-Flow.")


ui.guarded(render)
