"""Overview: the business problem, the headline results and how the system is built."""
import streamlit as st

from golden_fighter.dashboard import data, hebrew_reports, ui

ui.header("Golden Fighter AI: חיזוי נטישת מתאמנים",
          "בערך מתאמן אחד מכל ארבעה עוזב את המועדון. המערכת מזהה **מי צפוי לעזוב, כבר בתחילת החודש**, כדי שצוות "
          "השימור יוכל להציע עזרה בזמן, ומסבירה למה.")
st.info(ui.DATA_NOTE, icon=":material/database:")


def render():
    facts, evaluation = ui.load("facts"), ui.load("evaluation")
    test = evaluation["results"][evaluation["selected_model"]]["test"]

    data_col, model_col = st.columns([2, 3], gap="large")
    with data_col:
        st.markdown("##### הנתונים")
        cols = st.columns(2)
        cols[0].metric("מתאמנים שנותחו", f"{facts['members']:,}", border=True)
        cols[1].metric("עזבו את המועדון", ui.pct(facts["churn_rate"]), border=True,
                       help="שיעור המתאמנים שעזבו (churn) בנתונים.")
    with model_col:
        st.markdown("##### המודל, על מתאמנים שלא ראה באימון")
        cols = st.columns(3)
        cols[0].metric("עוזבים שזוהו", ui.pct(test["recall"]), border=True,
                       help="Recall: מכל 100 מתאמנים שבאמת עזבו, על כמה המודל התריע.")
        cols[1].metric("אזהרות נכונות", ui.pct(test["precision"]), border=True,
                       help="Precision: מכל 100 אזהרות של המודל, כמה מתאמנים באמת עזבו.")
        cols[2].metric("איכות הדירוג (ROC-AUC)", f"{test['roc_auc']:.3f}", border=True,
                       help="1.0 = דירוג מושלם של העוזבים מעל הנשארים; 0.5 = הטלת מטבע.")

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("מה מצאנו")
        st.markdown(hebrew_reports.key_findings(facts))
        st.caption("הסיכום נוצר בעברית מהמספרים המאומתים (`insight_facts.json`), בכל ריצה מחדש.")
        summary = data.agent_section(ui.paths().analyst_dir / "insights.md")
        if summary:
            with st.expander("הסיכום המקורי שכתב סוכן ה-AI (באנגלית)"):
                st.caption("לפני השמירה, הקוד בדק שכל מספר בטקסט הזה מופיע בעובדות שחושבו.")
                ui.english_document("agent_summary", summary)
    with right:
        st.subheader("הריצה המלאה האחרונה עם AI")
        history = data.run_history(ui.paths())
        full = history[history[data.KIND] == data.FULL_RUNS] if not history.empty else history
        if full.empty:
            st.info("עדיין לא נרשמה ריצה מלאה עם סוכני ה-AI.")
        else:
            run = full.iloc[0]
            with st.container(border=True):
                st.markdown(f"**{run['תוצאה']}** · {run[data.STARTED]} (UTC)  \n"
                            f"מודל שפה: `{run['מודל שפה']}`  \n"
                            f"**{run[data.API_CALLS]}** קריאות API בתשלום · **{run[data.TOOL_RUNS]}** הפעלות כלים "
                            f"מקומיות (חינם)  \n"
                            f"{run[data.TOKENS]:,} טוקנים · עלות משוערת: **${run[data.COST]:.4f}**")
        st.subheader("עיקרון התכנון")
        st.markdown("- **המספרים מגיעים מקוד Python נבדק.** סוכני ה-AI מפעילים את הקוד דרך כלים ומסבירים את "
                    "התוצאות.\n"
                    "- **שער הבדיקה הוא קוד רגיל, לא AI.** אף סוכן לא יכול לשכנע אותו לעבור.\n"
                    "- **צוות 2 רואה רק מה שצוות 1 פרסם**, כולל חוזה הנתונים, ואף פעם לא את הנתונים הגולמיים.")

    st.subheader("איך המערכת עובדת")
    diagram, steps = st.columns([1, 1], gap="large")
    with diagram:
        st.graphviz_chart("""
digraph {
  rankdir=TB; bgcolor="transparent"; nodesep=0.4; ranksep=0.4;
  node [shape=box, style="rounded,filled", fillcolor="#FFF8E1", color="#B8860B", fontname="Arial",
        fontsize=17, fixedsize=true, width=2.9, height=1.0, penwidth=1.5];
  edge [color="#555555", fontname="Arial", fontsize=16, penwidth=1.5];
  # fixedsize: Graphviz overestimates the width of Hebrew text, which made the whole diagram shrink
  raw [label="נתוני חדר כושר ציבוריים\\n4,000 מתאמנים", fillcolor="#F1F3F4", color="#9AA0A6", height=0.8];
  c1 [label="צוות 1: אנליסט נתונים\\n3 סוכני AI\\nניקוי, חקירה, חוזה נתונים"];
  gate [label="שער הבדיקה\\nקוד רגיל, 88 בדיקות", shape=diamond, fillcolor="#FFFFFF", width=3.2, height=1.3];
  c2 [label="צוות 2: מדען נתונים\\n3 סוכני AI\\nפיצ'רים, מודלים, כרטיס מודל"];
  out [label="מודל + דוחות", fillcolor="#E6F4EA", color="#1E8E3E", height=0.6];
  fail [label="עצירה\\nדוח כשל\\nאף מודל לא אומן", fillcolor="#FDECEA", color="#C62828", width=2.3];
  raw -> c1 -> gate; gate -> c2 [label="  עבר"]; c2 -> out; gate -> fail [label="  נכשל"];
  {rank=same; c2; fail}
}""", width="stretch")
    with steps:
        st.markdown(
            "**1. צוות 1: אנליסט נתונים (3 סוכני AI).** מנקה את הנתונים, מחפש מי עוזב ולמה, וכותב **חוזה "
            "נתונים**: קובץ שמגדיר בדיוק איך הנתונים הנקיים חייבים להיראות (עמודות, טווחים, יחידות מידה).\n\n"
            "**2. שער הבדיקה (בלי AI).** קוד Python רגיל שמשווה את הנתונים לחוזה ב-88 בדיקות. אם משהו לא תואם, "
            "למשל מחיר שהוכפל פי 100, השער **עוצר הכול** וכותב דוח כשל ברור.\n\n"
            "**3. צוות 2: מדען נתונים (3 סוכני AI).** מתחיל רק אם השער עבר. בונה פיצ'רים, מאמן ומשווה מודלים, "
            "וכותב כרטיס מודל עם המגבלות והשימוש המותר.\n\n"
            "**ההבדל בקצרה:** הצוותים **עושים את העבודה ומסבירים אותה**; השער **רק בודק**, תמיד באותו אופן, ולא "
            "ניתן לשכנע אותו.")
        st.caption("רוצים לראות את השער בפעולה? עברו ל**הדגמת שבירת נתונים** בתפריט למעלה.")


ui.guarded(render)
