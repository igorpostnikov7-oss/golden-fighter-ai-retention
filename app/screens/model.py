"""Model performance: the comparison, the selected model's test results, the leakage experiment, fairness."""
import streamlit as st

from golden_fighter.dashboard import data, hebrew, hebrew_reports, ui

ui.header("ביצועי המודל", "שלושה מודלים אומנו על 80% מהמתאמנים ונבדקו על 20% האחרים, שאותם לא ראו אף פעם. "
          "המודל הזוכה נבחר לפי נתוני האימון בלבד, אף פעם לא לפי נתוני המבחן.")

GLOSSARY = {
    "CV ROC-AUC (אימון)": "איכות הדירוג בממוצע על 5 חלקים של נתוני האימון. **לפי העמודה הזו נבחר המודל.**",
    "ROC-AUC (מבחן)": "איכות הדירוג על מתאמני המבחן, שהמודל לא ראה: 0.5 = הטלת מטבע, 1.0 = מושלם.",
    "Recall (מבחן)": "מכל העוזבים האמיתיים, איזה חלק המודל תפס.",
    "Precision (מבחן)": "מכל האזהרות של המודל, איזה חלק היו נכונות.",
    "F1 (מבחן)": "ציון אחד שמאזן בין Recall ל-Precision.",
}


def render():
    evaluation = ui.load("evaluation")
    selected = evaluation["selected_model"]
    test = evaluation["results"][selected]["test"]
    cm = test["confusion_matrix"]
    label = data.MODEL_LABELS.get(selected, selected)

    st.subheader(f"בשורה התחתונה: {label}")
    cols = st.columns(4)
    cols[0].metric("עוזבים שזוהו (Recall)", ui.pct(test["recall"]), border=True,
                   help=f"{cm['tp']} מתוך {cm['tp'] + cm['fn']} עוזבים בנתוני המבחן.")
    cols[1].metric("אזהרות נכונות (Precision)", ui.pct(test["precision"]), border=True,
                   help="מכל המתאמנים שהמודל סימן, איזה חלק באמת עזב.")
    cols[2].metric("איכות הדירוג (ROC-AUC)", f"{test['roc_auc']:.3f}", border=True,
                   help="0.5 = הטלת מטבע, 1.0 = מושלם.")
    cols[3].metric("ציון איזון (F1)", f"{test['f1']:.3f}", border=True,
                   help="ציון אחד שמאזן בין Recall ל-Precision (1.0 מושלם).")
    with st.container(border=True):
        st.markdown(
            "**מה חשוב לדעת לפני שמשתמשים במודל:**\n"
            f"- הוא תופס את רוב העוזבים, אבל גם מתריע לשווא על {cm['fp']} מתאמנים שנשארו. זה מקובל כאן, כי אזעקת "
            "שווא עולה רק הצעת שימור.\n"
            "- הסיכוי שהוא נותן הוא **הערכה לדירוג**, לא הבטחה. ערכים קיצוניים צריכים כיול (calibration) לפני שימוש "
            "אמיתי.\n"
            "- הוא למד מנתונים ציבוריים של רשת אחרת, בתקופה אחת. לפני שימוש אמיתי צריך לבדוק אותו על נתוני "
            "Golden Fighter.\n"
            "- מותר להשתמש בו רק כדי **להציע עזרה**, אף פעם לא כדי להעניש מתאמן.")
    with st.expander("מה המספרים האלה אומרים?"):
        st.markdown(
            "- **Recall**: מכל 100 מתאמנים שבאמת עוזבים, על כמה המודל מתריע.\n"
            "- **Precision**: מכל 100 מתאמנים שהמודל מתריע עליהם, כמה באמת עוזבים.\n"
            "- **ROC-AUC**: עד כמה המודל מדרג את העוזבים מעל הנשארים. 0.5 הוא הטלת מטבע, 1.0 הוא מושלם.\n"
            f"- מתאמן מסומן כשהסיכוי החזוי שיעזוב הוא {evaluation['threshold']:.0%} או יותר.")

    st.subheader("השוואה בין המודלים")
    table = data.model_table(evaluation)
    numbers = [c for c in table.columns if c not in ("מודל", "תפקיד")]
    styled = (table.style.format({c: "{:.3f}" for c in numbers})
              .apply(lambda row: ["background-color: #FFF4D6; font-weight: 600" if row["תפקיד"] == data.SELECTED
                                  else "color: #8a1c1c" if row["תפקיד"] == data.LEAKAGE_ROLE else ""] * len(row),
                     axis=1))
    ui.table(styled)
    st.caption(f"**כלל הבחירה:** {hebrew.text(hebrew.SELECTION_RULES, evaluation['selection_rule'])} "
               "השורה המודגשת היא המודל שנבחר.")
    with st.expander("מה אומרת כל עמודה?"):
        st.markdown("\n".join(f"- **{name}:** {text}" for name, text in GLOSSARY.items()))
    ablation = evaluation["ablation"]
    st.warning(f"**למה השורה האדומה, עם התוצאה הגבוהה ביותר, לא בשימוש:** ניסוי הדליפה מוסיף את "
               f"\"{hebrew.column(ablation['extra_columns'][0])}\", שמתמלאת רק בסוף החודש שבו המתאמן עוזב. התוצאה "
               f"שלו גבוהה יותר (ROC-AUC {ablation['test']['roc_auc']:.3f}) כי הוא רואה חלק מהתשובה. בתחילת החודש, "
               "כשהצוות צריך לפעול, הערך הזה עוד לא קיים.", icon=":material/visibility_off:")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("מטריצת בלבול (נתוני המבחן)")
        cm_table = data.confusion_matrix(evaluation)
        st.altair_chart(ui.confusion_heatmap(cm_table), width="stretch")
    with right:
        st.subheader(f"עקומת ROC (שטח מתחת לעקומה: {test['roc_auc']:.3f})")
        roc = data.roc_points(ui.paths())
        if roc is None:
            st.info("עקומת ה-ROC צריכה את `features.csv` (שורות ברמת המתאמן, לא נשמרות ב-Git). הריצו את ה-Flow "
                    "החינמי כדי ליצור אותו.")
        else:
            st.altair_chart(ui.roc_chart(roc), width="stretch")
            st.caption("ככל שהעקומה קרובה יותר לפינה השמאלית העליונה, המודל טוב יותר. הקו המקווקו הוא הטלת מטבע.")
    st.markdown("**ארבע התוצאות האפשריות:**")
    ui.table(cm_table.rename(columns={"short": "תוצאה", "members": "מתאמנים", "meaning": "מה זה אומר",
                                      "term": "המונח המקצועי"})[["תוצאה", "מתאמנים", "מה זה אומר", "המונח המקצועי"]])

    st.subheader(f"מה משפיע על התחזיות של המודל שנבחר ({label})")
    st.altair_chart(ui.importance_bars(data.feature_importance(evaluation)), width="stretch")
    st.caption("שיטת Permutation Importance על נתוני המבחן: מערבבים נתון אחד ובודקים כמה איכות הדירוג יורדת. "
               "ירידה גדולה = נתון חשוב. חושב רק למודל שנבחר, לא לניסוי הדליפה.")

    st.subheader("בדיקת הוגנות (נתוני המבחן)")
    st.markdown("המגדר **אינו** נתון קלט של המודל. הוא משמש כאן רק כדי לבדוק שהמודל מתייחס לקבוצות באופן דומה. "
                "בטבלאות: **עזבו בפועל** = שיעור העזיבה האמיתי בקבוצה; **סומנו בסיכון** = כמה מהקבוצה המודל סימן; "
                "**Recall** = כמה מהעוזבים בקבוצה זוהו; **Precision** = כמה מהסימונים בקבוצה היו נכונים.")
    gender, age = st.columns([2, 3], gap="large")
    with gender:
        st.markdown("**לפי מגדר** (קודים אנונימיים 0 / 1)")
        ui.table(data.fairness(evaluation, "gender"))
        gap = hebrew_reports.gender_recall_gap(evaluation)
        if gap is not None:
            st.caption(f"הפער ב-Recall בין שני הקודים: {gap:.3f}. הסבולת שנקבעה מראש: "
                       f"{hebrew_reports.FAIRNESS_RECALL_GAP}.")
    with age:
        st.markdown("**לפי קבוצת גיל**")
        ui.table(data.fairness(evaluation, "age_band"))
        st.caption("'אין עוזבים' = אף אחד בקבוצה לא עזב בתקופת המבחן, ולכן אי אפשר לחשב Recall ו-Precision. "
                   "בקבוצת הגיל 30-34 ה-Recall הנמוך ביותר: כדאי לעקוב.")

    st.subheader("כרטיס המודל ודוח ההערכה")
    st.caption("גרסאות עבריות שנוצרות מ-`evaluation.json`, אותו קובץ שממנו נכתבים המסמכים הרשמיים באנגלית. "
               "המסמכים באנגלית נשארים התוצר הרשמי, בלשונית האחרונה.")
    tabs = st.tabs(["כרטיס המודל", "דוח ההערכה", "המקור באנגלית"])
    with tabs[0]:
        st.markdown(hebrew_reports.model_card(evaluation))
    with tabs[1]:
        st.markdown(hebrew_reports.evaluation_report(evaluation))
    with tabs[2]:
        with st.expander("model_card.md (באנגלית)"):
            ui.english_document("model_card", data.document(ui.paths().scientist_dir / "model_card.md"))
        with st.expander("evaluation_report.md (באנגלית)"):
            ui.english_document("evaluation_report", data.document(ui.paths().scientist_dir / "evaluation_report.md"))


ui.guarded(render)
