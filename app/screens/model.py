"""Model performance: the comparison, the selected model's test results, the leakage experiment, fairness."""
import streamlit as st

from golden_fighter.dashboard import data, hebrew, ui

ui.header("ביצועי המודל", "שלושה מודלים אומנו על 80% מהמתאמנים ונבדקו על 20% האחרים, שאותם לא ראו מעולם. "
          "המודל הזוכה נבחר לפי נתוני האימון בלבד, אף פעם לא לפי נתוני המבחן.")


def render():
    evaluation = ui.load("evaluation")
    selected = evaluation["selected_model"]
    test = evaluation["results"][selected]["test"]
    label = data.MODEL_LABELS.get(selected, selected)

    st.subheader(f"המודל שנבחר: {label}")
    cols = st.columns(4)
    cols[0].metric("איכות הדירוג (ROC-AUC)", f"{test['roc_auc']:.3f}")
    cols[1].metric("Recall", ui.pct(test["recall"]), help="איזה חלק מהעוזבים האמיתיים המודל מסמן.")
    cols[2].metric("Precision", ui.pct(test["precision"]), help="איזה חלק מהמתאמנים שסומנו באמת עוזבים.")
    cols[3].metric("F1", f"{test['f1']:.3f}", help="איזון בין Recall ל-Precision (1.0 הוא מושלם).")
    with st.expander("מה המספרים האלה אומרים?"):
        st.markdown(
            "- **Recall**: מכל 100 מתאמנים שבאמת עוזבים, על כמה המודל מתריע.\n"
            "- **Precision**: מכל 100 מתאמנים שהמודל מתריע עליהם, כמה באמת עוזבים. "
            "אזעקת שווא עולה רק הצעת שימור שלא הייתה נחוצה.\n"
            "- **ROC-AUC**: עד כמה המודל מדרג את העוזבים מעל הנשארים. 0.5 הוא הטלת מטבע, 1.0 הוא מושלם.\n"
            f"- מתאמן מסומן כשהסיכוי החזוי שיעזוב הוא {evaluation['threshold']:.0%} או יותר.")

    st.subheader("השוואה בין המודלים")
    table = data.model_table(evaluation)
    ui.table(table, {c: "{:.3f}" for c in table.columns if c not in ("מודל", "תפקיד")})
    st.caption(f"כלל הבחירה: {hebrew.text(hebrew.SELECTION_RULES, evaluation['selection_rule'])}")
    ablation = evaluation["ablation"]
    st.warning(f"**למה השורה עם התוצאה הגבוהה ביותר לא בשימוש:** ניסוי הדליפה מוסיף את "
               f"`{', '.join(ablation['extra_columns'])}`, שמתמלאת רק בסוף החודש שבו המתאמן עוזב. התוצאה שלו "
               f"גבוהה יותר (ROC-AUC {ablation['test']['roc_auc']:.3f}) כי הוא רואה חלק מהתשובה. בתחילת החודש, "
               "כשהצוות צריך לפעול, הערך הזה עוד לא קיים.", icon=":material/visibility_off:")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("מטריצת בלבול (נתוני המבחן)")
        cm = data.confusion_matrix(evaluation)
        st.altair_chart(ui.confusion_heatmap(cm), width="stretch")
        for row in cm.itertuples():
            st.markdown(f"- **{row.members}** - {row.meaning}")
    with right:
        st.subheader("עקומת ROC")
        roc = data.roc_points(ui.paths())
        if roc is None:
            st.info("עקומת ה-ROC צריכה את `features.csv` (שורות ברמת המתאמן, לא נשמרות ב-Git). הריצו את ה-Flow "
                    "החינמי כדי ליצור אותו.")
        else:
            st.altair_chart(ui.roc_chart(roc, test["roc_auc"]), width="stretch")

    st.subheader(f"מה משפיע על התחזיות של המודל שנבחר ({label})")
    st.altair_chart(ui.importance_bars(data.feature_importance(evaluation)), width="stretch")
    st.caption("חשיבות בשיטת Permutation של המודל שנבחר על נתוני המבחן: בכמה יורד ה-ROC-AUC כשמערבבים נתון אחד. "
               "חושב רק למודל שנבחר, לא לניסוי הדליפה.")

    st.subheader("בדיקת הוגנות (נתוני המבחן)")
    st.markdown("המגדר **אינו** נתון קלט של המודל. הוא משמש כאן רק כדי לבדוק שהמודל מתייחס לקבוצות באופן דומה.")
    st.markdown("**לפי מגדר** (קודים אנונימיים 0 / 1)")
    ui.table(data.fairness(evaluation, "gender"))
    st.markdown("**לפי קבוצת גיל**")
    ui.table(data.fairness(evaluation, "age_band"))
    st.caption("'אין עוזבים' פירושו שאף אחד בקבוצה לא עזב בתקופת המבחן, ולכן אי אפשר לחשב Recall ו-Precision. "
               "בקבוצת הגיל 30-34 ה-Recall הנמוך ביותר: כדאי לעקוב.")

    with st.expander("כרטיס המודל (מטרה, נתונים, מגבלות, שימוש אתי; באנגלית)"):
        ui.english_document("model_card", data.document(ui.paths().scientist_dir / "model_card.md"))
    with st.expander("דוח ההערכה המלא (באנגלית)"):
        ui.english_document("evaluation_report", data.document(ui.paths().scientist_dir / "evaluation_report.md"))


ui.guarded(render)
