"""Predict churn for one member with the saved model (inputs are checked against the data contract)."""
import streamlit as st

from golden_fighter.dashboard import data, ui

ui.header("חיזוי עזיבה למתאמן אחד", "בחרו דוגמה או שנו את הערכים, ואז חשבו את הסיכון עם המודל השמור. "
          "הערכים נבדקים קודם מול חוזה הנתונים.")

YES_NO = {1: "כן", 0: "לא"}
BAND_TEXT = {"low": ("סיכון נמוך", "לא נדרשת פעולה כרגע."),
             "medium": ("סיכון בינוני", "כדאי ליצור קשר ידידותי."),
             "high": ("סיכון גבוה", "הציעו עזרה עכשיו: ליווי בתחילת הדרך, הזמנה לשיעור או מנוי ארוך יותר.")}


def render():
    bundle = data.model_bundle(ui.paths())
    example = st.segmented_control("התחילו מדוגמה", list(data.EXAMPLE_MEMBERS),
                                   default=list(data.EXAMPLE_MEMBERS)[0], key="example")
    values = data.EXAMPLE_MEMBERS[example or list(data.EXAMPLE_MEMBERS)[0]]

    with st.form("member"):
        member, cols = {}, st.columns(3)
        for i, field in enumerate(data.input_spec(bundle)):
            name, col = field["name"], cols[i % 3]
            key = f"{example}:{name}"
            if field["allowed_values"] == [0, 1]:
                member[name] = col.radio(field["label"], [1, 0], index=[1, 0].index(int(values[name])),
                                         format_func=YES_NO.get, horizontal=True, key=key)
            elif field["allowed_values"]:
                options = [int(v) for v in field["allowed_values"]]
                member[name] = col.selectbox(field["label"], options, index=options.index(int(values[name])), key=key)
            else:
                is_int = name in ("age", "month_to_end_contract", "lifetime")
                cast = int if is_int else float
                member[name] = col.number_input(
                    field["label"], min_value=cast(field["min"]) if field["min"] is not None else None,
                    max_value=cast(field["max"]) if field["max"] is not None else None, value=cast(values[name]),
                    step=1 if is_int else 0.5, key=key, help=f"יחידה: {field['unit']}")
        st.form_submit_button("חשבו את הסיכון של המתאמן", type="primary")

    try:
        result = data.predict_member(bundle, member)  # the form's current values (defaults = the example)
    except data.DataValidationError as err:
        issues = "\n".join(f"- {data.issue_in_hebrew(issue)}" for issue in err.issues)
        st.error(f"**הערכים האלה מפרים את חוזה הנתונים, ולכן המודל לא יחשב עבורם סיכון:**\n\n{issues}",
                 icon=":material/block:")
        return

    label, advice = BAND_TEXT[result["risk_band"]]
    left, right = st.columns([1, 2], gap="large")
    left.metric("הסיכוי לעזוב החודש", f"{result['probability']:.1%}" if result["probability"] >= 0.001
                else "פחות מ-0.1%")
    left.progress(min(max(result["probability"], 0.0), 1.0))
    with right:
        box = st.error if result["risk_band"] == "high" else st.warning if result["risk_band"] == "medium" else st.success
        box(f"**{label}.** {advice}  \nהמודל {'מסמן' if result['flagged'] else 'לא מסמן'} את המתאמן הזה "
            f"(סימון מ-{result['threshold']:.0%} ומעלה).")
        st.caption(f"המודל: {data.MODEL_LABELS.get(result['model'], result['model'])}. המגדר ומספר הביקורים "
                   "בחודש הנוכחי (שדולף מידע) אינם נתוני קלט. שימוש מותר: להציע עזרה; אף פעם לא להעניש מתאמן.")


ui.guarded(render)
