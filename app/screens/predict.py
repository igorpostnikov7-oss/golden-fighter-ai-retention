"""Predict churn for one member with the saved model (inputs are checked against the data contract)."""
import streamlit as st

from golden_fighter.dashboard import data, ui

ui.header("חיזוי עזיבה למתאמן אחד", "בחרו דוגמה או שנו את הערכים, ולחצו על **חשבו סיכון**. לפני החישוב, הערכים "
          "נבדקים מול חוזה הנתונים.")

YES_NO = {1: "כן", 0: "לא"}
GROUPS = {"המנוי": ["contract_period", "month_to_end_contract", "lifetime"],
          "פעילות והרגלים": ["group_visits", "avg_class_frequency_total", "avg_additional_charges_total"],
          "פרטים אישיים": ["age", "near_location", "partner", "promo_friends", "phone"]}
BANDS = {"low": ("סיכון נמוך", "green", "לא נדרשת פעולה כרגע."),
         "medium": ("סיכון בינוני", "orange", "כדאי ליצור קשר ידידותי ולשאול איך הולך."),
         "high": ("סיכון גבוה", "red", "להציע עזרה עכשיו: ליווי אישי, הזמנה לשיעור קבוצתי או מנוי ארוך יותר.")}
EXTREME = 0.05  # below 5% or above 95% the score is very confident; real use needs calibration first


def render():
    bundle = data.model_bundle(ui.paths())
    example = st.segmented_control("התחילו מדוגמה", list(data.EXAMPLE_MEMBERS),
                                   default=list(data.EXAMPLE_MEMBERS)[0], key="example")
    values = data.EXAMPLE_MEMBERS[example or list(data.EXAMPLE_MEMBERS)[0]]
    fields = {f["name"]: f for f in data.input_spec(bundle)}

    with st.form("member"):
        member = {}
        for column, (title, names) in zip(st.columns(3, gap="medium"), GROUPS.items()):
            with column.container(border=True):
                st.markdown(f"**{title}**")
                for name in names:
                    member[name] = _field(fields[name], values, example)
        rest = [n for n in fields if n not in member]  # any input the groups above do not list
        for name in rest:
            member[name] = _field(fields[name], values, example)
        st.form_submit_button("חשבו סיכון", type="primary", icon=":material/calculate:")

    try:
        result = data.predict_member(bundle, member)  # the form's current values (defaults = the example)
    except data.DataValidationError as err:
        issues = "\n".join(f"- {data.issue_in_hebrew(issue)}" for issue in err.issues)
        st.error(f"**הערכים האלה מפרים את חוזה הנתונים, ולכן המודל לא יחשב עבורם סיכון:**\n\n{issues}",
                 icon=":material/block:")
        return

    label, color, advice = BANDS[result["risk_band"]]
    probability = result["probability"]
    with st.container(border=True):
        st.markdown("#### התוצאה")
        left, right = st.columns([1, 2], gap="large")
        with left:
            st.metric("הסיכוי לעזוב החודש", f"{probability:.1%}" if probability >= 0.001 else "פחות מ-0.1%")
            st.progress(min(max(probability, 0.0), 1.0))
        with right:
            st.markdown(f"### :{color}[:material/flag: {label}]")
            st.markdown(f"**מה לעשות:** {advice}")
            st.markdown(f"המודל {'**מסמן**' if result['flagged'] else '**לא מסמן**'} את המתאמן הזה "
                        f"(סימון מ-{result['threshold']:.0%} ומעלה).")
    st.info("**זו הערכה, לא הבטחה.** המספר אומר עד כמה מתאמנים עם פרופיל דומה עזבו בנתוני האימון, ומשמש לדירוג "
            "מי לפנות אליו קודם. הוא לא קובע מה האדם הזה יעשה.", icon=":material/info:")
    if probability < EXTREME or probability > 1 - EXTREME:
        st.warning("**סיכוי קיצוני (קרוב ל-0% או ל-100%).** המודל לא עבר כיול הסתברויות (calibration), ולכן ערכים "
                   "קיצוניים עלולים להיות בטוחים מדי. לפני שימוש אמיתי צריך לכייל ולבדוק את המודל על נתוני "
                   "Golden Fighter.", icon=":material/tune:")
    st.caption(f"המודל: {data.MODEL_LABELS.get(result['model'], result['model'])}. המגדר ומספר הביקורים בחודש "
               "הנוכחי (שדולף מידע) אינם קלטים. שימוש מותר: להציע עזרה; אף פעם לא להעניש מתאמן.")


def _field(field: dict, values: dict, example: str):
    """One input widget, built from the contract's rule for that column."""
    name, key = field["name"], f"{example}:{field['name']}"
    if field["allowed_values"] == [0, 1]:
        return st.radio(field["label"], [1, 0], index=[1, 0].index(int(values[name])), format_func=YES_NO.get,
                        horizontal=True, key=key)
    if field["allowed_values"]:
        options = [int(v) for v in field["allowed_values"]]
        return st.selectbox(field["label"], options, index=options.index(int(values[name])), key=key)
    is_int = name in ("age", "month_to_end_contract", "lifetime")
    cast = int if is_int else float
    return st.number_input(field["label"], min_value=cast(field["min"]) if field["min"] is not None else None,
                           max_value=cast(field["max"]) if field["max"] is not None else None,
                           value=cast(values[name]), step=1 if is_int else 0.5, key=key,
                           help=f"יחידה: {field['unit']}")


ui.guarded(render)
