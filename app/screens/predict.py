"""Predict churn for one member with the saved model (inputs are checked against the data contract)."""
import streamlit as st

from golden_fighter.dashboard import data, ui

ui.header("Predict churn for one member", "Pick an example or change the values, then score the member with the "
          "saved model. The inputs are checked against the data contract first.")

YES_NO = {1: "Yes", 0: "No"}
BAND_TEXT = {"low": ("Low risk", "No action needed now."),
             "medium": ("Medium risk", "Worth a friendly check-in."),
             "high": ("High risk", "Offer help now: onboarding, a class invitation or a longer contract.")}


def render():
    bundle = data.model_bundle(ui.paths())
    example = st.segmented_control("Start from an example", list(data.EXAMPLE_MEMBERS),
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
                    step=1 if is_int else 0.5, key=key, help=f"Unit: {field['unit']}")
        st.form_submit_button("Score this member", type="primary")

    try:
        result = data.predict_member(bundle, member)  # the form's current values (defaults = the example)
    except data.DataValidationError as err:
        issues = "\n".join(f"- {issue}" for issue in err.issues)
        st.error(f"**These values break the data contract, so the model will not score them:**\n\n{issues}",
                 icon=":material/block:")
        return

    label, advice = BAND_TEXT[result["risk_band"]]
    left, right = st.columns([1, 2], gap="large")
    left.metric("Chance of leaving this month", f"{result['probability']:.1%}" if result["probability"] >= 0.001
                else "below 0.1%")
    left.progress(min(max(result["probability"], 0.0), 1.0))
    with right:
        box = st.error if result["risk_band"] == "high" else st.warning if result["risk_band"] == "medium" else st.success
        box(f"**{label}.** {advice}  \nThe model {'flags' if result['flagged'] else 'does not flag'} this member "
            f"(flag at {result['threshold']:.0%} or more).")
        st.caption(f"Model: {data.MODEL_LABELS.get(result['model'], result['model'])}. Gender and the leaky "
                   "current-month visit count are not inputs. Allowed use: offer help; never penalise a member.")


ui.guarded(render)
