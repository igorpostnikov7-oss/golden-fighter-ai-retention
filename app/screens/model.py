"""Model performance: the comparison, the selected model's test results, the leakage experiment, fairness."""
import streamlit as st

from golden_fighter.dashboard import data, ui

ui.header("Model performance", "Three models were trained on 80% of the members and tested on the other 20%, "
          "which they never saw. The winner was chosen on the training data only, never on the test set.")


def render():
    evaluation = ui.load("evaluation")
    selected = evaluation["selected_model"]
    test = evaluation["results"][selected]["test"]
    label = data.MODEL_LABELS.get(selected, selected)

    st.subheader(f"Selected model: {label}")
    cols = st.columns(4)
    cols[0].metric("Ranking quality (ROC-AUC)", f"{test['roc_auc']:.3f}")
    cols[1].metric("Recall", ui.pct(test["recall"]), help="Share of real leavers the model flags.")
    cols[2].metric("Precision", ui.pct(test["precision"]), help="Share of flagged members who really leave.")
    cols[3].metric("F1 score", f"{test['f1']:.3f}", help="Balance of recall and precision (1.0 is perfect).")
    with st.expander("What do these numbers mean?"):
        st.markdown(
            "- **Recall**: out of every 100 members who really leave, how many the model warns about.\n"
            "- **Precision**: out of every 100 members the model warns about, how many really leave. "
            "A false alarm only costs a retention offer that was not needed.\n"
            "- **ROC-AUC**: how well the model ranks leavers above stayers. 0.5 is a coin toss, 1.0 is perfect.\n"
            f"- A member is flagged when the predicted chance of leaving is at least {evaluation['threshold']:.0%}.")

    st.subheader("Comparison")
    table = data.model_table(evaluation)
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        c: st.column_config.NumberColumn(format="%.3f") for c in table.columns if c not in ("Model", "Role")})
    st.caption(f"Selection rule: {evaluation['selection_rule']}.")
    ablation = evaluation["ablation"]
    st.warning(f"**Why the best-looking row is not used:** the leakage experiment adds "
               f"`{', '.join(ablation['extra_columns'])}`, which is only complete at the end of the month the "
               f"member leaves. It scores higher (ROC-AUC {ablation['test']['roc_auc']:.3f}) because it partly "
               "sees the answer. At the start of the month, when the team must act, that value does not exist yet.",
               icon=":material/visibility_off:")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Confusion matrix (test set)")
        cm = data.confusion_matrix(evaluation)
        st.altair_chart(ui.confusion_heatmap(cm), width="stretch")
        for row in cm.itertuples():
            st.markdown(f"- **{row.members}** - {row.meaning}")
    with right:
        st.subheader("ROC curve")
        roc = data.roc_points(ui.paths())
        if roc is None:
            st.info("The ROC curve needs features.csv (member-level rows, not committed). Run the free Flow to "
                    "create it.")
        else:
            st.altair_chart(ui.roc_chart(roc, test["roc_auc"]), width="stretch")

    st.subheader(f"What drives the predictions of the selected model ({label})")
    st.altair_chart(ui.importance_bars(data.feature_importance(evaluation)), width="stretch")
    st.caption("Permutation importance of the selected model on the test set: how much the ROC-AUC drops when "
               "one input is shuffled. Computed for the selected model only, not for the leakage experiment.")

    st.subheader("Fairness check (test set)")
    st.markdown("Gender is **not** a model input. It is used here only to check that the model treats groups alike.")
    st.markdown("**By gender** (anonymised codes 0 / 1)")
    st.dataframe(data.fairness(evaluation, "gender"), hide_index=True, width="stretch")
    st.markdown("**By age band**")
    st.dataframe(data.fairness(evaluation, "age_band"), hide_index=True, width="stretch")
    st.caption("'no leavers' means nobody in that group left during the test period, so recall and precision "
               "cannot be computed. The 30-34 band has the lowest recall: worth monitoring.")

    with st.expander("Model card (purpose, data, limitations, ethical use)"):
        st.markdown(data.document(ui.paths().scientist_dir / "model_card.md"))
    with st.expander("Full evaluation report"):
        st.markdown(data.document(ui.paths().scientist_dir / "evaluation_report.md"))


ui.guarded(render)
