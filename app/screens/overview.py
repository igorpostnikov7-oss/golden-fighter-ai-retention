"""Overview: the business problem, the headline results and how the system is built."""
import streamlit as st

from golden_fighter.dashboard import data, ui

ui.header("Golden Fighter AI - Retention Intelligence",
          "About one gym member in four leaves. This project predicts **who is likely to leave at the start of a "
          "month**, so the retention team can offer help in time, and explains why.")


def render():
    facts, evaluation = ui.load("facts"), ui.load("evaluation")
    test = evaluation["results"][evaluation["selected_model"]]["test"]
    cols = st.columns(5)
    cols[0].metric("Members analysed", f"{facts['members']:,}")
    cols[1].metric("Left the club", ui.pct(facts["churn_rate"]), help="Share of members who churned.")
    cols[2].metric("Leavers caught", ui.pct(test["recall"]),
                   help="Recall: share of real leavers the model flags, on members it never saw in training.")
    cols[3].metric("Correct warnings", ui.pct(test["precision"]),
                   help="Precision: share of flagged members who really leave (test set).")
    cols[4].metric("ROC-AUC", f"{test['roc_auc']:.3f}",
                   help="1.0 = perfect ranking of leavers above stayers; 0.5 = a coin toss.")

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("What the AI agents found")
        summary = data.agent_section(ui.paths().analyst_dir / "insights.md")
        if summary:
            st.markdown(summary)
            st.caption("Written by the EDA & Insights agent. Before saving, code checked that every number in it "
                       "appears in the computed facts.")
        else:
            st.info("No agent-written summary yet (the free run writes reports without AI text).")
    with right:
        st.subheader("Latest full run")
        history = data.run_history(ui.paths())
        full = history[history["Kind"] == "Full runs (with AI agents)"] if not history.empty else history
        if full.empty:
            st.info("No full run with the AI agents is recorded yet.")
        else:
            run = full.iloc[0]
            st.markdown(f"**{run['Result']}** - {run['Started (UTC)']} UTC  \n"
                        f"Model: `{run['Model']}`  \n"
                        f"{run['Model API calls']} model API calls, {run['Agent tool executions']} agent tool "
                        f"executions, {run['Tokens']:,} tokens  \n"
                        f"Estimated cost: **${run['Est. cost (USD)']:.4f}**")
        st.subheader("Design rule")
        st.markdown("- **Numbers come from tested Python code.** AI agents run that code through tools and "
                    "explain the results.\n"
                    "- **The validation gate is plain Python.** No AI can talk its way past it.\n"
                    "- **Crew 2 only sees Crew 1's published files and the data contract**, never the raw data.")

    st.subheader("How the system works")
    st.graphviz_chart("""
digraph {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#FFF8E1", color="#B8860B", fontname="Helvetica",
        fontsize=16, margin="0.25,0.12"];
  edge [color="#666666", fontname="Helvetica", fontsize=14];
  raw [label="Public gym dataset\\n(4,000 members)"];
  c1 [label="Crew 1: Data Analyst\\n3 AI agents\\nclean, explore, write contract"];
  gate [label="Validation gate\\nplain Python, 88 checks", shape=diamond, fillcolor="#FFFFFF"];
  c2 [label="Crew 2: Data Scientist\\n3 AI agents\\nfeatures, models, model card"];
  out [label="Model + reports\\nrun_summary.json"];
  fail [label="STOP\\nfailure_report.md\\nno model trained", fillcolor="#FDECEA", color="#C62828"];
  raw -> c1 -> gate; gate -> c2 [label="passed"]; c2 -> out; gate -> fail [label="failed"];
}""", width="stretch")
    st.caption("Try it: the **Break-the-data demo** changes a copy of the data and shows the gate stopping the Flow.")


ui.guarded(render)
