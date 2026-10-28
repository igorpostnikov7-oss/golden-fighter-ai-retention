"""Flow runs & costs: what the CrewAI Flow did in each run, the agents, model API calls, tokens and cost."""
import pandas as pd
import streamlit as st

from golden_fighter.dashboard import data, ui

ui.header("Flow runs & costs", "The CrewAI Flow connects the two crews through the validation gate. Every run "
          "leaves a `run_summary.json`; this screen reads them.")


def render():
    st.subheader("The six AI agents and their tools")
    st.table(data.agents_overview().set_index("Agent"))  # a static table wraps the long goals
    st.caption("Agents decide which tool to call and write the explanations. The tools are tested Python code, so "
               "the numbers never depend on the AI.")

    p = ui.paths()
    history = data.run_history(p)
    st.subheader("Run history")
    if history.empty:
        st.info("No runs recorded yet. Use the button below to run the Flow for free.")
    else:
        full = history[history["Kind"] == "Full runs (with AI agents)"]
        cols = st.columns(4)
        cols[0].metric("Full runs with AI agents", len(full))
        cols[1].metric("Model API calls (all full runs)", int(full["Model API calls"].sum()))
        cols[2].metric("Tokens (all full runs)", f"{int(full['Tokens'].sum()):,}")
        cols[3].metric("Estimated API cost (all full runs)", f"${full['Est. cost (USD)'].sum():.4f}")
        st.dataframe(history, hide_index=True, width="stretch", column_config={
            "Est. cost (USD)": st.column_config.NumberColumn(format="$%.4f"),
            "Tokens": st.column_config.NumberColumn(format="%d")})
        with st.expander("Model API calls vs agent tool executions - what is the difference?"):
            for name, text in {"Model API calls": "Paid requests to the Claude API. They use tokens and cost money.",
                               "Agent tool executions": "Times an agent ran one of the project's Python tools "
                               "(cleaning, EDA, contract, training...). Free and deterministic.",
                               "Free runs and break demos": "Use no AI at all, so both counts are zero."}.items():
                st.markdown(f"- **{name}:** {text}")
        if len(full) > 0:
            st.altair_chart(ui.cost_bars(full), width="stretch")

        st.subheader("Run details")
        summaries = dict((s["run_id"], (kind, s)) for kind, s in data.run_summaries(p))
        run_id = st.selectbox("Choose a run", list(summaries), format_func=lambda r: f"{r} - {summaries[r][0]}")
        _show_run(*summaries[run_id])

    st.subheader("Run the Flow now")
    st.markdown("**Free run (no AI):** the same Flow with the tested pipelines doing the crews' work. It reads "
                "the raw dataset and writes to `artifacts/_no_llm/`, so the validated results above are not "
                "replaced. Takes about 10-30 seconds.")
    if st.button("Run the Flow without AI (free)", type="primary"):
        with st.spinner("Running preflight, Crew 1, the gate and Crew 2..."):
            st.session_state["free_run"] = data.run_free_flow().model_dump(mode="json")
    if "free_run" in st.session_state:
        st.markdown("**Result of your free run** (it is also listed in the run history after a refresh):")
        ui.show_flow_result(st.session_state["free_run"])
    st.info("**Runs with the AI agents cost money**, so this app cannot start them. They are started from the "
            "terminal, on purpose:\n\n`.\\.venv\\Scripts\\python.exe -m golden_fighter.flow.main --no-retries`\n\n"
            "With Claude Haiku 5.5 a full run costs about $0.007.", icon=":material/payments:")


def _show_run(kind: str, summary: dict) -> None:
    ui.show_flow_result(summary)
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Usage by crew**")
        rows = [{"Crew": name, "Model API calls": c.get("model_api_calls", 0),
                 "Agent tool executions": c.get("agent_tool_executions", 0),
                 "Source-check rejections": c.get("source_check_rejections", 0),
                 "Tokens": c.get("total_tokens", 0), "Est. cost (USD)": c.get("estimated_cost_usd") or 0.0}
                for name, c in summary.get("crews", {}).items()]
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                         column_config={"Est. cost (USD)": st.column_config.NumberColumn(format="$%.4f")})
        else:
            st.caption("No crew ran in this run.")
        if summary.get("metrics"):
            m = summary["metrics"]
            st.markdown(f"**Model from this run:** {data.MODEL_LABELS.get(m['selected_model'])}, test ROC-AUC "
                        f"{m['test']['roc_auc']:.3f}, recall {ui.pct(m['test']['recall'])}, precision "
                        f"{ui.pct(m['test']['precision'])}.")
    with right:
        st.markdown("**Files and whether this run produced them**")
        files = [{"Crew": group, "File": a["file"], "Produced by this run": "yes" if a["produced_by_this_run"]
                  else ("no - left over from an earlier run" if a["exists"] else "no - missing")}
                 for group, entries in summary.get("artifacts", {}).items() for a in entries]
        st.dataframe(pd.DataFrame(files), hide_index=True, width="stretch")


ui.guarded(render)
