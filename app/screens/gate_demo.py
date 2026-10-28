"""Break-the-data demo: damage a COPY of Crew 1's output and watch the gate stop the Flow."""
from pathlib import Path

import streamlit as st

from golden_fighter import config
from golden_fighter.dashboard import data, ui
from golden_fighter.flow.artifacts import sha256

ui.header("Break-the-data demo", "A real incident: someone changes the money column from dollars to cents. Every "
          "number is now 100x larger, but nothing crashes. Without a check, the model would quietly learn "
          "nonsense. Here the **validation gate** (plain Python, no AI) compares the data with its contract and "
          "stops the Flow before any model is trained.")


def _fingerprints() -> dict:
    files = sorted(config.ANALYST_DIR.glob("*")) + [config.RAW_DATA_PATH]
    return {f.name: sha256(f) for f in files if f.is_file()}


def render():
    st.subheader("1. The real data today")
    result = data.gate_result(ui.paths())
    if result.passed:
        st.success(f"The current data passes the gate: {result.checks_run} checks, 0 errors.",
                   icon=":material/verified:")
    else:
        st.error(f"The current data fails the gate ({len(result.errors)} errors).", icon=":material/block:")
        for issue in result.errors:
            st.markdown(f"- `{issue}`")

    st.subheader("2. Break a copy and run the Flow")
    scenarios = data.break_scenarios()
    scenario = st.selectbox("What goes wrong?", list(scenarios), format_func=lambda s: f"{s}: {scenarios[s]}",
                            index=list(scenarios).index("unit_x100"))
    st.caption("The original files are never changed: the damage is applied to a copy in `artifacts/_broken/`. "
               "This demo uses no AI and costs nothing.")
    if st.button("Break a copy and run the Flow", type="primary"):
        before = _fingerprints()
        with st.spinner("Copying, breaking the copy and running the Flow..."):
            state = data.run_break_demo(scenario)
        st.session_state["demo"] = (scenario, state.model_dump(mode="json"), before == _fingerprints())

    if "demo" in st.session_state:
        scenario, summary, untouched = st.session_state["demo"]
        st.subheader(f"3. Result for '{scenario}'")
        ui.show_flow_result(summary)
        cols = st.columns(3)
        cols[0].metric("Model API calls", summary["totals"].get("model_api_calls", 0))
        cols[1].metric("Model trained", "no" if summary["status"] != "succeeded" else "yes")
        cols[2].metric("Original data unchanged", "yes" if untouched else "NO")
        report = summary.get("paths", {}).get("failure_report")
        if report:
            path = Path(report) if Path(report).is_absolute() else config.ROOT / report
            with st.expander(f"The failure report the Flow wrote ({report})"):
                st.markdown(path.read_text(encoding="utf-8") if path.exists() else "The report file was removed.")


ui.guarded(render)
