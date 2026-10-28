"""Shared Streamlit pieces for the app: cached loaders, error boxes and the charts (Altair, one accent hue)."""
from collections.abc import Callable
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from golden_fighter.dashboard import data

ACCENT = "#B8860B"      # Golden Fighter gold: the one data hue
MUTED = "#9AA0A6"       # neutral for the comparison bar ("No", "stayed")
NO_AI_NOTE = "This app never calls the Claude API. Every number on screen comes from files written by tested code."


def paths() -> data.Paths:
    return data.Paths.current()


def _stamp(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def _cached(kind: str, path: str, stamp: float):
    p = data.Paths.current()
    return {"facts": data.facts, "evaluation": data.evaluation, "contract": data.contract}[kind](p)


def load(kind: str):
    """Cached facts / evaluation / contract; the cache refreshes when the file changes."""
    p = paths()
    file = {"facts": p.analyst_dir / "insight_facts.json", "evaluation": p.scientist_dir / "evaluation.json",
            "contract": p.analyst_dir / "dataset_contract.json"}[kind]
    return _cached(kind, str(file), _stamp(file))


def guarded(render: Callable[[], None]) -> None:
    """Run a screen; a missing artifact or bad input becomes a clear message instead of a stack trace."""
    try:
        render()
    except data.MissingArtifact as err:
        st.warning(f"**Some results are not available yet.** {err}", icon=":material/info:")
    except data.DataValidationError as err:
        st.error("**The data did not pass the checks:**\n\n" + "\n".join(f"- {i}" for i in err.issues),
                 icon=":material/block:")


def header(title: str, intro: str) -> None:
    st.title(title)
    st.markdown(intro)


def pct(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{value * 100:.{digits}f}%"


# --- charts ------------------------------------------------------------------------------------------------

def bar_by_group(df: pd.DataFrame, group_title: str, order: list[str] | None = None) -> alt.Chart:
    base = alt.Chart(df).encode(
        x=alt.X("group:N", title=group_title, sort=order or list(df["group"]), axis=alt.Axis(labelAngle=0)),
        y=alt.Y("churn_rate_pct:Q", title="Churn rate (%)", scale=alt.Scale(domain=[0, 100])),
        tooltip=[alt.Tooltip("group:N", title=group_title), alt.Tooltip("members:Q", title="Members", format=","),
                 alt.Tooltip("churn_rate_pct:Q", title="Churn rate (%)")])
    bars = base.mark_bar(color=ACCENT, cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=42)
    labels = base.mark_text(dy=-8).encode(text=alt.Text("churn_rate_pct:Q", format=".1f"))
    return (bars + labels).properties(height=280)


def bar_yes_no(df: pd.DataFrame) -> alt.Chart:
    base = alt.Chart(df).encode(
        y=alt.Y("factor:N", title=None, sort=None, axis=alt.Axis(labelLimit=260)),
        yOffset=alt.YOffset("answer:N", sort=["Yes", "No"]),
        x=alt.X("churn_rate_pct:Q", title="Churn rate (%)", scale=alt.Scale(domain=[0, 60])),
        tooltip=[alt.Tooltip("factor:N", title="Factor"), alt.Tooltip("answer:N", title="Answer"),
                 alt.Tooltip("churn_rate_pct:Q", title="Churn rate (%)")])
    bars = base.mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
        color=alt.Color("answer:N", title="Answer", sort=["Yes", "No"],
                        scale=alt.Scale(domain=["Yes", "No"], range=[ACCENT, MUTED])))
    labels = base.mark_text(align="left", dx=4).encode(text=alt.Text("churn_rate_pct:Q", format=".1f"))
    return (bars + labels).properties(height=360)


def confusion_heatmap(df: pd.DataFrame) -> alt.Chart:
    """2x2 grid: correct answers in gold, mistakes in grey (size is in the number, not the colour)."""
    df = df.assign(said=df["predicted"].str.replace("Predicted to ", "Said: "),
                   outcome=["Correct", "Mistake", "Mistake", "Correct"])
    base = alt.Chart(df).encode(
        x=alt.X("said:N", title="What the model said", sort=["Said: stay", "Said: leave"],
                axis=alt.Axis(labelAngle=0, orient="top", labelFontSize=13)),
        y=alt.Y("actual:N", title="What really happened", sort=["Stayed", "Left"], axis=alt.Axis(labelFontSize=13)),
        tooltip=[alt.Tooltip("meaning:N", title="Meaning"), alt.Tooltip("members:Q", title="Members")])
    cells = base.mark_rect(cornerRadius=6, stroke="white", strokeWidth=4).encode(
        color=alt.Color("outcome:N", title=None, scale=alt.Scale(domain=["Correct", "Mistake"],
                                                                 range=["#F2D27A", "#E3E5E8"]),
                        legend=alt.Legend(orient="bottom")))
    text = base.mark_text(fontSize=26, fontWeight="bold", color="#202124").encode(text="members:Q")
    return (cells + text).properties(height=280, width=380)


def importance_bars(df: pd.DataFrame) -> alt.Chart:
    base = alt.Chart(df).encode(
        y=alt.Y("label:N", title=None, sort="-x", axis=alt.Axis(labelLimit=320)),
        x=alt.X("auc_drop_mean:Q", title="Drop in ROC-AUC when this input is shuffled"),
        tooltip=[alt.Tooltip("label:N", title="Input"), alt.Tooltip("auc_drop_mean:Q", title="ROC-AUC drop"),
                 alt.Tooltip("auc_drop_std:Q", title="Spread (std)")])
    bars = base.mark_bar(color=ACCENT, cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
    labels = base.mark_text(align="left", dx=4).encode(text=alt.Text("auc_drop_mean:Q", format=".3f"))
    return (bars + labels).properties(height=320)


def roc_chart(df: pd.DataFrame, auc: float) -> alt.Chart:
    curve = alt.Chart(df).mark_line(color=ACCENT, strokeWidth=2).encode(
        x=alt.X("false_alarm_rate:Q", title="False alarms (share of loyal members flagged)"),
        y=alt.Y("churners_caught:Q", title="Churners caught (recall)"),
        tooltip=[alt.Tooltip("false_alarm_rate:Q", format=".2f", title="False alarm rate"),
                 alt.Tooltip("churners_caught:Q", format=".2f", title="Churners caught")])
    chance = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color=MUTED, strokeDash=[4, 4]).encode(
        x="x:Q", y="y:Q")
    return (chance + curve).properties(height=320, title=f"ROC curve of the selected model (area = {auc:.3f})")


def cost_bars(df: pd.DataFrame) -> alt.Chart:
    """Estimated API cost per full run. Column names with dots or brackets break Vega-Lite fields, so rename."""
    plot = df.rename(columns={"Est. cost (USD)": "cost", "Model API calls": "calls"})[
        ["Run", "Model", "calls", "Tokens", "cost"]]
    base = alt.Chart(plot).encode(
        x=alt.X("Run:N", title="Full run", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("cost:Q", title="Estimated cost (USD)", axis=alt.Axis(format="$.3f")),
        tooltip=["Run", "Model", alt.Tooltip("calls:Q", title="Model API calls"), "Tokens",
                 alt.Tooltip("cost:Q", title="Estimated cost (USD)", format="$.4f")])
    return (base.mark_bar(color=ACCENT, cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=42)
            + base.mark_text(dy=-8).encode(text=alt.Text("cost:Q", format="$.4f"))).properties(height=240)


STAGE_LABELS = {"preflight": "1. Preflight checks", "analyst": "2. Crew 1 - Data Analyst",
                "validation_gate": "3. Validation gate", "scientist": "4. Crew 2 - Data Scientist"}
STATUS_ICONS = {"ok": "✅ passed", "skipped": "⏭️ not run", "validation_failed": "⛔ stopped by data checks",
                "error": "❌ error"}


def stages_table(summary: dict) -> pd.DataFrame:
    return pd.DataFrame([{"Step": STAGE_LABELS.get(s["name"], s["name"]),
                          "Result": STATUS_ICONS.get(s["status"], s["status"]),
                          "Seconds": s["seconds"], "Detail": (s.get("detail") or "").splitlines()[0][:160]
                          if s.get("detail") else ""} for s in summary["stages"]])


def show_flow_result(summary: dict) -> None:
    """Outcome of one Flow run in plain words, then the details."""
    status = summary["status"]
    if status == "succeeded":
        st.success(f"**The Flow finished successfully** (run `{summary['run_id']}`).", icon=":material/check_circle:")
    elif status == "validation_failed":
        st.error(f"**The Flow stopped at: {STAGE_LABELS.get(summary['failed_stage'], summary['failed_stage'])}.** "
                 "The data broke the contract, so the Data Scientist crew never started and no model was trained.",
                 icon=":material/block:")
    else:
        st.error(f"**The Flow stopped with an error at {summary['failed_stage']}:** {summary['failure_reason']}",
                 icon=":material/error:")
    st.dataframe(stages_table(summary), hide_index=True, width="stretch")
    if summary.get("validation_issues"):
        st.markdown("**What the gate found** (each line names the column and the rule it breaks):")
        for issue in summary["validation_issues"]:
            st.markdown(f"- `{issue}`")
