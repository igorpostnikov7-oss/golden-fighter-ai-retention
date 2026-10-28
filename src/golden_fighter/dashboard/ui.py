"""Shared Streamlit pieces for the app: right-to-left layout, cached loaders, error boxes, tables and the charts
(Altair, one accent hue)."""
import re
from collections.abc import Callable
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from golden_fighter.dashboard import data, hebrew

ACCENT = "#B8860B"      # Golden Fighter gold: the one data hue
MUTED = "#9AA0A6"       # neutral for the comparison bar ("No", "stayed")
NO_AI_NOTE = "האפליקציה אף פעם לא פונה ל-Claude API. כל מספר שמוצג כאן מגיע מקבצים שנכתבו על ידי קוד נבדק."

# Streamlit has no right-to-left switch, so the app sets it with CSS on the page content and the sidebar content
# (not on the whole app: that flips Streamlit's own layout and the page slides under the sidebar). Charts and
# the embedded EDA report keep left-to-right (their axes and numbers read that way); code, file names and English
# documents (containers with a key starting with "ltr") stay left-to-right so their characters keep their order.
RTL_CSS = """
<style>
[data-testid="stMain"], [data-testid="stSidebarContent"] { direction: rtl; }
[data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"], [data-testid="stHeading"],
[data-testid="stMetricLabel"], [data-testid="stWidgetLabel"], [data-testid="stExpander"] summary,
[data-testid="stTable"] th, [data-testid="stTable"] td { text-align: right; }
code, pre { direction: ltr; unicode-bidi: isolate; }
[data-testid="stVegaLiteChart"], [data-testid="stIFrame"],
[data-testid="stNumberInputContainer"] { direction: ltr; }
[class*="st-key-ltr"], [class*="st-key-ltr"] [data-testid="stMarkdownContainer"] { direction: ltr; text-align: left; }
</style>
"""


def apply_rtl() -> None:
    st.html(RTL_CSS)


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
        st.warning(f"**חלק מהתוצאות עדיין לא זמינות.** {err}", icon=":material/info:")
    except data.DataValidationError as err:
        st.error("**הנתונים לא עברו את הבדיקות:**\n\n" + "\n".join(f"- `{i}`" for i in err.issues),
                 icon=":material/block:")


def header(title: str, intro: str) -> None:
    st.title(title)
    st.markdown(intro)


def pct(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{value * 100:.{digits}f}%"


def english_document(key: str, markdown: str) -> None:
    """An English artifact (report, model card) shown as written, left to right."""
    with st.container(key=f"ltr_{key}"):
        st.markdown(markdown)


def table(df: pd.DataFrame, formats: dict[str, str] | None = None) -> None:
    """A static table (it follows the right-to-left layout, unlike the interactive grid). formats: column -> format."""
    shown = df.copy()
    for column, fmt in (formats or {}).items():
        shown[column] = [fmt.format(v) if pd.notna(v) else "-" for v in shown[column]]
    st.table(shown, hide_index=True)


# --- charts ------------------------------------------------------------------------------------------------
# Charts are drawn left to right, which scrambles Hebrew text that contains English, digits or brackets.
# Wrapping it in right-to-left embedding marks (RLE ... PDF) keeps it in reading order.

def _r(text: str) -> str:
    return f"\u202b{text}\u202c" if text and re.search("[\u0590-\u05ff]", text) else text


def _rtl_columns(df: pd.DataFrame, *columns: str) -> pd.DataFrame:
    return df.assign(**{c: [_r(str(v)) for v in df[c]] for c in columns})



def bar_by_group(df: pd.DataFrame, group_title: str, order: list[str] | None = None) -> alt.Chart:
    df = _rtl_columns(df, "group")
    base = alt.Chart(df).encode(
        x=alt.X("group:N", title=_r(group_title), sort=order or list(df["group"]), axis=alt.Axis(labelAngle=0)),
        y=alt.Y("churn_rate_pct:Q", title=_r("שיעור עזיבה (%)"), scale=alt.Scale(domain=[0, 100])),
        tooltip=[alt.Tooltip("group:N", title=_r(group_title)),
                 alt.Tooltip("members:Q", title=_r("מתאמנים"), format=","),
                 alt.Tooltip("churn_rate_pct:Q", title=_r("שיעור עזיבה (%)"))])
    bars = base.mark_bar(color=ACCENT, cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=42)
    labels = base.mark_text(dy=-8).encode(text=alt.Text("churn_rate_pct:Q", format=".1f"))
    return (bars + labels).properties(height=280)


def bar_yes_no(df: pd.DataFrame) -> alt.Chart:
    df, yes_no = _rtl_columns(df, "factor", "answer"), [_r("כן"), _r("לא")]
    base = alt.Chart(df).encode(
        y=alt.Y("factor:N", title=None, sort=None, axis=alt.Axis(labelLimit=260)),
        yOffset=alt.YOffset("answer:N", sort=yes_no),
        x=alt.X("churn_rate_pct:Q", title=_r("שיעור עזיבה (%)"), scale=alt.Scale(domain=[0, 60])),
        tooltip=[alt.Tooltip("factor:N", title=_r("גורם")), alt.Tooltip("answer:N", title=_r("תשובה")),
                 alt.Tooltip("churn_rate_pct:Q", title=_r("שיעור עזיבה (%)"))])
    bars = base.mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
        color=alt.Color("answer:N", title=_r("תשובה"), sort=yes_no,
                        scale=alt.Scale(domain=yes_no, range=[ACCENT, MUTED])))
    labels = base.mark_text(align="left", dx=4).encode(text=alt.Text("churn_rate_pct:Q", format=".1f"))
    return (bars + labels).properties(height=360)


def confusion_heatmap(df: pd.DataFrame) -> alt.Chart:
    """2x2 grid: correct answers in gold, mistakes in grey (size is in the number, not the colour)."""
    right, wrong = _r("נכון"), _r("טעות")
    df = _rtl_columns(df.assign(outcome=[right, wrong, wrong, right]), "predicted", "actual", "meaning")
    base = alt.Chart(df).encode(
        x=alt.X("predicted:N", title=_r("מה המודל אמר"), sort=[_r(data.SAID_STAY), _r(data.SAID_LEAVE)],
                axis=alt.Axis(labelAngle=0, orient="top", labelFontSize=13)),
        y=alt.Y("actual:N", title=_r("מה קרה בפועל"), sort=[_r(data.STAYED), _r(data.LEFT)],
                axis=alt.Axis(labelFontSize=13)),
        tooltip=[alt.Tooltip("meaning:N", title=_r("משמעות")), alt.Tooltip("members:Q", title=_r("מתאמנים"))])
    cells = base.mark_rect(cornerRadius=6, stroke="white", strokeWidth=4).encode(
        color=alt.Color("outcome:N", title=None, scale=alt.Scale(domain=[right, wrong],
                                                                 range=["#F2D27A", "#E3E5E8"]),
                        legend=alt.Legend(orient="bottom")))
    text = base.mark_text(fontSize=26, fontWeight="bold", color="#202124").encode(text="members:Q")
    return (cells + text).properties(height=280, width=380)


def importance_bars(df: pd.DataFrame) -> alt.Chart:
    df = _rtl_columns(df, "label")
    base = alt.Chart(df).encode(
        y=alt.Y("label:N", title=None, sort="-x", axis=alt.Axis(labelLimit=320)),
        x=alt.X("auc_drop_mean:Q", title=_r("הירידה ב-ROC-AUC כשמערבבים את הנתון הזה")),
        tooltip=[alt.Tooltip("label:N", title=_r("נתון")),
                 alt.Tooltip("auc_drop_mean:Q", title=_r("ירידה ב-ROC-AUC")),
                 alt.Tooltip("auc_drop_std:Q", title=_r("פיזור (סטיית תקן)"))])
    bars = base.mark_bar(color=ACCENT, cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
    labels = base.mark_text(align="left", dx=4).encode(text=alt.Text("auc_drop_mean:Q", format=".3f"))
    return (bars + labels).properties(height=320)


def roc_chart(df: pd.DataFrame, auc: float) -> alt.Chart:
    curve = alt.Chart(df).mark_line(color=ACCENT, strokeWidth=2).encode(
        x=alt.X("false_alarm_rate:Q", title=_r("אזעקות שווא (חלק מהנשארים שסומנו)")),
        y=alt.Y("churners_caught:Q", title=_r("עוזבים שזוהו (Recall)")),
        tooltip=[alt.Tooltip("false_alarm_rate:Q", format=".2f", title=_r("שיעור אזעקות שווא")),
                 alt.Tooltip("churners_caught:Q", format=".2f", title=_r("עוזבים שזוהו"))])
    chance = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color=MUTED, strokeDash=[4, 4]).encode(
        x="x:Q", y="y:Q")
    return (chance + curve).properties(height=320, title=_r(f"עקומת ROC של המודל שנבחר (שטח = {auc:.3f})"))


def cost_bars(df: pd.DataFrame) -> alt.Chart:
    """Estimated API cost per full run. Vega-Lite field names must be plain (no dots or brackets), so rename."""
    plot = df.rename(columns={data.RUN: "run", "מודל שפה": "model", data.API_CALLS: "calls", data.TOKENS: "tokens",
                              data.COST: "cost"})[["run", "model", "calls", "tokens", "cost"]]
    base = alt.Chart(plot).encode(
        x=alt.X("run:N", title=_r("ריצה מלאה"), axis=alt.Axis(labelAngle=0)),
        y=alt.Y("cost:Q", title=_r("עלות משוערת (USD)"), axis=alt.Axis(format="$.3f")),
        tooltip=[alt.Tooltip("run:N", title=_r("ריצה")), alt.Tooltip("model:N", title=_r("מודל שפה")),
                 alt.Tooltip("calls:Q", title=_r("קריאות API למודל")),
                 alt.Tooltip("tokens:Q", title=_r("טוקנים"), format=","),
                 alt.Tooltip("cost:Q", title=_r("עלות משוערת (USD)"), format="$.4f")])
    return (base.mark_bar(color=ACCENT, cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=42)
            + base.mark_text(dy=-8).encode(text=alt.Text("cost:Q", format="$.4f"))).properties(height=240)


# --- Flow results ------------------------------------------------------------------------------------------

def stages_table(summary: dict) -> pd.DataFrame:
    return pd.DataFrame([{"שלב": hebrew.text(hebrew.STAGES, s["name"]),
                          "תוצאה": hebrew.text(hebrew.STAGE_STATUS, s["status"]),
                          "שניות": s["seconds"], "פירוט": (s.get("detail") or "").splitlines()[0][:160]
                          if s.get("detail") else ""} for s in summary["stages"]])


def show_flow_result(summary: dict) -> None:
    """Outcome of one Flow run in plain words, then the details."""
    status = summary["status"]
    if status == "succeeded":
        st.success(f"**ה-Flow הסתיים בהצלחה** (ריצה `{summary['run_id']}`).", icon=":material/check_circle:")
    elif status == "validation_failed":
        st.error(f"**ה-Flow נעצר בשלב: {hebrew.text(hebrew.STAGES, summary['failed_stage'])}.** "
                 "הנתונים הפרו את החוזה, ולכן צוות מדעני הנתונים לא התחיל ואף מודל לא אומן.",
                 icon=":material/block:")
    else:
        st.error(f"**ה-Flow נעצר בגלל שגיאה בשלב {hebrew.text(hebrew.STAGES, summary['failed_stage'])}:** "
                 f"`{summary['failure_reason']}`", icon=":material/error:")
    table(stages_table(summary), {"שניות": "{:.1f}"})
    if summary.get("validation_issues"):
        st.markdown("**מה השער מצא** (כל שורה מציינת את העמודה ואת הכלל שהיא מפרה; ההודעות הטכניות נשארות "
                    "באנגלית, כמו ביומן ובדוח הכשל):")
        for issue in summary["validation_issues"]:
            st.markdown(f"- `{issue}`")
