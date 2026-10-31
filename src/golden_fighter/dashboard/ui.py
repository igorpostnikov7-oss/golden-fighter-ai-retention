"""Shared Streamlit pieces for the app: right-to-left layout, cached loaders, error boxes, tables and the charts
(Altair, one accent hue)."""
import re
from collections.abc import Callable
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from pandas.io.formats.style import Styler

from golden_fighter.dashboard import data, hebrew

ACCENT = "#B8860B"      # Golden Fighter gold: the one data hue
MUTED = "#9AA0A6"       # neutral for the comparison bar ("No", "stayed")
NO_AI_NOTE = "האפליקציה אף פעם לא פונה ל-Claude API. כל מספר שמוצג כאן מגיע מקבצים שנכתבו על ידי קוד נבדק."
DATA_NOTE = ("הנתונים: מערך נתונים ציבורי מ-Kaggle של רשת חדרי כושר חיצונית (4,000 מתאמנים אנונימיים). "
             "**אלה לא לקוחות אמיתיים של Golden Fighter**, ואין כאן נתונים פרטיים או חיבור ל-CRM.")

# Streamlit has no right-to-left switch. Two official features do most of the work: the navigation sits in the
# top bar (st.navigation position="top"), so there is no left sidebar to flip, and the theme in
# .streamlit/config.toml sets a Hebrew-friendly font and darker alert text. This small stylesheet only sets the
# writing direction on the page content and the top bar, using Streamlit's stable data-testid names; it changes no
# layout sizes. Charts and the number inputs keep left-to-right (axes and digits read that way); code, file names
# and English documents (containers with a key starting with "ltr") stay left-to-right so their order is kept.
RTL_CSS = """
<style>
[data-testid="stMain"], [data-testid="stToolbar"] { direction: rtl; }
[data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"], [data-testid="stHeading"],
[data-testid="stMetricLabel"], [data-testid="stWidgetLabel"], [data-testid="stExpander"] summary,
[data-testid="stTable"] th, [data-testid="stTable"] td { text-align: right; }
code, pre { direction: ltr; unicode-bidi: isolate; }
[data-testid="stVegaLiteChart"], [data-testid="stIFrame"], [data-testid="stNumberInputContainer"] { direction: ltr; }
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
    return {"facts": data.facts, "evaluation": data.evaluation, "contract": data.contract,
            "cleaning": data.cleaning_report, "clean_data": data.clean_data}[kind](p)


def load(kind: str):
    """Cached facts / evaluation / contract / cleaning report / clean data; refreshes when the file changes."""
    p = paths()
    file = {"facts": p.analyst_dir / "insight_facts.json", "evaluation": p.scientist_dir / "evaluation.json",
            "contract": p.analyst_dir / "dataset_contract.json", "cleaning": p.analyst_dir / "cleaning_report.json",
            "clean_data": p.analyst_dir / "clean_data.csv"}[kind]
    return _cached(kind, str(file), _stamp(file))


def guarded(render: Callable[[], None]) -> None:
    """Run a screen; a missing artifact or bad input becomes a clear message instead of a stack trace."""
    try:
        render()
    except data.MissingArtifact as err:
        st.warning(f"**חלק מהתוצאות עדיין לא זמינות.** {err}", icon=":material/info:")
    except data.DataValidationError as err:
        st.error("**הנתונים לא עברו את הבדיקות:**\n\n"
                 + "\n".join(f"- {hebrew.gate_issue(i)['text']}" for i in err.issues), icon=":material/block:")
        technical_details(err.issues)


def header(title: str, intro: str) -> None:
    st.title(title)
    st.markdown(intro)


def footer() -> None:
    st.divider()
    st.caption(f"{NO_AI_NOTE} {DATA_NOTE}")


def pct(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{value * 100:.{digits}f}%"


def english_document(key: str, markdown: str) -> None:
    """An English artifact (report, model card) shown as written, left to right."""
    with st.container(key=f"ltr_{key}"):
        st.markdown(markdown)


def table(df: pd.DataFrame | Styler, formats: dict[str, str] | None = None) -> None:
    """A static table (it follows the right-to-left layout, unlike the interactive grid). formats: column -> format.
    Cells are rendered as Markdown, so a cell must never start with "1. " (that becomes a list)."""
    if isinstance(df, Styler):
        st.table(df, hide_index=True)
        return
    shown = df.copy()
    for column, fmt in (formats or {}).items():
        shown[column] = [fmt.format(v) if pd.notna(v) else "-" for v in shown[column]]
    st.table(shown, hide_index=True)


def technical_details(issues: list[str], label: str = "הפרטים הטכניים המקוריים (באנגלית, לביקורת)") -> None:
    with st.expander(label):
        st.code("\n".join(issues), language=None, wrap_lines=True)


def explain_gate_issues(issues: list[str]) -> None:
    """Each gate issue in plain Hebrew, then the exact original lines for auditing."""
    for item in map(hebrew.gate_issue, issues):
        kind = f"**{item['check_he']}:** " if item["check_he"] else ""
        st.markdown(f"- {kind}{item['text']}")
    technical_details(issues)


# --- charts ------------------------------------------------------------------------------------------------
# Charts are drawn left to right, which scrambles Hebrew text that contains English, digits or brackets.
# Wrapping it in right-to-left embedding marks (RLE ... PDF) keeps it in reading order.

def _r(text: str) -> str:
    return f"‫{text}‬" if text and re.search("[֐-׿]", text) else text


def _rtl_columns(df: pd.DataFrame, *columns: str) -> pd.DataFrame:
    return df.assign(**{c: [_r(str(v)) for v in df[c]] for c in columns})


def _style(chart) -> alt.Chart:
    """Larger, calmer chart text for every chart in the app."""
    return (chart.configure_axis(labelFontSize=13, titleFontSize=14, titleFontWeight="normal", titleColor="#3c4043",
                                 labelColor="#3c4043")
            .configure_legend(labelFontSize=13, titleFontSize=13, titleFontWeight="normal")
            .configure_view(strokeWidth=0))


def bar_by_group(df: pd.DataFrame, group_title: str, order: list[str] | None = None) -> alt.Chart:
    df = _rtl_columns(df, "group")
    base = alt.Chart(df).encode(
        x=alt.X("group:N", title=_r(group_title), sort=order or list(df["group"]),
                axis=alt.Axis(labelAngle=0, labelLimit=200)),
        y=alt.Y("churn_rate_pct:Q", title=_r("שיעור עזיבה (%)"), scale=alt.Scale(domain=[0, 100])),
        tooltip=[alt.Tooltip("group:N", title=_r(group_title)),
                 alt.Tooltip("members:Q", title=_r("מתאמנים"), format=","),
                 alt.Tooltip("churn_rate_pct:Q", title=_r("שיעור עזיבה (%)"))])
    bars = base.mark_bar(color=ACCENT, cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=48)
    labels = base.mark_text(dy=-10, fontSize=14, fontWeight="bold").encode(
        text=alt.Text("churn_rate_pct:Q", format=".1f"))
    return _style((bars + labels).properties(height=300))


def bar_yes_no(df: pd.DataFrame) -> alt.Chart:
    df, yes_no = _rtl_columns(df, "factor", "answer"), [_r("כן"), _r("לא")]
    base = alt.Chart(df).encode(
        y=alt.Y("factor:N", title=None, sort=None, axis=alt.Axis(labelLimit=320)),
        yOffset=alt.YOffset("answer:N", sort=yes_no),
        x=alt.X("churn_rate_pct:Q", title=_r("שיעור עזיבה (%)"), scale=alt.Scale(domain=[0, 60])),
        tooltip=[alt.Tooltip("factor:N", title=_r("גורם")), alt.Tooltip("answer:N", title=_r("תשובה")),
                 alt.Tooltip("churn_rate_pct:Q", title=_r("שיעור עזיבה (%)"))])
    bars = base.mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
        color=alt.Color("answer:N", title=_r("יש את המאפיין?"), sort=yes_no,
                        scale=alt.Scale(domain=yes_no, range=[ACCENT, MUTED]), legend=alt.Legend(orient="top")))
    labels = base.mark_text(align="left", dx=4, fontSize=12).encode(text=alt.Text("churn_rate_pct:Q", format=".1f"))
    return _style((bars + labels).properties(height=380))


def confusion_heatmap(df: pd.DataFrame) -> alt.Chart:
    """2x2 grid: correct answers in gold, mistakes in grey; each cell names its outcome under the count."""
    right, wrong = _r("תחזית נכונה"), _r("טעות")
    df = _rtl_columns(df.assign(outcome=[right, wrong, wrong, right]), "predicted", "actual", "meaning", "short")
    base = alt.Chart(df).encode(
        x=alt.X("predicted:N", title=_r("מה המודל חזה"), sort=[_r(data.SAID_STAY), _r(data.SAID_LEAVE)],
                axis=alt.Axis(labelAngle=0, orient="top", labelFontSize=14)),
        y=alt.Y("actual:N", title=_r("מה קרה בפועל"), sort=[_r(data.STAYED), _r(data.LEFT)],
                axis=alt.Axis(labelFontSize=14, labelAngle=0)),
        tooltip=[alt.Tooltip("meaning:N", title=_r("משמעות")), alt.Tooltip("members:Q", title=_r("מתאמנים"))])
    cells = base.mark_rect(cornerRadius=6, stroke="white", strokeWidth=4).encode(
        color=alt.Color("outcome:N", title=None, scale=alt.Scale(domain=[right, wrong], range=["#F2D27A", "#E3E5E8"]),
                        legend=alt.Legend(orient="bottom")))
    count = base.mark_text(fontSize=30, fontWeight="bold", color="#202124", dy=-10).encode(text="members:Q")
    name = base.mark_text(fontSize=13, color="#3c4043", dy=18).encode(text="short:N")
    return _style((cells + count + name).properties(height=320))


def importance_bars(df: pd.DataFrame) -> alt.Chart:
    df = _rtl_columns(df, "label")
    base = alt.Chart(df).encode(
        y=alt.Y("label:N", title=None, sort="-x", axis=alt.Axis(labelLimit=420)),
        x=alt.X("auc_drop_mean:Q", title=_r("כמה ROC-AUC יורד כשמערבבים את הנתון (יותר = חשוב יותר)")),
        tooltip=[alt.Tooltip("label:N", title=_r("נתון")),
                 alt.Tooltip("auc_drop_mean:Q", title=_r("ירידה ב-ROC-AUC")),
                 alt.Tooltip("auc_drop_std:Q", title=_r("פיזור (סטיית תקן)"))])
    bars = base.mark_bar(color=ACCENT, cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
    labels = base.mark_text(align="left", dx=4, fontSize=12).encode(text=alt.Text("auc_drop_mean:Q", format=".3f"))
    return _style((bars + labels).properties(height=380))


def roc_chart(df: pd.DataFrame) -> alt.Chart:
    curve = alt.Chart(df).mark_line(color=ACCENT, strokeWidth=3).encode(
        x=alt.X("false_alarm_rate:Q", title=_r("אזעקות שווא (חלק מהנשארים שסומנו)"), scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("churners_caught:Q", title=_r("עוזבים שזוהו (Recall)"), scale=alt.Scale(domain=[0, 1])),
        tooltip=[alt.Tooltip("false_alarm_rate:Q", format=".2f", title=_r("שיעור אזעקות שווא")),
                 alt.Tooltip("churners_caught:Q", format=".2f", title=_r("עוזבים שזוהו"))])
    chance = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color=MUTED, strokeDash=[4, 4]).encode(
        x="x:Q", y="y:Q")
    return _style((chance + curve).properties(height=340))


def cost_bars(df: pd.DataFrame) -> alt.Chart:
    """Estimated API cost per full run. Vega-Lite field names must be plain (no dots or brackets), so rename."""
    plot = df.rename(columns={data.STARTED: "when", "מודל שפה": "model", data.API_CALLS: "calls",
                              data.TOKENS: "tokens", data.COST: "cost"})[["when", "model", "calls", "tokens", "cost"]]
    base = alt.Chart(plot).encode(
        x=alt.X("when:N", title=_r("ריצה מלאה (תאריך ושעה, UTC)"), axis=alt.Axis(labelAngle=0, labelLimit=300)),
        y=alt.Y("cost:Q", title=_r("עלות משוערת (USD)"), axis=alt.Axis(format="$.3f")),
        tooltip=[alt.Tooltip("when:N", title=_r("מתי")), alt.Tooltip("model:N", title=_r("מודל שפה")),
                 alt.Tooltip("calls:Q", title=_r("קריאות API בתשלום")),
                 alt.Tooltip("tokens:Q", title=_r("טוקנים"), format=","),
                 alt.Tooltip("cost:Q", title=_r("עלות משוערת (USD)"), format="$.4f")])
    return _style((base.mark_bar(color=ACCENT, cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=56)
                   + base.mark_text(dy=-10, fontSize=14, fontWeight="bold").encode(
                       text=alt.Text("cost:Q", format="$.4f"))).properties(height=260))


def correlation_heatmap(df: pd.DataFrame) -> alt.Chart:
    """Correlation matrix: red = move together, blue = move in opposite directions (same colours as the EDA)."""
    df = _rtl_columns(df.assign(label=[f"{v:.2f}" for v in df["value"]]), "row", "col")
    order = list(dict.fromkeys(df["row"]))
    base = alt.Chart(df).encode(
        x=alt.X("col:N", title=None, sort=order,
                axis=alt.Axis(labelAngle=-55, labelLimit=220, orient="top", labelOverlap=False)),
        y=alt.Y("row:N", title=None, sort=order, axis=alt.Axis(labelLimit=220)),
        tooltip=[alt.Tooltip("row:N", title=_r("עמודה")), alt.Tooltip("col:N", title=_r("מול")),
                 alt.Tooltip("value:Q", title=_r("מתאם"), format=".3f")])
    cells = base.mark_rect(stroke="white", strokeWidth=1).encode(
        color=alt.Color("value:Q", title=_r("מתאם"), scale=alt.Scale(domain=[-1, 0, 1],
                                                                    range=["#2a78d6", "#f0efec", "#e34948"])))
    text = base.mark_text(fontSize=11).encode(
        text="label:N", color=alt.condition("abs(datum.value) > 0.6", alt.value("white"), alt.value("#202124")))
    return _style((cells + text).properties(height=620))


def auc_chart(df: pd.DataFrame) -> alt.Chart:
    """Single-column AUC from the 0.5 'no signal' line to at most the meaningful maximum of 1.0."""
    df = _rtl_columns(df, "column", "risk")
    high = _r("סיכון גבוה")
    base = alt.Chart(df).encode(
        y=alt.Y("column:N", title=None, sort="-x", axis=alt.Axis(labelLimit=320)),
        tooltip=[alt.Tooltip("column:N", title=_r("עמודה")), alt.Tooltip("auc:Q", title="AUC", format=".3f"),
                 alt.Tooltip("risk:N", title=_r("סיכון דליפה"))])
    bars = base.mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
        x=alt.X("auc:Q", title=_r("AUC של העמודה לבדה (0.5 = אין אות, 1.0 = מושלם)"),
                scale=alt.Scale(domain=[0.5, 1.0], clamp=True)),
        x2=alt.datum(0.5),
        color=alt.condition(alt.datum.risk == high, alt.value("#C62828"), alt.value(MUTED)))
    labels = base.mark_text(align="left", dx=4, fontSize=12).encode(x="auc:Q", text=alt.Text("auc:Q", format=".3f"))
    return _style((bars + labels).properties(height=420))


def distribution_chart(df: pd.DataFrame, column: str) -> alt.Chart:
    """Share of stayed vs churned members per value range of one column (from clean_data.csv)."""
    plot = pd.DataFrame({"value": df[column], "group": df["churn"].map({0: _r("נשארו"), 1: _r("עזבו")})})
    return _style(alt.Chart(plot).mark_bar(opacity=0.75, binSpacing=1).encode(
        x=alt.X("value:Q", bin=alt.Bin(maxbins=25), title=_r(hebrew.column(column))),
        y=alt.Y("count():Q", title=_r("מתאמנים"), stack=None),
        color=alt.Color("group:N", title=None, scale=alt.Scale(domain=[_r("נשארו"), _r("עזבו")],
                                                               range=[MUTED, ACCENT]),
                        legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("group:N", title=_r("קבוצה")), alt.Tooltip("count():Q", title=_r("מתאמנים"))],
    ).properties(height=300))


# --- Flow results ------------------------------------------------------------------------------------------

def stages_table(summary: dict) -> pd.DataFrame:
    return pd.DataFrame([{"שלב": hebrew.text(hebrew.STAGES, s["name"]),
                          "תוצאה": hebrew.text(hebrew.STAGE_STATUS, s["status"]),
                          "שניות": s["seconds"], "פירוט": hebrew.stage_detail(s.get("detail") or "")}
                         for s in summary["stages"]])


def show_flow_result(summary: dict) -> None:
    """Outcome of one Flow run in plain words, then the details."""
    status = summary["status"]
    if status == "succeeded":
        st.success(f"**ה-Flow הסתיים בהצלחה.** כל ארבעת השלבים עברו (ריצה `{summary['run_id']}`).",
                   icon=":material/check_circle:")
    elif status == "validation_failed":
        st.error(f"**ה-Flow נעצר בשלב: {hebrew.text(hebrew.STAGES, summary['failed_stage'])}.** "
                 "הנתונים הפרו את החוזה, ולכן צוות 2 (מדען הנתונים) לא התחיל ואף מודל לא אומן.",
                 icon=":material/block:")
    else:
        st.error(f"**ה-Flow נעצר בגלל שגיאה בשלב {hebrew.text(hebrew.STAGES, summary['failed_stage'])}:** "
                 f"{hebrew.stage_detail(summary['failure_reason'] or '')}", icon=":material/error:")
    table(stages_table(summary), {"שניות": "{:.1f}"})
    if summary.get("validation_issues"):
        st.markdown("**למה השער עצר את הנתונים:**")
        explain_gate_issues(summary["validation_issues"])
