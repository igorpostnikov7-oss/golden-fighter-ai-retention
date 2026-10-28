"""Self-contained EDA report (eda_report.html): descriptive statistics plus charts embedded as PNG."""
import base64
import html
import io
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from golden_fighter.analysis.schema_spec import ID_COLUMN, TARGET  # noqa: E402

log = logging.getLogger(__name__)

# Palette (dataviz reference palette, light mode): stayed = slot 1 blue, churned = slot 2 orange.
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
STAYED = "#2a78d6"
CHURNED = "#eb6834"
ORDINAL_LOW = "#86b6ef"
ORDINAL_HIGH = "#1c5cab"
CRITICAL = "#d03b3b"
NEUTRAL_MARK = "#9e9c96"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#2a78d6", "#f0efec", "#e34948"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_MUTED, "xtick.color": TEXT_MUTED, "ytick.color": TEXT_MUTED,
    "text.color": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
})


def _to_png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _bar_labels(ax, bars, fmt="{:.0%}"):
    for bar in bars:
        ax.annotate(fmt.format(bar.get_height()), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9, color=TEXT, xytext=(0, 2), textcoords="offset points")


def _rate_chart(table: pd.DataFrame, title: str, xlabel: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 3.2))
    bars = ax.bar(table["group"], table["churn_rate"], color=CHURNED, width=0.55, edgecolor=SURFACE, linewidth=2)
    _bar_labels(ax, bars)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Churn rate")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="x", visible=False)
    return _to_png(fig)


def _split_chart(df: pd.DataFrame) -> str:
    counts = df[TARGET].value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 2.2))
    bars = ax.barh(["Stayed", "Churned"], counts.values, color=[STAYED, CHURNED], height=0.55,
                   edgecolor=SURFACE, linewidth=2)
    for bar, n in zip(bars, counts.values):
        ax.annotate(f"{n:,} ({n / len(df):.1%})", (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    va="center", fontsize=9, xytext=(4, 0), textcoords="offset points")
    ax.set_title("Members who stayed vs churned")
    ax.set_xlabel("Members")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, counts.max() * 1.25)
    return _to_png(fig)


def _flag_chart(flags: dict) -> str:
    labels = [f["label"] for f in flags.values()]
    no = [f["rate_if_no"] for f in flags.values()]
    yes = [f["rate_if_yes"] for f in flags.values()]
    y = range(len(labels))
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.barh([i - 0.2 for i in y], yes, height=0.38, color=ORDINAL_HIGH, label="With attribute",
            edgecolor=SURFACE, linewidth=2)
    ax.barh([i + 0.2 for i in y], no, height=0.38, color=ORDINAL_LOW, label="Without attribute",
            edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(list(y), labels)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Churn rate")
    ax.set_title("Churn rate with vs without each member attribute")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    return _to_png(fig)


def _distribution_chart(df: pd.DataFrame, columns: list[str], titles: list[str]) -> str:
    fig, axes = plt.subplots(1, len(columns), figsize=(3.3 * len(columns), 2.9))
    for ax, col, title in zip(axes, columns, titles):
        bins = 25 if df[col].nunique() > 25 else sorted(df[col].unique().tolist()) + [df[col].max() + 1]
        for value, color, label in ((0, STAYED, "Stayed"), (1, CHURNED, "Churned")):
            ax.hist(df.loc[df[TARGET] == value, col], bins=bins, density=True, histtype="step", linewidth=2,
                    color=color, label=label)
        ax.set_title(title, fontsize=10)
        ax.set_yticks([])
        ax.grid(visible=False)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return _to_png(fig)


def _correlation_chart(corr: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 6.6))
    im = ax.imshow(corr.values, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr)), corr.index, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color=TEXT)
    ax.grid(visible=False)
    ax.set_title("Correlation matrix (red = positive, blue = negative)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    return _to_png(fig)


def _leakage_chart(findings: list[dict]) -> str:
    rows = sorted(findings, key=lambda f: f["univariate_auc"])
    fig, ax = plt.subplots(figsize=(6.5, 4))
    colors = [CRITICAL if f["risk"] == "high" else NEUTRAL_MARK for f in rows]
    # Bars grow from the 0.5 "no signal" baseline, not from 0.
    ax.barh([f["column"] for f in rows], [f["univariate_auc"] - 0.5 for f in rows], left=0.5, color=colors,
            height=0.6, edgecolor=SURFACE, linewidth=2)
    for i, f in enumerate(rows):
        tag = "  HIGH leakage risk" if f["risk"] == "high" else ""
        ax.annotate(f"{f['univariate_auc']:.3f}{tag}", (f["univariate_auc"], i), va="center", fontsize=8,
                    xytext=(3, 0), textcoords="offset points")
    ax.axvline(0.5, color=TEXT_MUTED, linewidth=1)
    ax.set_xlim(0.5, 1.1)
    ax.set_xlabel("Single-column AUC (0.5 = no signal, 1.0 = perfect)")
    ax.set_title("How well each column alone separates churners")
    ax.grid(axis="y", visible=False)
    return _to_png(fig)


def _table(df: pd.DataFrame) -> str:
    return df.to_html(border=0, classes="data", float_format=lambda x: f"{x:,.3f}", na_rep="")


def _figure(png: str, alt: str, table_html: str, caption: str) -> str:
    return (f'<figure><img src="data:image/png;base64,{png}" alt="{html.escape(alt)}">'
            f"<figcaption>{caption}</figcaption>"
            f"<details><summary>Data table</summary>{table_html}</details></figure>")


CSS = """
:root{--bg:#fcfcfb;--panel:#ffffff;--ink:#0b0b0b;--ink-2:#52514e;--line:#e4e3df;--accent:#1c5cab;color-scheme:light}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,"Segoe UI",Roboto,Arial,sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:36px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
.sub{color:var(--ink-2);margin:0 0 18px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.kpi b{display:block;font-size:24px}.kpi span{color:var(--ink-2);font-size:13px}
figure{margin:18px 0;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px}
figure img{max-width:100%;height:auto;display:block}
figcaption{color:var(--ink-2);font-size:13.5px;margin-top:6px}
details{margin-top:6px}summary{cursor:pointer;color:var(--accent);font-size:13px}
.scroll{overflow-x:auto}
table.data{border-collapse:collapse;font-size:12.5px;margin:6px 0}
table.data th,table.data td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:right}
table.data th:first-child,table.data td:first-child{text-align:left}
ul{padding-left:20px}
"""


def describe_features(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics per column (the report's section 2; the app shows the same table)."""
    return df.drop(columns=[ID_COLUMN]).describe().T[["mean", "std", "min", "50%", "max"]]


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlations between all columns except the id (the report's section 6)."""
    return df.drop(columns=[ID_COLUMN]).corr().round(3)


def build_eda_report(df: pd.DataFrame, cleaning_report: dict, facts: dict) -> str:
    features = df.drop(columns=[ID_COLUMN])
    rate_tables = {key: pd.DataFrame(facts[key]) for key in
                   ("churn_by_contract_period", "churn_by_lifetime", "churn_by_age_band")}
    flags_df = pd.DataFrame(facts["churn_by_flag"]).T[["label", "rate_if_no", "rate_if_yes"]]
    corr = correlations(df)
    leakage_df = pd.DataFrame(facts["leakage_findings"]).set_index("column")
    describe = describe_features(df)
    seg = facts["high_risk_segment"]

    dropped = (cleaning_report["dropped_non_numeric_or_null"] + cleaning_report["dropped_duplicates"]
               + sum(cleaning_report["dropped_by_rule"].values()))
    quality_items = [
        f"Rows in raw file: {cleaning_report['rows_in']:,}; rows after cleaning: {cleaning_report['rows_out']:,} "
        f"({dropped:,} removed).",
        f"Missing or non-numeric values removed: {cleaning_report['dropped_non_numeric_or_null']:,}.",
        f"Exact duplicate rows removed: {cleaning_report['dropped_duplicates']:,}.",
        "Domain-rule violations removed: "
        + (", ".join(f"{k} ({v})" for k, v in cleaning_report["dropped_by_rule"].items()) or "none") + ".",
        "Columns renamed to snake_case; <code>member_id</code> added (raw row number) for traceability.",
        *[html.escape(n) for n in cleaning_report["notes"]],
    ]

    sections = [
        "<h1>Golden Fighter: Exploratory Data Analysis</h1>",
        '<p class="sub">Gym membership churn · produced by the Data Analyst crew from clean_data.csv</p>',
        '<div class="kpis">'
        f'<div class="kpi"><b>{facts["members"]:,}</b><span>members</span></div>'
        f'<div class="kpi"><b>{facts["churn_rate"]:.1%}</b><span>churn rate</span></div>'
        f'<div class="kpi"><b>{features.shape[1] - 1}</b><span>candidate features</span></div>'
        f'<div class="kpi"><b>{seg["churn_rate"]:.0%}</b><span>churn in high-risk segment '
        f'({seg["members"]:,} members)</span></div></div>',
        "<h2>1. Data quality</h2><ul>" + "".join(f"<li>{i}</li>" for i in quality_items) + "</ul>",
        "<h2>2. Descriptive statistics</h2>",
        f'<div class="scroll">{_table(describe)}</div>',
        "<h2>3. Churn overview</h2>",
        _figure(_split_chart(df), "Bar chart of members who stayed vs churned",
                _table(df[TARGET].value_counts().rename("members").to_frame()),
                "About one in four members churned. The classes are imbalanced, so models must be judged on "
                "recall and ROC-AUC, not accuracy alone."),
        "<h2>4. What separates churners</h2>",
        _figure(_rate_chart(rate_tables["churn_by_contract_period"], "Churn rate by contract length",
                            "Contract period (months)"),
                "Churn rate by contract length", _table(rate_tables["churn_by_contract_period"]),
                "Longer contracts are associated with much lower churn."),
        _figure(_rate_chart(rate_tables["churn_by_lifetime"], "Churn rate by membership age",
                            "Months since first visit"),
                "Churn rate by membership age", _table(rate_tables["churn_by_lifetime"]),
                "Most churn happens in the first months of membership."),
        _figure(_rate_chart(rate_tables["churn_by_age_band"], "Churn rate by age band", "Age"),
                "Churn rate by age band", _table(rate_tables["churn_by_age_band"]),
                "Younger members churn more often."),
        _figure(_flag_chart(facts["churn_by_flag"]), "Churn rate with vs without member attributes",
                _table(flags_df), "Group classes, referrals and partner discounts go together with lower churn; "
                                  "having a phone number on file makes no difference."),
        "<h2>5. Distributions: stayed vs churned</h2>",
        _figure(_distribution_chart(df, ["lifetime", "age", "avg_additional_charges_total"],
                                    ["Lifetime (months)", "Age (years)", "Extra spend"]),
                "Distribution of lifetime, age and extra spend for stayed vs churned members",
                _table(pd.DataFrame(facts["mean_stayed_vs_churned"]).T),
                "Churners are newer, younger and spend less on extra services."),
        _figure(_distribution_chart(df, ["avg_class_frequency_total", "avg_class_frequency_current_month"],
                                    ["Visits/week: whole membership", "Visits/week: current month"]),
                "Distribution of visit frequency for stayed vs churned members",
                _table(pd.DataFrame([facts["visit_change_current_vs_total"]], index=["current minus total"])),
                "Churners' current-month visits collapse toward zero; this is measured during the churn month "
                "itself, which is why that column is treated as a leakage risk."),
        "<h2>6. Correlations</h2>",
        _figure(_correlation_chart(corr), "Correlation heatmap of all columns", _table(corr),
                "contract_period and month_to_end_contract are highly correlated (they describe the same "
                "contract); the two visit-frequency columns are too."),
        "<h2>7. Leakage scan</h2>",
        _figure(_leakage_chart(facts["leakage_findings"]), "Single-column AUC per feature with leakage flags",
                _table(leakage_df),
                "No column predicts churn suspiciously perfectly on its own. The current-month visit frequency is "
                "flagged high-risk for a timing reason (it is measured in the churn month), not because of its AUC."),
    ]
    page = (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>Golden Fighter EDA</title><style>{CSS}</style></head>"
            f"<body><main>{''.join(sections)}</main></body></html>\n")
    log.info("Built EDA report (%d KB)", len(page) // 1024)
    return page
