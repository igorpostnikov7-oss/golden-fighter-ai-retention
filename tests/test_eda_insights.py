import json
import re

import pandas as pd

from golden_fighter.analysis.schema_spec import TARGET


def test_all_four_crew1_artifacts_exist(analyst_out):
    for name in ("clean_data.csv", "eda_report.html", "insights.md", "dataset_contract.json"):
        assert (analyst_out / name).stat().st_size > 0, name


def test_eda_report_is_self_contained_with_charts_and_stats(analyst_out):
    page = (analyst_out / "eda_report.html").read_text(encoding="utf-8")
    assert len(re.findall(r'<img src="data:image/png;base64,', page)) >= 8
    assert "Descriptive statistics" in page and "Leakage scan" in page
    assert not re.search(r'(src|href)="https?://', page), "report must not load external resources"


def test_every_chart_has_a_table_view(analyst_out):
    page = (analyst_out / "eda_report.html").read_text(encoding="utf-8")
    assert page.count("<figure>") == page.count("<summary>Data table</summary>")


def test_insights_numbers_match_the_data(analyst_out):
    clean = pd.read_csv(analyst_out / "clean_data.csv")
    facts = json.loads((analyst_out / "insight_facts.json").read_text(encoding="utf-8"))
    insights = (analyst_out / "insights.md").read_text(encoding="utf-8")
    assert facts["churn_rate"] == round(clean[TARGET].mean(), 4)
    assert f"({clean[TARGET].mean():.1%})" in insights
    for heading in ("## Executive summary", "## Retention levers", "## Caveats for the modeling crew"):
        assert heading in insights


def test_strongest_signals_exclude_leaky_and_protected_columns(analyst_out):
    facts = json.loads((analyst_out / "insight_facts.json").read_text(encoding="utf-8"))
    names = [s["column"] for s in facts["strongest_safe_signals"]]
    assert "avg_class_frequency_current_month" not in names
    assert "gender" not in names
