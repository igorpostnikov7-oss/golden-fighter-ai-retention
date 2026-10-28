"""M6: the Streamlit app. Data layer unit tests + every screen rendered headlessly (streamlit AppTest).

All screens run against a throwaway copy of the project (config is pointed at tmp_path), so the tests never
touch the real artifacts, and no test makes an API call.
"""
import json
import shutil
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from golden_fighter import config
from golden_fighter.dashboard import data
from golden_fighter.errors import DataValidationError

APP = Path(__file__).resolve().parents[1] / "app"
PAGES = ["overview", "insights", "model", "predict", "flow", "gate_demo"]


@pytest.fixture
def project(tmp_path, monkeypatch, analyst_out, scientist_out, raw_csv) -> Path:
    """A private project with Crew 1 and Crew 2 artifacts and the raw data; config points at it."""
    root = tmp_path / "project"
    shutil.copytree(analyst_out, root / "artifacts" / "analyst")
    shutil.copytree(scientist_out, root / "artifacts" / "scientist")
    for name, value in {"ARTIFACTS_DIR": root / "artifacts", "ANALYST_DIR": root / "artifacts" / "analyst",
                        "SCIENTIST_DIR": root / "artifacts" / "scientist", "RAW_DATA_PATH": raw_csv,
                        "LOGS_DIR": tmp_path / "logs"}.items():
        monkeypatch.setattr(config, name, value)
    return root


@pytest.fixture
def paths(project) -> data.Paths:
    return data.Paths.current()


def _page(name: str) -> AppTest:
    return AppTest.from_file(str(APP / "screens" / f"{name}.py"), default_timeout=180)


def _texts(at: AppTest) -> str:
    parts = [e.value for kind in ("markdown", "caption", "info", "warning", "error", "success", "title", "subheader")
             for e in getattr(at, kind)]
    return "\n".join(str(p) for p in parts)


# --- data layer -----------------------------------------------------------------------------------------------

def test_model_table_spells_out_each_models_role(paths):
    evaluation = data.evaluation(paths)
    table = data.model_table(evaluation)
    roles = dict(zip(table["מודל"], table["תפקיד"]))
    assert roles[data.MODEL_LABELS[evaluation["selected_model"]]] == data.SELECTED
    assert roles[data.MODEL_LABELS["baseline_majority"]] == data.REFERENCE
    assert list(table["תפקיד"]).count(data.LEAKAGE_ROLE) == 1
    assert table.loc[table["תפקיד"] == data.SELECTED, "ROC-AUC (מבחן)"].item() == \
        evaluation["results"][evaluation["selected_model"]]["test"]["roc_auc"]


def test_confusion_matrix_covers_the_whole_test_set(paths):
    evaluation = data.evaluation(paths)
    assert data.confusion_matrix(evaluation)["members"].sum() == evaluation["data"]["test_rows"]


def test_example_members_are_valid_and_ranked_sensibly(paths):
    bundle = data.model_bundle(paths)
    new, loyal = (data.predict_member(bundle, m) for m in data.EXAMPLE_MEMBERS.values())
    assert new["probability"] > loyal["probability"]
    assert {f["name"] for f in data.input_spec(bundle)} == set(bundle["input_features"])


def test_prediction_refuses_values_that_break_the_contract(paths):
    member = dict(data.EXAMPLE_MEMBERS["מתאמן חדש, מנוי לחודש אחד"], month_to_end_contract=9)
    with pytest.raises(DataValidationError, match="cannot exceed"):
        data.predict_member(data.model_bundle(paths), member)


def test_prediction_problems_are_shown_in_hebrew_and_unknown_ones_stay_readable():
    assert data.issue_in_hebrew("'month_to_end_contract' cannot exceed 'contract_period'") == \
        "מספר החודשים שנותרו למנוי לא יכול להיות גדול מתקופת המנוי."
    assert data.issue_in_hebrew("'age' above the contract maximum 100 years") == \
        "גיל (שנים): גבוה מהמקסימום שבחוזה (100 שנים)."
    assert data.issue_in_hebrew("something new") == "something new"  # never hidden


def test_every_agent_scenario_and_stage_has_a_hebrew_text():
    from golden_fighter.contract.breakers import SCENARIOS
    from golden_fighter.dashboard import hebrew

    assert set(data.agents_overview()["שם ב-CrewAI"]) == set(hebrew.AGENTS)
    assert set(SCENARIOS) == set(hebrew.SCENARIOS)
    assert all(text in hebrew.SCENARIOS.values() for text in data.break_scenarios().values())


def test_documents_are_shown_without_markers_or_source_tags(tmp_path):
    doc = tmp_path / "model_card.md"
    doc.write_text("# Card\n\n<!-- agent-summary:start -->\n## Summary (agent)\n\n- Recall 90.1%. "
                   "[source: train_and_compare_models]\n<!-- agent-summary:end -->\n\n## Details\n", encoding="utf-8")
    assert "<!--" not in data.document(doc) and "[source:" not in data.document(doc)
    assert data.agent_section(doc) == "- Recall 90.1%."


def test_old_run_summaries_are_read_with_the_current_names():
    old = {"run_id": "r1", "status": "succeeded", "crews": {"analyst": {"llm_calls": 9, "tool_calls": [
        {"tool": "clean_data", "status": "ok"}, {"tool": "source_check", "status": "rejected"},
        {"tool": "run_eda", "status": "ok"}]}}, "totals": {"llm_calls": 9}}
    summary = data.normalize_summary(old)
    crew = summary["crews"]["analyst"]
    assert crew["model_api_calls"] == 9 and crew["agent_tool_executions"] == 2 and crew["source_check_rejections"] == 1
    assert summary["totals"]["model_api_calls"] == 9 and summary["totals"]["agent_tool_executions"] == 2


def test_run_history_lists_every_kind_newest_first_and_skips_damaged_files(paths, project):
    for sub, run_id, started in (("runs", "full", "2026-10-28T10:00:00"), ("_no_llm/runs", "free", "2026-10-28T11:00:00"),
                                 ("_broken/runs", "demo", "2026-10-28T12:00:00")):
        folder = project / "artifacts" / sub / run_id
        folder.mkdir(parents=True)
        (folder / "run_summary.json").write_text(json.dumps(
            {"run_id": run_id, "started_at": started, "status": "succeeded", "crews": {},
             "totals": {"model_api_calls": 3 if run_id == "full" else 0, "estimated_cost_usd": 0.001}}),
            encoding="utf-8")
    (project / "artifacts" / "runs" / "broken").mkdir()
    (project / "artifacts" / "runs" / "broken" / "run_summary.json").write_text("{not json", encoding="utf-8")
    history = data.run_history(paths)
    assert list(history[data.RUN]) == ["demo", "free", "full"]
    assert history.set_index(data.RUN).loc["full", data.API_CALLS] == 3


def test_agents_overview_lists_six_agents_with_their_tools():
    agents = data.agents_overview()
    assert len(agents) == 6 and agents["צוות"].nunique() == 2
    assert agents.set_index("שם ב-CrewAI").loc["Model Trainer & Evaluator", "כלים"] == "train_and_compare_models"


def test_missing_artifacts_raise_a_message_that_says_what_to_do(tmp_path):
    empty = data.Paths(tmp_path / "a", tmp_path / "s", tmp_path)
    with pytest.raises(data.MissingArtifact, match="Flow החינמי"):
        data.evaluation(empty)


# --- screens --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("page", PAGES)
def test_every_screen_renders_without_errors(project, page):
    at = _page(page).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title  # every screen explains itself with a title and an introduction


def test_main_app_opens_on_the_overview(project):
    at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=180).run()
    assert not at.exception and at.title[0].value == "Golden Fighter AI: חיזוי נטישת מתאמנים"
    assert "אף פעם לא פונה ל-Claude API" in _texts(at) + "".join(c.value for c in at.sidebar.caption)


def test_screens_explain_missing_results_instead_of_crashing(tmp_path, monkeypatch):
    for name in ("ARTIFACTS_DIR", "ANALYST_DIR", "SCIENTIST_DIR"):
        monkeypatch.setattr(config, name, tmp_path / "empty" / name.lower())
    for page in ("overview", "insights", "model", "predict"):
        at = _page(page).run()
        assert not at.exception and "עדיין לא זמינות" in _texts(at), page


def test_overview_shows_the_agent_summary_and_headline_metrics(project):
    at = _page("overview").run()
    labels = [m.label for m in at.metric]
    assert "עוזבים שזוהו" in labels and "ROC-AUC" in labels


def test_prediction_screen_scores_and_refuses_invalid_input(project):
    at = _page("predict").run()
    assert at.metric[0].label == "הסיכוי לעזוב החודש"
    months_left = next(n for n in at.number_input if n.label == "חודשים שנותרו למנוי")
    months_left.set_value(9)
    at.button[0].click().run()  # the form's submit button
    assert "מפרים את חוזה הנתונים" in _texts(at) and "לא יכול להיות גדול מתקופת המנוי" in _texts(at)


def test_break_demo_stops_the_flow_and_leaves_the_data_untouched(project):
    at = _page("gate_demo").run()
    assert "עוברים את השער" in _texts(at)
    assert at.selectbox[0].value == "unit_x100"
    at.button[0].click().run()
    text = _texts(at)
    assert "נעצר בשלב: 3 - שער הבדיקה" in text and "UNIT CHANGE" in text
    metrics = {m.label: m.value for m in at.metric}
    assert metrics == {"קריאות API למודל": "0", "אומן מודל": "לא", "הנתונים המקוריים לא השתנו": "כן"}
    assert not (project / "artifacts" / "_broken" / "unit_x100_scientist" / "model.joblib").exists()


def test_free_flow_button_never_builds_the_paid_llm(project, monkeypatch):
    import golden_fighter.flow.main as flow_main

    def no_paid_llm(*_args, **_kwargs):
        raise AssertionError("the app tried to build the paid LLM")

    monkeypatch.setattr(flow_main, "build_llm", no_paid_llm)
    validated = {p: p.read_bytes() for p in (project / "artifacts" / "scientist").iterdir()}
    at = _page("flow").run()
    at.button[0].click().run()
    assert not at.exception and "הסתיים בהצלחה" in _texts(at)
    assert (project / "artifacts" / "_no_llm" / "scientist" / "model.joblib").exists()
    assert {p: p.read_bytes() for p in (project / "artifacts" / "scientist").iterdir()} == validated


def test_app_code_cannot_start_a_paid_run_or_read_secrets():
    """Static guard: the only Flow entry points the app uses are the free ones, and it never reads env vars."""
    sources = [p.read_text(encoding="utf-8") for p in list(APP.rglob("*.py"))
               + list((APP.parents[0] / "src" / "golden_fighter" / "dashboard").glob("*.py"))]
    code = "\n".join(sources)
    assert "build_llm" not in code and "run_analyst_crew" not in code and "run_scientist_crew" not in code
    assert "getenv" not in code and "os.environ" not in code and "ANTHROPIC_API_KEY" not in code
    assert "run_flow(no_llm_settings())" in code and "run_flow(demo_settings(scenario))" in code
