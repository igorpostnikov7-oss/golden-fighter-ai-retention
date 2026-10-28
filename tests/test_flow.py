"""M5: the CrewAI Flow (Crew 1 -> gate -> Crew 2), driven by the scripted mock LLM. No API key, no network."""
import json
import shutil
from pathlib import Path

import anthropic
import httpx
import pandas as pd
import pytest

from golden_fighter import config
from golden_fighter.contract.breakers import make_broken_copy
from golden_fighter.flow import artifacts
from golden_fighter.flow import main as flow_main
from golden_fighter.flow.main import FlowSettings, run_flow
from golden_fighter.flow.reports import FAILURE_REPORT_FILE, SUMMARY_FILE
from crew_scripts import analyst_script, scientist_script
from mock_llm import final, scripted_llm, tool

ANALYST_FILES = ["clean_data.csv", "eda_report.html", "insights.md", "dataset_contract.json"]
SCIENTIST_FILES = ["features.csv", "model.joblib", "evaluation_report.md", "model_card.md"]


@pytest.fixture(autouse=True)
def isolated_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path / "logs")


def _settings(tmp_path: Path, **overrides) -> FlowSettings:
    values = {"analyst_dir": tmp_path / "analyst", "scientist_dir": tmp_path / "scientist",
              "runs_dir": tmp_path / "runs"} | overrides
    return FlowSettings(**values)


def _both_crews(tmp_path: Path, **llm_kwargs):
    return scripted_llm(analyst_script(tmp_path / "analyst") | scientist_script(tmp_path / "scientist"),
                        **llm_kwargs)


def _run_dir(state) -> Path:
    return Path(state.paths["run_summary"]).parent


def _broken_inputs(tmp_path: Path, analyst_out: Path, scenario: str = "unit_x100") -> Path:
    return make_broken_copy(analyst_out, scenario, tmp_path / "broken")


# --- the happy path: both crews in one Flow ----------------------------------------------------------------

def test_flow_runs_both_crews_through_the_gate(tmp_path, raw_csv):
    llm = _both_crews(tmp_path, model="mock/claude-haiku-5-5")
    state = run_flow(_settings(tmp_path, raw_path=raw_csv), llm=llm)

    assert state.status == "succeeded" and state.exit_code == 0 and state.failed_stage is None
    assert [(s.name, s.status) for s in state.stages] == [
        ("preflight", "ok"), ("analyst", "ok"), ("validation_gate", "ok"), ("scientist", "ok")]
    assert state.gate["passed"] and state.gate["checks_run"] > 0
    # Crew 1 finished before Crew 2 made its first model call
    roles = [c["role"] for c in llm.calls]
    first_scientist = roles.index("Feature Engineer")
    assert set(roles[:first_scientist]) == {"Data Quality Engineer", "EDA & Insights Analyst",
                                            "Data Contract Specialist"}
    # every expected artifact was produced by THIS run
    for group, names in (("analyst", ANALYST_FILES), ("scientist", SCIENTIST_FILES)):
        entries = state.artifacts[group]
        assert [e["file"] for e in entries] == names
        assert all(e["produced_by_this_run"] and e["exists"] and e["sha256"] for e in entries)


def test_run_summary_reports_usage_cost_and_metrics(tmp_path, raw_csv):
    llm = _both_crews(tmp_path, model="mock/claude-haiku-5-5")
    state = run_flow(_settings(tmp_path, raw_path=raw_csv), llm=llm)

    run_dir = _run_dir(state)
    summary = json.loads((run_dir / SUMMARY_FILE).read_text(encoding="utf-8"))
    assert summary == state.model_dump(mode="json")
    assert not (run_dir / FAILURE_REPORT_FILE).exists()
    analyst, scientist = summary["crews"]["analyst"], summary["crews"]["scientist"]
    assert analyst["model_api_calls"] > 0 and scientist["model_api_calls"] > 0
    totals = summary["totals"]
    assert totals["model_api_calls"] == analyst["model_api_calls"] + scientist["model_api_calls"] == len(llm.calls)
    # paid model calls and free tool runs are counted separately (mock: analyst runs 7 tools, scientist 5)
    assert analyst["agent_tool_executions"] == len(analyst["tool_log"]) - analyst["source_check_rejections"] == 7
    assert scientist["agent_tool_executions"] == scientist["successful_tool_executions"] == 5
    assert totals["agent_tool_executions"] == 12
    assert {"model_api_calls", "agent_tool_executions"} <= set(summary["glossary"])
    assert summary["totals"]["total_tokens"] == analyst["total_tokens"] + scientist["total_tokens"] > 0
    assert summary["totals"]["estimated_cost_usd"] == pytest.approx(
        analyst["estimated_cost_usd"] + scientist["estimated_cost_usd"])
    assert all(c["status"] == "ok" for c in (analyst, scientist))
    # metrics are copied from the deterministic evaluation.json, never from an agent
    evaluation = json.loads((tmp_path / "scientist" / "evaluation.json").read_text(encoding="utf-8"))
    selected = evaluation["selected_model"]
    assert summary["metrics"]["selected_model"] == selected
    assert summary["metrics"]["test"]["roc_auc"] == evaluation["results"][selected]["test"]["roc_auc"]
    assert summary["inputs"]["raw_sha256"] == artifacts.sha256(raw_csv)


def test_no_llm_mode_runs_the_same_flow_with_the_deterministic_pipelines(tmp_path, raw_csv):
    state = run_flow(_settings(tmp_path, raw_path=raw_csv, use_llm=False))
    assert state.status == "succeeded" and state.totals["model_api_calls"] == 0
    assert state.crews["analyst"]["mode"].startswith("deterministic")
    assert all(e["produced_by_this_run"] for group in state.artifacts.values() for e in group)


# --- the gate stops the Flow --------------------------------------------------------------------------------

def test_failed_gate_stops_the_flow_before_crew_2(tmp_path, analyst_out):
    broken = _broken_inputs(tmp_path, analyst_out)
    llm = _both_crews(tmp_path)
    state = run_flow(_settings(tmp_path, analyst_dir=broken, from_artifacts=True), llm=llm)

    assert state.status == "validation_failed" and state.exit_code == 1
    assert state.failed_stage == "validation_gate"
    assert [(s.name, s.status) for s in state.stages] == [
        ("preflight", "ok"), ("analyst", "skipped"), ("validation_gate", "validation_failed"), ("scientist", "skipped")]
    assert llm.calls == []  # Crew 2 never made a single model call
    assert not (tmp_path / "scientist").exists() or not any((tmp_path / "scientist").iterdir())
    assert any("UNIT CHANGE" in issue for issue in state.validation_issues)

    report = (_run_dir(state) / FAILURE_REPORT_FILE).read_text(encoding="utf-8")
    assert "Validation gate" in report and "UNIT CHANGE" in report
    assert "The Data Scientist crew did not start" in report and "No model API calls were made" in report
    summary = json.loads((_run_dir(state) / SUMMARY_FILE).read_text(encoding="utf-8"))
    assert summary["status"] == "validation_failed" and summary["exit_code"] == 1


@pytest.mark.parametrize("scenario", ["rename_column", "corrupt_contract", "drop_column"])
def test_other_break_scenarios_also_stop_the_flow(tmp_path, analyst_out, scenario):
    broken = _broken_inputs(tmp_path, analyst_out, scenario)
    state = run_flow(_settings(tmp_path, analyst_dir=broken, from_artifacts=True, use_llm=False))
    assert state.status == "validation_failed" and state.failed_stage == "validation_gate"
    assert not (tmp_path / "scientist" / "model.joblib").exists()


def test_leftover_artifacts_from_an_earlier_run_are_never_reported_as_new(tmp_path, analyst_out, scientist_out):
    old = Path(shutil.copytree(scientist_out, tmp_path / "scientist"))  # a previous, successful run
    before = artifacts.snapshot([old / name for name in SCIENTIST_FILES])
    broken = _broken_inputs(tmp_path, analyst_out)
    state = run_flow(_settings(tmp_path, analyst_dir=broken, from_artifacts=True, use_llm=False))

    assert state.status == "validation_failed"
    entries = state.artifacts["scientist"]
    assert all(e["exists"] and not e["produced_by_this_run"] for e in entries)
    assert artifacts.snapshot([old / name for name in SCIENTIST_FILES]) == before  # left untouched
    report = (_run_dir(state) / FAILURE_REPORT_FILE).read_text(encoding="utf-8")
    assert "left over from an earlier run" in report and "model.joblib" in report


def test_crew_that_does_not_write_its_outputs_stops_the_flow(tmp_path, raw_csv):
    """Stale-output protection end to end: Crew 1 'finishes' without running its tools."""
    llm = scripted_llm({role: [final("All four files are written.")] for role in
                        ("Data Quality Engineer", "EDA & Insights Analyst", "Data Contract Specialist")})
    state = run_flow(_settings(tmp_path, raw_path=raw_csv), llm=llm)
    assert state.status == "error" and state.exit_code == 2 and state.failed_stage == "analyst"
    assert "not written in this run" in state.failure_reason
    assert [s.status for s in state.stages] == ["ok", "error", "skipped", "skipped"]


def test_not_rewritten_detects_untouched_and_missing_files(tmp_path):
    kept, rewritten, missing = (tmp_path / n for n in ("kept.csv", "rewritten.csv", "missing.csv"))
    kept.write_text("a"), rewritten.write_text("a")
    before = artifacts.snapshot([kept, rewritten, missing])
    rewritten.write_text("ab")
    assert artifacts.not_rewritten(before) == ["kept.csv", "missing.csv"]


# --- other ways to stop -------------------------------------------------------------------------------------

def test_missing_raw_data_stops_at_preflight_without_model_calls(tmp_path):
    llm = _both_crews(tmp_path)
    state = run_flow(_settings(tmp_path, raw_path=tmp_path / "missing.csv"), llm=llm)
    assert state.status == "validation_failed" and state.failed_stage == "preflight"
    assert llm.calls == [] and [s.status for s in state.stages] == ["validation_failed", "skipped", "skipped",
                                                                    "skipped"]
    assert (_run_dir(state) / FAILURE_REPORT_FILE).exists()


def test_missing_api_key_fails_fast_before_any_work(tmp_path, raw_csv, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    state = run_flow(_settings(tmp_path, raw_path=raw_csv))  # no injected LLM: the real one would be built
    assert state.status == "error" and state.exit_code == 2 and state.failed_stage == "preflight"
    assert "ANTHROPIC_API_KEY" in state.failure_reason
    assert not (tmp_path / "analyst").exists()


def test_api_error_in_crew_2_is_reported_with_its_usage(tmp_path, analyst_out):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    error = anthropic.AuthenticationError("invalid x-api-key", response=httpx.Response(401, request=request),
                                          body=None)
    analyst = Path(shutil.copytree(analyst_out, tmp_path / "analyst"))
    state = run_flow(_settings(tmp_path, analyst_dir=analyst, from_artifacts=True),
                     llm=scripted_llm({}, fail_with=error))
    assert state.status == "error" and state.failed_stage == "scientist" and state.exit_code == 2
    assert "API key was rejected" in state.failure_reason
    assert state.crews["scientist"]["model_api_calls"] == 1
    report = (_run_dir(state) / FAILURE_REPORT_FILE).read_text(encoding="utf-8")
    assert "started but did not finish" in report


def test_data_changed_during_crew_2_is_a_validation_failure_with_usage(tmp_path, analyst_out):
    analyst = Path(shutil.copytree(analyst_out, tmp_path / "analyst"))

    def tamper_then_train(_messages):
        clean = pd.read_csv(analyst / "clean_data.csv")
        clean["avg_additional_charges_total"] *= 100
        clean.to_csv(analyst / "clean_data.csv", index=False)
        return tool("train_and_compare_models")

    script = scientist_script(tmp_path / "scientist")
    script["Model Trainer & Evaluator"] = [tamper_then_train, final("Trained.")]
    state = run_flow(_settings(tmp_path, analyst_dir=analyst, from_artifacts=True), llm=scripted_llm(script))
    assert state.status == "validation_failed" and state.failed_stage == "scientist"
    assert state.crews["scientist"]["model_api_calls"] > 0  # usage of the stopped crew is still reported
    assert not (tmp_path / "scientist" / "model.joblib").exists()


# --- CLI: break demo and the free mode never touch the validated artifacts -----------------------------------

@pytest.fixture
def project_copy(tmp_path, monkeypatch, analyst_out, scientist_out, raw_csv) -> Path:
    """Point config at a throwaway project so the CLI can be tested without touching the real artifacts."""
    root = tmp_path / "project"
    shutil.copytree(analyst_out, root / "artifacts" / "analyst")
    shutil.copytree(scientist_out, root / "artifacts" / "scientist")
    monkeypatch.setattr(config, "ARTIFACTS_DIR", root / "artifacts")
    monkeypatch.setattr(config, "ANALYST_DIR", root / "artifacts" / "analyst")
    monkeypatch.setattr(config, "RAW_DATA_PATH", raw_csv)
    return root


def _fingerprints(*dirs: Path) -> dict:
    return {p: artifacts.sha256(p) for d in dirs for p in sorted(d.iterdir())}


def test_break_demo_proves_the_flow_stops_before_modelling(project_copy):
    validated = _fingerprints(project_copy / "artifacts" / "analyst", project_copy / "artifacts" / "scientist")
    assert flow_main.main(["--demo-break", "unit_x100"]) == 1
    demo = project_copy / "artifacts" / "_broken"
    assert not (demo / "unit_x100_scientist" / "model.joblib").exists()
    (summary_path,) = demo.glob("runs/*/run_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["demo_scenario"] == "unit_x100" and summary["failed_stage"] == "validation_gate"
    assert (summary_path.parent / FAILURE_REPORT_FILE).exists()
    assert _fingerprints(project_copy / "artifacts" / "analyst",
                         project_copy / "artifacts" / "scientist") == validated


def test_no_llm_cli_writes_beside_the_validated_artifacts(project_copy):
    validated = _fingerprints(project_copy / "artifacts" / "analyst", project_copy / "artifacts" / "scientist")
    assert flow_main.main(["--no-llm", "--from-artifacts"]) == 0
    assert (project_copy / "artifacts" / "_no_llm" / "scientist" / "model.joblib").exists()
    assert _fingerprints(project_copy / "artifacts" / "analyst",
                         project_copy / "artifacts" / "scientist") == validated
