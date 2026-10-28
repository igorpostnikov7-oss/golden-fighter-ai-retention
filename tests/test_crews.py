"""M4: the two CrewAI crews, driven by a scripted mock LLM (tests/mock_llm.py). No API key, no network.

These run the real Agents, Tasks, Crews and tools; only the language model is replaced.
"""
import json
import shutil
from pathlib import Path

import anthropic
import httpx
import pandas as pd
import pytest

from golden_fighter import config
from golden_fighter.analysis.pipeline import run_analyst_pipeline
from golden_fighter.contract.breakers import make_broken_copy
from golden_fighter.contract.validator import run_gate
from golden_fighter.crews import run as crew_cli
from golden_fighter.crews.analyst_crew.crew import build_analyst_crew, run_analyst_crew
from golden_fighter.crews.runner import RUN_LOG_FILE, CrewRunError
from golden_fighter.crews.scientist_crew.crew import build_scientist_crew, run_scientist_crew
from golden_fighter.errors import DataValidationError
from golden_fighter.tools.attribution import SOURCE_RULES, attribution_problems
from golden_fighter.tools.base import ToolRunLog
from golden_fighter.tools.scientist_tools import scientist_tools
from mock_llm import final, scripted_llm, tool

ANALYST_ROLES = ["Data Quality Engineer", "EDA & Insights Analyst", "Data Contract Specialist"]
SCIENTIST_ROLES = ["Feature Engineer", "Model Trainer & Evaluator", "Model Card Specialist"]


@pytest.fixture(autouse=True)
def isolated_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path / "logs")
    return tmp_path / "logs"


@pytest.fixture
def analyst_copy(tmp_path, analyst_out) -> Path:
    """A private copy of Crew 1's artifacts (the session fixture must stay read-only)."""
    return Path(shutil.copytree(analyst_out, tmp_path / "analyst"))


def _run_records(log_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (log_dir / RUN_LOG_FILE).read_text(encoding="utf-8").splitlines()]


def _insights_summary(out_dir: Path):
    def step(_messages):
        facts = json.loads((out_dir / "insight_facts.json").read_text(encoding="utf-8"))
        top = facts["churn_by_contract_period"][0]
        return tool("write_insights_summary", {"summary": (
            f"- Overall churn is {facts['churn_rate'] * 100:.1f}% of {facts['members']:,} members.\n"
            f"- Members on {top['group']}-month contracts churn at {top['churn_rate'] * 100:.1f}%.\n"
            "- Longer contracts and group classes go together with staying (association, not cause).")})
    return step


def _analyst_script(out_dir: Path) -> dict:
    return {
        "Data Quality Engineer": [tool("profile_raw_data"), tool("clean_data"), final("Cleaned the data.")],
        "EDA & Insights Analyst": [
            tool("run_eda"),
            tool("write_insights_summary", {"summary": "- Churn is 77.7% for everybody, a number nobody computed.\n"
                                                       "- It only exists to check that invented numbers are refused "
                                                       "by the tool before anything is written to insights.md."}),
            _insights_summary(out_dir),
            final("Summary saved."),
        ],
        "Data Contract Specialist": [tool("build_contract"), tool("contract_self_check"), final("Contract ok.")],
    }


def _card_summary(out_dir: Path):
    def step(_messages):
        report = json.loads((out_dir / "evaluation.json").read_text(encoding="utf-8"))
        test = report["results"][report["selected_model"]]["test"]
        return tool("write_model_card_summary", {"summary": (
            f"- Predicts which members are likely to leave at the start of a month.\n"
            f"- Finds {test['recall'] * 100:.1f}% of churners; {test['precision'] * 100:.1f}% of flagged members "
            "really leave.\n- Use the score to offer help, never to penalise a member.")})
    return step


def _scientist_script(out_dir: Path) -> dict:
    return {
        "Feature Engineer": [tool("inspect_contract"), tool("build_features"), final("Features built.")],
        "Model Trainer & Evaluator": [tool("train_and_compare_models"), final("Logistic regression selected.")],
        "Model Card Specialist": [tool("write_model_reports"), _card_summary(out_dir), final("Card written.")],
    }


# --- structure -------------------------------------------------------------------------------------------

def test_each_crew_has_three_real_agents_with_their_own_tools(tmp_path, analyst_copy):
    llm = scripted_llm({})
    analyst = build_analyst_crew(llm, ToolRunLog(), tmp_path / "raw.csv", tmp_path / "out")
    scientist = build_scientist_crew(llm, ToolRunLog(), analyst_copy, tmp_path / "sci")
    for crew, roles in ((analyst, ANALYST_ROLES), (scientist, SCIENTIST_ROLES)):
        assert [agent.role for agent in crew.agents] == roles
        assert len(crew.tasks) == 3 and [task.agent.role for task in crew.tasks] == roles
        for agent in crew.agents:
            assert agent.goal and agent.backstory and agent.tools
            assert agent.allow_delegation is False
            assert agent.max_iter == config.AGENT_MAX_ITER
            assert agent.max_retry_limit == config.AGENT_MAX_RETRY_LIMIT
    tool_names = {a.role: [t.name for t in a.tools] for a in analyst.agents + scientist.agents}
    assert tool_names["Data Quality Engineer"] == ["profile_raw_data", "clean_data"]
    assert tool_names["Model Trainer & Evaluator"] == ["train_and_compare_models"]


def test_scientist_tools_cannot_be_pointed_at_raw_data(tmp_path, analyst_copy):
    for tool_obj in scientist_tools(ToolRunLog(), analyst_copy, tmp_path).values():
        assert set(tool_obj.args_schema.model_fields) <= {"summary"}  # the agent never chooses a path
        assert not hasattr(tool_obj, "raw_path")
    with pytest.raises(DataValidationError, match="may not read raw data"):
        scientist_tools(ToolRunLog(), config.RAW_DATA_PATH.parent, tmp_path)


# --- Crew 1 ----------------------------------------------------------------------------------------------

def test_analyst_crew_runs_tools_and_produces_valid_artifacts(tmp_path, raw_csv, isolated_logs):
    out_dir = tmp_path / "analyst"
    llm = scripted_llm(_analyst_script(out_dir), model="mock/claude-haiku-5-5")
    result = run_analyst_crew(llm=llm, raw_path=raw_csv, out_dir=out_dir)

    assert result.status == "ok" and result.warnings == []
    assert [c["tool"] for c in result.tool_calls if c["status"] == "ok"] == [
        "profile_raw_data", "clean_data", "run_eda", "write_insights_summary", "build_contract", "contract_self_check"]
    assert run_gate(out_dir).passed
    insights = (out_dir / "insights.md").read_text(encoding="utf-8")
    assert "Analyst summary (written by the EDA & Insights agent)" in insights and "77.7" not in insights
    # token usage and cost are logged for every run
    record = _run_records(isolated_logs)[-1]
    assert record["status"] == "ok" and record["llm_calls"] == len(llm.calls) == 10
    assert record["usage"]["total_tokens"] > 0 and record["estimated_cost_usd"] > 0


def test_analyst_crew_numbers_come_from_the_deterministic_code(tmp_path, raw_csv):
    """The crew's data and contract are byte-identical to the plain-Python pipeline's."""
    crew_dir, plain_dir = tmp_path / "crew", tmp_path / "plain"
    run_analyst_crew(llm=scripted_llm(_analyst_script(crew_dir)), raw_path=raw_csv, out_dir=crew_dir)
    run_analyst_pipeline(raw_csv, plain_dir)
    for name in ("clean_data.csv", "dataset_contract.json", "insight_facts.json", "cleaning_report.json"):
        assert (crew_dir / name).read_bytes() == (plain_dir / name).read_bytes(), name


def test_rejected_summary_is_logged_and_never_saved(tmp_path, raw_csv):
    out_dir = tmp_path / "analyst"
    result = run_analyst_crew(llm=scripted_llm(_analyst_script(out_dir)), raw_path=raw_csv, out_dir=out_dir)
    rejected = [c for c in result.tool_calls if c["status"] == "rejected"]
    assert len(rejected) == 1 and "77.7" in rejected[0]["detail"]


def test_agent_cannot_talk_its_way_past_a_validation_failure(tmp_path, isolated_logs):
    """The raw file is missing. The agent ignores the STOP and claims success; the runner still stops."""
    out_dir = tmp_path / "analyst"
    llm = scripted_llm({role: [tool("profile_raw_data"), tool("clean_data"), tool("run_eda"),
                               final("Everything went perfectly.")] for role in ANALYST_ROLES})
    with pytest.raises(DataValidationError, match="raw"):
        run_analyst_crew(llm=llm, raw_path=tmp_path / "missing.csv", out_dir=out_dir)
    assert not (out_dir / "clean_data.csv").exists()
    statuses = [c["status"] for c in _run_records(isolated_logs)[-1]["tool_calls"]]
    assert statuses[0] == "validation_failed" and set(statuses[1:]) == {"refused"}
    assert _run_records(isolated_logs)[-1]["status"] == "validation_failed"


def test_crew_that_skips_its_tools_is_caught(tmp_path, raw_csv):
    out_dir = tmp_path / "analyst"
    llm = scripted_llm({role: [final("Done, all four files are written.")] for role in ANALYST_ROLES})
    with pytest.raises(CrewRunError, match="not written in this run"):
        run_analyst_crew(llm=llm, raw_path=raw_csv, out_dir=out_dir)


def test_stale_artifacts_from_an_earlier_run_do_not_count(tmp_path, raw_csv):
    out_dir = tmp_path / "analyst"
    run_analyst_pipeline(raw_csv, out_dir)  # artifacts already exist from before
    llm = scripted_llm({role: [final("Done.")] for role in ANALYST_ROLES})
    with pytest.raises(CrewRunError, match="clean_data.csv"):
        run_analyst_crew(llm=llm, raw_path=raw_csv, out_dir=out_dir)


def _leakage_answer(out_dir: Path, credit: str):
    """The contract agent's final answer from the live test, with the leakage AUC credited by `credit`."""
    def step(_messages):
        facts = json.loads((out_dir / "insight_facts.json").read_text(encoding="utf-8"))
        auc = next(f["univariate_auc"] for f in facts["leakage_findings"] if f["risk"] == "high")
        return final("Contract built, gate PASSED.\n- avg_class_frequency_current_month leaks the outcome. "
                     + credit.replace("AUC", f"AUC {auc}"))
    return step


MISCREDITED = "The cleaning report notes it has AUC and partly records the churn itself."
CORRECTED = "The EDA facts give it a single-column AUC. [source: run_eda]"


def test_miscredited_metric_is_sent_back_and_corrected(tmp_path, raw_csv, isolated_logs):
    """The live-test slip: an AUC from the EDA facts credited to the cleaning report. Code catches it."""
    out_dir = tmp_path / "analyst"
    script = _analyst_script(out_dir)
    script["Data Contract Specialist"] = [tool("build_contract"), tool("contract_self_check"),
                                          _leakage_answer(out_dir, MISCREDITED), _leakage_answer(out_dir, CORRECTED)]
    result = run_analyst_crew(llm=scripted_llm(script), raw_path=raw_csv, out_dir=out_dir)

    assert "[source: run_eda]" in result.final_answer and "cleaning report" not in result.final_answer
    checks = [c for c in result.tool_calls if c["tool"] == "source_check"]
    assert len(checks) == 1 and "contract_task" in checks[0]["detail"]
    assert "not found in the result of build_contract, contract_self_check" in checks[0]["detail"]
    assert "They appear in: run_eda" in checks[0]["detail"]


def test_agent_that_keeps_inventing_a_source_stops_the_crew(tmp_path, raw_csv, isolated_logs):
    out_dir = tmp_path / "analyst"
    script = _analyst_script(out_dir)
    script["Data Contract Specialist"] = [tool("build_contract"), tool("contract_self_check"),
                                          _leakage_answer(out_dir, MISCREDITED)]  # repeats the same answer
    with pytest.raises(CrewRunError, match="failed the source check"):
        run_analyst_crew(llm=scripted_llm(script), raw_path=raw_csv, out_dir=out_dir)
    record = _run_records(isolated_logs)[-1]
    assert record["status"] == "error"
    assert sum(c["tool"] == "source_check" for c in record["tool_calls"]) == 1 + config.SOURCE_CHECK_MAX_RETRIES


def test_contract_tools_give_the_contracts_own_reasons(tmp_path, analyst_copy):
    """Agents are asked for a reason per excluded column, so the tools must supply it (no guessing)."""
    contract = json.loads((analyst_copy / "dataset_contract.json").read_text(encoding="utf-8"))
    log = ToolRunLog()
    scientist_tools(log, analyst_copy, tmp_path)["inspect_contract"]._run()
    reasons = log.results["inspect_contract"]["reasons"]
    assert set(reasons) == set(contract["forbidden_features"] + contract["excluded_from_primary_model"])
    assert any("protected attribute" in note for note in reasons["gender"])
    assert any("same month" in note for note in reasons["avg_class_frequency_current_month"])


# The two Feature Engineer answers the live Haiku test (28.10) rejected, verbatim. 80/20 is true (TEST_SIZE = 0.2),
# but build_features did not return it, so the source check correctly refused it twice and the crew stopped.
LIVE_FIRST_ANSWER = (
    "**Row counts [source: build_features]:**\n- Train: 3200 rows\n- Test: 800 rows\n- Total: 4000 rows\n"
    "**Output columns in features.csv [source: build_features]:** member_id, split, near_location, partner, "
    "promo_friends, phone, contract_period, group_visits, age, avg_additional_charges_total, "
    "month_to_end_contract, lifetime, avg_class_frequency_total, contract_elapsed_share, is_new_member, "
    "social_ties, churn. The split is stratified 80/20.")
LIVE_REWRITE = (
    "The contract is version 1.1 with target `churn` [source: inspect_contract].\n"
    "**Row counts [source: build_features]:**\n- Train: 3200 rows\n- Test: 800 rows\n- Total: 4000 rows\n"
    "The split label is a stratified train/test split. The 80/20 ratio is not stated in any tool result, "
    "so it is not reported here.")


@pytest.fixture
def feature_results(tmp_path, analyst_copy) -> dict:
    """Real results of the Feature Engineer's two tools on the test data."""
    log = ToolRunLog()
    tools = scientist_tools(log, analyst_copy, tmp_path / "sci")
    for name in ("inspect_contract", "build_features"):
        tools[name]._run()
    return log.results


@pytest.mark.parametrize("answer", [LIVE_FIRST_ANSWER, LIVE_REWRITE])
def test_live_split_answers_failed_without_split_settings_and_pass_with_them(answer, raw_df, feature_results):
    own = ["inspect_contract", "build_features"]
    rows = feature_results["build_features"]["split_counts"]
    # the test data is smaller than the real data: swap in its row counts, keep everything else verbatim
    answer = (answer.replace("Train: 3200", f"Train: {rows['train']}").replace("Test: 800", f"Test: {rows['test']}")
              .replace("Total: 4000", f"Total: {sum(rows.values())}"))
    before = {**feature_results, "build_features": {k: v for k, v in feature_results["build_features"].items()
                                                    if k != "split_settings"}}
    assert any("80, 20 not found" in p for p in attribution_problems(answer, before, own))
    assert attribution_problems(answer, feature_results, own) == []


def test_build_features_reports_the_split_the_code_used(feature_results):
    from golden_fighter.modeling.features import TEST_SIZE
    settings = feature_results["build_features"]["split_settings"]
    assert settings == {"train_share": 1 - TEST_SIZE, "test_share": TEST_SIZE, "stratified_on": "target",
                        "random_seed": config.SEED}
    counts = feature_results["build_features"]["split_counts"]
    assert counts["test"] / sum(counts.values()) == pytest.approx(TEST_SIZE, abs=0.01)


@pytest.mark.parametrize("answer, bad", [
    ("The split is stratified 70/30.", "70, 30"),                       # a different ratio is still invented
    ("The split is stratified 80/20 with 3,201 training rows.", "3,201"),
    ("The split is 80/20. [source: run_eda]", "did not produce a result"),  # invented source still refused
    ("Random forest recall is 0.81. [source: build_features]", "0.81 not found"),
])
def test_split_settings_do_not_open_the_door_to_invented_numbers(answer, bad, feature_results):
    problems = attribution_problems(answer, feature_results, ["inspect_contract", "build_features"])
    assert len(problems) == 1 and bad in problems[0]


def test_every_task_is_told_to_omit_unsupported_numbers_completely(tmp_path, analyst_copy):
    crew = build_scientist_crew(scripted_llm({}), ToolRunLog(), analyst_copy, tmp_path / "sci")
    assert "not even in a sentence saying it is missing" in SOURCE_RULES
    assert all(task.expected_output.endswith(SOURCE_RULES) for task in crew.tasks)


def test_feature_engineer_can_state_the_split_ratio(tmp_path, analyst_copy):
    """The crew-level replay: the live first answer's 80/20 sentence now passes with no rewrite."""
    out_dir = tmp_path / "scientist"
    script = _scientist_script(out_dir)
    script["Feature Engineer"] = [tool("inspect_contract"), tool("build_features"),
                                  final("Features built. The split is stratified 80/20.")]
    result = run_scientist_crew(llm=scripted_llm(script), analyst_dir=analyst_copy, out_dir=out_dir)
    assert result.status == "ok" and not [c for c in result.tool_calls if c["tool"] == "source_check"]


# --- limits and API errors --------------------------------------------------------------------------------

def test_call_budget_stops_a_runaway_agent(tmp_path, raw_csv, monkeypatch, isolated_logs):
    monkeypatch.setattr(config, "MAX_LLM_CALLS_PER_CREW", 3)
    llm = scripted_llm({role: [tool("profile_raw_data")] for role in ANALYST_ROLES})  # never answers
    with pytest.raises(CrewRunError, match="budget of 3 calls"):
        run_analyst_crew(llm=llm, raw_path=raw_csv, out_dir=tmp_path / "analyst")
    assert len(llm.calls) == 3  # the 4th call was blocked before it reached the model
    record = _run_records(isolated_logs)[-1]
    assert record["status"] == "error" and record["llm_calls"] == 3


def test_api_error_is_reported_in_plain_words(tmp_path, raw_csv, isolated_logs):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    error = anthropic.AuthenticationError("invalid x-api-key", response=httpx.Response(401, request=request),
                                          body=None)
    llm = scripted_llm({}, fail_with=error)
    with pytest.raises(CrewRunError, match="API key was rejected"):
        run_analyst_crew(llm=llm, raw_path=raw_csv, out_dir=tmp_path / "analyst")
    assert len(llm.calls) == 1  # no automatic task retries
    assert _run_records(isolated_logs)[-1]["status"] == "error"


# --- Crew 2 ----------------------------------------------------------------------------------------------

def test_scientist_crew_refuses_a_broken_handoff_before_any_model_call(tmp_path, analyst_copy):
    broken = make_broken_copy(analyst_copy, "unit_x100", tmp_path / "broken")
    llm = scripted_llm(_scientist_script(tmp_path / "sci"))
    with pytest.raises(DataValidationError, match="UNIT CHANGE"):
        run_scientist_crew(llm=llm, analyst_dir=broken, out_dir=tmp_path / "sci")
    assert llm.calls == []  # the gate is plain Python: a bad hand-off costs nothing
    assert not (tmp_path / "sci" / "model.joblib").exists()


def test_scientist_crew_trains_and_documents_from_contract_only(tmp_path, analyst_copy):
    out_dir = tmp_path / "scientist"
    llm = scripted_llm(_scientist_script(out_dir))
    result = run_scientist_crew(llm=llm, analyst_dir=analyst_copy, out_dir=out_dir)

    assert result.status == "ok" and result.warnings == []
    for name in ("features.csv", "model.joblib", "evaluation_report.md", "model_card.md"):
        assert (out_dir / name).exists()
    card = (out_dir / "model_card.md").read_text(encoding="utf-8")
    assert card.index("Summary for the retention team") < card.index("## Model details")
    features = pd.read_csv(out_dir / "features.csv")
    assert not {"gender", "avg_class_frequency_current_month"} & set(features.columns)
    prompts = json.dumps([c["messages"] for c in llm.calls], default=str)
    assert "gym_churn.csv" not in prompts and "data/raw" not in prompts


def test_scientist_crew_stops_if_data_changes_mid_run(tmp_path, analyst_copy):
    """Data altered after the features step: training's own gate catches it, whatever the agent says."""
    out_dir = tmp_path / "scientist"

    def tamper_then_train(_messages):
        clean = pd.read_csv(analyst_copy / "clean_data.csv")
        clean["avg_additional_charges_total"] *= 100
        clean.to_csv(analyst_copy / "clean_data.csv", index=False)
        return tool("train_and_compare_models")

    script = _scientist_script(out_dir)
    script["Model Trainer & Evaluator"] = [tamper_then_train, final("Trained a great model, AUC 0.99.")]
    with pytest.raises(DataValidationError, match="UNIT CHANGE"):
        run_scientist_crew(llm=scripted_llm(script), analyst_dir=analyst_copy, out_dir=out_dir)
    assert (out_dir / "features.csv").exists() and not (out_dir / "model.joblib").exists()


# --- CLI ---------------------------------------------------------------------------------------------------

def test_cli_without_api_key_fails_fast_without_calling_the_api(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    assert crew_cli.main(["analyst"]) == 2


# --- feature importance belongs to the selected model, not the ablation (live test 28.10) -----------------------

@pytest.fixture(scope="module")
def training_run(analyst_out, tmp_path_factory) -> tuple[dict, Path]:
    """train_and_compare_models run once on a private copy of Crew 1's artifacts: (tool result, output dir)."""
    base = tmp_path_factory.mktemp("training_run")
    analyst = Path(shutil.copytree(analyst_out, base / "analyst"))
    log = ToolRunLog()
    tools = scientist_tools(log, analyst, base / "sci")
    for name in ("build_features", "train_and_compare_models"):
        tools[name]._run()
    return log.results["train_and_compare_models"], base / "sci"


def test_feature_importance_is_the_selected_models_on_the_test_split(training_run):
    """Recomputed from the saved model: the tool's numbers are the selected model's, on primary features only."""
    import joblib

    from golden_fighter.modeling.evaluate import feature_importance
    result, sci = training_run
    block = result["selected_model_feature_importance"]
    bundle = joblib.load(sci / "model.joblib")
    test = pd.read_csv(sci / "features.csv").query("split == 'test'")
    assert block["model"] == result["selected_model"] == bundle["model_name"]
    assert not set(result["leakage_ablation"]["extra_columns"]) & set(bundle["input_features"])
    recomputed = feature_importance(bundle["pipeline"], test[bundle["input_features"]], test["churn"])[:5]
    assert block["top_5"] == recomputed
    assert "top_features" not in result  # the unlabelled key the agent misread is gone
    assert "not computed" in result["leakage_ablation"]["feature_importance"]


def _importance_line(result: dict, framing: str) -> str:
    pairs = ", ".join(f"{f['feature']} {f['auc_drop_mean']}"
                      for f in result["selected_model_feature_importance"]["top_5"])
    return f"- **Feature importance {framing} (AUC drop when permuted):** {pairs}. [source: train_and_compare_models]"


def test_selected_model_importance_presented_as_ablation_is_caught(training_run):
    result, _ = training_run
    results, own = {"train_and_compare_models": result}, ["train_and_compare_models"]
    # the live mistake, verbatim framing: rejected with a pointer to the right section
    live = _importance_line(result, "in the ablation").replace(
        ". [source", ". These come from the ablation run, which is diagnostic only. [source")
    problems = attribution_problems(live, results, own)
    assert len(problems) == 1 and "belong to selected_model_feature_importance" in problems[0]
    # with the old, unlabelled tool output the same line passed: this is why the output was renamed
    old = {k: v for k, v in result.items() if k != "selected_model_feature_importance"}
    old["top_features"] = result["selected_model_feature_importance"]["top_5"]
    assert attribution_problems(live, {"train_and_compare_models": old}, own) == []
    # correct framings still pass
    assert attribution_problems(_importance_line(result, "of the selected model"), results, own) == []
    ablation = result["leakage_ablation"]
    for ok in (f"The leakage ablation scored test ROC-AUC {ablation['test_roc_auc']}.",
               f"The ablation's CV ROC-AUC {ablation['cv_roc_auc']['mean']} is diagnostic only."):
        assert attribution_problems(ok, results, own) == []
