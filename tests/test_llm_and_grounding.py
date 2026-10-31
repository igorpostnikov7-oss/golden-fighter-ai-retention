"""M4 guardrails without any crew: number grounding, source check, LLM setup, cost estimate, API errors."""
import anthropic
import httpx
import pytest

from golden_fighter.llm import LLMSetupError, build_llm, describe_api_error, estimate_cost_usd
from golden_fighter.tools.attribution import attribution_problems
from golden_fighter.tools.grounding import SECTION_START, ungrounded_numbers, upsert_agent_section

FACTS = {"members": 4000, "churn_rate": 0.2652, "by_contract": [{"group": "1", "churn_rate": 0.4232}],
         "age_band": {"30-34": {"recall": 0.762}}, "flag": True}


@pytest.mark.parametrize("text", [
    "Churn is 26.5% overall.", "Churn is 26.52%.", "Churn is 27%.", "4,000 members", "4000 members",
    "1-month contracts churn at 42.3%", "members aged 30-34", "recall 0.762", "a 5-fold CV and 12-month contracts",
    "F1 and ROC-AUC are reported",
])
def test_grounded_numbers_pass(text):
    assert ungrounded_numbers(text, FACTS) == []


@pytest.mark.parametrize("text, bad", [
    ("Churn is 31.4% overall.", ["31.4"]),
    ("Churn is 26.6%.", ["26.6"]),          # wrong rounding of 26.52 is still a different number
    ("About 5,000 members", ["5,000"]),
    ("AUC of 0.99", ["0.99"]),
])
def test_invented_numbers_are_caught(text, bad):
    assert ungrounded_numbers(text, FACTS) == bad


def test_agent_section_is_inserted_once_and_replaced_on_rerun():
    doc = "# Title\n\nintro\n\n## First\nbody\n"
    once = upsert_agent_section(doc, "Agent summary", "- v1")
    twice = upsert_agent_section(once, "Agent summary", "- v2")
    assert once.index("## Agent summary") < once.index("## First")
    assert twice.count(SECTION_START) == 1 and "- v2" in twice and "- v1" not in twice
    assert twice.replace("- v2", "- v1") == once


def test_build_llm_without_key_fails_before_any_call(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "  ")
    with pytest.raises(LLMSetupError, match="ANTHROPIC_API_KEY is not set"):
        build_llm("anthropic/claude-haiku-5-5")


def test_only_anthropic_models_are_accepted(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder")
    with pytest.raises(LLMSetupError, match="Only Anthropic"):
        build_llm("openai/gpt-x")


def test_cost_estimate_uses_the_price_table():
    assert estimate_cost_usd("anthropic/claude-haiku-5-5", 1_000_000, 1_000_000) == pytest.approx(0.60)
    assert estimate_cost_usd("anthropic/claude-sonnet-5-5", 10_000, 2_000) == pytest.approx(0.04)
    assert estimate_cost_usd("mock/scripted", 10_000, 2_000) is None


def test_api_errors_are_found_anywhere_in_the_exception_chain():
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    rate_limited = anthropic.RateLimitError("slow down", response=httpx.Response(429, request=request), body=None)
    try:
        try:
            raise rate_limited
        except anthropic.RateLimitError as inner:
            raise RuntimeError("crew wrapper") from inner
    except RuntimeError as outer:
        assert describe_api_error(outer).startswith("rate limit reached")
    assert describe_api_error(ValueError("unrelated")) is None


def test_no_retries_mode_disables_sdk_and_crewai_retries(monkeypatch):
    import crewai.llms.retry as crewai_retry

    monkeypatch.setattr(crewai_retry, "_LLM_RATE_LIMIT_MAX_ATTEMPTS", 3)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder-not-a-key")
    llm = build_llm("anthropic/claude-haiku-5-5", retries=False)  # builds the client only, no request
    assert llm.max_retries == 0 and crewai_retry._LLM_RATE_LIMIT_MAX_ATTEMPTS == 1


# --- source check on final answers (tools/attribution.py) ---------------------------------------------------

RESULTS = {"clean_data": {"rows_in": 4000, "rows_out": 4000},
           "run_eda": {"churn_rate": 0.2652, "leakage_findings": [{"column": "x", "auc": 0.7749}]},
           "build_contract": {"contract_version": "1.1", "row_count_min": 2000, "column_count": 15}}
CONTRACT_TOOLS = ["build_contract", "contract_self_check"]


@pytest.mark.parametrize("answer", [
    "Contract version 1.1, at least 2,000 rows, 15 columns.",                       # own tools, no tag needed
    "The clean data has 4,000 rows. [source: clean_data]",                           # another agent's tool, credited
    "The leakage column has a single-column AUC of 0.7749 in the EDA facts. [source: run_eda]",
    "Built from clean_data.csv; target is churn.",                                   # no numbers, nothing to check
])
def test_correctly_sourced_answers_pass(answer):
    assert attribution_problems(answer, RESULTS, CONTRACT_TOOLS) == []


@pytest.mark.parametrize("answer, expected", [
    # the live-test slip: the AUC is credited to the contract agent's own tools; the hint names the real source
    ("The cleaning report notes it has AUC 0.7749.", "0.7749 not found in the result of build_contract"),
    ("It has AUC 0.7749. [source: clean_data]", "They appear in: run_eda"),
    # right number, right tag, but the prose names a different report
    ("The cleaning report notes AUC 0.7749. [source: run_eda]", "names cleaning report"),
    # a source that never ran in this run is an invented source
    ("Recall is 0.81. [source: train_and_compare_models]", "did not produce a result in this run"),
    ("Churn is 31.4% overall. [source: run_eda]", "No tool produced them"),
])
def test_wrong_or_invented_sources_are_caught(answer, expected):
    problems = attribution_problems(answer, RESULTS, CONTRACT_TOOLS)
    assert len(problems) == 1 and expected in problems[0]
