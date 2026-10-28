"""Scripted mock-LLM runs of both crews (tests/mock_llm.py), shared by the crew and Flow tests."""
import json
from pathlib import Path

from mock_llm import final, tool


def insights_summary(out_dir: Path):
    def step(_messages):
        facts = json.loads((out_dir / "insight_facts.json").read_text(encoding="utf-8"))
        top = facts["churn_by_contract_period"][0]
        return tool("write_insights_summary", {"summary": (
            f"- Overall churn is {facts['churn_rate'] * 100:.1f}% of {facts['members']:,} members.\n"
            f"- Members on {top['group']}-month contracts churn at {top['churn_rate'] * 100:.1f}%.\n"
            "- Longer contracts and group classes go together with staying (association, not cause).")})
    return step


def analyst_script(out_dir: Path) -> dict:
    return {
        "Data Quality Engineer": [tool("profile_raw_data"), tool("clean_data"), final("Cleaned the data.")],
        "EDA & Insights Analyst": [
            tool("run_eda"),
            tool("write_insights_summary", {"summary": "- Churn is 77.7% for everybody, a number nobody computed.\n"
                                                       "- It only exists to check that invented numbers are refused "
                                                       "by the tool before anything is written to insights.md."}),
            insights_summary(out_dir),
            final("Summary saved."),
        ],
        "Data Contract Specialist": [tool("build_contract"), tool("contract_self_check"), final("Contract ok.")],
    }


def card_summary(out_dir: Path):
    def step(_messages):
        report = json.loads((out_dir / "evaluation.json").read_text(encoding="utf-8"))
        test = report["results"][report["selected_model"]]["test"]
        return tool("write_model_card_summary", {"summary": (
            f"- Predicts which members are likely to leave at the start of a month.\n"
            f"- Finds {test['recall'] * 100:.1f}% of churners; {test['precision'] * 100:.1f}% of flagged members "
            "really leave.\n- Use the score to offer help, never to penalise a member.")})
    return step


def scientist_script(out_dir: Path) -> dict:
    return {
        "Feature Engineer": [tool("inspect_contract"), tool("build_features"), final("Features built.")],
        "Model Trainer & Evaluator": [tool("train_and_compare_models"), final("Logistic regression selected.")],
        "Model Card Specialist": [tool("write_model_reports"), card_summary(out_dir), final("Card written.")],
    }
