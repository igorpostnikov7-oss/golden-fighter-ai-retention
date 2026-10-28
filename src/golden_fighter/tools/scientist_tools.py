"""CrewAI tools for the Data Scientist crew. Each wraps a step of modeling/pipeline.py.

These tools only know `artifacts/analyst/` (Crew 1's published output) and `artifacts/scientist/`. They have
no raw-data path at all, refuse an analyst directory inside data/raw, and every data-reading step runs the
validation gate first.
"""
import json
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from golden_fighter.modeling import pipeline as steps
from golden_fighter.modeling.features import ENGINEERED_FEATURES, ensure_crew1_artifact_dir, load_validated_inputs
from golden_fighter.tools.base import PipelineTool, ToolInputRejected, contract_reasons
from golden_fighter.tools.grounding import ungrounded_numbers, upsert_agent_section

SUMMARY_MIN_CHARS = 150
SUMMARY_MAX_CHARS = 1500
CARD_SUMMARY_HEADING = "Summary for the retention team (written by the Model Card agent)"
TEST_METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")


class ScientistTool(PipelineTool):
    analyst_dir: Path
    out_dir: Path

    @model_validator(mode="after")
    def _only_crew1_artifacts(self):
        ensure_crew1_artifact_dir(self.analyst_dir)
        return self


class InspectContractTool(ScientistTool):
    name: str = "inspect_contract"
    description: str = ("Runs the validation gate, then returns what the dataset contract allows: target, primary "
                        "model features, forbidden and leakage-excluded columns with the contract's reasons, and the "
                        "engineered features.")

    def execute(self) -> dict:
        _, contract = load_validated_inputs(self.analyst_dir)
        return {"contract_version": contract.contract_version, "target": contract.target.name,
                "prediction_time": contract.prediction_time,
                "primary_features": contract.primary_feature_columns,
                "forbidden_features": contract.forbidden_features,
                "excluded_from_primary_model": contract.excluded_from_primary_model,
                "reasons": contract_reasons(contract.model_dump()),
                "engineered_features": ENGINEERED_FEATURES}


class BuildFeaturesTool(ScientistTool):
    name: str = "build_features"
    description: str = ("Runs the validation gate, then writes features.csv: member_id, a stratified 80/20 "
                        "train/test split label, the contract's primary features, engineered features and the target.")

    def execute(self) -> dict:
        return steps.features_step(self.analyst_dir, self.out_dir)


class TrainAndCompareTool(ScientistTool):
    name: str = "train_and_compare_models"
    description: str = ("Runs the gate again, trains a baseline, logistic regression and random forest with 5-fold "
                        "CV on the training split, selects by CV ROC-AUC (never the test set), runs the leakage "
                        "ablation, and writes model.joblib + evaluation.json. Returns the comparison.")

    def execute(self) -> dict:
        report = steps.training_step(self.analyst_dir, self.out_dir)
        selected = report["selected_model"]
        return {
            "selected_model": selected,
            "selection_rule": report["selection_rule"],
            "models": {name: {"cv_roc_auc": r["cv"]["roc_auc"],
                              "test": {m: r["test"][m] for m in TEST_METRICS}}
                       for name, r in report["results"].items()},
            "selected_confusion_matrix": report["results"][selected]["test"]["confusion_matrix"],
            "leakage_ablation": {"extra_columns": report["ablation"]["extra_columns"],
                                 "cv_roc_auc": report["ablation"]["cv"]["roc_auc"],
                                 "test_roc_auc": report["ablation"]["test"]["roc_auc"]},
            "top_features": report["feature_importance"][:5],
            "warnings": report["warnings"],
        }


class WriteModelReportsTool(ScientistTool):
    name: str = "write_model_reports"
    description: str = ("Writes evaluation_report.md and model_card.md from evaluation.json (metrics, fairness, "
                        "limitations, ethics). Returns the fairness results for your summary.")

    def execute(self) -> dict:
        steps.reports_step(self.out_dir)
        report = json.loads((self.out_dir / steps.EVALUATION_JSON).read_text(encoding="utf-8"))
        return {"files": [steps.EVALUATION_FILE, steps.MODEL_CARD_FILE], "fairness": report["fairness"]}


class SummaryInput(BaseModel):
    summary: str = Field(description=f"Markdown bullet points, {SUMMARY_MIN_CHARS}-{SUMMARY_MAX_CHARS} characters. "
                                     "Use only numbers returned by the training and report tools.")


class WriteModelCardSummaryTool(ScientistTool):
    name: str = "write_model_card_summary"
    description: str = ("Saves your plain-language summary at the top of model_card.md. Every number is checked "
                        "against evaluation.json; a summary with an invented number is rejected.")
    args_schema: type[BaseModel] = SummaryInput
    max_usage_count: int | None = 3

    def execute(self, summary: str) -> dict:
        summary = summary.strip()
        if not SUMMARY_MIN_CHARS <= len(summary) <= SUMMARY_MAX_CHARS:
            raise ToolInputRejected(f"The summary must be {SUMMARY_MIN_CHARS}-{SUMMARY_MAX_CHARS} characters, "
                                    f"got {len(summary)}.")
        report = json.loads((self.out_dir / steps.EVALUATION_JSON).read_text(encoding="utf-8"))
        bad = ungrounded_numbers(summary, report)
        if bad:
            raise ToolInputRejected(f"These numbers are not in evaluation.json: {', '.join(bad)}.")
        path = self.out_dir / steps.MODEL_CARD_FILE
        path.write_text(upsert_agent_section(path.read_text(encoding="utf-8"), CARD_SUMMARY_HEADING, summary),
                        encoding="utf-8", newline="\n")
        return {"saved": path.name, "characters": len(summary)}


def scientist_tools(run_log, analyst_dir: Path, out_dir: Path) -> dict[str, PipelineTool]:
    """All scientist tools for one run, sharing one ToolRunLog. Keys are tool names."""
    kwargs = {"run_log": run_log, "analyst_dir": analyst_dir, "out_dir": out_dir}
    tools = [InspectContractTool(**kwargs), BuildFeaturesTool(**kwargs), TrainAndCompareTool(**kwargs),
             WriteModelReportsTool(**kwargs), WriteModelCardSummaryTool(**kwargs)]
    return {tool.name: tool for tool in tools}
