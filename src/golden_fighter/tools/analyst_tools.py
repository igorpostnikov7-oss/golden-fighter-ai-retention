"""CrewAI tools for the Data Analyst crew. Each wraps a step of analysis/pipeline.py; numbers come from code."""
import json
from pathlib import Path

from pydantic import BaseModel, Field

from golden_fighter.analysis import pipeline as steps
from golden_fighter.analysis.cleaning import load_raw
from golden_fighter.contract.validator import run_gate
from golden_fighter.tools.base import PipelineTool, ToolInputRejected
from golden_fighter.tools.grounding import ungrounded_numbers, upsert_agent_section

SUMMARY_MIN_CHARS = 150
SUMMARY_MAX_CHARS = 1500
INSIGHTS_SUMMARY_HEADING = "Analyst summary (written by the EDA & Insights agent)"


class ProfileRawDataTool(PipelineTool):
    name: str = "profile_raw_data"
    description: str = ("Read-only first look at the raw CSV: row count, columns, missing values and duplicate rows. "
                        "Fails with a validation error if the file is missing, empty or has the wrong columns.")
    raw_path: Path

    def execute(self) -> dict:
        raw = load_raw(self.raw_path)
        return {"rows": len(raw), "columns": list(raw.columns), "missing_values": int(raw.isna().sum().sum()),
                "duplicate_rows": int(raw.duplicated().sum())}


class CleanDataTool(PipelineTool):
    name: str = "clean_data"
    description: str = ("Cleans the raw CSV with the tested cleaning rules (snake_case names, numeric types, nulls, "
                        "duplicates, domain rules; stops if more than 5% of rows are invalid). "
                        "Writes clean_data.csv and cleaning_report.json and returns the cleaning report.")
    raw_path: Path
    out_dir: Path

    def execute(self) -> dict:
        report = steps.clean_step(self.raw_path, self.out_dir)
        report.pop("renamed_columns", None)  # long and uninteresting for the agent
        return report


class RunEdaTool(PipelineTool):
    name: str = "run_eda"
    description: str = ("Builds eda_report.html (charts + statistics) and insights.md from clean_data.csv, "
                        "and returns the computed churn facts (rates are fractions: 0.2652 = 26.52%).")
    out_dir: Path

    def execute(self) -> dict:
        facts = steps.eda_and_insights_step(self.out_dir)
        facts["leakage_findings"] = [f for f in facts["leakage_findings"] if f["risk"] != "none"]
        facts.pop("mean_stayed_vs_churned", None)
        return facts


class SummaryInput(BaseModel):
    summary: str = Field(description=f"Markdown bullet points, {SUMMARY_MIN_CHARS}-{SUMMARY_MAX_CHARS} characters. "
                                     "Use only numbers returned by run_eda.")


class WriteInsightsSummaryTool(PipelineTool):
    name: str = "write_insights_summary"
    description: str = ("Saves your business summary at the top of insights.md. Every number in it is checked "
                        "against the computed facts; a summary with an invented number is rejected.")
    args_schema: type[BaseModel] = SummaryInput
    out_dir: Path
    max_usage_count: int | None = 3

    def execute(self, summary: str) -> dict:
        summary = summary.strip()
        if not SUMMARY_MIN_CHARS <= len(summary) <= SUMMARY_MAX_CHARS:
            raise ToolInputRejected(f"The summary must be {SUMMARY_MIN_CHARS}-{SUMMARY_MAX_CHARS} characters, "
                                    f"got {len(summary)}.")
        facts = json.loads((self.out_dir / steps.FACTS_FILE).read_text(encoding="utf-8"))
        bad = ungrounded_numbers(summary, facts)
        if bad:
            raise ToolInputRejected(f"These numbers are not in the computed facts: {', '.join(bad)}.")
        path = self.out_dir / steps.INSIGHTS_FILE
        path.write_text(upsert_agent_section(path.read_text(encoding="utf-8"), INSIGHTS_SUMMARY_HEADING, summary),
                        encoding="utf-8", newline="\n")
        return {"saved": str(path.name), "characters": len(summary)}


class BuildContractTool(PipelineTool):
    name: str = "build_contract"
    description: str = ("Builds dataset_contract.json from clean_data.csv: types, allowed values, ranges, units, "
                        "target, forbidden and leakage-excluded columns, cross-column rules. Returns a summary.")
    out_dir: Path

    def execute(self) -> dict:
        contract = steps.contract_step(self.out_dir)
        return {key: contract[key] for key in ("contract_version", "prediction_time", "row_count_min",
                                               "target", "forbidden_features", "excluded_from_primary_model",
                                               "cross_column_rules")} | {
            "columns": [c["name"] for c in contract["columns"]]}


class ContractSelfCheckTool(PipelineTool):
    name: str = "contract_self_check"
    description: str = ("Read-only: runs the deterministic validation gate on the four analyst artifacts and returns "
                        "PASSED/FAILED with every issue. You cannot change its verdict.")
    out_dir: Path

    def execute(self) -> dict:
        result = run_gate(self.out_dir)
        result.raise_if_failed()  # a failure blocks the run (see tools/base.py)
        return {"passed": True, "checks_run": result.checks_run,
                "warnings": [str(w) for w in result.warnings]}


def analyst_tools(run_log, raw_path: Path, out_dir: Path) -> dict[str, PipelineTool]:
    """All analyst tools for one run, sharing one ToolRunLog. Keys are tool names."""
    tools = [ProfileRawDataTool(run_log=run_log, raw_path=raw_path),
             CleanDataTool(run_log=run_log, raw_path=raw_path, out_dir=out_dir),
             RunEdaTool(run_log=run_log, out_dir=out_dir),
             WriteInsightsSummaryTool(run_log=run_log, out_dir=out_dir),
             BuildContractTool(run_log=run_log, out_dir=out_dir),
             ContractSelfCheckTool(run_log=run_log, out_dir=out_dir)]
    return {tool.name: tool for tool in tools}
