"""What a Flow run leaves behind: run_summary.json always, failure_report.md when the run did not succeed."""
import json
from pathlib import Path

from golden_fighter.flow.state import FlowState

SUMMARY_FILE = "run_summary.json"
FAILURE_REPORT_FILE = "failure_report.md"

_WHAT_HAPPENED = {
    "validation_failed": "The data broke the dataset contract, so the Flow stopped before modelling.",
    "error": "A stage could not finish (crew, API or setup problem), so the Flow stopped.",
}
_STAGE_NAMES = {"preflight": "Preflight (raw data and setup)", "analyst": "Crew 1 - Data Analyst",
                "validation_gate": "Validation gate (pure Python)", "scientist": "Crew 2 - Data Scientist"}


def write_summary(state: FlowState, run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / SUMMARY_FILE
    path.write_text(json.dumps(state.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8", newline="\n")
    return path


def render_failure_report(state: FlowState) -> str:
    lines = [
        "# Flow run failed",
        "",
        f"- **Run:** `{state.run_id}` ({state.started_at})",
        f"- **Status:** `{state.status}` (exit code {state.exit_code})",
        f"- **Stopped at:** {_STAGE_NAMES.get(state.failed_stage or '', state.failed_stage)}",
        f"- **Mode:** {state.mode}" + (" from existing artifacts" if state.from_artifacts else "")
        + (f", break demo `{state.demo_scenario}`" if state.demo_scenario else ""),
        "",
        "## What happened",
        "",
        _WHAT_HAPPENED.get(state.status, "The Flow did not finish."),
        "",
        f"**Reason:** {(state.failure_reason or '').splitlines()[0] if state.failure_reason else 'unknown'}",
        "",
    ]
    if state.validation_issues:
        lines += ["## Every problem found", ""] + [f"- {issue}" for issue in state.validation_issues] + [""]
    scientist_ran = any(s.name == "scientist" and s.status != "skipped" for s in state.stages)
    lines += ["## What this means", ""]
    if not scientist_ran:
        lines.append("- **The Data Scientist crew did not start. No features were built and no model was trained "
                     "in this run.**")
    else:
        lines.append("- The Data Scientist crew started but did not finish; its outputs from this run are not valid.")
    stale = [a for a in state.artifacts.get("scientist", []) if a["exists"] and not a["produced_by_this_run"]]
    if stale:
        lines.append("- These scientist files are left over from an earlier run. They were **not** produced or "
                     "validated by this run, so do not use them as this run's results:")
        lines += [f"  - `{a['path']}` (last modified {a.get('modified', '?')})" for a in stale]
    lines += ["", "## Stages", "", "| Stage | Status | Seconds | Detail |", "|---|---|---:|---|"]
    lines += [f"| {_STAGE_NAMES.get(s.name, s.name)} | {s.status} | {s.seconds:.1f} | "
              f"{s.detail.splitlines()[0][:150] if s.detail else ''} |" for s in state.stages]
    totals = state.totals
    usage = (f"{totals['llm_calls']} model calls, {totals.get('total_tokens', 0)} tokens, estimated cost "
             f"{totals.get('estimated_cost_usd')} USD." if totals.get("llm_calls")
             else "No model calls were made in this run (cost: 0 USD).")
    lines += ["", "## Model usage in this run", "", usage, "", "## How to fix", ""]
    if state.status == "validation_failed":
        lines += ["1. Read the problems above: each names the column and the contract rule it breaks.",
                  "2. Fix the data at its source, or publish a new contract version if the change is intended "
                  "(for example a deliberate unit change).",
                  "3. Run the Flow again. The gate decides in plain Python; nobody can override it."]
    else:
        lines += ["1. Read the reason above (for an API problem it says what to check, e.g. the key or credit).",
                  "2. Fix it and run the Flow again. Failed runs are never retried automatically."]
    return "\n".join(lines) + "\n"


def write_failure_report(state: FlowState, run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / FAILURE_REPORT_FILE
    path.write_text(render_failure_report(state), encoding="utf-8", newline="\n")
    return path
