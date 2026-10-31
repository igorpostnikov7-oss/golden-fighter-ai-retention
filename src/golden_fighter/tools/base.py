"""Shared base for the crews' tools.

Every tool is a thin CrewAI wrapper around an existing, tested pipeline function. File paths are fixed
when the tool is built, never chosen by the agent, so an agent cannot point a tool at other data.

A data-validation failure is final: CrewAI turns tool exceptions into text the agent may ignore, so the
failure is also recorded in a shared ToolRunLog. Once blocked, every later tool call in the run refuses
to work, and the crew runner raises the failure after the crew finishes, whatever the agent answered.
"""
import json
import logging
import time
from dataclasses import dataclass, field

from crewai.tools import BaseTool
from pydantic import BaseModel

from golden_fighter.errors import DataValidationError

log = logging.getLogger(__name__)

MAX_RESULT_CHARS = 4000  # keeps tool results (and so prompt tokens) small


class ToolInputRejected(ValueError):
    """The agent's input was refused (e.g. an ungrounded number); the agent may fix it and retry."""


class NoArgs(BaseModel):
    """This tool takes no arguments: call it with an empty input {}."""


@dataclass
class ToolRunLog:
    calls: list[dict] = field(default_factory=list)
    blocked_by: DataValidationError | None = None
    results: dict[str, dict] = field(default_factory=dict)  # last successful result per tool (source check)

    def record(self, tool: str, status: str, seconds: float, detail: str = "") -> None:
        self.calls.append({"tool": tool, "status": status, "seconds": round(seconds, 2), "detail": detail[:300]})

    def succeeded(self, tool: str) -> bool:
        return any(c["tool"] == tool and c["status"] == "ok" for c in self.calls)


SOURCE_CHECK = "source_check"  # entries the source check adds to the log; they are verdicts, not tool runs


def tool_execution_counts(calls: list[dict]) -> dict:
    """Agent tool executions (our free Python tools) kept apart from source-check verdicts."""
    runs = [c for c in calls if c["tool"] != SOURCE_CHECK]
    return {"agent_tool_executions": len(runs), "successful_tool_executions": sum(c["status"] == "ok" for c in runs),
            "source_check_rejections": sum(c["tool"] == SOURCE_CHECK for c in calls)}


def contract_reasons(contract: dict) -> dict[str, list[str]]:
    """The contract's own words on why each forbidden or excluded column is kept out of the model."""
    reasons = {}
    for col in contract["columns"]:
        name = col["name"]
        if name not in contract["forbidden_features"] + contract["excluded_from_primary_model"]:
            continue
        notes = [f"role: {col['role']}", col["description"]]
        if col.get("sensitive"):
            notes.append("sensitive attribute")
        notes += [col[key] for key in ("leakage_note", "availability_note") if col.get(key)]
        notes += [a for a in contract.get("assumptions", []) if name in a]
        reasons[name] = notes
    return reasons


class PipelineTool(BaseTool):
    """Runs `execute()` and returns a compact JSON result; turns failures into explicit STOP / ERROR text."""

    args_schema: type[BaseModel] = NoArgs
    run_log: ToolRunLog
    max_usage_count: int | None = 2

    def execute(self, **kwargs) -> dict:
        raise NotImplementedError

    def _run(self, **kwargs) -> str:
        start = time.perf_counter()
        if self.run_log.blocked_by is not None:
            self.run_log.record(self.name, "refused", 0.0, "run already stopped by a validation failure")
            return ("REFUSED: this run was stopped by a data validation failure. Do not call more tools; "
                    "report the failure in your final answer.")
        try:
            result = self.execute(**kwargs)
        except DataValidationError as err:
            self.run_log.blocked_by = err
            self.run_log.record(self.name, "validation_failed", time.perf_counter() - start, str(err))
            log.error("Tool %s stopped the run: %s", self.name, err)
            return ("STOP - DATA VALIDATION FAILED. The run cannot continue and this cannot be worked around. "
                    f"Report this failure as your final answer:\n{err}")
        except ToolInputRejected as err:
            self.run_log.record(self.name, "rejected", time.perf_counter() - start, str(err))
            return f"REJECTED: {err} Fix your input and call {self.name} again."
        except Exception as err:  # unexpected bug or I/O problem: report it, never invent a result
            self.run_log.record(self.name, "error", time.perf_counter() - start, f"{type(err).__name__}: {err}")
            log.exception("Tool %s failed", self.name)
            return f"ERROR: {self.name} failed with {type(err).__name__}: {err}. Do not invent results."
        self.run_log.record(self.name, "ok", time.perf_counter() - start)
        self.run_log.results[self.name] = result
        text = json.dumps(result, ensure_ascii=False, default=str)
        return text if len(text) <= MAX_RESULT_CHARS else text[:MAX_RESULT_CHARS] + " ...(truncated)"
