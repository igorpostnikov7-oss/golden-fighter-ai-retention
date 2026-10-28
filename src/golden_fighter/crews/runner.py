"""Builds a crew from its YAML config and runs it under hard limits, then verifies the result in Python.

The crew's final answer is never trusted on its own. After kickoff the runner checks, deterministically:
  1. no tool hit a data-validation failure (if one did, that DataValidationError is raised);
  2. the model-call budget was not exceeded and the API did not fail (else CrewRunError);
  3. every expected artifact was (re)written during this run (else CrewRunError).
Each task's final answer also passes a source check (tools/attribution.py) before the next task starts.
Token usage and the estimated cost are logged to logs/crew_runs.jsonl for every run, even a failed one.
"""
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml
from crewai import Agent, Crew, Process, Task

from golden_fighter import config
from golden_fighter.llm import describe_api_error, estimate_cost_usd, llm_call_budget
from golden_fighter.tools.attribution import SOURCE_RULES, source_check
from golden_fighter.tools.base import ToolRunLog

log = logging.getLogger(__name__)

RUN_LOG_FILE = "crew_runs.jsonl"
FINAL_ANSWER_LOG_CHARS = 2000


class CrewRunError(RuntimeError):
    """The crew could not finish (API error, budget, missing artifacts). `record` holds usage and cost."""

    def __init__(self, crew: str, reason: str, record: dict):
        super().__init__(f"[{crew} crew] {reason}")
        self.crew = crew
        self.reason = reason
        self.record = record


@dataclass
class CrewRunResult:
    crew: str
    model: str
    status: str
    final_answer: str
    llm_calls: int
    usage: dict
    estimated_cost_usd: float | None
    duration_seconds: float
    tool_calls: list[dict]
    outputs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_crew(config_dir: Path, llm, tools: dict, verbose: bool = False) -> Crew:
    """Agents and tasks from agents.yaml / tasks.yaml; each agent gets exactly the tools its YAML names.

    Every task gets the source rules and a deterministic source check on its final answer.
    """
    run_log = next(iter(tools.values())).run_log  # all tools of one run share one log
    agents_cfg = _load_yaml(config_dir / "agents.yaml")
    tasks_cfg = _load_yaml(config_dir / "tasks.yaml")
    agents = {
        key: Agent(role=cfg["role"].strip(), goal=cfg["goal"].strip(), backstory=cfg["backstory"].strip(),
                   tools=[tools[name] for name in cfg["tools"]], llm=llm,
                   max_iter=config.AGENT_MAX_ITER, max_retry_limit=config.AGENT_MAX_RETRY_LIMIT,
                   allow_delegation=False, verbose=verbose)
        for key, cfg in agents_cfg.items()
    }
    tasks: dict[str, Task] = {}
    for key, cfg in tasks_cfg.items():
        own_tools = agents_cfg[cfg["agent"]]["tools"]
        tasks[key] = Task(name=key, description=cfg["description"].strip(),
                          expected_output=cfg["expected_output"].strip() + "\n" + SOURCE_RULES,
                          agent=agents[cfg["agent"]], context=[tasks[name] for name in cfg.get("context", [])],
                          guardrail=source_check(run_log, own_tools, key),
                          guardrail_max_retries=config.SOURCE_CHECK_MAX_RETRIES)
    return Crew(agents=list(agents.values()), tasks=list(tasks.values()), process=Process.sequential,
                max_rpm=config.CREW_MAX_RPM, cache=False, verbose=verbose)


def _snapshot(paths: list[Path]) -> dict[Path, tuple[int, int] | None]:
    return {p: (p.stat().st_mtime_ns, p.stat().st_size) if p.exists() else None for p in paths}


def _append_run_log(record: dict) -> None:
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)  # read at call time so tests can redirect it
    with (config.LOGS_DIR / RUN_LOG_FILE).open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def run_crew(name: str, crew: Crew, llm, run_log: ToolRunLog, expected_outputs: list[Path],
             optional_tools: tuple[str, ...] = (), max_llm_calls: int | None = None) -> CrewRunResult:
    max_llm_calls = max_llm_calls or config.MAX_LLM_CALLS_PER_CREW
    model = getattr(llm, "model", "unknown")
    before = _snapshot(expected_outputs)
    usage_before = llm.get_token_usage_summary().model_dump()
    started = time.perf_counter()
    error: Exception | None = None
    final_answer = ""
    log.info("Starting %s crew with %s (budget: %d model calls)", name, model, max_llm_calls)
    with llm_call_budget(max_llm_calls) as budget:
        try:
            final_answer = crew.kickoff().raw or ""
        except Exception as exc:  # classified below; never swallowed
            error = exc

    usage_after = llm.get_token_usage_summary().model_dump()
    usage = {k: usage_after[k] - usage_before.get(k, 0) for k in usage_after if isinstance(usage_after[k], int)}
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "crew": name, "model": model, "llm_calls": budget["calls"], "usage": usage,
        "estimated_cost_usd": estimate_cost_usd(model, usage.get("prompt_tokens", 0),
                                                usage.get("completion_tokens", 0)),
        "duration_seconds": round(time.perf_counter() - started, 1),
        "tool_calls": run_log.calls, "final_answer": final_answer[:FINAL_ANSWER_LOG_CHARS],
    }

    def fail(status: str, reason: str):
        record.update(status=status, reason=reason)
        _append_run_log(record)
        log.error("%s crew %s: %s", name, status, reason)

    if run_log.blocked_by is not None:  # the gate said no: that verdict wins over anything the agent wrote
        fail("validation_failed", str(run_log.blocked_by))
        run_log.blocked_by.crew_record = record  # so the Flow can still report this run's usage and cost
        raise run_log.blocked_by
    if error is not None:
        source_failures = [c["detail"] for c in run_log.calls if c["tool"] == "source_check"]
        if budget["exceeded"]:
            reason = f"stopped after the model-call budget of {max_llm_calls} calls was used up"
        elif source_failures and "guardrail" in str(error):
            reason = f"an agent's final answer failed the source check after a rewrite: {source_failures[-1]}"
        else:
            reason = describe_api_error(error) or f"crew failed: {type(error).__name__}: {str(error)[:300]}"
        fail("error", reason)
        raise CrewRunError(name, reason, record) from error

    after = _snapshot(expected_outputs)
    stale = [p.name for p in expected_outputs if after[p] is None or after[p] == before[p]]
    if stale:
        reason = f"the agents finished but these artifacts were not written in this run: {', '.join(stale)}"
        fail("incomplete", reason)
        raise CrewRunError(name, reason, record)

    warnings = [f"optional tool {tool} did not succeed; the document has no agent-written summary"
                for tool in optional_tools if not run_log.succeeded(tool)]
    record.update(status="ok", warnings=warnings)
    _append_run_log(record)
    log.info("%s crew finished: %d model calls, %s tokens, estimated cost %s USD", name, budget["calls"],
             usage.get("total_tokens"), record["estimated_cost_usd"])
    return CrewRunResult(crew=name, model=model, status="ok", final_answer=final_answer,
                         llm_calls=budget["calls"], usage=usage, estimated_cost_usd=record["estimated_cost_usd"],
                         duration_seconds=record["duration_seconds"], tool_calls=run_log.calls,
                         outputs=[str(p) for p in expected_outputs], warnings=warnings)
