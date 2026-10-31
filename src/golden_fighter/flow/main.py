"""The Golden Fighter CrewAI Flow: Crew 1 -> validation gate -> Crew 2, with a hard stop in between.

  preflight -> analyst -> validation_gate --"passed"--> scientist -> check_scientist --"succeeded"--> finalize
                              (router)   \\--"failed"--> handle_failure <--"failed"--/   (router)

Usage:
  python -m golden_fighter.flow.main                    # full run with the real LLM (costs money)
  python -m golden_fighter.flow.main --no-llm           # same Flow, deterministic pipelines instead of agents
                                                        # (writes to artifacts/_no_llm/, never over the
                                                        #  validated, agent-written artifacts)
  python -m golden_fighter.flow.main --from-artifacts   # skip Crew 1: gate + Crew 2 on existing artifacts
  python -m golden_fighter.flow.main --demo-break unit_x100   # break a COPY of the data; the Flow must stop

Every run writes artifacts/runs/<run_id>/run_summary.json; a failed run also writes failure_report.md.
Exit code: 0 = succeeded, 1 = validation failed (Crew 2 never started), 2 = crew/API/setup error.
"""
import argparse
import json
import logging
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from crewai.flow.flow import Flow, listen, router, start
from pydantic import PrivateAttr

from golden_fighter import config
from golden_fighter.analysis.cleaning import load_raw
from golden_fighter.analysis.pipeline import run_analyst_pipeline
from golden_fighter.contract.breakers import SCENARIOS, make_broken_copy
from golden_fighter.contract.validator import REQUIRED_ARTIFACTS, run_gate
from golden_fighter.crews.analyst_crew.crew import OUTPUT_FILES as ANALYST_FILES
from golden_fighter.crews.analyst_crew.crew import run_analyst_crew
from golden_fighter.crews.runner import CrewRunError
from golden_fighter.crews.scientist_crew.crew import OUTPUT_FILES as SCIENTIST_FILES
from golden_fighter.crews.scientist_crew.crew import run_scientist_crew
from golden_fighter.errors import DataValidationError
from golden_fighter.flow import artifacts, reports
from golden_fighter.flow.state import EXIT_CODES, FlowState, StageRecord
from golden_fighter.llm import LLMSetupError, build_llm
from golden_fighter.logging_setup import configure_logging
from golden_fighter.modeling import pipeline as modeling
from golden_fighter.modeling.pipeline import run_scientist_pipeline
from golden_fighter.tools.base import tool_execution_counts

log = logging.getLogger(__name__)

USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens", "cached_prompt_tokens")
COUNT_KEYS = ("model_api_calls", "agent_tool_executions", "successful_tool_executions", "source_check_rejections")
# Written into every run_summary.json so a reader never confuses the two kinds of "call".
GLOSSARY = {
    "model_api_calls": "Paid requests to the Claude API (these cost money and use tokens).",
    "agent_tool_executions": "Times an agent ran one of the project's Python tools (free, deterministic code).",
    "source_check_rejections": "Agent answers sent back because a number was not credited to its real source.",
}


@dataclass
class FlowSettings:
    raw_path: Path = config.RAW_DATA_PATH
    analyst_dir: Path = config.ANALYST_DIR
    scientist_dir: Path = config.SCIENTIST_DIR
    runs_dir: Path = config.RUNS_DIR
    use_llm: bool = True            # False: the deterministic pipelines do the crews' work (no API key needed)
    from_artifacts: bool = False    # True: skip Crew 1 and start at the gate with the existing analyst artifacts
    retries: bool = True            # False: no automatic API retries
    verbose: bool = False
    demo_scenario: str | None = None


class GoldenFighterFlow(Flow[FlowState]):
    """Each stage catches its own failure and records it in the state; the routers send a failed run to
    handle_failure, so Crew 2 can only start through the gate's "passed" route."""

    _settings: FlowSettings = PrivateAttr()
    _llm: object = PrivateAttr(default=None)
    _fresh: dict = PrivateAttr(default_factory=dict)
    _t0: float = PrivateAttr(default=0.0)

    def __init__(self, settings: FlowSettings | None = None, llm=None, **kwargs):
        super().__init__(**kwargs)
        self._settings = settings or FlowSettings()
        self._llm = llm  # tests inject the scripted mock LLM here; None + use_llm builds the real one
        self._fresh = {"analyst": set(), "scientist": set()}

    # --- helpers ----------------------------------------------------------------------------------------

    @contextmanager
    def _stage(self, name: str):
        record = StageRecord(name=name, status="ok")
        started = time.perf_counter()
        try:
            yield record
        finally:
            record.seconds = round(time.perf_counter() - started, 2)
            if self.state.failed and self.state.failed_stage == name:
                record.status = self.state.status
                record.detail = record.detail or (self.state.failure_reason or "")
            self.state.stages.append(record)

    def _fail(self, stage: str, status: str, reason: str, issues: list[str] | None = None) -> None:
        self.state.status, self.state.failed_stage, self.state.failure_reason = status, stage, reason
        self.state.validation_issues = list(issues or [])
        log.error("Flow stopped at %s (%s): %s", stage, status, reason.splitlines()[0])

    def _skip(self, name: str, why: str) -> None:
        self.state.stages.append(StageRecord(name=name, status="skipped", detail=why))

    def _run_stage(self, name: str, out_dir: Path, files: tuple[str, ...], run) -> None:
        """Run one crew (or its deterministic pipeline), classify any failure, and check freshness."""
        paths = [out_dir / f for f in files]
        before = artifacts.snapshot(paths)
        try:
            result = run()
        except DataValidationError as err:
            self.state.crews[name] = self._usage_entry(getattr(err, "crew_record", None))
            self._fail(name, "validation_failed", str(err), err.issues)
            return
        except (CrewRunError, LLMSetupError) as err:
            self.state.crews[name] = self._usage_entry(getattr(err, "record", None))
            self._fail(name, "error", str(err))
            return
        except Exception as err:  # an unexpected bug still ends in a readable report, never a silent pass
            log.exception("%s stage crashed", name)
            self._fail(name, "error", f"{type(err).__name__}: {err}")
            return
        self.state.crews[name] = self._usage_entry(result.to_dict() if hasattr(result, "to_dict") else None)
        stale = artifacts.not_rewritten(before)
        if stale:
            self._fail(name, "error", f"these outputs were not written in this run: {', '.join(stale)}")
            return
        self._fresh[name] = set(files)

    def _usage_entry(self, record: dict | None) -> dict:
        if not self._settings.use_llm:
            return {"mode": "deterministic pipeline (no LLM)"} | dict.fromkeys(COUNT_KEYS, 0)
        record = record or {}
        tool_log = record.get("tool_calls", [])
        entry = {"mode": "crewai crew", "model": record.get("model"), "status": record.get("status"),
                 "model_api_calls": record.get("llm_calls", 0), **tool_execution_counts(tool_log),
                 "estimated_cost_usd": record.get("estimated_cost_usd"),
                 "duration_seconds": record.get("duration_seconds"),
                 "tool_log": tool_log, "warnings": record.get("warnings", [])}
        usage = record.get("usage") or {}
        return entry | {key: usage.get(key, 0) for key in USAGE_KEYS}

    def _finish(self) -> None:
        s, state = self._settings, self.state
        state.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        state.duration_seconds = round(time.perf_counter() - self._t0, 1)
        state.exit_code = EXIT_CODES[state.status]
        crews = state.crews.values()
        costs = [c["estimated_cost_usd"] for c in crews if c.get("estimated_cost_usd") is not None]
        state.totals = {**{key: sum(c.get(key, 0) for c in crews) for key in COUNT_KEYS + USAGE_KEYS},
                        "estimated_cost_usd": round(sum(costs), 6) if costs else None}
        state.artifacts = {
            "analyst": artifacts.describe([s.analyst_dir / f for f in ANALYST_FILES], self._fresh["analyst"]),
            "scientist": artifacts.describe([s.scientist_dir / f for f in SCIENTIST_FILES], self._fresh["scientist"]),
        }
        run_dir = s.runs_dir / state.run_id
        state.paths["run_dir"] = artifacts.display_path(run_dir)
        if state.failed:
            state.paths["failure_report"] = artifacts.display_path(run_dir / reports.FAILURE_REPORT_FILE)
        state.paths["run_summary"] = artifacts.display_path(run_dir / reports.SUMMARY_FILE)
        if state.failed:
            reports.write_failure_report(state, run_dir)
        reports.write_summary(state, run_dir)

    # --- the Flow -----------------------------------------------------------------------------------------

    @start()
    def preflight(self):
        s, state = self._settings, self.state
        self._t0 = time.perf_counter()
        now = datetime.now(timezone.utc)
        state.run_id = f"{now:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
        state.started_at = now.isoformat(timespec="seconds")
        state.mode = "llm" if s.use_llm else "no_llm"
        state.from_artifacts, state.demo_scenario = s.from_artifacts, s.demo_scenario
        state.paths = {"analyst_dir": artifacts.display_path(s.analyst_dir),
                       "scientist_dir": artifacts.display_path(s.scientist_dir)}
        state.inputs = {"seed": config.SEED}
        state.glossary = GLOSSARY
        log.info("Flow run %s started (mode %s%s)", state.run_id, state.mode,
                 ", from existing artifacts" if s.from_artifacts else "")
        with self._stage("preflight") as stage:
            if not s.from_artifacts:
                try:
                    load_raw(s.raw_path)  # file exists, readable, raw schema sane
                except DataValidationError as err:
                    self._fail("preflight", "validation_failed", str(err), err.issues)
                    return
                state.inputs |= {"raw_data": artifacts.display_path(s.raw_path),
                                 "raw_sha256": artifacts.sha256(s.raw_path)}
            if s.use_llm:
                try:
                    self._llm = self._llm or build_llm(retries=s.retries)  # no API call; fails fast without a key
                except LLMSetupError as err:
                    self._fail("preflight", "error", str(err))
                    return
                state.model = getattr(self._llm, "model", None)
            stage.detail = "raw data readable" if not s.from_artifacts else "using existing analyst artifacts"

    @listen(preflight)
    def analyst(self):
        s = self._settings
        if self.state.failed:
            return self._skip("analyst", "not started: an earlier stage failed")
        if s.from_artifacts:
            return self._skip("analyst", f"skipped: re-using {artifacts.display_path(s.analyst_dir)} "
                                         "(the gate re-validates it)")
        with self._stage("analyst"):
            if s.use_llm:
                self._run_stage("analyst", s.analyst_dir, ANALYST_FILES, lambda: run_analyst_crew(
                    llm=self._llm, raw_path=s.raw_path, out_dir=s.analyst_dir, verbose=s.verbose))
            else:
                self._run_stage("analyst", s.analyst_dir, ANALYST_FILES,
                                lambda: run_analyst_pipeline(s.raw_path, s.analyst_dir))

    @router(analyst)
    def validation_gate(self):
        if self.state.failed:
            self._skip("validation_gate", "not run: an earlier stage failed")
            return "failed"
        with self._stage("validation_gate") as stage:
            result = run_gate(self._settings.analyst_dir)  # pure Python; no LLM can influence it
            self.state.gate = result.to_dict()
            stage.detail = result.render().splitlines()[0]
            if not result.passed:
                self._fail("validation_gate", "validation_failed",
                           f"the validation gate found {len(result.errors)} contract violation(s)",
                           [str(issue) for issue in result.errors])
                return "failed"
        log.info("%s", stage.detail)
        return "passed"

    @listen("passed")
    def scientist(self):
        s = self._settings
        with self._stage("scientist"):
            if s.use_llm:
                self._run_stage("scientist", s.scientist_dir, SCIENTIST_FILES, lambda: run_scientist_crew(
                    llm=self._llm, analyst_dir=s.analyst_dir, out_dir=s.scientist_dir, verbose=s.verbose))
            else:
                self._run_stage("scientist", s.scientist_dir, SCIENTIST_FILES,
                                lambda: run_scientist_pipeline(s.analyst_dir, s.scientist_dir))
        if not self.state.failed:
            self.state.metrics = _headline_metrics(s.scientist_dir / modeling.EVALUATION_JSON)

    @router(scientist)
    def check_scientist(self):
        return "failed" if self.state.failed else "succeeded"

    @listen("succeeded")
    def finalize(self):
        self.state.status = "succeeded"
        self._finish()
        log.info("Flow run %s SUCCEEDED in %.0f s. Summary: %s", self.state.run_id, self.state.duration_seconds,
                 self.state.paths["run_summary"])
        return self.state.status

    @listen("failed")
    def handle_failure(self):
        if self.state.failed_stage != "scientist":
            self._skip("scientist", "not started: the Flow stopped before modelling")
        self._finish()
        log.error("Flow run %s FAILED at %s. Report: %s", self.state.run_id, self.state.failed_stage,
                  self.state.paths["failure_report"])
        return self.state.status


def _headline_metrics(evaluation_json: Path) -> dict:
    """The selected model's numbers, copied from the deterministic evaluation.json (never from an agent)."""
    report = json.loads(evaluation_json.read_text(encoding="utf-8"))
    selected = report["selected_model"]
    test = report["results"][selected]["test"]
    return {"selected_model": selected, "selection_rule": report["selection_rule"],
            "cv_roc_auc_mean": report["results"][selected]["cv"]["roc_auc"]["mean"],
            "test": {m: test[m] for m in ("roc_auc", "pr_auc", "recall", "precision", "f1", "accuracy")},
            "leakage_ablation_test_roc_auc": report["ablation"]["test"]["roc_auc"],
            "train_rows": report["data"]["train_rows"], "test_rows": report["data"]["test_rows"]}


def run_flow(settings: FlowSettings | None = None, llm=None) -> FlowState:
    """Run the Flow once and return its final state (also written to run_summary.json)."""
    flow = GoldenFighterFlow(settings, llm=llm)
    flow.kickoff()
    return flow.state


def demo_source_dir() -> Path:
    """The Crew 1 artifacts a break demo copies: the validated ones, or on a fresh clone (where the member-level
    clean_data.csv is not committed) the complete set the free --no-llm run wrote. Neither is ever modified."""
    for folder in (config.ANALYST_DIR, config.ARTIFACTS_DIR / "_no_llm" / "analyst"):
        if all((folder / name).is_file() and (folder / name).stat().st_size for name in REQUIRED_ARTIFACTS):
            return folder
    raise FileNotFoundError("The break demo needs a complete set of Crew 1 artifacts, including clean_data.csv. "
                            "Run the free Flow first: python -m golden_fighter.flow.main --no-llm")


def _new_demo_dir(base: Path, scenario: str) -> Path:
    """A new, uniquely named folder for one break demo. Existing folders, including ones written by earlier
    versions without the demo-copy marker, are never reused, changed or deleted."""
    base.mkdir(parents=True, exist_ok=True)
    for _ in range(5):
        folder = base / f"{scenario}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}-{uuid.uuid4().hex[:6]}"
        try:
            folder.mkdir()  # fails instead of reusing if the name is already taken
            return folder
        except FileExistsError:
            continue
    raise FileExistsError(f"could not create a new demo folder in {base}")


def demo_settings(scenario: str) -> FlowSettings:
    """Break a COPY of Crew 1's artifacts in a fresh folder and point the Flow at it. Real artifacts are never
    touched; the run summaries of all demos stay together in artifacts/_broken/runs/."""
    base = config.ARTIFACTS_DIR / "_broken"  # gitignored
    source = demo_source_dir()  # checked first, so a demo that cannot start leaves no empty folder
    demo_dir = _new_demo_dir(base, scenario)
    broken = make_broken_copy(source, scenario, demo_dir / "analyst")
    return FlowSettings(analyst_dir=broken, scientist_dir=demo_dir / "scientist", runs_dir=base / "runs",
                        use_llm=False, from_artifacts=True, demo_scenario=scenario)


def no_llm_settings(from_artifacts: bool = False) -> FlowSettings:
    """Free run without agents. Outputs go to artifacts/_no_llm/ (gitignored) so the validated artifacts, which
    hold the agents' summaries, are never replaced by agent-less versions. With from_artifacts the existing
    analyst artifacts are only read."""
    base = config.ARTIFACTS_DIR / "_no_llm"
    return FlowSettings(raw_path=config.RAW_DATA_PATH,
                        analyst_dir=config.ANALYST_DIR if from_artifacts else base / "analyst",
                        scientist_dir=base / "scientist", runs_dir=base / "runs", use_llm=False,
                        from_artifacts=from_artifacts)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # CrewAI's console output uses emoji; Windows consoles may not
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-llm", action="store_true", help="deterministic pipelines instead of agents (free)")
    parser.add_argument("--from-artifacts", action="store_true", help="skip Crew 1; gate + Crew 2 on existing files")
    parser.add_argument("--demo-break", choices=list(SCENARIOS), metavar="SCENARIO",
                        help=f"break a copy of the data and show the Flow stopping ({', '.join(SCENARIOS)})")
    parser.add_argument("--no-retries", action="store_true", help="never repeat a failed API request")
    parser.add_argument("--verbose", action="store_true", help="show the agents' reasoning and tool calls")
    args = parser.parse_args(argv)
    configure_logging(config.LOGS_DIR / "flow.log")

    if args.demo_break:
        try:
            settings = demo_settings(args.demo_break)
        except FileNotFoundError as err:
            log.error("%s", err)
            return EXIT_CODES["error"]
        log.info("Break demo '%s': %s", args.demo_break, SCENARIOS[args.demo_break][0])
    elif args.no_llm:
        settings = no_llm_settings(args.from_artifacts)
    else:
        settings = FlowSettings(from_artifacts=args.from_artifacts, retries=not args.no_retries, verbose=args.verbose)
    state = run_flow(settings)
    if state.failed:
        for issue in state.validation_issues:
            log.error("  %s", issue)
    totals = state.totals
    if totals.get("model_api_calls"):
        log.info("Model API calls: %s (%s tokens, estimated cost %s USD). Agent tool executions: %s.",
                 totals["model_api_calls"], totals.get("total_tokens"), totals.get("estimated_cost_usd"),
                 totals.get("agent_tool_executions"))
    else:
        log.info("Model API calls: none (cost 0 USD)")
    return state.exit_code


if __name__ == "__main__":
    sys.exit(main())
