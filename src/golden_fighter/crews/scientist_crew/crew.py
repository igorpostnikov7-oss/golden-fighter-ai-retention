"""Crew 2 - Data Scientist: Feature Engineer -> Model Trainer & Evaluator -> Model Card Specialist.

Reads only Crew 1's artifacts. The validation gate runs in plain Python before the crew starts, so a bad
hand-off costs no model calls, and again inside every data-reading tool.
"""
from pathlib import Path

from crewai import Crew

from golden_fighter import config
from golden_fighter.contract.validator import run_gate
from golden_fighter.crews.runner import CrewRunResult, build_crew, run_crew
from golden_fighter.llm import build_llm
from golden_fighter.modeling import pipeline as steps
from golden_fighter.modeling.features import ensure_crew1_artifact_dir
from golden_fighter.tools.base import ToolRunLog
from golden_fighter.tools.scientist_tools import scientist_tools

CONFIG_DIR = Path(__file__).parent / "config"
OUTPUT_FILES = (steps.FEATURES_FILE, steps.MODEL_FILE, steps.EVALUATION_FILE, steps.MODEL_CARD_FILE)


def build_scientist_crew(llm, run_log: ToolRunLog, analyst_dir: Path = config.ANALYST_DIR,
                         out_dir: Path = config.SCIENTIST_DIR, verbose: bool = False) -> Crew:
    return build_crew(CONFIG_DIR, llm, scientist_tools(run_log, analyst_dir, out_dir), verbose)


def run_scientist_crew(llm=None, analyst_dir: Path = config.ANALYST_DIR, out_dir: Path = config.SCIENTIST_DIR,
                       verbose: bool = False) -> CrewRunResult:
    """Gate first (no LLM), then run Crew 2 and verify its four artifacts. `llm=None` uses the real API."""
    ensure_crew1_artifact_dir(analyst_dir)
    run_gate(analyst_dir).raise_if_failed()
    llm = llm or build_llm()
    run_log = ToolRunLog()
    out_dir.mkdir(parents=True, exist_ok=True)
    crew = build_scientist_crew(llm, run_log, analyst_dir, out_dir, verbose)
    return run_crew("scientist", crew, llm, run_log, [out_dir / name for name in OUTPUT_FILES],
                    optional_tools=("write_model_card_summary",))
