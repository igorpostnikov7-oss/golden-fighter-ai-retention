"""Crew 1 - Data Analyst: Data Quality Engineer -> EDA & Insights Analyst -> Data Contract Specialist."""
from pathlib import Path

from crewai import Crew

from golden_fighter import config
from golden_fighter.analysis import pipeline as steps
from golden_fighter.crews.runner import CrewRunResult, build_crew, run_crew
from golden_fighter.llm import build_llm
from golden_fighter.tools.analyst_tools import analyst_tools
from golden_fighter.tools.base import ToolRunLog

CONFIG_DIR = Path(__file__).parent / "config"
OUTPUT_FILES = (steps.CLEAN_FILE, steps.EDA_FILE, steps.INSIGHTS_FILE, steps.CONTRACT_FILE)


def build_analyst_crew(llm, run_log: ToolRunLog, raw_path: Path = config.RAW_DATA_PATH,
                       out_dir: Path = config.ANALYST_DIR, verbose: bool = False) -> Crew:
    return build_crew(CONFIG_DIR, llm, analyst_tools(run_log, raw_path, out_dir), verbose)


def run_analyst_crew(llm=None, raw_path: Path = config.RAW_DATA_PATH, out_dir: Path = config.ANALYST_DIR,
                     verbose: bool = False) -> CrewRunResult:
    """Run Crew 1 and verify its four artifacts. `llm=None` builds the real Anthropic LLM (costs money)."""
    llm = llm or build_llm()
    run_log = ToolRunLog()
    crew = build_analyst_crew(llm, run_log, raw_path, out_dir, verbose)
    return run_crew("analyst", crew, llm, run_log, [out_dir / name for name in OUTPUT_FILES],
                    optional_tools=("write_insights_summary",))
