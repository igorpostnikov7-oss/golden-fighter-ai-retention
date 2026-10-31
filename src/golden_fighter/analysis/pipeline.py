"""Crew 1 pipeline in three steps, one per Data Analyst agent.

Steps hand work to each other only through files in the output directory, exactly as the
agents will in M4 (each step becomes a CrewAI tool):

  1. Data Quality Engineer -> clean_step()           -> clean_data.csv, cleaning_report.json
  2. EDA & Insights Analyst -> eda_and_insights_step() -> eda_report.html, insights.md, insight_facts.json
  3. Data Contract Steward  -> contract_step()        -> dataset_contract.json

Run all three: python -m golden_fighter.analysis.pipeline
"""
import json
import logging
from pathlib import Path

import pandas as pd

from golden_fighter import config
from golden_fighter.analysis.cleaning import clean_data, load_raw, write_clean_csv
from golden_fighter.analysis.eda import build_eda_report
from golden_fighter.analysis.insights import compute_insight_facts, render_insights_md
from golden_fighter.analysis.leakage import scan_leakage
from golden_fighter.contract.builder import build_contract, write_contract
from golden_fighter.logging_setup import configure_logging

log = logging.getLogger(__name__)

CLEAN_FILE = config.CLEAN_DATA_PATH.name
CLEANING_REPORT_FILE = "cleaning_report.json"
FACTS_FILE = "insight_facts.json"
EDA_FILE = config.EDA_REPORT_PATH.name
INSIGHTS_FILE = config.INSIGHTS_PATH.name
CONTRACT_FILE = config.CONTRACT_PATH.name


def _write_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def _read_clean(out_dir: Path) -> pd.DataFrame:
    return pd.read_csv(out_dir / CLEAN_FILE)


def clean_step(raw_path: Path = config.RAW_DATA_PATH, out_dir: Path = config.ANALYST_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    clean, report = clean_data(load_raw(raw_path))
    write_clean_csv(clean, out_dir / CLEAN_FILE)
    _write_json(report.to_dict(), out_dir / CLEANING_REPORT_FILE)
    log.info("Step 1/3 done: wrote %s and %s", CLEAN_FILE, CLEANING_REPORT_FILE)
    return report.to_dict()


def eda_and_insights_step(out_dir: Path = config.ANALYST_DIR) -> dict:
    clean = _read_clean(out_dir)
    cleaning_report = json.loads((out_dir / CLEANING_REPORT_FILE).read_text(encoding="utf-8"))
    facts = compute_insight_facts(clean)
    _write_json(facts, out_dir / FACTS_FILE)
    (out_dir / EDA_FILE).write_text(build_eda_report(clean, cleaning_report, facts), encoding="utf-8",
                                    newline="\n")
    (out_dir / INSIGHTS_FILE).write_text(render_insights_md(facts), encoding="utf-8", newline="\n")
    log.info("Step 2/3 done: wrote %s, %s and %s", EDA_FILE, INSIGHTS_FILE, FACTS_FILE)
    return facts


def contract_step(out_dir: Path = config.ANALYST_DIR) -> dict:
    clean = _read_clean(out_dir)
    contract = build_contract(clean, out_dir / CLEAN_FILE, scan_leakage(clean))
    write_contract(contract, out_dir / CONTRACT_FILE)
    log.info("Step 3/3 done: wrote %s", CONTRACT_FILE)
    return contract.model_dump(mode="json")


def run_analyst_pipeline(raw_path: Path = config.RAW_DATA_PATH, out_dir: Path = config.ANALYST_DIR) -> dict:
    clean_step(raw_path, out_dir)
    eda_and_insights_step(out_dir)
    contract_step(out_dir)
    return {name: out_dir / name for name in (CLEAN_FILE, EDA_FILE, INSIGHTS_FILE, CONTRACT_FILE)}


if __name__ == "__main__":
    configure_logging(config.LOGS_DIR / "analyst_pipeline.log")
    for name, path in run_analyst_pipeline().items():
        log.info("Artifact ready: %s", path)
