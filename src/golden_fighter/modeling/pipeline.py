"""Crew 2 pipeline in three steps, one per Data Scientist agent.

Every step reads Crew 1's artifacts only through the validation gate, and hands work to the next
step through files in the output directory (each step becomes a CrewAI tool in M4):

  1. Feature Engineer          -> features_step()  -> features.csv
  2. Model Trainer & Evaluator -> training_step()  -> model.joblib, evaluation.json
  3. Model Card Author         -> reports_step()   -> evaluation_report.md, model_card.md

Run all three: python -m golden_fighter.modeling.pipeline
"""
import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from golden_fighter import config
from golden_fighter.logging_setup import configure_logging
from golden_fighter.modeling.features import build_feature_table, load_validated_inputs
from golden_fighter.modeling.reports import render_evaluation_report, render_model_card
from golden_fighter.modeling.train import train_and_evaluate

log = logging.getLogger(__name__)

FEATURES_FILE = config.FEATURES_PATH.name
MODEL_FILE = config.MODEL_PATH.name
EVALUATION_JSON = "evaluation.json"
EVALUATION_FILE = config.EVALUATION_REPORT_PATH.name
MODEL_CARD_FILE = config.MODEL_CARD_PATH.name


def features_step(analyst_dir: Path = config.ANALYST_DIR, out_dir: Path = config.SCIENTIST_DIR) -> dict:
    clean, contract = load_validated_inputs(analyst_dir)
    table = build_feature_table(clean, contract)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / FEATURES_FILE, index=False, lineterminator="\n")
    log.info("Step 1/3 done: wrote %s (%d rows, %d columns)", FEATURES_FILE, *table.shape)
    return {"rows": len(table), "columns": list(table.columns),
            "split_counts": table["split"].value_counts().to_dict()}


def training_step(analyst_dir: Path = config.ANALYST_DIR, out_dir: Path = config.SCIENTIST_DIR) -> dict:
    clean, contract = load_validated_inputs(analyst_dir)  # gate runs again before every training
    features = pd.read_csv(out_dir / FEATURES_FILE)
    if not features["member_id"].equals(clean["member_id"]):
        raise ValueError("features.csv does not match clean_data.csv row-for-row; re-run the features step")
    outcome = train_and_evaluate(features, clean, contract)
    joblib.dump(outcome["bundle"], out_dir / MODEL_FILE)
    (out_dir / EVALUATION_JSON).write_text(json.dumps(outcome["report"], indent=2) + "\n", encoding="utf-8",
                                           newline="\n")
    log.info("Step 2/3 done: wrote %s (%s) and %s", MODEL_FILE, outcome["report"]["selected_model"],
             EVALUATION_JSON)
    return outcome["report"]


def reports_step(out_dir: Path = config.SCIENTIST_DIR) -> dict:
    report = json.loads((out_dir / EVALUATION_JSON).read_text(encoding="utf-8"))
    (out_dir / EVALUATION_FILE).write_text(render_evaluation_report(report), encoding="utf-8", newline="\n")
    (out_dir / MODEL_CARD_FILE).write_text(render_model_card(report), encoding="utf-8", newline="\n")
    log.info("Step 3/3 done: wrote %s and %s", EVALUATION_FILE, MODEL_CARD_FILE)
    return {"files": [EVALUATION_FILE, MODEL_CARD_FILE]}


def run_scientist_pipeline(analyst_dir: Path = config.ANALYST_DIR, out_dir: Path = config.SCIENTIST_DIR) -> dict:
    features_step(analyst_dir, out_dir)
    training_step(analyst_dir, out_dir)
    reports_step(out_dir)
    return {name: out_dir / name for name in (FEATURES_FILE, MODEL_FILE, EVALUATION_FILE, MODEL_CARD_FILE)}


if __name__ == "__main__":
    configure_logging(config.LOGS_DIR / "scientist_pipeline.log")
    for path in run_scientist_pipeline().values():
        log.info("Artifact ready: %s", path)
