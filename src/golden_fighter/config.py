"""Single source of truth for paths, seed and runtime settings."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

SEED = 42

RAW_DATA_PATH = ROOT / "data" / "raw" / "gym_churn.csv"

ARTIFACTS_DIR = ROOT / "artifacts"
ANALYST_DIR = ARTIFACTS_DIR / "analyst"
SCIENTIST_DIR = ARTIFACTS_DIR / "scientist"
RUNS_DIR = ARTIFACTS_DIR / "runs"
LOGS_DIR = ROOT / "logs"

CLEAN_DATA_PATH = ANALYST_DIR / "clean_data.csv"
EDA_REPORT_PATH = ANALYST_DIR / "eda_report.html"
INSIGHTS_PATH = ANALYST_DIR / "insights.md"
CONTRACT_PATH = ANALYST_DIR / "dataset_contract.json"

FEATURES_PATH = SCIENTIST_DIR / "features.csv"
MODEL_PATH = SCIENTIST_DIR / "model.joblib"
EVALUATION_REPORT_PATH = SCIENTIST_DIR / "evaluation_report.md"
MODEL_CARD_PATH = SCIENTIST_DIR / "model_card.md"

# Haiku for development (cheap); set GF_LLM_MODEL=anthropic/claude-sonnet-5-5 for final runs.
LLM_MODEL = os.getenv("GF_LLM_MODEL", "anthropic/claude-haiku-5-5")
