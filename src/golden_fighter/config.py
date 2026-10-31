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

# Hard limits that keep an agent run cheap and bounded (see golden_fighter/llm.py).
LLM_MAX_TOKENS = 2048          # per model response
LLM_TIMEOUT_SECONDS = 90       # per API request
LLM_MAX_RETRIES = 2            # SDK retries for 429/5xx/connection errors, with backoff
AGENT_MAX_ITER = 5             # tool-use rounds per task before the agent must answer
AGENT_MAX_RETRY_LIMIT = 0      # CrewAI re-running a whole failed task would double the cost
CREW_MAX_RPM = 20              # requests per minute across a crew
MAX_LLM_CALLS_PER_CREW = 20    # budget: the 21st model call in one crew run is blocked
SOURCE_CHECK_MAX_RETRIES = 1   # an answer failing the source check gets one rewrite, then the crew stops

# USD per 1M tokens (input, output), prompts <= 100K tokens. Used for the cost estimate only.
LLM_PRICES_PER_MTOK = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
}

# No CrewAI telemetry or trace upload: this project only talks to the Anthropic API.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
