# Golden Fighter AI — Retention Intelligence

A CrewAI Flow that predicts gym-membership churn, built from **two crews with a machine-enforced contract between them**.

- **Crew 1 — Data Analyst** cleans and explores the data, then publishes `dataset_contract.json`.
- **Validation gate** (pure Python) checks the clean data against the contract. If anything is off, the Flow stops with a readable error.
- **Crew 2 — Data Scientist** engineers features, trains and compares models, and writes a model card.
- **Streamlit app** shows the results and a live "break the data" demo.

> Status: work in progress. M1 done: deterministic Data Analyst pipeline + dataset contract.
> M2 done: validation gate + 13 "break it on purpose" scenarios.
> M3 done: baseline vs Logistic Regression vs Random Forest, evaluation report and model card.
> M4 done: two CrewAI crews (3 agents each) that run the tested pipeline through tools; tested with a mock LLM.
> M5 done: the CrewAI Flow connecting Crew 1 -> validation gate -> Crew 2, with failure report and run summary.
> M6 done: Streamlit dashboard (offline; never calls the Claude API).

## Open the dashboard
```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py     # then open http://localhost:8501
```
Six screens: Overview, Churn insights, Model performance, Predict churn, Flow runs & costs, Break-the-data demo.
The app reads the committed results and never calls the Claude API. Its two buttons are free and safe: a Flow
run without AI (writes to `artifacts/_no_llm/`) and the break demo (works on a copy in `artifacts/_broken/`).

## Run the Flow
```powershell
# Free, no API key: the same Flow with the deterministic pipelines doing the crews' work.
# Writes to artifacts/_no_llm/ so the validated, agent-written artifacts are never replaced.
.\.venv\Scripts\python.exe -m golden_fighter.flow.main --no-llm

# "Break it on purpose": dollars -> cents in a COPY of the clean data. The gate refuses, Crew 2 never starts,
# exit code 1, and failure_report.md explains every violation (artifacts/_broken/runs/<run_id>/).
.\.venv\Scripts\python.exe -m golden_fighter.flow.main --demo-break unit_x100

# Full run with the agents (costs money, Claude via ANTHROPIC_API_KEY; model from GF_LLM_MODEL):
.\.venv\Scripts\python.exe -m golden_fighter.flow.main --no-retries
```
Every run writes `run_summary.json` (status, stages, gate result, which files THIS run produced, model calls,
tokens, estimated cost, headline metrics) to `artifacts/runs/<run_id>/`. Exit code: 0 succeeded,
1 validation failed (no model trained), 2 crew/API/setup error.

## Dataset
Public Kaggle dataset "Gym Customers Features and Churn". See [`data/raw/DATA_SOURCE.md`](data/raw/DATA_SOURCE.md).
No private or CRM data is used.

## Quick start (Windows / PowerShell)
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
copy .env.example .env   # then add your ANTHROPIC_API_KEY
.\.venv\Scripts\python.exe -m pytest -q
```

Run the Data Analyst pipeline (plain Python, no API calls). It reads `data/raw/gym_churn.csv` and writes
`clean_data.csv`, `eda_report.html`, `insights.md` and `dataset_contract.json` to `artifacts/analyst/`:
```powershell
.\.venv\Scripts\python.exe -m golden_fighter.analysis.pipeline
```

Check the artifacts against the contract (exit code 0 = passed, 1 = failed), then break a copy on purpose:
```powershell
.\.venv\Scripts\python.exe -m golden_fighter.contract.gate
.\.venv\Scripts\python.exe scripts\break_data.py unit_x100     # dollars -> cents: the gate refuses
.\.venv\Scripts\python.exe scripts\break_data.py --list        # all 13 scenarios
```

Run the Data Scientist pipeline (plain Python, no API calls). It runs the gate first, then writes
`features.csv`, `model.joblib`, `evaluation_report.md` and `model_card.md` to `artifacts/scientist/`:
```powershell
.\.venv\Scripts\python.exe -m golden_fighter.modeling.pipeline
```

Run one crew with real Claude agents (uses the Anthropic API and costs money; the model comes from
`GF_LLM_MODEL`, default Claude Haiku 5.5). Token usage and the estimated cost are appended to `logs/crew_runs.jsonl`:
```powershell
.\.venv\Scripts\python.exe -m golden_fighter.crews.run analyst
.\.venv\Scripts\python.exe -m golden_fighter.crews.run scientist
```

Run instructions for the Flow and the app will be added as those milestones land.
