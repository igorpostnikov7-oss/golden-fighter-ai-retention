# Golden Fighter AI — Retention Intelligence

A CrewAI Flow that predicts gym-membership churn, built from **two crews with a machine-enforced contract between them**.

- **Crew 1 — Data Analyst** cleans and explores the data, then publishes `dataset_contract.json`.
- **Validation gate** (pure Python) checks the clean data against the contract. If anything is off, the Flow stops with a readable error.
- **Crew 2 — Data Scientist** engineers features, trains and compares models, and writes a model card.
- **Streamlit app** shows the results and a live "break the data" demo.

> Status: work in progress. M1 done: deterministic Data Analyst pipeline + dataset contract.
> M2 done: validation gate + 13 "break it on purpose" scenarios.
> M3 done: baseline vs Logistic Regression vs Random Forest, evaluation report and model card.

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

Run instructions for the Flow and the app will be added as those milestones land.
