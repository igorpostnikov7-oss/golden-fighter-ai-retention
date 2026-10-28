# Golden Fighter AI — Retention Intelligence

A CrewAI Flow that predicts gym-membership churn, built from **two crews with a machine-enforced contract between them**.

- **Crew 1 — Data Analyst** cleans and explores the data, then publishes `dataset_contract.json`.
- **Validation gate** (pure Python) checks the clean data against the contract. If anything is off, the Flow stops with a readable error.
- **Crew 2 — Data Scientist** engineers features, trains and compares models, and writes a model card.
- **Streamlit app** shows the results and a live "break the data" demo.

> Status: work in progress (milestone M0: project setup).

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

Run instructions for the Flow and the app will be added as those milestones land.
