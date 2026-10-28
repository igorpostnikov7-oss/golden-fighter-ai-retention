# Course requirements checklist

Source: the course brief "Industry-Simulated AI Product Workflow" (Harbor & Vale) and the submission note
(GitHub link + short presentation on Slack by 10.11). Checked on 2026-10-28 against the code, the committed
artifacts, the test suite (228 tests) and a fresh-clone install.

Status: ✅ done · ⏳ waiting for the project owner's approval (publishing) · ➖ optional, not done

## Crew 1: Data Analyst (at least 3 agents)

| Requirement | Status | Evidence |
|---|---|---|
| At least 3 agents | ✅ | Data Quality Engineer, EDA & Insights Analyst, Data Contract Specialist (`crews/analyst_crew/config/agents.yaml`) |
| Ingest and validate a chosen dataset | ✅ | Kaggle gym churn data; `analysis/cleaning.py load_raw` (raw schema sanity), Flow `preflight` |
| Clean, preprocess, document | ✅ | `clean_data` (rename, dedupe, domain rules, stop if > 5% invalid); `cleaning_report.json` |
| Descriptive analytics and EDA | ✅ | `eda_report.html` (descriptive statistics and 9 charts, including correlations and a leakage scan) |
| Visual insights and business summaries | ✅ | `eda_report.html`, `insights.md` (agent-written summary, numbers grounded in `insight_facts.json`) |
| Dataset contract for Crew 2 | ✅ | `dataset_contract.json` (pydantic schema `contract/schema.py`) |
| Outputs: clean_data.csv, eda_report.html, insights.md, dataset_contract.json | ✅ | All four produced by the verified live run and committed (`clean_data.csv` under CC BY-NC-SA 4.0, see `data/LICENSE-DATA.md`) |

## Crew 2: Data Scientist (at least 3 agents)

| Requirement | Status | Evidence |
|---|---|---|
| At least 3 agents | ✅ | Feature Engineer, Model Trainer & Evaluator, Model Card Specialist |
| Built on Crew 1's artifacts, never on raw data | ✅ | Tools fixed to `artifacts/analyst/`; tests `test_crew2_refuses_raw_data_directory`, scientist tools refuse `data/raw` (`test_crews.py`) |
| Read and validate clean data and contract | ✅ | Gate runs before features and again before training (`modeling/pipeline.py`) |
| Feature engineering | ✅ | `modeling/features.py`: 3 engineered features inside the model pipeline |
| Train at least one model | ✅ | `model.joblib` (logistic regression, selected) |
| Compare at least two variations | ✅ | Baseline vs logistic regression vs random forest, plus a leakage ablation (`evaluation_report.md`) |
| Model card: purpose, data, metrics, limitations, ethics | ✅ | `model_card.md` |
| Outputs: features.csv, model.joblib, evaluation_report.md, model_card.md | ✅ | All four produced by the verified live run and committed (`features.csv` under CC BY-NC-SA 4.0) |

## The Flow

| Requirement | Status | Evidence |
|---|---|---|
| Automates the handoff Crew 1 → Crew 2 | ✅ | `flow/main.py` CrewAI `Flow` (`@start`, `@listen`, `@router`) |
| Validation: contract matches the clean data | ✅ | `validate_dataset`: 88 checks on the real data |
| Validation: required features exist before modeling | ✅ | `validate_model_features` (missing, unknown, forbidden, leaky, target) |
| Deterministic steps | ✅ | `SEED = 42`; byte-identical reruns tested (`test_pipeline_is_deterministic`, `test_training_is_reproducible`) |
| Clear logging | ✅ | `logging` everywhere; `run_summary.json` per run; `logs/crew_runs.jsonl` for live crews |
| Artifacts saved inside the repo | ✅ | All 8 deliverables + `run_summary.json` of the verified live run are committed; the dataset too |
| Fails gracefully | ✅ | Gate failure → `failure_report.md`, exit code 1, Crew 2 never starts; API/setup errors → exit code 2 |
| Crew 2 never reaches past the contract | ✅ | Contract is the only shared interface; tools have fixed paths; tests above |

## Tech stack and good habits

| Requirement | Status | Evidence |
|---|---|---|
| CrewAI, Python | ✅ | crewai 1.15.32, Python 3.12 |
| Pandas, scikit-learn, Matplotlib / Seaborn | ✅ | Pinned in `requirements.txt`; charts in the EDA report use Matplotlib |
| Streamlit or Flask | ✅ | Streamlit app, 6 screens (`app/`) |
| Git + GitHub with Pull Requests | ✅ / ⏳ | 34 local commits on stacked milestone branches (`m1-data-layer` ... `m7-docs`). GitHub repo, push and one PR per milestone wait for approval (`docs/github_plan.md`) |
| The contract is a file, not a conversation | ✅ | Crew 2 knows only `dataset_contract.json` |
| Break it on purpose | ✅ | 13 scenarios (`contract/breakers.py`), each tested; `--demo-break` and the app's demo screen |
| Secrets stay out of the repo | ✅ | `.env` gitignored, `.env.example` empty; full history scanned: no keys, no `.env`, no data dumps |
| Deployment (Streamlit Cloud, Railway, Supabase) | ➖ | Optional; not done |

## "You'll know it's done when..."

| Check | Status | Evidence |
|---|---|---|
| Clone, run the Flow once, see Crew 1's 4 artifacts, the gate pass, Crew 2's 4 artifacts | ✅ | Fresh-clone install test (PROJECT_PROGRESS.md). The live run with the agents needs an API key; without one, `--no-llm` runs the same Flow free. The verified live run `20261028T162349Z-07b240` is committed |
| Results visible in the app | ✅ | Streamlit app reads the committed artifacts |
| Change one unit in the clean data, run again, the Flow refuses to train | ✅ | `--demo-break unit_x100`: exit 1, gate names the column and the ×100 unit change, no model trained |

## Submission

| Item | Status |
|---|---|
| GitHub link | ⏳ needs approval to create the public repo and push |
| Short presentation | ✅ prepared in M8 (`docs/presentation/`) |
