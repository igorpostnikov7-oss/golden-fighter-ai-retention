"""Everything the dashboard shows, read from the artifacts the tested M1-M5 code wrote.

Nothing here computes a metric: numbers come from insight_facts.json, evaluation.json, run_summary.json and
the saved model. The app never calls the Claude API. Its only actions are free and work on copies:
the break demo (artifacts/_broken/) and a no-LLM Flow run (artifacts/_no_llm/).
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from golden_fighter import config
from golden_fighter.contract.breakers import SCENARIOS
from golden_fighter.contract.validator import ValidationResult, run_gate
from golden_fighter.errors import DataValidationError
from golden_fighter.modeling.predict import RISK_BANDS, load_model, predict_churn
from golden_fighter.tools.grounding import strip_source_tags

# Plain-English names for the model inputs (the contract holds the technical descriptions).
FEATURE_LABELS = {
    "near_location": "Lives or works near the club",
    "partner": "Works for a partner company",
    "promo_friends": "Joined through a friend's referral",
    "phone": "Phone number on file",
    "contract_period": "Contract length (months)",
    "group_visits": "Attends group classes",
    "age": "Age (years)",
    "avg_additional_charges_total": "Average extra spending (cafe, services)",
    "month_to_end_contract": "Months left on the contract",
    "lifetime": "Months since joining",
    "avg_class_frequency_total": "Average visits per week since joining",
}
MODEL_LABELS = {"baseline_majority": "Baseline (always guesses)", "logistic_regression": "Logistic regression",
                "random_forest": "Random forest"}
RUN_KINDS = {"Full runs (with AI agents)": "runs", "Free runs (no AI)": "_no_llm/runs", "Break demos": "_broken/runs"}

# Two example members for the prediction screen (values inside the contract's ranges).
EXAMPLE_MEMBERS = {
    "New member on a 1-month contract": {
        "near_location": 1, "partner": 0, "promo_friends": 0, "phone": 1, "contract_period": 1, "group_visits": 0,
        "age": 24, "avg_additional_charges_total": 60.0, "month_to_end_contract": 1, "lifetime": 0,
        "avg_class_frequency_total": 1.0},
    "Loyal member on a 12-month contract": {
        "near_location": 1, "partner": 1, "promo_friends": 1, "phone": 1, "contract_period": 12, "group_visits": 1,
        "age": 33, "avg_additional_charges_total": 220.0, "month_to_end_contract": 9, "lifetime": 14,
        "avg_class_frequency_total": 3.0},
}


@dataclass
class Paths:
    """Resolved at call time from config, so tests can point the app at a throwaway project."""
    analyst_dir: Path
    scientist_dir: Path
    artifacts_dir: Path

    @classmethod
    def current(cls) -> "Paths":
        return cls(config.ANALYST_DIR, config.SCIENTIST_DIR, config.ARTIFACTS_DIR)


class MissingArtifact(FileNotFoundError):
    """An artifact the screen needs does not exist yet; the message says how to create it."""


def _read_json(path: Path, how_to_create: str) -> dict:
    if not path.exists():
        raise MissingArtifact(f"{path.name} was not found in {path.parent.name}/. {how_to_create}")
    return json.loads(path.read_text(encoding="utf-8"))


def facts(paths: Paths) -> dict:
    return _read_json(paths.analyst_dir / "insight_facts.json",
                      "Run the free Flow (screen 'Flow runs & costs') to create it.")


def evaluation(paths: Paths) -> dict:
    return _read_json(paths.scientist_dir / "evaluation.json",
                      "Run the free Flow (screen 'Flow runs & costs') to train the model.")


def contract(paths: Paths) -> dict:
    return _read_json(paths.analyst_dir / "dataset_contract.json",
                      "Run the free Flow (screen 'Flow runs & costs') to create it.")


def document(path: Path) -> str:
    """A markdown artifact for display: agent-section markers and any source tags removed."""
    if not path.exists():
        raise MissingArtifact(f"{path.name} was not found in {path.parent.name}/.")
    text = re.sub(r"<!-- agent-summary:(start|end) -->\n?", "", path.read_text(encoding="utf-8"))
    return strip_source_tags(text)


def agent_section(path: Path) -> str | None:
    """Only the part an AI agent wrote (between the agent-summary markers), or None."""
    if not path.exists():
        return None
    match = re.search(r"<!-- agent-summary:start -->(.*?)<!-- agent-summary:end -->", path.read_text(encoding="utf-8"),
                      re.S)
    if not match:
        return None
    body = re.sub(r"^\s*## .*\n", "", match.group(1).strip() + "\n", count=1)
    return strip_source_tags(body)


# --- insights -------------------------------------------------------------------------------------------------

def churn_by(facts_: dict, key: str) -> pd.DataFrame:
    """A churn-by-group table (group, members, churn rate in %) from insight_facts.json."""
    df = pd.DataFrame(facts_[key])
    df["churn_rate_pct"] = (df["churn_rate"] * 100).round(1)
    return df


def churn_by_flags(facts_: dict) -> pd.DataFrame:
    rows = []
    for item in facts_["churn_by_flag"].values():
        rows += [{"factor": item["label"], "answer": "Yes", "churn_rate_pct": round(item["rate_if_yes"] * 100, 1)},
                 {"factor": item["label"], "answer": "No", "churn_rate_pct": round(item["rate_if_no"] * 100, 1)}]
    return pd.DataFrame(rows)


# --- model ----------------------------------------------------------------------------------------------------

def model_table(eval_: dict) -> pd.DataFrame:
    """One row per model plus the leakage experiment, with its role spelled out."""
    rows = []
    for name, result in eval_["results"].items():
        role = ("Selected" if name == eval_["selected_model"]
                else "Reference only" if name == "baseline_majority" else "Not selected")
        rows.append(_model_row(MODEL_LABELS.get(name, name), role, result))
    rows.append(_model_row(f"{MODEL_LABELS.get(eval_['selected_model'])} + current-month visits",
                           "Leakage experiment - never used", eval_["ablation"]))
    return pd.DataFrame(rows)


def _model_row(label: str, role: str, result: dict) -> dict:
    test = result["test"]
    return {"Model": label, "Role": role, "CV ROC-AUC (train)": result["cv"]["roc_auc"]["mean"],
            "Test ROC-AUC": test["roc_auc"], "Test recall": test["recall"], "Test precision": test["precision"],
            "Test F1": test["f1"]}


def confusion_matrix(eval_: dict) -> pd.DataFrame:
    cm = eval_["results"][eval_["selected_model"]]["test"]["confusion_matrix"]
    return pd.DataFrame([
        {"actual": "Stayed", "predicted": "Predicted to stay", "members": cm["tn"], "meaning": "Correctly left alone"},
        {"actual": "Stayed", "predicted": "Predicted to leave", "members": cm["fp"],
         "meaning": "False alarm: an offer that was not needed"},
        {"actual": "Left", "predicted": "Predicted to stay", "members": cm["fn"], "meaning": "Missed churner"},
        {"actual": "Left", "predicted": "Predicted to leave", "members": cm["tp"], "meaning": "Churner caught in time"},
    ])


def feature_importance(eval_: dict) -> pd.DataFrame:
    df = pd.DataFrame(eval_["feature_importance"])
    df["label"] = df["feature"].map(FEATURE_LABELS).fillna(df["feature"])
    return df


def fairness(eval_: dict, key: str) -> pd.DataFrame:
    df = pd.DataFrame(eval_["fairness"][key])
    for col in ("actual_churn_rate", "flagged_rate", "recall", "precision"):
        df[col] = [f"{v:.1%}" if pd.notna(v) else "- (no leavers)" for v in df[col]]
    return df.rename(columns={"group": "Group", "members": "Test members", "actual_churn_rate": "Actual churn rate",
                              "flagged_rate": "Flagged rate", "recall": "Recall", "precision": "Precision"})


def roc_points(paths: Paths) -> pd.DataFrame | None:
    """ROC curve of the saved model on the saved test split, or None when features.csv is not available
    (it holds member-level rows, so it is not committed)."""
    features_path, model_path = paths.scientist_dir / "features.csv", paths.scientist_dir / "model.joblib"
    if not (features_path.exists() and model_path.exists()):
        return None
    from sklearn.metrics import roc_curve

    bundle = load_model(model_path)
    test = pd.read_csv(features_path).query("split == 'test'")
    proba = bundle["pipeline"].predict_proba(test[bundle["input_features"]])[:, 1]
    fpr, tpr, _ = roc_curve(test[bundle["target"]], proba)
    return pd.DataFrame({"false_alarm_rate": fpr, "churners_caught": tpr})


# --- prediction -----------------------------------------------------------------------------------------------

def model_bundle(paths: Paths) -> dict:
    path = paths.scientist_dir / "model.joblib"
    if not path.exists():
        raise MissingArtifact("model.joblib was not found. Run the free Flow (screen 'Flow runs & costs').")
    return load_model(path)


def input_spec(bundle: dict) -> list[dict]:
    """Form fields for the prediction screen, straight from the contract ranges stored in the model."""
    spec = []
    for name in bundle["input_features"]:
        rule = bundle["input_contract"][name]
        spec.append({"name": name, "label": FEATURE_LABELS.get(name, name), "unit": rule["unit"],
                     "allowed_values": rule["allowed_values"], "min": rule["min"], "max": rule["max"]})
    return spec


def predict_member(bundle: dict, member: dict) -> dict:
    """Score one member with the saved model. Raises DataValidationError with readable issues."""
    row = predict_churn(bundle, pd.DataFrame([member])).iloc[0]
    return {"probability": float(row["churn_probability"]), "flagged": bool(row["churn_prediction"]),
            "risk_band": row["risk_band"], "threshold": bundle["threshold"], "model": bundle["model_name"]}


# --- gate, demo and runs --------------------------------------------------------------------------------------

def gate_result(paths: Paths) -> ValidationResult:
    """The validation gate on the current analyst artifacts (read-only)."""
    return run_gate(paths.analyst_dir)


def break_scenarios() -> dict[str, str]:
    return {name: description for name, (description, _) in SCENARIOS.items()}


def run_break_demo(scenario: str):
    """Break a COPY of Crew 1's artifacts and run the Flow on it (no AI, free). Returns the FlowState."""
    from golden_fighter.flow.main import demo_settings, run_flow
    if not (config.ANALYST_DIR / "clean_data.csv").exists():
        raise MissingArtifact("clean_data.csv is needed to make a broken copy. Run the free Flow first.")
    return run_flow(demo_settings(scenario))


def run_free_flow():
    """The Flow with the deterministic pipelines instead of agents (free). Writes to artifacts/_no_llm/."""
    from golden_fighter.flow.main import no_llm_settings, run_flow
    if not config.RAW_DATA_PATH.exists():
        raise MissingArtifact("The raw dataset data/raw/gym_churn.csv is missing. See data/raw/DATA_SOURCE.md "
                              "for the download link.")
    return run_flow(no_llm_settings())


def normalize_summary(raw: dict) -> dict:
    """Map older run summaries (llm_calls / tool_calls) to the current names (model_api_calls / tool_log)."""
    summary = json.loads(json.dumps(raw))
    for entry in list(summary.get("crews", {}).values()) + [summary.setdefault("totals", {})]:
        if "model_api_calls" not in entry and "llm_calls" in entry:
            entry["model_api_calls"] = entry.pop("llm_calls")
        if "tool_log" not in entry and "tool_calls" in entry:
            entry["tool_log"] = entry.pop("tool_calls")
        if "agent_tool_executions" not in entry and "tool_log" in entry:
            runs = [c for c in entry["tool_log"] if c["tool"] != "source_check"]
            entry["agent_tool_executions"] = len(runs)
            entry["source_check_rejections"] = len(entry["tool_log"]) - len(runs)
    totals, crews = summary["totals"], summary.get("crews", {}).values()
    for key in ("model_api_calls", "agent_tool_executions"):
        totals.setdefault(key, sum(c.get(key, 0) for c in crews))
    return summary


def run_summaries(paths: Paths) -> list[tuple[str, dict]]:
    """(kind, normalized summary) for every run on disk, newest first."""
    found = []
    for kind, sub in RUN_KINDS.items():
        for path in (paths.artifacts_dir / sub).glob("*/run_summary.json"):
            try:
                found.append((kind, normalize_summary(json.loads(path.read_text(encoding="utf-8")))))
            except (json.JSONDecodeError, OSError, KeyError, TypeError):
                continue  # a damaged summary is skipped, never shown as a run
    return sorted(found, key=lambda item: item[1].get("started_at", ""), reverse=True)


def run_history(paths: Paths) -> pd.DataFrame:
    rows = []
    for kind, s in run_summaries(paths):
        totals = s.get("totals", {})
        rows.append({"Run": s["run_id"], "Kind": kind, "Started (UTC)": s.get("started_at", "")[:19].replace("T", " "),
                     "Result": _status_label(s.get("status")), "Stopped at": s.get("failed_stage") or "-",
                     "Model": s.get("model") or "none (free)",
                     "Model API calls": totals.get("model_api_calls", 0),
                     "Agent tool executions": totals.get("agent_tool_executions", 0),
                     "Tokens": totals.get("total_tokens", 0), "Est. cost (USD)": totals.get("estimated_cost_usd") or 0.0,
                     "Seconds": s.get("duration_seconds")})
    return pd.DataFrame(rows)


def _status_label(status: str | None) -> str:
    return {"succeeded": "Succeeded", "validation_failed": "Stopped by the gate",
            "error": "Stopped by an error"}.get(status or "", status or "unknown")


def agents_overview() -> pd.DataFrame:
    """The six agents, their goals and the tools they may use, from the crews' YAML."""
    rows = []
    base = Path(__file__).resolve().parents[1] / "crews"
    for crew, folder in (("Crew 1 - Data Analyst", "analyst_crew"), ("Crew 2 - Data Scientist", "scientist_crew")):
        agents = yaml.safe_load((base / folder / "config" / "agents.yaml").read_text(encoding="utf-8"))
        for cfg in agents.values():
            rows.append({"Crew": crew, "Agent": cfg["role"], "Goal": " ".join(cfg["goal"].split()),
                         "Tools": ", ".join(cfg["tools"])})
    return pd.DataFrame(rows)


def risk_band_limits() -> list[tuple[float, str]]:
    return list(RISK_BANDS)


__all__ = ["DataValidationError", "MissingArtifact", "Paths"]
