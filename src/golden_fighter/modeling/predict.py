"""Score new members with the saved model. Inputs are checked against the contract first."""
from pathlib import Path

import joblib
import pandas as pd

from golden_fighter.errors import DataValidationError

RISK_BANDS = ((0.3, "low"), (0.6, "medium"), (1.01, "high"))


def load_model(path: Path) -> dict:
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise DataValidationError("load_model", [f"{path.name} is not a Golden Fighter model bundle"])
    return bundle


def check_inputs(bundle: dict, rows: pd.DataFrame) -> list[str]:
    issues = []
    for name, rule in bundle["input_contract"].items():
        if name not in rows.columns:
            issues.append(f"missing input column '{name}'")
            continue
        values = pd.to_numeric(rows[name], errors="coerce")
        if values.isna().any():
            issues.append(f"'{name}' must be a number for every member")
            continue
        if rule["allowed_values"] is not None and not values.isin(rule["allowed_values"]).all():
            issues.append(f"'{name}' must be one of {rule['allowed_values']}")
        if rule["min"] is not None and (values < rule["min"]).any():
            issues.append(f"'{name}' below the contract minimum {rule['min']:g} {rule['unit']}")
        if rule["max"] is not None and (values > rule["max"]).any():
            issues.append(f"'{name}' above the contract maximum {rule['max']:g} {rule['unit']}")
    if not issues and "month_to_end_contract" in rows and "contract_period" in rows:
        if (rows["month_to_end_contract"] > rows["contract_period"]).any():
            issues.append("'month_to_end_contract' cannot exceed 'contract_period'")
    return issues


def predict_churn(bundle: dict, rows: pd.DataFrame) -> pd.DataFrame:
    """Return churn probability, 0/1 prediction and a risk band per row, in the input order."""
    issues = check_inputs(bundle, rows)
    if issues:
        raise DataValidationError("predict", issues)
    X = rows[bundle["input_features"]].apply(pd.to_numeric)
    proba = bundle["pipeline"].predict_proba(X)[:, 1]
    out = pd.DataFrame({"churn_probability": proba.round(4)}, index=rows.index)
    out["churn_prediction"] = (proba >= bundle["threshold"]).astype(int)
    out["risk_band"] = [next(label for limit, label in RISK_BANDS if p < limit) for p in proba]
    return out
