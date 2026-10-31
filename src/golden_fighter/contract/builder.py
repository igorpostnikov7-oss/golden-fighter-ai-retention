"""Build dataset_contract.json from the clean data plus the analysts' domain knowledge."""
import hashlib
import json
import logging
from pathlib import Path

import pandas as pd

from golden_fighter.analysis.schema_spec import COLUMN_SPECS, CROSS_COLUMN_RULES, ID_COLUMN, TARGET
from golden_fighter.contract.schema import (
    ColumnContract,
    CrossColumnRule,
    DatasetContract,
    ObservedStats,
    TargetContract,
)

log = logging.getLogger(__name__)

# Tolerances for the distribution checks. Wide enough for next month's data, tight enough
# that a unit change (cents vs dollars, x100) or a broken join is caught.
CONTINUOUS_MEAN_TOLERANCE = 0.25  # +/- 25% of the observed mean
BINARY_MEAN_TOLERANCE = 0.15  # +/- 15 percentage points
TARGET_RATE_TOLERANCE = 0.10
MIN_ROW_SHARE = 0.5  # a delivery with < 50% of today's rows is treated as incomplete

# Protected attribute: kept in the data for fairness evaluation only.
SENSITIVE_EXCLUDED = ["gender"]


def sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean_range(spec, mean: float) -> tuple[float, float] | None:
    if spec.role != "feature":
        return None
    if spec.allowed_values is not None and set(spec.allowed_values) == {0, 1}:
        return (round(max(0.0, mean - BINARY_MEAN_TOLERANCE), 4), round(min(1.0, mean + BINARY_MEAN_TOLERANCE), 4))
    return (round(mean * (1 - CONTINUOUS_MEAN_TOLERANCE), 4), round(mean * (1 + CONTINUOUS_MEAN_TOLERANCE), 4))


def build_contract(clean: pd.DataFrame, clean_path: Path, leakage_findings: list[dict]) -> DatasetContract:
    risk_by_column = {f["column"]: f for f in leakage_findings}
    columns = []
    for spec in COLUMN_SPECS:
        col = clean[spec.name]
        mean = float(col.mean())
        finding = risk_by_column.get(spec.name)
        columns.append(ColumnContract(
            name=spec.name,
            dtype=spec.dtype,
            role=spec.role,
            description=spec.description,
            unit=spec.unit,
            allowed_values=list(spec.allowed_values) if spec.allowed_values else None,
            min=spec.min,
            max=spec.max,
            expected_mean_range=_mean_range(spec, mean),
            sensitive=spec.sensitive,
            leakage_risk=finding["risk"] if finding else spec.leakage_risk,
            leakage_note=(finding["reason"] if finding and finding["risk"] != "none" else spec.leakage_note),
            observed=ObservedStats(min=round(float(col.min()), 4), max=round(float(col.max()), 4),
                                   mean=round(mean, 4)),
        ))

    rate = float(clean[TARGET].mean())
    high_risk = [c.name for c in columns if c.leakage_risk == "high"]
    contract = DatasetContract(
        dataset_name="Gym customers features and churn (Kaggle)",
        produced_by="Data Analyst Crew",
        clean_data_file=clean_path.name,
        clean_data_sha256=sha256_of_file(clean_path),
        row_count_min=int(len(clean) * MIN_ROW_SHARE),
        row_count_observed=len(clean),
        target=TargetContract(
            name=TARGET,
            expected_rate_range=(round(max(0.0, rate - TARGET_RATE_TOLERANCE), 4),
                                 round(min(1.0, rate + TARGET_RATE_TOLERANCE), 4)),
            observed_rate=round(rate, 4),
        ),
        columns=columns,
        cross_column_rules=[CrossColumnRule(left=l, operator=op, right=r, description=d)
                            for l, op, r, d in CROSS_COLUMN_RULES],
        forbidden_features=[TARGET, ID_COLUMN, *SENSITIVE_EXCLUDED],
        excluded_from_primary_model=high_risk,
        assumptions=[
            "One row per member; member_id is the row number in the raw file and carries no information.",
            "Snapshot data: features describe the member up to the churn month, churn is the label for that month.",
            "Monetary values are in the source's currency units and must not be rescaled without a new contract.",
            "Visit frequencies are average visits per week.",
            "contract_period only takes the values 1, 6 or 12 months.",
            "gender is a protected attribute: never a model input, used only to check fairness.",
            "Columns in excluded_from_primary_model may only be used in a clearly labelled leakage ablation.",
            "The Data Scientist crew reads only clean_data.csv and this contract, never the raw data.",
        ],
    )
    log.info("Built contract: %d columns, target '%s' (rate %.3f), forbidden %s, excluded %s",
             len(columns), TARGET, rate, contract.forbidden_features, high_risk)
    return contract


def write_contract(contract: DatasetContract, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(contract.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def load_contract(path: Path) -> DatasetContract:
    return DatasetContract.model_validate_json(path.read_text(encoding="utf-8"))
