"""Inputs and feature engineering for the Data Scientist crew.

Crew 2 only ever reads Crew 1's artifacts, and only after the validation gate passes. Feature
engineering is row-wise (no statistics learned from the data), so it cannot leak test information
and is safe to run inside the saved model pipeline at prediction time.
"""
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from golden_fighter import config
from golden_fighter.contract.schema import DatasetContract
from golden_fighter.contract.validator import run_gate, validate_model_features
from golden_fighter.errors import DataValidationError

log = logging.getLogger(__name__)

TEST_SIZE = 0.2
SPLIT_COLUMN = "split"

ENGINEERED_FEATURES = {
    "contract_elapsed_share": "Share of the current contract already used: (contract_period - months left) / "
                              "contract_period. 0 = just signed, close to 1 = about to expire.",
    "is_new_member": "1 if the member joined 0-1 months ago (the highest-churn period in the EDA).",
    "social_ties": "Number of social links to the club: partner company + friend referral + group classes (0-3).",
}


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise derived features. Used both when building features.csv and inside the saved model."""
    out = df.copy()
    out["contract_elapsed_share"] = (out["contract_period"] - out["month_to_end_contract"]) / out["contract_period"]
    out["is_new_member"] = (out["lifetime"] <= 1).astype("int64")
    out["social_ties"] = out["partner"] + out["promo_friends"] + out["group_visits"]
    return out


def ensure_crew1_artifact_dir(analyst_dir: Path) -> None:
    """The boundary from the brief: Crew 2 must never reach into the raw data."""
    raw_dir = config.RAW_DATA_PATH.parent.resolve()
    target = analyst_dir.resolve()
    if target == raw_dir or raw_dir in target.parents:
        raise DataValidationError("crew2_inputs", [f"Crew 2 may not read raw data ({analyst_dir}); "
                                                   "use Crew 1's artifacts in artifacts/analyst"])


def load_validated_inputs(analyst_dir: Path) -> tuple[pd.DataFrame, DatasetContract]:
    """Run the validation gate, then load the clean data and the contract. Stops on any failure."""
    ensure_crew1_artifact_dir(analyst_dir)
    run_gate(analyst_dir).raise_if_failed()
    contract = DatasetContract.model_validate_json((analyst_dir / "dataset_contract.json").read_text("utf-8"))
    df = pd.read_csv(analyst_dir / contract.clean_data_file)
    log.info("Gate passed; loaded %d rows and contract v%s", len(df), contract.contract_version)
    return df, contract


def select_features(contract: DatasetContract, df: pd.DataFrame, mode: str = "primary") -> list[str]:
    """Model inputs straight from the contract; checked by the gate's feature rules before use."""
    features = contract.primary_feature_columns
    if mode == "ablation":
        features = features + contract.excluded_from_primary_model
    check = validate_model_features(features, contract, df.columns, mode=mode)
    check.raise_if_failed()
    for warning in check.warnings:
        log.warning("Feature check: %s", warning)
    return features


def split_train_test(df: pd.DataFrame, target: str, seed: int = config.SEED) -> pd.Series:
    """Stratified 80/20 split, returned as a 'train'/'test' label per row (keeps churn ratio equal)."""
    train_idx, _ = train_test_split(df.index, test_size=TEST_SIZE, stratify=df[target], random_state=seed)
    return pd.Series("test", index=df.index).where(~df.index.isin(train_idx), "train")


def build_feature_table(df: pd.DataFrame, contract: DatasetContract, id_column: str = "member_id") -> pd.DataFrame:
    """features.csv: id, split, contract inputs, engineered features and the target (for training only)."""
    inputs = select_features(contract, df)
    target = contract.target.name
    table = add_engineered_features(df[inputs])
    table.insert(0, SPLIT_COLUMN, split_train_test(df, target))
    table.insert(0, id_column, df[id_column])
    table[target] = df[target]
    return table
