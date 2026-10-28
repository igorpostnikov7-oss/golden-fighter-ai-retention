import json

import pandas as pd
import pytest
from pydantic import ValidationError

from golden_fighter.analysis.schema_spec import ID_COLUMN, TARGET
from golden_fighter.contract.builder import load_contract, sha256_of_file
from golden_fighter.contract.schema import DatasetContract


@pytest.fixture
def contract(analyst_out):
    return load_contract(analyst_out / "dataset_contract.json")


@pytest.fixture
def clean(analyst_out):
    return pd.read_csv(analyst_out / "clean_data.csv")


def test_contract_matches_clean_columns_and_types(contract, clean):
    assert [c.name for c in contract.columns] == list(clean.columns)
    for col in contract.columns:
        assert str(clean[col.name].dtype) == ("int64" if col.dtype == "int" else "float64"), col.name


def test_clean_data_satisfies_its_own_contract(contract, clean):
    """Round trip: data the contract was built from must pass every rule in it."""
    assert contract.row_count_min <= len(clean)
    for col in contract.columns:
        values = clean[col.name]
        assert not values.isna().any()
        if col.allowed_values is not None:
            assert values.isin(col.allowed_values).all(), col.name
        if col.min is not None:
            assert values.min() >= col.min, col.name
        if col.max is not None:
            assert values.max() <= col.max, col.name
        if col.expected_mean_range is not None:
            low, high = col.expected_mean_range
            assert low <= values.mean() <= high, col.name
    for rule in contract.cross_column_rules:
        assert (clean[rule.left] <= clean[rule.right]).all()
    low, high = contract.target.expected_rate_range
    assert low <= clean[TARGET].mean() <= high


def test_target_is_never_a_feature(contract):
    assert contract.target.name == TARGET
    assert contract.column(TARGET).role == "target"
    assert TARGET in contract.forbidden_features
    assert TARGET not in contract.feature_columns


def test_identifier_and_protected_attribute_are_not_features(contract):
    assert {ID_COLUMN, "gender"} <= set(contract.forbidden_features)
    assert ID_COLUMN not in contract.feature_columns
    assert "gender" not in contract.feature_columns


def test_current_month_frequency_is_flagged_as_leakage(contract):
    col = contract.column("avg_class_frequency_current_month")
    assert col.leakage_risk == "high"
    assert "avg_class_frequency_current_month" in contract.excluded_from_primary_model


def test_every_column_documents_unit_and_description(contract):
    for col in contract.columns:
        assert col.unit and col.description, col.name


def test_hash_matches_clean_file(contract, analyst_out):
    assert contract.clean_data_sha256 == sha256_of_file(analyst_out / "clean_data.csv")


def test_unit_change_would_be_detectable(contract, clean):
    """Design check for the M2 gate: x100 on the money column leaves the expected mean range."""
    low, high = contract.column("avg_additional_charges_total").expected_mean_range
    assert not low <= (clean["avg_additional_charges_total"] * 100).mean() <= high


def test_schema_rejects_unknown_fields(analyst_out):
    data = json.loads((analyst_out / "dataset_contract.json").read_text())
    data["surprise"] = 1
    with pytest.raises(ValidationError):
        DatasetContract.model_validate(data)


def test_schema_rejects_target_allowed_as_feature(analyst_out):
    data = json.loads((analyst_out / "dataset_contract.json").read_text())
    data["forbidden_features"].remove(TARGET)
    with pytest.raises(ValidationError, match="forbidden_features"):
        DatasetContract.model_validate(data)


def test_schema_rejects_missing_section(analyst_out):
    data = json.loads((analyst_out / "dataset_contract.json").read_text())
    del data["columns"]
    with pytest.raises(ValidationError):
        DatasetContract.model_validate(data)
