"""Unit tests for each validation rule, on a contract built from the synthetic dataset."""
import json

import pandas as pd
import pytest
from pydantic import ValidationError

from golden_fighter.contract.builder import load_contract
from golden_fighter.contract.schema import DatasetContract
from golden_fighter.contract.validator import validate_dataset, validate_model_features
from golden_fighter.errors import DataValidationError

MONEY = "avg_additional_charges_total"
LEAKY = "avg_class_frequency_current_month"


@pytest.fixture
def contract(analyst_out):
    return load_contract(analyst_out / "dataset_contract.json")


@pytest.fixture
def clean(analyst_out):
    return pd.read_csv(analyst_out / "clean_data.csv")


def _checks(result):
    return {(issue.check, issue.column) for issue in result.errors}


def test_clean_data_passes(clean, contract, analyst_out):
    result = validate_dataset(clean, contract, analyst_out / "clean_data.csv")
    assert result.passed, result.render()
    assert not result.warnings
    assert "PASSED" in result.render()


def test_missing_column(clean, contract):
    result = validate_dataset(clean.drop(columns=["age"]), contract)
    assert ("schema", "age") in _checks(result)


def test_renamed_column_reports_both_sides(clean, contract):
    result = validate_dataset(clean.rename(columns={MONEY: "spend"}), contract)
    assert ("schema", MONEY) in _checks(result)
    assert ("schema", "spend") in _checks(result)
    assert "renamed column?" in result.render()


def test_decimals_in_integer_column(clean, contract):
    broken = clean.assign(age=clean["age"] + 0.5)
    assert ("dtype", "age") in _checks(validate_dataset(broken, contract))


def test_text_in_numeric_column(clean, contract):
    broken = clean.astype({"lifetime": object})
    broken.loc[0, "lifetime"] = "n/a"
    result = validate_dataset(broken, contract)
    assert ("dtype", "lifetime") in _checks(result)
    assert "'n/a'" in result.render()


def test_nulls(clean, contract):
    broken = clean.astype({"age": float})
    broken.loc[:4, "age"] = None
    assert ("nulls", "age") in _checks(validate_dataset(broken, contract))


def test_value_outside_allowed_set(clean, contract):
    broken = clean.copy()
    broken.loc[0, "group_visits"] = 7
    assert ("allowed_values", "group_visits") in _checks(validate_dataset(broken, contract))


def test_below_minimum_and_above_maximum(clean, contract):
    broken = clean.copy()
    broken.loc[0, "lifetime"] = -1
    broken.loc[1, "age"] = 120
    checks = _checks(validate_dataset(broken, contract))
    assert ("range", "lifetime") in checks
    assert ("range", "age") in checks


@pytest.mark.parametrize("factor", [100, 0.01])
def test_unit_change_is_named_as_such(clean, contract, factor):
    broken = clean.assign(**{MONEY: clean[MONEY] * factor})
    result = validate_dataset(broken, contract)
    assert ("unit_or_distribution", MONEY) in _checks(result)
    assert "UNIT CHANGE" in result.render()


def test_cross_column_rule(clean, contract):
    broken = clean.copy()
    one_month = broken.index[broken["contract_period"] == 1][0]
    broken.loc[one_month, "month_to_end_contract"] = 6
    assert ("cross_column", None) in _checks(validate_dataset(broken, contract))


def test_target_rate_outside_expected_range(clean, contract):
    result = validate_dataset(clean.assign(churn=1), contract)
    assert ("target_rate", "churn") in _checks(result)


def test_duplicate_member_ids(clean, contract):
    broken = clean.copy()
    broken.loc[1, "member_id"] = broken.loc[0, "member_id"]
    assert ("identifier", "member_id") in _checks(validate_dataset(broken, contract))


def test_too_few_rows(clean, contract):
    assert ("row_count", None) in _checks(validate_dataset(clean.head(10), contract))


def test_small_legitimate_edit_only_warns(clean, contract, tmp_path):
    edited = clean.copy()
    edited.loc[0, "age"] = 30
    path = tmp_path / "clean_data.csv"
    edited.to_csv(path, index=False, lineterminator="\n")
    result = validate_dataset(edited, contract, path)
    assert result.passed
    assert [w.check for w in result.warnings] == ["fingerprint"]


def test_raise_if_failed_gives_readable_error(clean, contract):
    result = validate_dataset(clean.drop(columns=["lifetime"]), contract)
    with pytest.raises(DataValidationError) as err:
        result.raise_if_failed()
    assert "lifetime" in str(err.value) and "missing" in str(err.value)


# ---- feature checks: target, forbidden, leakage and prediction-time availability ----

def test_primary_features_exclude_target_id_gender_and_leaky(contract):
    features = contract.primary_feature_columns
    assert not {"churn", "member_id", "gender", LEAKY} & set(features)
    assert "lifetime" in features and MONEY in features


def test_contract_primary_features_pass(contract, clean):
    assert validate_model_features(contract.primary_feature_columns, contract, clean.columns).passed


@pytest.mark.parametrize("bad, check", [("churn", "leakage"), ("member_id", "forbidden"), ("gender", "forbidden"),
                                        ("loyalty_score", "unknown_feature")])
def test_forbidden_or_unknown_feature_fails(contract, clean, bad, check):
    result = validate_model_features(contract.primary_feature_columns + [bad], contract, clean.columns)
    assert (check, bad) in _checks(result)


def test_feature_not_available_before_churn_fails_primary(contract, clean):
    result = validate_model_features(contract.primary_feature_columns + [LEAKY], contract, clean.columns)
    checks = _checks(result)
    assert ("temporal", LEAKY) in checks
    assert ("leakage", LEAKY) in checks
    assert "prediction time" in result.render()


def test_leaky_feature_allowed_only_in_ablation_with_warning(contract, clean):
    result = validate_model_features(contract.primary_feature_columns + [LEAKY], contract, clean.columns,
                                     mode="ablation")
    assert result.passed
    assert {w.check for w in result.warnings} == {"temporal", "leakage_ablation"}


def test_every_primary_feature_is_available_before_churn(contract):
    for name in contract.primary_feature_columns:
        assert contract.column(name).available_before_churn, name


def test_required_feature_missing_from_data(contract, clean):
    result = validate_model_features(contract.primary_feature_columns, contract, clean.drop(columns="age").columns)
    assert ("missing_feature", "age") in _checks(result)


def test_empty_feature_list(contract, clean):
    assert ("features", None) in _checks(validate_model_features([], contract, clean.columns))


def test_contract_schema_rejects_unblocked_late_feature(analyst_out):
    data = json.loads((analyst_out / "dataset_contract.json").read_text())
    data["excluded_from_primary_model"] = []
    with pytest.raises(ValidationError, match="not available before churn"):
        DatasetContract.model_validate(data)
