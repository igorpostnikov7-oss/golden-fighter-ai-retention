import hashlib

import pandas as pd
import pytest

from golden_fighter.analysis.cleaning import clean_data
from golden_fighter.analysis.pipeline import run_analyst_pipeline
from golden_fighter.analysis.schema_spec import CLEAN_COLUMN_ORDER, ID_COLUMN, SPEC_BY_NAME
from golden_fighter.errors import DataValidationError


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_raw_file_is_never_modified(tmp_path, raw_csv):
    before = _sha(raw_csv)
    run_analyst_pipeline(raw_csv, tmp_path / "out")
    assert _sha(raw_csv) == before


def test_clean_schema_and_types(raw_df):
    clean, report = clean_data(raw_df)
    assert list(clean.columns) == CLEAN_COLUMN_ORDER
    assert not clean.isna().any().any()
    for col in clean.columns:
        expected = "int64" if SPEC_BY_NAME[col].dtype == "int" else "float64"
        assert str(clean[col].dtype) == expected, col
    assert report.rows_in == report.rows_out == len(raw_df)


def test_input_dataframe_not_mutated(raw_df):
    copy = raw_df.copy()
    clean_data(raw_df)
    pd.testing.assert_frame_equal(raw_df, copy)


def test_duplicates_removed(raw_df):
    dirty = pd.concat([raw_df, raw_df.head(5)], ignore_index=True)
    clean, report = clean_data(dirty)
    assert report.dropped_duplicates == 5
    assert len(clean) == len(raw_df)


def test_few_invalid_rows_are_dropped_and_reported(raw_df):
    dirty = raw_df.copy()
    dirty.loc[0, "Contract_period"] = 3
    dirty.loc[0, "Month_to_end_contract"] = 1  # keep row 0 breaking only the allowed-values rule
    dirty.loc[1, "Age"] = -5
    dirty.loc[2, "Month_to_end_contract"] = 12
    dirty.loc[2, "Contract_period"] = 1
    dirty.loc[3, "Churn"] = None
    clean, report = clean_data(dirty)
    assert len(clean) == len(raw_df) - 4
    assert report.dropped_by_rule["contract_period not in [1, 6, 12]"] == 1
    assert report.dropped_by_rule["age < 14"] == 1
    assert report.dropped_by_rule["not (month_to_end_contract <= contract_period)"] == 1
    assert report.dropped_non_numeric_or_null == 1


def test_systematic_corruption_stops_the_pipeline(raw_df):
    dirty = raw_df.copy()
    dirty["Avg_additional_charges_total"] *= 100  # cents vs dollars: most rows exceed the domain max
    with pytest.raises(DataValidationError) as err:
        clean_data(dirty)
    assert "avg_additional_charges_total > 1000" in str(err.value)


def test_non_numeric_values_are_reported(raw_df):
    dirty = raw_df.astype({"Age": "object"})
    dirty.loc[:100, "Age"] = "unknown"
    with pytest.raises(DataValidationError, match="non-numeric"):
        clean_data(dirty)


def test_member_id_traces_back_to_raw_row(raw_df):
    dirty = raw_df.copy()
    dirty.loc[0, "Age"] = -1
    clean, _ = clean_data(dirty)
    assert clean[ID_COLUMN].iloc[0] == 2
    assert clean[ID_COLUMN].is_unique


def test_pipeline_is_deterministic(tmp_path, raw_csv):
    first = run_analyst_pipeline(raw_csv, tmp_path / "a")
    second = run_analyst_pipeline(raw_csv, tmp_path / "b")
    for name in first:
        assert first[name].read_bytes() == second[name].read_bytes(), name
