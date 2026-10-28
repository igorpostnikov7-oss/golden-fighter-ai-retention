"""Invalid raw input must stop the pipeline with a readable DataValidationError."""
import pytest

from golden_fighter.analysis.cleaning import load_raw
from golden_fighter.errors import DataValidationError


def test_missing_file(tmp_path):
    with pytest.raises(DataValidationError, match="not found"):
        load_raw(tmp_path / "nope.csv")


def test_empty_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(DataValidationError, match="empty"):
        load_raw(path)


def test_header_only(tmp_path, raw_df):
    path = tmp_path / "header.csv"
    raw_df.head(0).to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="no rows"):
        load_raw(path)


def test_renamed_column_is_named_in_error(tmp_path, raw_df):
    path = tmp_path / "renamed.csv"
    raw_df.rename(columns={"Churn": "churn_flag"}).to_csv(path, index=False)
    with pytest.raises(DataValidationError) as err:
        load_raw(path)
    assert "Missing required column 'Churn'" in str(err.value)
    assert "churn_flag" in str(err.value)


def test_dropped_column(tmp_path, raw_df):
    path = tmp_path / "dropped.csv"
    raw_df.drop(columns=["Lifetime"]).to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="Lifetime"):
        load_raw(path)


def test_wrong_delimiter(tmp_path, raw_df):
    path = tmp_path / "semicolon.csv"
    raw_df.to_csv(path, index=False, sep=";")
    with pytest.raises(DataValidationError, match="Missing required column"):
        load_raw(path)


def test_valid_file_loads(raw_csv):
    assert len(load_raw(raw_csv)) == 600
