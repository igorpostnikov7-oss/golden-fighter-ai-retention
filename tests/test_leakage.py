from golden_fighter.analysis.cleaning import clean_data
from golden_fighter.analysis.leakage import scan_leakage
from golden_fighter.analysis.schema_spec import ID_COLUMN, TARGET


def _by_column(findings):
    return {f["column"]: f for f in findings}


def test_target_and_id_are_not_scanned_as_features(raw_df):
    clean, _ = clean_data(raw_df)
    columns = _by_column(scan_leakage(clean))
    assert TARGET not in columns
    assert ID_COLUMN not in columns


def test_copy_of_target_is_flagged(raw_df):
    clean, _ = clean_data(raw_df)
    clean["churn_copy"] = clean[TARGET]
    finding = _by_column(scan_leakage(clean))["churn_copy"]
    assert finding["risk"] == "high"
    assert "identical to the target" in finding["reason"]


def test_near_perfect_proxy_is_flagged(raw_df):
    clean, _ = clean_data(raw_df)
    clean["days_since_cancellation_request"] = clean[TARGET] * 30 + (clean[ID_COLUMN] % 3)
    finding = _by_column(scan_leakage(clean))["days_since_cancellation_request"]
    assert finding["risk"] == "high"
    assert "suspiciously perfect" in finding["reason"]


def test_domain_flag_for_current_month_frequency(raw_df):
    clean, _ = clean_data(raw_df)
    assert _by_column(scan_leakage(clean))["avg_class_frequency_current_month"]["risk"] == "high"
