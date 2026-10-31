"""The saved model scores new members with the same processing steps, and rejects bad input."""
import json

import numpy as np
import pandas as pd
import pytest

from golden_fighter.errors import DataValidationError
from golden_fighter.modeling.evaluate import compute_metrics
from golden_fighter.modeling.predict import load_model, predict_churn


@pytest.fixture(scope="module")
def bundle(scientist_out):
    return load_model(scientist_out / "model.joblib")


@pytest.fixture(scope="module")
def clean(analyst_out):
    return pd.read_csv(analyst_out / "clean_data.csv")


@pytest.fixture
def new_member():
    return pd.DataFrame([{
        "near_location": 1, "partner": 0, "promo_friends": 0, "phone": 1, "contract_period": 1,
        "group_visits": 0, "age": 24, "avg_additional_charges_total": 60.0, "month_to_end_contract": 1,
        "lifetime": 0, "avg_class_frequency_total": 0.8,
    }])


def test_predicts_a_brand_new_member(bundle, new_member):
    out = predict_churn(bundle, new_member)
    assert list(out.columns) == ["churn_probability", "churn_prediction", "risk_band"]
    assert 0 <= out.loc[0, "churn_probability"] <= 1
    assert out.loc[0, "risk_band"] in ("low", "medium", "high")


def test_loyal_member_scores_lower_than_new_risky_member(bundle, new_member):
    loyal = new_member.assign(contract_period=12, month_to_end_contract=6, lifetime=24, group_visits=1,
                              promo_friends=1, age=33, avg_class_frequency_total=3.0)
    risky_p = predict_churn(bundle, new_member).loc[0, "churn_probability"]
    loyal_p = predict_churn(bundle, loyal).loc[0, "churn_probability"]
    assert loyal_p < risky_p


def test_saved_model_reproduces_reported_test_metrics(bundle, scientist_out, clean):
    features = pd.read_csv(scientist_out / "features.csv")
    test_ids = features.loc[features["split"] == "test", "member_id"]
    rows = clean[clean["member_id"].isin(test_ids)]
    out = predict_churn(bundle, rows)
    report = json.loads((scientist_out / "evaluation.json").read_text(encoding="utf-8"))
    expected = report["results"][report["selected_model"]]["test"]
    got = compute_metrics(rows["churn"], out["churn_probability"].to_numpy())
    assert got["confusion_matrix"] == expected["confusion_matrix"]
    assert got["roc_auc"] == pytest.approx(expected["roc_auc"], abs=1e-3)


def test_extra_columns_such_as_the_target_are_ignored(bundle, clean):
    rows = clean.head(20)
    with_target = predict_churn(bundle, rows)
    flipped = predict_churn(bundle, rows.assign(churn=1 - rows["churn"], gender=0))
    pd.testing.assert_frame_equal(with_target, flipped)


def test_same_steps_as_training_no_refit(bundle, clean):
    """Scoring one row or many gives the same probability for that row (nothing is re-fitted)."""
    many = predict_churn(bundle, clean.head(50))
    one = predict_churn(bundle, clean.head(50).iloc[[10]])
    assert np.isclose(many.iloc[10]["churn_probability"], one.iloc[0]["churn_probability"])


@pytest.mark.parametrize("change, message", [
    (lambda df: df.drop(columns=["lifetime"]), "missing input column 'lifetime'"),
    (lambda df: df.assign(avg_additional_charges_total=6000.0), "above the contract maximum"),
    (lambda df: df.assign(contract_period=3), "must be one of [1, 6, 12]"),
    (lambda df: df.assign(age="twenty"), "'age' must be a number"),
    (lambda df: df.assign(month_to_end_contract=12), "cannot exceed 'contract_period'"),
])
def test_invalid_input_is_rejected_with_a_clear_message(bundle, new_member, change, message):
    with pytest.raises(DataValidationError) as err:
        predict_churn(bundle, change(new_member))
    assert message in str(err.value)
