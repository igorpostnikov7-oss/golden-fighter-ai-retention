"""Crew 2: gate before training, leakage-safe features and split, model selection, reports."""
import json

import pandas as pd
import pytest

from golden_fighter import config
from golden_fighter.contract.breakers import make_broken_copy
from golden_fighter.errors import DataValidationError
from golden_fighter.modeling.evaluate import leakage_warnings
from golden_fighter.modeling.features import add_engineered_features, ensure_crew1_artifact_dir
from golden_fighter.modeling.pipeline import features_step, run_scientist_pipeline, training_step
from golden_fighter.modeling.predict import load_model
from golden_fighter.modeling.train import make_pipeline

LEAKY = "avg_class_frequency_current_month"
NEVER_INPUTS = {"churn", "member_id", "gender", LEAKY}


@pytest.fixture(scope="module")
def evaluation(scientist_out):
    return json.loads((scientist_out / "evaluation.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def features(scientist_out):
    return pd.read_csv(scientist_out / "features.csv")


def test_all_four_crew2_artifacts_exist(scientist_out):
    for name in ("features.csv", "model.joblib", "evaluation_report.md", "model_card.md"):
        assert (scientist_out / name).stat().st_size > 0, name


# ---- the gate and the crew boundary ----

def test_gate_blocks_feature_building_on_broken_data(analyst_out, tmp_path):
    broken = make_broken_copy(analyst_out, "unit_x100", tmp_path / "broken")
    with pytest.raises(DataValidationError, match="UNIT CHANGE"):
        features_step(broken, tmp_path / "out")
    assert not (tmp_path / "out" / "features.csv").exists()


def test_gate_runs_again_before_training(analyst_out, tmp_path):
    out_dir = tmp_path / "out"
    features_step(analyst_out, out_dir)
    broken = make_broken_copy(analyst_out, "drop_column", tmp_path / "broken")
    with pytest.raises(DataValidationError, match="lifetime"):
        training_step(broken, out_dir)
    assert not (out_dir / "model.joblib").exists()


def test_crew2_refuses_raw_data_directory():
    with pytest.raises(DataValidationError, match="may not read raw data"):
        ensure_crew1_artifact_dir(config.RAW_DATA_PATH.parent)


# ---- leakage-safe features and split ----

def test_model_inputs_exclude_target_ids_gender_and_leaky_columns(scientist_out, evaluation):
    bundle = load_model(scientist_out / "model.joblib")
    assert not NEVER_INPUTS & set(bundle["input_features"])
    assert not NEVER_INPUTS & set(evaluation["input_features"])


def test_engineered_features_match_the_saved_pipeline_steps(features, evaluation):
    recomputed = add_engineered_features(features[evaluation["input_features"]])
    for col in evaluation["engineered_features"]:
        pd.testing.assert_series_equal(features[col], recomputed[col], check_names=False)


def test_engineered_features_are_row_wise(features, evaluation):
    """Computing on one row must give the same as computing on the full table: no dataset statistics."""
    row = features[evaluation["input_features"]].iloc[[7]]
    alone = add_engineered_features(row)
    together = add_engineered_features(features[evaluation["input_features"]]).iloc[[7]]
    pd.testing.assert_frame_equal(alone, together)


def test_split_is_stratified_and_disjoint(features):
    train, test = features[features["split"] == "train"], features[features["split"] == "test"]
    assert len(test) == pytest.approx(0.2 * len(features), abs=1)
    assert abs(train["churn"].mean() - test["churn"].mean()) < 0.02
    assert set(train["member_id"]).isdisjoint(test["member_id"])


def test_scaler_lives_inside_the_pipeline():
    """Scaling is a pipeline step, so cross-validation refits it on each training fold only."""
    steps = [name for name, _ in make_pipeline("logistic_regression").steps]
    assert steps == ["features", "scale", "model"]


# ---- selection, metrics, ablation ----

def test_selection_uses_cross_validation_not_test(evaluation):
    candidates = {n: r for n, r in evaluation["results"].items() if n != "baseline_majority"}
    best_cv = max(candidates, key=lambda n: candidates[n]["cv"]["roc_auc"]["mean"])
    assert evaluation["selected_model"] == best_cv
    assert "test set was not used for selection" in evaluation["selection_rule"]


def test_all_required_metrics_and_confusion_matrix(evaluation):
    test_rows = evaluation["data"]["test_rows"]
    for name, result in evaluation["results"].items():
        for metric in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
            assert 0 <= result["test"][metric] <= 1, (name, metric)
        assert sum(result["test"]["confusion_matrix"].values()) == test_rows


def test_models_beat_the_baseline(evaluation):
    baseline = evaluation["results"]["baseline_majority"]["test"]
    selected = evaluation["results"][evaluation["selected_model"]]["test"]
    assert baseline["roc_auc"] == 0.5 and baseline["recall"] == 0
    assert selected["roc_auc"] > 0.75 and selected["recall"] > 0.5


def test_ablation_is_reported_but_never_selected(evaluation):
    assert evaluation["ablation"]["extra_columns"] == [LEAKY]
    assert evaluation["selected_model"] in ("logistic_regression", "random_forest")


def test_suspiciously_perfect_auc_raises_a_warning():
    cv = {"roc_auc": {"mean": 0.995}}
    assert leakage_warnings("too_good", cv, {"roc_auc": 0.97})
    assert not leakage_warnings("fine", {"roc_auc": {"mean": 0.9}}, {"roc_auc": 0.9})


def test_training_is_reproducible(analyst_out, tmp_path, scientist_out):
    run_scientist_pipeline(analyst_out, tmp_path / "again")
    first = json.loads((scientist_out / "evaluation.json").read_text(encoding="utf-8"))
    second = json.loads((tmp_path / "again" / "evaluation.json").read_text(encoding="utf-8"))
    assert first == second
    assert (scientist_out / "features.csv").read_bytes() == (tmp_path / "again" / "features.csv").read_bytes()


# ---- reports ----

def test_evaluation_report_compares_all_variations(scientist_out):
    text = (scientist_out / "evaluation_report.md").read_text(encoding="utf-8")
    for needle in ("Baseline (majority)", "Logistic Regression", "Random Forest", "Confusion matrix",
                   "Leakage ablation", "PR-AUC", "Fairness check"):
        assert needle in text, needle


def test_model_card_has_required_sections(scientist_out):
    text = (scientist_out / "model_card.md").read_text(encoding="utf-8")
    for heading in ("## Intended use", "## Training data", "## Performance", "## Limitations",
                    "## Ethical considerations", "## Fairness"):
        assert heading in text, heading
