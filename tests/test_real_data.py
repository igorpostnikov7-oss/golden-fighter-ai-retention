"""Checks on the real Kaggle file. Skipped when it is absent (e.g. a fresh clone before download)."""
import json

import pandas as pd
import pytest

from golden_fighter import config
from golden_fighter.analysis.pipeline import run_analyst_pipeline
from golden_fighter.contract.builder import load_contract
from golden_fighter.modeling.pipeline import run_scientist_pipeline

pytestmark = pytest.mark.skipif(not config.RAW_DATA_PATH.exists(), reason="real dataset not downloaded")


@pytest.fixture(scope="module")
def real_out(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("real")
    run_analyst_pipeline(config.RAW_DATA_PATH, out_dir)
    return out_dir


def test_real_data_cleans_without_losing_rows(real_out):
    clean = pd.read_csv(real_out / "clean_data.csv")
    assert len(clean) == 4000
    assert round(clean["churn"].mean(), 4) == 0.2652


def test_real_model_quality_and_no_leakage_warning(real_out, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("real_scientist")
    run_scientist_pipeline(real_out, out_dir)
    report = json.loads((out_dir / "evaluation.json").read_text(encoding="utf-8"))
    selected = report["results"][report["selected_model"]]
    assert selected["test"]["roc_auc"] > 0.9
    assert report["warnings"] == []
    assert report["ablation"]["test"]["roc_auc"] > selected["test"]["roc_auc"]


def test_real_contract_flags(real_out):
    contract = load_contract(real_out / "dataset_contract.json")
    assert contract.excluded_from_primary_model == ["avg_class_frequency_current_month"]
    assert contract.column("contract_period").allowed_values == [1, 6, 12]
