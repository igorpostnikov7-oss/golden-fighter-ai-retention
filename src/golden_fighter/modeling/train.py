"""Train and compare model variations; select on cross-validation, report on the held-out test set.

Every candidate is a full sklearn Pipeline (feature engineering -> scaling -> model), so scaling is
fitted inside each CV fold on training rows only, and the saved model repeats exactly the same
steps at prediction time.
"""
import logging

import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from golden_fighter import config
from golden_fighter.contract.schema import DatasetContract
from golden_fighter.modeling.evaluate import (
    compute_metrics,
    fairness_slices,
    feature_importance,
    leakage_warnings,
)
from golden_fighter.modeling.features import (
    ENGINEERED_FEATURES,
    SPLIT_COLUMN,
    add_engineered_features,
    select_features,
)

log = logging.getLogger(__name__)

CV_FOLDS = 5
THRESHOLD = 0.5
SELECTION_METRIC = "roc_auc"  # ranking quality: what a retention team needs to prioritise calls
# zero_division=0: the majority baseline never predicts churn, so precision/F1 are 0, not undefined.
CV_SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "recall": "recall",
              "precision": make_scorer(precision_score, zero_division=0),
              "f1": make_scorer(f1_score, zero_division=0), "accuracy": "accuracy"}

CANDIDATES = {
    "baseline_majority": "Always predicts the overall churn rate (DummyClassifier). Any real model must beat it.",
    "logistic_regression": "Scaled inputs, L2-regularised logistic regression, balanced class weights.",
    "random_forest": "300 trees, min 5 members per leaf, balanced class weights.",
}


def make_pipeline(name: str) -> Pipeline:
    if name == "baseline_majority":
        model = DummyClassifier(strategy="prior")
        steps = [("features", FunctionTransformer(add_engineered_features)), ("model", model)]
    elif name == "logistic_regression":
        steps = [("features", FunctionTransformer(add_engineered_features)), ("scale", StandardScaler()),
                 ("model", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))]
    elif name == "random_forest":
        steps = [("features", FunctionTransformer(add_engineered_features)),
                 ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced",
                                                  random_state=config.SEED, n_jobs=-1))]
    else:
        raise ValueError(f"Unknown model '{name}'")
    return Pipeline(steps)


def _cross_validate(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=config.SEED)
    scores = cross_validate(pipeline, X, y, cv=folds, scoring=CV_SCORING, n_jobs=1)
    return {metric: {"mean": round(float(scores[f"test_{metric}"].mean()), 4),
                     "std": round(float(scores[f"test_{metric}"].std()), 4)} for metric in CV_SCORING}


def _fit_and_score(name: str, X_train, y_train, X_test, y_test, label: str | None = None
                   ) -> tuple[Pipeline, dict]:
    label = label or name
    pipeline = make_pipeline(name)
    cv = _cross_validate(pipeline, X_train, y_train)
    pipeline.fit(X_train, y_train)
    test = compute_metrics(y_test, pipeline.predict_proba(X_test)[:, 1], THRESHOLD)
    log.info("%-30s CV ROC-AUC %.4f +/- %.4f | test ROC-AUC %.4f, recall %.3f",
             label, cv["roc_auc"]["mean"], cv["roc_auc"]["std"], test["roc_auc"], test["recall"])
    return pipeline, {"cv": cv, "test": test, "warnings": leakage_warnings(label, cv, test)}


def train_and_evaluate(features: pd.DataFrame, clean: pd.DataFrame, contract: DatasetContract) -> dict:
    """Returns the selected model bundle and a results dict that feeds the reports."""
    target = contract.target.name
    inputs = select_features(contract, clean)
    train = features[features[SPLIT_COLUMN] == "train"]
    test = features[features[SPLIT_COLUMN] == "test"]
    X_train, y_train, X_test, y_test = train[inputs], train[target], test[inputs], test[target]

    models, results = {}, {}
    for name in CANDIDATES:
        models[name], results[name] = _fit_and_score(name, X_train, y_train, X_test, y_test)

    # Select on validation (CV on the training set only); the baseline is a reference, not a candidate.
    selectable = [n for n in CANDIDATES if n != "baseline_majority"]
    selected = max(selectable, key=lambda n: (results[n]["cv"][SELECTION_METRIC]["mean"],
                                              results[n]["cv"]["pr_auc"]["mean"]))
    log.info("Selected %s (best mean CV %s)", selected, SELECTION_METRIC)

    # Leakage ablation: same model family plus the contract-excluded columns. Reported, never selectable.
    ablation_inputs = select_features(contract, clean, mode="ablation")
    ablation_X_train = clean.loc[train.index, ablation_inputs]
    ablation_X_test = clean.loc[test.index, ablation_inputs]
    _, ablation = _fit_and_score(selected, ablation_X_train, y_train, ablation_X_test, y_test,
                                 label=f"{selected} + leaky (ablation)")
    ablation["extra_columns"] = contract.excluded_from_primary_model

    proba = models[selected].predict_proba(X_test)[:, 1]
    audit_cols = [c for c in ("gender",) if c in clean.columns]
    audit = clean.loc[test.index, audit_cols].copy()
    audit["age_band"] = pd.cut(clean.loc[test.index, "age"], [0, 24, 29, 34, 200],
                               labels=["under 25", "25-29", "30-34", "35+"])

    contract_ranges = {}
    for name in inputs:
        col = contract.column(name)
        contract_ranges[name] = {"min": col.min, "max": col.max, "allowed_values": col.allowed_values,
                                 "unit": col.unit}

    bundle = {
        "model_name": selected,
        "pipeline": models[selected],
        "input_features": inputs,
        "engineered_features": list(ENGINEERED_FEATURES),
        "threshold": THRESHOLD,
        "target": target,
        "input_contract": contract_ranges,
        "contract_version": contract.contract_version,
        "contract_sha256": contract.clean_data_sha256,
        "sklearn_version": sklearn.__version__,
        "seed": config.SEED,
    }
    report = {
        "selected_model": selected,
        "selection_rule": f"highest mean {CV_FOLDS}-fold CV {SELECTION_METRIC} on the training set "
                          "(tie-break: CV PR-AUC); the test set was not used for selection",
        "threshold": THRESHOLD,
        "candidates": CANDIDATES,
        "results": results,
        "ablation": ablation,
        "input_features": inputs,
        "engineered_features": bundle["engineered_features"],
        "excluded_columns": {"forbidden": contract.forbidden_features,
                             "leakage_or_late": contract.excluded_from_primary_model},
        "data": {"rows": len(features), "train_rows": len(train), "test_rows": len(test),
                 "train_churn_rate": round(float(y_train.mean()), 4),
                 "test_churn_rate": round(float(y_test.mean()), 4),
                 "contract_version": contract.contract_version, "clean_data_sha256": contract.clean_data_sha256},
        "feature_importance": feature_importance(models[selected], X_test, y_test),
        "fairness": fairness_slices(audit, y_test, proba, THRESHOLD),
        "warnings": [w for r in results.values() for w in r["warnings"]],
        "sklearn_version": sklearn.__version__,
        "seed": config.SEED,
    }
    return {"bundle": bundle, "report": report}
