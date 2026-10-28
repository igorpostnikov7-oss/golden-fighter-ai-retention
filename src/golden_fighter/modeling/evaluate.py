"""Metrics, fairness slices and feature importance. All numbers for the reports come from here."""
import logging

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from golden_fighter import config

log = logging.getLogger(__name__)

# A churn model this good on tabular member data would be a red flag for leakage, not a success.
SUSPICIOUS_AUC = 0.99
METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")


def compute_metrics(y_true: pd.Series, proba: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def leakage_warnings(name: str, cv: dict, test: dict) -> list[str]:
    warnings = []
    for label, auc in (("cross-validation", cv["roc_auc"]["mean"]), ("test", test["roc_auc"])):
        if auc > SUSPICIOUS_AUC:
            warnings.append(f"{name}: {label} ROC-AUC {auc:.4f} > {SUSPICIOUS_AUC}; suspiciously perfect, "
                            "check for leakage before trusting this model")
    for w in warnings:
        log.warning(w)
    return warnings


def fairness_slices(groups: pd.DataFrame, y_true: pd.Series, proba: np.ndarray, threshold: float = 0.5) -> dict:
    """Per-group recall/precision on the test set. `groups` holds audit-only columns (e.g. gender)."""
    pred = pd.Series((proba >= threshold).astype(int), index=y_true.index)
    out = {}
    for column in groups.columns:
        rows = []
        for value, idx in groups.groupby(column, observed=True).groups.items():
            y, p = y_true.loc[idx], pred.loc[idx]
            # None (not 0) when undefined: no churners in the group, or nobody flagged.
            rows.append({"group": str(value), "members": int(len(idx)),
                         "actual_churn_rate": round(float(y.mean()), 4),
                         "flagged_rate": round(float(p.mean()), 4),
                         "recall": round(float(recall_score(y, p)), 4) if y.sum() else None,
                         "precision": round(float(precision_score(y, p)), 4) if p.sum() else None})
        out[column] = rows
    return out


def feature_importance(model, X: pd.DataFrame, y: pd.Series, top: int = 10) -> list[dict]:
    """Permutation importance: how much test ROC-AUC drops when one input column is shuffled."""
    result = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=10, random_state=config.SEED,
                                    n_jobs=1)
    order = np.argsort(-result.importances_mean)[:top]
    return [{"feature": X.columns[i], "auc_drop_mean": round(float(result.importances_mean[i]), 4),
             "auc_drop_std": round(float(result.importances_std[i]), 4)} for i in order]
