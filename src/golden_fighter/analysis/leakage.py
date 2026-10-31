"""Leakage scan: find columns that could leak the churn label into a model."""
import logging

import pandas as pd
from sklearn.metrics import roc_auc_score

from golden_fighter.analysis.schema_spec import SPEC_BY_NAME, TARGET

log = logging.getLogger(__name__)

# A single raw column that separates churners almost perfectly is far more likely to be a
# leak than a real signal.
SUSPICIOUS_AUC = 0.95


def univariate_auc(df: pd.DataFrame, column: str, target: str = TARGET) -> float:
    """ROC-AUC of one column used directly as a score, folded so 0.5 = useless and 1.0 = perfect."""
    if df[column].nunique() < 2:
        return 0.5
    auc = roc_auc_score(df[target], df[column])
    return round(max(auc, 1 - auc), 4)


def scan_leakage(df: pd.DataFrame, target: str = TARGET) -> list[dict]:
    """Combine domain knowledge (schema_spec) with data checks into one finding per feature."""
    findings = []
    for column in df.columns:
        spec = SPEC_BY_NAME.get(column)
        if column == target or (spec is not None and spec.role != "feature"):
            continue
        auc = univariate_auc(df, column, target)
        risk = spec.leakage_risk if spec else "none"
        reasons = [spec.leakage_note] if spec and spec.leakage_note else []
        if df[column].equals(df[target]):
            risk = "high"
            reasons.append("Column is identical to the target.")
        elif auc >= SUSPICIOUS_AUC:
            risk = "high"
            reasons.append(f"Single-column AUC {auc:.3f} >= {SUSPICIOUS_AUC}: suspiciously perfect predictor.")
        findings.append({"column": column, "univariate_auc": auc, "risk": risk,
                         "reason": " ".join(reasons) or "No leakage indication."})
    flagged = [f["column"] for f in findings if f["risk"] in ("medium", "high")]
    log.info("Leakage scan: %d features checked, flagged medium/high: %s", len(findings), flagged or "none")
    return findings
