"""Bias audit: performance and decision rates by group on the reference (holdout) set.

Protected attributes (SEX, MARRIAGE) are NOT model inputs; they are used here only
to audit outcomes. Groups smaller than ``min_group_size`` are suppressed to limit
re-identification risk and unstable estimates.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

GROUP_LABELS = {
    "SEX": {1: "masculino", 2: "feminino"},
    "MARRIAGE": {1: "casado(a)", 2: "solteiro(a)", 3: "outros"},
    "EDUCATION": {1: "pós-graduação", 2: "universitário", 3: "ensino médio", 4: "outros"},
}
AGE_BINS = [0, 29, 44, 59, 200]
AGE_LABELS = ["até 29", "30-44", "45-59", "60+"]


def age_band(age: pd.Series) -> pd.Series:
    """Bucketize age into coarse bands (generalization reduces identifiability)."""
    return pd.cut(age, bins=AGE_BINS, labels=AGE_LABELS).astype(str)


def group_metrics(frame: pd.DataFrame, group_col: str, target: str, min_group_size: int) -> pd.DataFrame:
    """Per-group decision and error rates; suppress small groups."""
    rows = []
    for group, part in frame.groupby(group_col, observed=True):
        n = len(part)
        if n < min_group_size:
            rows.append({"attribute": group_col, "group": str(group), "n": n, "suppressed": True})
            continue
        y, pred, score = part[target].to_numpy(), part["predicted_label"].to_numpy(), part["prediction"].to_numpy()
        positives, negatives = (y == 1).sum(), (y == 0).sum()
        rows.append(
            {
                "attribute": group_col,
                "group": str(group),
                "n": n,
                "suppressed": False,
                "observed_default_rate": round(float(y.mean()), 4),
                "rejection_rate": round(float(pred.mean()), 4),
                "approval_rate": round(float(1 - pred.mean()), 4),
                "tpr": round(float(pred[y == 1].mean()), 4) if positives else np.nan,
                "fpr": round(float(pred[y == 0].mean()), 4) if negatives else np.nan,
                "roc_auc": round(float(roc_auc_score(y, score)), 4) if positives and negatives else np.nan,
            }
        )
    table = pd.DataFrame(rows)
    valid = table[~table["suppressed"]]
    if not valid.empty:
        best_approval = valid["approval_rate"].max()
        table["approval_ratio_vs_best"] = (table["approval_rate"] / best_approval).round(4)
        table["fpr_gap_vs_min"] = (table["fpr"] - valid["fpr"].min()).round(4)
    return table


def fairness_audit(scored_reference: pd.DataFrame, target: str, min_group_size: int = 100) -> dict[str, Any]:
    """Run the audit across SEX, MARRIAGE, EDUCATION and age bands."""
    frame = scored_reference.copy()
    frame["AGE_BAND"] = age_band(frame["AGE"])
    for column, labels in GROUP_LABELS.items():
        frame[column] = frame[column].map(labels)
    tables = [group_metrics(frame, col, target, min_group_size) for col in ["SEX", "MARRIAGE", "EDUCATION", "AGE_BAND"]]
    table = pd.concat(tables, ignore_index=True)
    findings = []
    for attribute, part in table[~table["suppressed"]].groupby("attribute"):
        min_ratio = float(part["approval_ratio_vs_best"].min())
        auc_spread = float(part["roc_auc"].max() - part["roc_auc"].min())
        findings.append(
            {
                "attribute": attribute,
                "min_approval_ratio_vs_best": round(min_ratio, 4),
                "four_fifths_rule_ok": bool(min_ratio >= 0.8),
                "roc_auc_spread": round(auc_spread, 4),
                "max_fpr_gap": round(float(part["fpr_gap_vs_min"].max()), 4),
            }
        )
    return {"min_group_size": min_group_size, "groups": table, "findings": findings}
