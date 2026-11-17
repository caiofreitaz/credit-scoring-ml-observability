"""Concept Drift evidence: change in P(Y | score) and performance comparison.

The model score s(x) summarizes X. Comparing the observed default rate inside the
same score bins (bins learned on the reference) between reference and production
tests whether P(Y | s(X)) changed. Under pure covariate shift (P(Y|X) stable) the
per-bin rates should match; under Concept Drift they diverge. This requires
observed outcomes, which in real credit portfolios arrive with delay.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def calibration_shift_test(
    reference_scores: np.ndarray,
    reference_labels: np.ndarray,
    current_scores: np.ndarray,
    current_labels: np.ndarray,
    bins: int = 10,
) -> dict[str, Any]:
    """Per-bin two-proportion z-tests combined into a chi-square statistic."""
    reference_scores = np.asarray(reference_scores, dtype="float64")
    current_scores = np.asarray(current_scores, dtype="float64")
    reference_labels = np.asarray(reference_labels).astype(int)
    current_labels = np.asarray(current_labels).astype(int)

    edges = np.unique(np.quantile(reference_scores, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_bin = np.digitize(reference_scores, edges[1:-1])
    cur_bin = np.digitize(current_scores, edges[1:-1])

    rows, chi2_stat, dof = [], 0.0, 0
    for b in range(len(edges) - 1):
        ref_mask, cur_mask = ref_bin == b, cur_bin == b
        n_ref, n_cur = int(ref_mask.sum()), int(cur_mask.sum())
        if n_ref == 0 or n_cur == 0:
            continue
        p_ref = reference_labels[ref_mask].mean()
        p_cur = current_labels[cur_mask].mean()
        pooled = (reference_labels[ref_mask].sum() + current_labels[cur_mask].sum()) / (n_ref + n_cur)
        variance = pooled * (1 - pooled) * (1 / n_ref + 1 / n_cur)
        z = (p_cur - p_ref) / np.sqrt(variance) if variance > 0 else 0.0
        chi2_stat += z**2
        dof += 1
        rows.append(
            {
                "bin": b,
                "reference_n": n_ref,
                "reference_default_rate": round(float(p_ref), 4),
                "current_n": n_cur,
                "current_default_rate": round(float(p_cur), 4),
                "difference_pp": round(float(p_cur - p_ref) * 100, 2),
                "z": round(float(z), 3),
            }
        )
    p_value = float(stats.chi2.sf(chi2_stat, dof)) if dof else 1.0
    table = pd.DataFrame(rows)
    weighted_gap = (
        float(np.average(table["difference_pp"].abs(), weights=table["current_n"])) if not table.empty else 0.0
    )
    return {
        "calibration_chi2": round(float(chi2_stat), 4),
        "calibration_dof": dof,
        "calibration_pvalue": float(f"{p_value:.3e}"),
        "calibration_mean_abs_gap_pp": round(weighted_gap, 3),
        "table": table,
    }


def compare_performance(reference_metrics: dict[str, float], current_metrics: dict[str, float]) -> dict[str, float]:
    """Absolute deltas (current - reference) for the main metrics."""
    keys = ["roc_auc", "accuracy", "precision", "recall", "f1", "brier", "observed_default_rate"]
    return {f"delta_{k}": round(current_metrics[k] - reference_metrics[k], 5) for k in keys}


def diagnose(
    n_drifted_features: int,
    prediction_psi: float,
    auc_drop: float,
    calibration_pvalue: float,
    drift_cfg: dict[str, Any],
    alert_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Separate Data Drift, Prediction Drift, Concept Drift and performance degradation."""
    data_drift = n_drifted_features > 0
    prediction_drift = prediction_psi >= alert_cfg["prediction_psi_warning"]
    degradation = auc_drop >= alert_cfg["auc_drop_warning"]
    concept_signal = calibration_pvalue < drift_cfg["calibration_pvalue_threshold"]
    labels = []
    if data_drift:
        labels.append("Data Drift (P(X))")
    if prediction_drift:
        labels.append("Prediction Drift")
    if concept_signal:
        labels.append("Concept Drift confirmado (P(Y|X))" if degradation else "Concept Drift suspeito (P(Y|X))")
    if degradation:
        labels.append("Degradação de performance")
    if not labels:
        labels.append("Estável")
    return {
        "data_drift": data_drift,
        "prediction_drift": prediction_drift,
        "concept_drift_signal": concept_signal,
        "performance_degradation": degradation,
        "diagnosis": " + ".join(labels),
    }
