"""Statistical drift tests: PSI, Kolmogorov-Smirnov and chi-square."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

EPSILON = 1e-4


def _proportions(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    props = counts / total if total > 0 else np.zeros_like(counts, dtype="float64")
    return np.clip(props, EPSILON, None)


def psi_continuous(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index with quantile bins learned on the reference sample."""
    reference = np.asarray(reference, dtype="float64")
    current = np.asarray(current, dtype="float64")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return psi_discrete(reference, current)
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = np.histogram(reference, bins=edges)[0].astype("float64")
    cur_counts = np.histogram(current, bins=edges)[0].astype("float64")
    ref_p, cur_p = _proportions(ref_counts), _proportions(cur_counts)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def psi_discrete(reference: np.ndarray, current: np.ndarray) -> float:
    """PSI over the union of observed categories."""
    categories = np.union1d(np.unique(reference), np.unique(current))
    ref_counts = np.array([(reference == c).sum() for c in categories], dtype="float64")
    cur_counts = np.array([(current == c).sum() for c in categories], dtype="float64")
    ref_p, cur_p = _proportions(ref_counts), _proportions(cur_counts)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def chi_square_test(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Chi-square test of homogeneity for categorical/ordinal codes."""
    categories = np.union1d(np.unique(reference), np.unique(current))
    table = np.array(
        [[(reference == c).sum() for c in categories], [(current == c).sum() for c in categories]],
        dtype="float64",
    )
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2:
        return 0.0, 1.0
    result = stats.chi2_contingency(table)
    return float(result.statistic), float(result.pvalue)


def ks_test(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test for continuous variables."""
    result = stats.ks_2samp(reference, current)
    return float(result.statistic), float(result.pvalue)


def psi_severity(psi: float, moderate: float, severe: float) -> str:
    """Classify PSI magnitude (industry rule of thumb: 0.10 / 0.25)."""
    if psi >= severe:
        return "severe"
    if psi >= moderate:
        return "moderate"
    return "none"


def is_discrete(series: pd.Series, max_unique: int) -> bool:
    """Treat low-cardinality integer columns as discrete codes."""
    return series.nunique(dropna=True) <= max_unique


def feature_drift_table(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Compute PSI + significance test for each feature and classify drift.

    A feature is flagged as drifted when PSI >= ``psi_moderate`` AND the
    statistical test rejects equality (p-value < ``pvalue_threshold``).
    Requiring both avoids flagging statistically significant but negligible
    shifts (large samples) and large-looking PSI from sampling noise.
    """
    rows = []
    for feature in features:
        ref = reference[feature].to_numpy()
        cur = current[feature].to_numpy()
        discrete = is_discrete(reference[feature], cfg["discrete_max_unique"])
        if discrete:
            psi = psi_discrete(ref, cur)
            statistic, p_value = chi_square_test(ref, cur)
            test = "chi-square"
        else:
            psi = psi_continuous(ref, cur, cfg["psi_bins"])
            statistic, p_value = ks_test(ref, cur)
            test = "kolmogorov-smirnov"
        severity = psi_severity(psi, cfg["psi_moderate"], cfg["psi_severe"])
        drifted = severity != "none" and p_value < cfg["pvalue_threshold"]
        ref_mean, cur_mean = float(np.mean(ref)), float(np.mean(cur))
        rows.append(
            {
                "feature": feature,
                "type": "discrete" if discrete else "continuous",
                "psi": round(psi, 5),
                "test": test,
                "statistic": round(statistic, 5),
                "p_value": float(f"{p_value:.3e}"),
                "severity": severity if drifted else "none",
                "drift_detected": bool(drifted),
                "reference_mean": round(ref_mean, 3),
                "current_mean": round(cur_mean, 3),
                "mean_change_pct": round((cur_mean - ref_mean) / abs(ref_mean) * 100, 2) if ref_mean else None,
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def prediction_drift(reference_scores: np.ndarray, current_scores: np.ndarray, bins: int = 10) -> dict[str, float]:
    """PSI and KS of the model score distribution (Prediction Drift)."""
    statistic, p_value = ks_test(reference_scores, current_scores)
    return {
        "prediction_psi": round(psi_continuous(reference_scores, current_scores, bins), 5),
        "prediction_ks_statistic": round(statistic, 5),
        "prediction_ks_pvalue": float(f"{p_value:.3e}"),
        "reference_mean_score": round(float(np.mean(reference_scores)), 5),
        "current_mean_score": round(float(np.mean(current_scores)), 5),
    }
