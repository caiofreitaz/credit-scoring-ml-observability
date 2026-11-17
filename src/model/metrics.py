"""Classification metrics used for training evaluation and production monitoring."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, proba, threshold: float) -> dict[str, float]:
    """Compute threshold-dependent and ranking metrics for a binary classifier."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype="float64")
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    has_both_classes = len(np.unique(y_true)) == 2
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)) if has_both_classes else float("nan"),
        "log_loss": float(log_loss(y_true, np.clip(proba, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, proba)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_positive_rate": float(y_pred.mean()),
        "observed_default_rate": float(y_true.mean()),
        "mean_score": float(proba.mean()),
    }


def bootstrap_auc_ci(y_true, proba, n_resamples: int = 300, seed: int = 42, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for ROC-AUC (quantifies sampling noise)."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype="float64")
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(roc_auc_score(y_true[idx], proba[idx]))
    low, high = np.quantile(scores, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def best_f1_threshold(y_true, proba, start: float, stop: float, step: float) -> tuple[float, float]:
    """Return the threshold (and F1) that maximizes F1 on a validation set."""
    best_threshold, best_f1 = start, -1.0
    for threshold in np.arange(start, stop + 1e-9, step):
        score = f1_score(y_true, (np.asarray(proba) >= threshold).astype(int), zero_division=0)
        if score > best_f1 + 1e-12:
            best_threshold, best_f1 = float(round(threshold, 4)), float(score)
    return best_threshold, best_f1
