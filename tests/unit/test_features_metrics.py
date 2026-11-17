"""Feature engineering and metric helpers."""

import numpy as np
import pandas as pd
import pytest

from src.model.features import ENGINEERED, CreditFeatureEngineer, signed_log1p
from src.model.metrics import best_f1_threshold, bootstrap_auc_ci, classification_metrics


def test_signed_log1p_keeps_sign():
    out = signed_log1p(np.array([-9.0, 0.0, 9.0]))
    assert out[0] == pytest.approx(-np.log(10)) and out[1] == 0 and out[2] == pytest.approx(np.log(10))


def test_feature_engineer_adds_columns(synthetic_raw):
    frame = synthetic_raw.drop(columns=["ID", "SEX", "MARRIAGE", "default_next_month"]).head(20)
    engineer = CreditFeatureEngineer().fit(frame)
    out = engineer.transform(frame)
    assert set(ENGINEERED) <= set(out.columns)
    assert list(engineer.get_feature_names_out()) == list(frame.columns) + ENGINEERED
    expected_util = (frame["BILL_AMT1"] / frame["LIMIT_BAL"]).clip(-1, 5)
    assert np.allclose(out["utilization_last"], expected_util)
    assert out["months_delayed"].between(0, 6).all()
    assert not out.isna().any().any()


def test_classification_metrics_on_known_example():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.6, 0.4, 0.9])
    m = classification_metrics(y, proba, threshold=0.5)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (1, 1, 1, 1)
    assert m["accuracy"] == 0.5 and m["precision"] == 0.5 and m["recall"] == 0.5
    assert m["roc_auc"] == pytest.approx(0.75)


def test_metrics_single_class_auc_is_nan():
    m = classification_metrics([0, 0, 0], [0.1, 0.2, 0.3], 0.5)
    assert np.isnan(m["roc_auc"])


def test_best_f1_threshold_finds_separating_cut():
    y = np.array([0] * 50 + [1] * 50)
    proba = np.concatenate([np.full(50, 0.3), np.full(50, 0.7)])
    threshold, f1 = best_f1_threshold(y, proba, 0.2, 0.8, 0.05)
    assert 0.3 < threshold <= 0.7 and f1 == 1.0


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    proba = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1)
    low, high = bootstrap_auc_ci(y, proba, n_resamples=100)
    point = classification_metrics(y, proba, 0.5)["roc_auc"]
    assert low < point < high
    assert pd.notna(low)
