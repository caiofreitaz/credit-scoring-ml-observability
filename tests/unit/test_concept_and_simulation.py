"""Concept Drift evidence and the scenario simulation mechanisms."""

import numpy as np
import pandas as pd
import pytest

from src.data.prepare import clean_dataset
from src.drift.concept import calibration_shift_test, compare_performance, diagnose
from src.drift.simulation import (
    SCENARIOS,
    apply_concept_shift,
    build_scenarios,
    customer_mix_weights,
    load_scenario,
    write_scenarios,
)
from src.validation.gate import validate_batch

DRIFT_CFG = {"calibration_pvalue_threshold": 0.01}
ALERT_CFG = {"prediction_psi_warning": 0.10, "auc_drop_warning": 0.02}


def _labels(scores, rng):
    return (rng.random(len(scores)) < scores).astype(int)


def test_calibration_stable_when_relationship_unchanged():
    rng = np.random.default_rng(1)
    ref_s, cur_s = rng.random(4000), rng.random(4000)
    result = calibration_shift_test(ref_s, _labels(ref_s, rng), cur_s, _labels(cur_s, rng))
    assert result["calibration_pvalue"] > 0.01
    assert len(result["table"]) == 10


def test_calibration_detects_changed_relationship():
    rng = np.random.default_rng(2)
    ref_s, cur_s = rng.random(4000), rng.random(4000)
    changed = (rng.random(4000) < np.clip(cur_s + 0.2, 0, 1)).astype(int)
    result = calibration_shift_test(ref_s, _labels(ref_s, rng), cur_s, changed)
    assert result["calibration_pvalue"] < 1e-6
    assert result["calibration_mean_abs_gap_pp"] > 10


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((0, 0.01, 0.0, 0.5), "Estável"),
        ((3, 0.30, 0.0, 0.5), "Data Drift (P(X)) + Prediction Drift"),
        ((0, 0.01, 0.10, 1e-9), "Concept Drift confirmado (P(Y|X)) + Degradação de performance"),
        ((0, 0.01, 0.00, 1e-9), "Concept Drift suspeito (P(Y|X))"),
    ],
)
def test_diagnose_distinguishes_drift_types(args, expected):
    assert diagnose(*args, DRIFT_CFG, ALERT_CFG)["diagnosis"] == expected


def test_compare_performance():
    keys = ["roc_auc", "accuracy", "precision", "recall", "f1", "brier", "observed_default_rate"]
    ref = dict.fromkeys(keys, 0.5)
    deltas = compare_performance(ref, {**ref, "roc_auc": 0.4})
    assert deltas["delta_roc_auc"] == pytest.approx(-0.1)


@pytest.fixture()
def pool(synthetic_raw):
    cleaned, _ = clean_dataset(synthetic_raw, "salt")
    return cleaned.iloc[:1500].reset_index(drop=True)


@pytest.fixture()
def scenarios(pool, project):
    return build_scenarios(pool, project["config"])


def test_scenarios_are_reproducible(pool, project):
    first, second = build_scenarios(pool, project["config"]), build_scenarios(pool, project["config"])
    for name in SCENARIOS:
        pd.testing.assert_frame_equal(first[name].features, second[name].features)
        pd.testing.assert_frame_equal(first[name].outcomes, second[name].outcomes)


def test_data_drift_shifts_limit_and_age_but_keeps_real_labels(scenarios, pool):
    normal, shifted = scenarios["normal"].features, scenarios["data_drift"].features
    assert shifted["LIMIT_BAL"].mean() < 0.8 * normal["LIMIT_BAL"].mean()
    assert shifted["AGE"].mean() < normal["AGE"].mean()
    merged = scenarios["data_drift"].outcomes.merge(pool[["client_id", "default_next_month"]], on="client_id")
    assert (merged["default_next_month_x"] == merged["default_next_month_y"]).all()


def test_concept_drift_keeps_features_and_changes_labels(scenarios):
    pd.testing.assert_frame_equal(scenarios["normal"].features, scenarios["concept_drift"].features)
    before = scenarios["normal"].outcomes["default_next_month"]
    after = scenarios["concept_drift"].outcomes["default_next_month"]
    assert (before != after).sum() > 0
    mech = scenarios["concept_drift"].mechanism
    assert (before != after).sum() == mech["new_defaults"] + mech["cures"]


def test_combined_scenario_uses_shifted_rows(scenarios):
    pd.testing.assert_frame_equal(scenarios["data_drift"].features, scenarios["data_and_concept_drift"].features)


def test_concept_mechanism_targets_expected_clients():
    frame = pd.DataFrame(
        {
            "LIMIT_BAL": [100_000] * 4,
            "BILL_AMT1": [90_000, 90_000, 10_000, 10_000],
            "PAY_0": [0, 0, 3, 3],
            "y": [0, 0, 1, 1],
        }
    )
    cfg = {
        "utilization_threshold": 0.6,
        "p_new_default_high_utilization": 1.0,
        "late_payment_status_threshold": 2,
        "p_cure_late_payers": 1.0,
    }
    shifted, stats = apply_concept_shift(frame, "y", cfg, np.random.default_rng(0))
    assert shifted["y"].tolist() == [1, 1, 0, 0]
    assert stats["new_defaults"] == 2 and stats["cures"] == 2


def test_mix_weights_favour_low_limits_and_young(pool, project):
    weights = customer_mix_weights(pool, project["config"]["simulation"]["data_drift"])
    assert weights.sum() == pytest.approx(1.0)
    young_low = (pool["AGE"] < 30) & (pool["LIMIT_BAL"] < 50_000)
    assert weights[young_low].mean() > weights[~young_low].mean()


def test_scenarios_respect_contracts_and_persist(scenarios, project):
    manifest = write_scenarios(scenarios, project["paths"])
    for name in SCENARIOS:
        features, outcomes = load_scenario(name, project["paths"])
        assert validate_batch(features, "inference", name).is_valid
        assert validate_batch(outcomes, "outcome", name).is_valid
        assert manifest[name]["rows"] == len(features)


def test_load_missing_scenario(project):
    with pytest.raises(FileNotFoundError, match="simulate-drift"):
        load_scenario("normal", project["paths"])
