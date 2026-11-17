"""PSI / KS / chi-square tests, drift identification and the no-drift case."""

import numpy as np
import pandas as pd
import pytest

from src.drift.statistics import (
    chi_square_test,
    feature_drift_table,
    ks_test,
    prediction_drift,
    psi_continuous,
    psi_discrete,
    psi_severity,
)

CFG = {"psi_bins": 10, "psi_moderate": 0.10, "psi_severe": 0.25, "pvalue_threshold": 0.05, "discrete_max_unique": 12}


@pytest.fixture()
def rng():
    return np.random.default_rng(42)


def test_psi_identical_is_zero(rng):
    sample = rng.normal(size=5000)
    assert psi_continuous(sample, sample) == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_with_shift(rng):
    ref = rng.normal(size=5000)
    small = psi_continuous(ref, rng.normal(0.1, 1, 5000))
    large = psi_continuous(ref, rng.normal(1.0, 1, 5000))
    assert small < 0.1 < 0.25 < large


def test_psi_discrete_and_constant_fallback():
    ref = np.array([1] * 80 + [2] * 20)
    assert psi_discrete(ref, ref) == pytest.approx(0.0, abs=1e-9)
    assert psi_discrete(ref, np.array([1] * 20 + [2] * 80)) > 0.25
    constant = np.zeros(100)
    assert psi_continuous(constant, constant) == pytest.approx(0.0, abs=1e-9)


def test_statistical_tests(rng):
    ref = rng.normal(size=2000)
    assert ks_test(ref, rng.normal(size=2000))[1] > 0.01
    assert ks_test(ref, rng.normal(0.5, 1, 2000))[1] < 1e-6
    cats = rng.integers(0, 3, 2000)
    assert chi_square_test(cats, rng.integers(0, 3, 2000))[1] > 0.01
    assert chi_square_test(np.ones(10), np.ones(10)) == (0.0, 1.0)


def test_severity_levels():
    assert psi_severity(0.05, 0.1, 0.25) == "none"
    assert psi_severity(0.15, 0.1, 0.25) == "moderate"
    assert psi_severity(0.30, 0.1, 0.25) == "severe"


def test_feature_drift_table_identifies_only_shifted_features(rng):
    n = 3000
    ref = pd.DataFrame(
        {"income": rng.lognormal(10, 0.5, n), "age": rng.integers(21, 70, n), "status": rng.integers(-2, 3, n)}
    )
    cur = pd.DataFrame(
        {"income": rng.lognormal(10, 0.5, n) * 0.6, "age": rng.integers(21, 70, n), "status": rng.integers(-2, 3, n)}
    )
    table = feature_drift_table(ref, cur, ["income", "age", "status"], CFG).set_index("feature")
    assert table.loc["income", "drift_detected"] and table.loc["income", "severity"] == "severe"
    assert not table.loc["age", "drift_detected"]
    assert table.loc["status", "type"] == "discrete" and table.loc["status", "test"] == "chi-square"
    assert table.loc["income", "test"] == "kolmogorov-smirnov"


def test_no_drift_scenario_flags_nothing(rng):
    n = 3000
    ref = pd.DataFrame({"x": rng.normal(size=n), "y": rng.exponential(size=n)})
    cur = pd.DataFrame({"x": rng.normal(size=n), "y": rng.exponential(size=n)})
    assert not feature_drift_table(ref, cur, ["x", "y"], CFG)["drift_detected"].any()


def test_prediction_drift(rng):
    scores = rng.beta(2, 5, 3000)
    stable = prediction_drift(scores, rng.beta(2, 5, 3000))
    shifted = prediction_drift(scores, rng.beta(4, 3, 3000))
    assert stable["prediction_psi"] < 0.1 < shifted["prediction_psi"]
    assert shifted["current_mean_score"] > shifted["reference_mean_score"]
