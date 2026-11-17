"""Alerts, structured logging, stage observability, fairness and Evidently parsing."""

import json
import re

import pandas as pd
import pytest

from src.drift.evidently_reports import summarize_snapshot
from src.model.fairness import age_band, fairness_audit
from src.monitoring import alerts as al
from src.monitoring.logger import configure_logging, get_logger, new_run_id, read_log_events, set_run_id
from src.monitoring.observability import StageTracker

DRIFT = {"psi_moderate": 0.10, "psi_severe": 0.25, "calibration_pvalue_threshold": 0.01}
ALERTS = {
    "auc_drop_warning": 0.02,
    "auc_drop_critical": 0.05,
    "prediction_psi_warning": 0.10,
    "prediction_psi_critical": 0.25,
    "default_rate_shift_warning_pp": 3.0,
    "latency_ms_per_1k_warning": 1000,
}


def test_data_drift_rule_severity_levels():
    assert al.rule_data_drift("s", 0, 0, [], DRIFT).severity == al.Severity.INFO
    assert al.rule_data_drift("s", 2, 0, ["a", "b"], DRIFT).severity == al.Severity.WARNING
    severe = al.rule_data_drift("s", 1, 1, ["a", "b"], DRIFT, ["a"])
    assert severe.severity == al.Severity.CRITICAL and "a" in severe.message


@pytest.mark.parametrize(
    ("ref", "cur", "expected"), [(0.76, 0.75, None), (0.76, 0.735, "WARNING"), (0.76, 0.70, "CRITICAL")]
)
def test_performance_rule(ref, cur, expected):
    alert = al.rule_performance("s", ref, cur, ALERTS)
    assert (alert.severity.value if alert else None) == expected


@pytest.mark.parametrize(("psi", "expected"), [(0.05, None), (0.15, "WARNING"), (0.4, "CRITICAL")])
def test_prediction_drift_rule(psi, expected):
    alert = al.rule_prediction_drift("s", psi, ALERTS)
    assert (alert.severity.value if alert else None) == expected


def test_concept_rule_suspected_vs_confirmed():
    assert al.rule_concept_drift("s", 0.5, True, DRIFT) is None
    assert al.rule_concept_drift("s", 1e-5, False, DRIFT).code == "CONCEPT_DRIFT_SUSPECTED"
    assert al.rule_concept_drift("s", 1e-5, True, DRIFT).code == "CONCEPT_DRIFT_CONFIRMED"


def test_other_rules():
    assert al.rule_default_rate_shift("s", 0.22, 0.23, ALERTS) is None
    assert al.rule_default_rate_shift("s", 0.22, 0.30, ALERTS).value == pytest.approx(8.0)
    assert al.rule_latency("s", 10, ALERTS) is None
    assert al.rule_latency("s", 5000, ALERTS).code == "HIGH_INFERENCE_LATENCY"
    assert al.rule_contract_violation("b", 3, ["R05"]).severity == al.Severity.CRITICAL
    assert al.rule_stage_failure("x", "boom").code == "PIPELINE_STAGE_FAILED"


def test_alert_manager_counts_logs_and_saves(tmp_path):
    configure_logging(tmp_path)
    set_run_id("run-xyz")
    manager = al.AlertManager()
    manager.extend([al.rule_data_drift("a", 0, 0, [], DRIFT), al.rule_contract_violation("b", 1, ["R04"])])
    assert manager.counts() == {"INFO": 1, "WARNING": 0, "CRITICAL": 1}
    assert manager.counts("b")["CRITICAL"] == 1 and len(manager.for_scope("a")) == 1
    saved = json.loads(manager.save(tmp_path / "alerts.json").read_text())
    assert saved[1]["run_id"] == "run-xyz" and saved[1]["severity"] == "CRITICAL"
    events = read_log_events(tmp_path / "pipeline.jsonl")
    assert any(e["level"] == "ERROR" and e["alert"]["code"] == "DATA_CONTRACT_VIOLATION" for e in events)


def test_structured_logs_are_json_lines_with_run_id(tmp_path):
    configure_logging(tmp_path)
    run_id = new_run_id()
    assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{8}", run_id)
    set_run_id(run_id)
    get_logger().info("hello", extra={"stage": "unit", "rows": 3})
    set_run_id("other")
    get_logger().warning("other run")
    mine = read_log_events(tmp_path / "pipeline.jsonl", run_id=run_id)
    assert len(mine) == 1 and mine[0]["stage"] == "unit" and mine[0]["rows"] == 3
    assert read_log_events(tmp_path / "missing.jsonl") == []


def test_stage_tracker_success_failure_and_blocked(tmp_path):
    configure_logging(tmp_path)
    manager = al.AlertManager()
    tracker = StageTracker(manager)
    with tracker.stage("ok", records_in=5) as record:
        record.records_out = 5
    with pytest.raises(RuntimeError), tracker.stage("broken"):
        raise RuntimeError("boom")
    with tracker.stage("gate") as record:
        record.status = "BLOCKED"
    summary = tracker.summary()
    assert [s["status"] for s in summary["stages"]] == ["SUCCESS", "FAILED", "BLOCKED"]
    assert summary["n_failed"] == 1 and summary["n_blocked"] == 1
    assert summary["stages"][0]["duration_s"] >= 0 and summary["stages"][0]["resources"]["process_rss_mb"] > 0
    assert manager.alerts[-1].code == "PIPELINE_STAGE_FAILED"


def test_fairness_audit_suppresses_small_groups():
    n = 400
    frame = pd.DataFrame(
        {
            "SEX": [1, 2] * (n // 2),
            "MARRIAGE": [1] * (n - 10) + [3] * 10,
            "EDUCATION": [1, 2, 3, 4] * (n // 4),
            "AGE": list(range(21, 61)) * (n // 40),
            "default_next_month": [0, 1, 0, 0] * (n // 4),
            "prediction": [0.2, 0.7, 0.4, 0.1] * (n // 4),
            "predicted_label": [0, 1, 0, 0] * (n // 4),
        }
    )
    audit = fairness_audit(frame, "default_next_month", min_group_size=50)
    groups = audit["groups"]
    assert groups.loc[groups["group"] == "outros", "suppressed"].any()
    assert {f["attribute"] for f in audit["findings"]} == {"SEX", "MARRIAGE", "EDUCATION", "AGE_BAND"}
    assert age_band(pd.Series([25, 35, 50, 70])).tolist() == ["até 29", "30-44", "45-59", "60+"]


def test_summarize_evidently_snapshot():
    snapshot = {
        "metrics": [
            {"metric_name": "DriftedColumnsCount(drift_share=0.25)", "value": {"count": 1.0, "share": 0.5}},
            {"metric_name": "ValueDrift(column=LIMIT_BAL,method=PSI,threshold=0.1)", "value": 1.2},
            {"metric_name": "ValueDrift(column=AGE,method=PSI,threshold=0.1)", "value": 0.02},
            {"metric_name": "RocAuc(classification_name=default)", "value": 0.7},
        ]
    }
    summary = summarize_snapshot(snapshot, 0.1)
    assert summary["drifted_columns"] == ["LIMIT_BAL"] and summary["drifted_columns_count"] == 1.0
    assert summary["method"] == "PSI" and summary["classification"]["RocAuc"] == 0.7
