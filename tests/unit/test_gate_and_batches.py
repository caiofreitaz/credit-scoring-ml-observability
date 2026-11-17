"""Validation gate side effects and the demo batches."""

import json

import pytest

from src.data.batches import INJECTED_VIOLATIONS, build_invalid_batch, build_valid_batch, to_inference_frame
from src.data.prepare import clean_dataset
from src.monitoring.alerts import AlertManager
from src.monitoring.logger import read_log_events
from src.validation.gate import DataContractViolation, enforce_contract, read_batch, validate_batch


@pytest.fixture()
def pool(synthetic_raw):
    cleaned, _ = clean_dataset(synthetic_raw.head(600), "salt")
    return cleaned


def test_valid_demo_batch_is_accepted(pool, base_config):
    batch = build_valid_batch(pool, base_config, size=100)
    assert validate_batch(batch, "inference", "valid").is_valid
    assert {"SEX", "MARRIAGE", "default_next_month"}.isdisjoint(batch.columns)


def test_invalid_demo_batch_violates_every_injected_rule(pool, base_config):
    result = validate_batch(build_invalid_batch(pool, base_config, size=60), "inference", "invalid")
    expected = {desc.split(":")[0].split(" + ")[0] for _, _, desc in INJECTED_VIOLATIONS}
    assert not result.is_valid
    assert expected <= set(result.rules_violated)


def test_enforce_contract_accepts_and_writes_report(pool, base_config, project):
    batch = to_inference_frame(pool.head(20), base_config)
    result = enforce_contract(batch, "inference", "ok_batch", project["paths"])
    report = json.loads((project["paths"].reports_dir / "validation" / "ok_batch_validation.json").read_text())
    assert result.is_valid and report["is_valid"] is True


def test_enforce_contract_blocks_quarantines_logs_and_alerts(pool, base_config, project):
    paths = project["paths"]
    alerts = AlertManager()
    batch = build_invalid_batch(pool, base_config, size=60)
    with pytest.raises(DataContractViolation) as excinfo:
        enforce_contract(batch, "inference", "bad_batch", paths, alerts)
    assert excinfo.value.result.n_failures > 0
    assert (paths.reports_dir / "validation" / "bad_batch_failure_cases.csv").exists()
    assert list(paths.quarantine_dir.glob("bad_batch__*.csv"))
    assert alerts.alerts[0].code == "DATA_CONTRACT_VIOLATION"
    assert alerts.alerts[0].severity.value == "CRITICAL"
    errors = [e for e in read_log_events(paths.logs_dir / "pipeline.jsonl") if e["level"] == "ERROR"]
    assert any(e.get("status") == "rejected" and e.get("batch") == "bad_batch" for e in errors)


def test_read_batch_keeps_numeric_looking_ids_as_strings(tmp_path):
    path = tmp_path / "b.csv"
    path.write_text("client_id,AGE\n0123456789012345,30\n", encoding="utf-8")
    assert read_batch(path)["client_id"].iloc[0] == "0123456789012345"
