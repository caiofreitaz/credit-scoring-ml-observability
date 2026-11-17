"""End-to-end: full pipeline run (session fixture) and the CLI flow."""

import json

import pytest
from mlflow.tracking import MlflowClient

from src.cli import EXIT_CONTRACT_VIOLATION, EXIT_ERROR, EXIT_OK, main
from src.monitoring.logger import read_log_events
from src.monitoring.tracker import configure_mlflow


def test_pipeline_succeeds_and_blocks_invalid_batch(pipeline_run):
    summary = pipeline_run["summary"]
    assert summary["status"] == "SUCCESS"
    statuses = {b["batch"]: b for b in summary["batches"]}
    assert statuses["batch_valid"]["status"] == "accepted" and statuses["batch_valid"]["predictions_generated"] > 0
    assert statuses["batch_invalid"]["status"] == "rejected"
    assert statuses["batch_invalid"]["predictions_generated"] == 0
    assert summary["pipeline"]["n_failed"] == 0 and summary["pipeline"]["n_blocked"] == 1


def test_scenarios_are_diagnosed(pipeline_run):
    by_name = {s["scenario"]: s for s in pipeline_run["summary"]["scenarios"]}
    assert set(by_name) == {"normal", "data_drift", "concept_drift", "data_and_concept_drift"}
    assert by_name["normal"]["n_drifted_features"] == 0
    assert "LIMIT_BAL" in by_name["data_drift"]["drifted_features"]
    assert by_name["concept_drift"]["n_drifted_features"] == 0
    assert by_name["concept_drift"]["diagnosis"]["concept_drift_signal"] is True
    assert by_name["concept_drift"]["performance"]["auc_drop"] > by_name["normal"]["performance"]["auc_drop"]


def test_alerts_have_expected_severities(pipeline_run):
    alerts = pipeline_run["summary"]["alerts"]
    codes = {(a["scope"], a["code"], a["severity"]) for a in alerts}
    assert ("batch_invalid", "DATA_CONTRACT_VIOLATION", "CRITICAL") in codes
    assert ("scenario:normal", "DATA_DRIFT_NONE", "INFO") in codes
    assert any(
        a["scope"] == "scenario:data_drift"
        and a["code"].startswith("DATA_DRIFT_")
        and a["severity"] in {"WARNING", "CRITICAL"}
        for a in alerts
    )
    assert any(a["scope"] == "scenario:concept_drift" and a["code"] == "CONCEPT_DRIFT_CONFIRMED" for a in alerts)
    assert all(a["run_id"] == pipeline_run["summary"]["run_id"] for a in alerts)


def test_reports_are_generated(pipeline_run):
    reports = pipeline_run["paths"].reports_dir
    for name in ("normal", "data_drift", "concept_drift", "data_and_concept_drift"):
        html = (reports / "evidently" / f"{name}_report.html").read_text(encoding="utf-8")
        assert len(html) > 100_000 and "<html" in html.lower()
        assert (reports / "drift" / f"{name}_feature_drift.csv").exists()
    dashboard = (reports / "monitoring_dashboard.html").read_text(encoding="utf-8")
    assert pipeline_run["summary"]["run_id"] in dashboard and "data:image/png;base64" in dashboard
    assert json.loads((reports / "alerts.json").read_text(encoding="utf-8"))
    for figure in ("psi_heatmap.png", "scenario_auc.png", "invalid_batch_violations.png", "roc_reference.png"):
        assert (reports / "figures" / figure).stat().st_size > 1000


def test_logs_are_correlated_by_run_id(pipeline_run):
    run_id = pipeline_run["summary"]["run_id"]
    events = read_log_events(pipeline_run["paths"].logs_dir / "pipeline.jsonl", run_id=run_id)
    stages = {e.get("stage") for e in events}
    assert {"prepare", "training", "validation", "inference", "monitoring", "alerting"} <= stages
    assert any(e["level"] == "ERROR" and e.get("status") == "rejected" for e in events)


def test_mlflow_has_successful_and_failed_runs(pipeline_run):
    configure_mlflow(pipeline_run["paths"])
    client = MlflowClient()
    experiment = client.get_experiment_by_name(pipeline_run["config"]["mlflow"]["monitoring_experiment"])
    runs = client.search_runs(
        [experiment.experiment_id], f"tags.pipeline_run_id = '{pipeline_run['summary']['run_id']}'"
    )
    by_name = {r.info.run_name: r for r in runs}
    assert by_name["batch-batch_invalid"].info.status == "FAILED"
    assert by_name["batch-batch_valid"].info.status == "FINISHED"
    monitor = by_name["monitor-data_drift"]
    assert monitor.data.metrics["n_drifted_features"] > 0 and "psi_LIMIT_BAL" in monitor.data.metrics
    assert by_name[f"pipeline-{pipeline_run['summary']['run_id']}"].data.metrics["batches_rejected"] == 1


def test_cli_step_by_step_flow(project, capsys):
    base = ["--config", str(project["config_path"]), "--root", str(project["root"])]
    assert main([*base, "download-data"]) == EXIT_OK
    assert main([*base, "prepare-data"]) == EXIT_OK
    root = project["root"]
    valid, invalid = str(root / "data/production/batch_valid.csv"), str(root / "data/invalid/batch_invalid.csv")
    assert main([*base, "validate", "--input", str(root / "data/reference/train.csv"), "--mode", "training"]) == EXIT_OK
    assert main([*base, "validate", "--input", invalid]) == EXIT_CONTRACT_VIOLATION
    assert main([*base, "train"]) == EXIT_OK
    assert main([*base, "predict", "--input", valid]) == EXIT_OK
    assert main([*base, "predict", "--input", invalid]) == EXIT_CONTRACT_VIOLATION
    assert "Nenhuma predição foi gerada" in capsys.readouterr().out
    assert main([*base, "simulate-drift"]) == EXIT_OK
    assert main([*base, "monitor", "--scenario", "normal", "concept_drift"]) == EXIT_OK
    assert (root / "reports" / "monitoring_dashboard.html").exists()


def test_cli_run_all_reuses_existing_model(pipeline_run, capsys):
    base = ["--config", str(pipeline_run["config_path"]), "--root", str(pipeline_run["root"])]
    assert main([*base, "run-all", "--skip-download", "--skip-train"]) == EXIT_OK
    assert "RESUMO DA EXECUÇÃO" in capsys.readouterr().out


def test_cli_reports_missing_inputs(tmp_path, project):
    base = ["--config", str(project["config_path"]), "--root", str(tmp_path / "empty")]
    assert main([*base, "train"]) == EXIT_ERROR


@pytest.mark.parametrize("command", [["validate"], ["unknown"]])
def test_cli_rejects_bad_arguments(command):
    with pytest.raises(SystemExit):
        main(command)
