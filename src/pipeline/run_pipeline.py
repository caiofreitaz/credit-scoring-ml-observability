"""End-to-end orchestration with stage-level observability."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from src.config import ProjectPaths, load_config, resolve_paths
from src.data.batches import INVALID_BATCH_FILE, VALID_BATCH_FILE, write_demo_batches
from src.data.eda import run_eda
from src.data.ingest import download_dataset, load_raw_dataset
from src.data.prepare import load_partition, prepare_data
from src.drift.monitor import monitor_scenario
from src.drift.simulation import SCENARIOS, build_scenarios, load_scenario, write_scenarios
from src.model.fairness import fairness_audit
from src.model.predict import load_model, predict_batch
from src.model.train import REFERENCE_SCORED_FILE, coefficient_table, train_baseline
from src.monitoring import plots
from src.monitoring.alerts import AlertManager
from src.monitoring.dashboard import build_dashboard
from src.monitoring.logger import configure_logging, get_logger, get_run_id, new_run_id, set_run_id
from src.monitoring.observability import StageTracker, resource_snapshot
from src.monitoring.tracker import configure_mlflow, ensure_experiment, log_numeric_metrics
from src.validation.gate import DataContractViolation, read_batch

FIGURES_DIR = "figures"


@dataclass
class PipelineContext:
    """Shared state of one pipeline execution."""

    config: dict[str, Any]
    paths: ProjectPaths
    run_id: str
    alerts: AlertManager = field(default_factory=AlertManager)
    tracker: StageTracker | None = None

    def __post_init__(self) -> None:
        if self.tracker is None:
            self.tracker = StageTracker(self.alerts)


def create_context(config_path: str | None = None, root: str | Path | None = None) -> PipelineContext:
    """Load configuration, create folders, configure logging and bind a run id."""
    config = load_config(config_path)
    paths = resolve_paths(config, root)
    paths.ensure()
    configure_logging(paths.logs_dir)
    run_id = new_run_id()
    set_run_id(run_id)
    get_logger().info("pipeline context created", extra={"stage": "setup", "root": str(paths.root)})
    return PipelineContext(config=config, paths=paths, run_id=run_id)


# ---------------------------------------------------------------------------
# Individual stages (each also exposed through the CLI)
# ---------------------------------------------------------------------------


def stage_download(ctx: PipelineContext, force: bool = False) -> dict[str, Any]:
    with ctx.tracker.stage("download") as record:
        manifest = download_dataset(ctx.config, ctx.paths, force=force)
        record.records_out = manifest["rows"]
        record.details = {"source": manifest["source_name"], "sha256": manifest["canonical_sha256"]}
    return manifest


def stage_prepare(ctx: PipelineContext) -> dict[str, Any]:
    with ctx.tracker.stage("prepare") as record:
        raw = load_raw_dataset(ctx.paths)
        record.records_in = len(raw)
        manifest = prepare_data(raw, ctx.config, ctx.paths)
        record.records_out = manifest["rows_total"]
    with ctx.tracker.stage("eda") as record:
        train = load_partition(ctx.paths, "train")
        record.records_in = len(train)
        run_eda(train, ctx.config, ctx.paths)
    with ctx.tracker.stage("demo_batches") as record:
        pool = load_partition(ctx.paths, "production")
        record.details = write_demo_batches(pool, ctx.config, ctx.paths)
    return manifest


def stage_train(ctx: PipelineContext) -> dict[str, Any]:
    with ctx.tracker.stage("train") as record:
        result = train_baseline(ctx.config, ctx.paths)
        record.records_in = result.metadata["params"]["n_train"]
        record.details = {"model_version": result.metadata["model_version"]}
    with ctx.tracker.stage("training_reports"):
        _training_figures(ctx, result.metadata, result.reference_scored, result.pipeline)
    with ctx.tracker.stage("fairness_audit") as record:
        audit = fairness_audit(result.reference_scored, ctx.config["features"]["target"])
        output = ctx.paths.reports_dir / "fairness"
        output.mkdir(parents=True, exist_ok=True)
        audit["groups"].to_csv(output / "fairness_by_group.csv", index=False)
        (output / "fairness_findings.json").write_text(
            json.dumps(
                {"min_group_size": audit["min_group_size"], "findings": audit["findings"]}, indent=2, ensure_ascii=False
            ),
            encoding="utf-8",
        )
        plots.plot_fairness(audit["groups"], ctx.paths.reports_dir / FIGURES_DIR / "fairness_approval_rate.png")
        record.details = {"findings": audit["findings"]}
    return result.metadata


def _training_figures(ctx: PipelineContext, metadata: dict, reference_scored: pd.DataFrame, pipeline) -> None:
    figures = ctx.paths.reports_dir / FIGURES_DIR
    target = ctx.config["features"]["target"]
    plots.plot_roc(
        reference_scored[target],
        reference_scored["prediction"],
        figures / "roc_reference.png",
        metadata["reference_metrics"]["roc_auc"],
    )
    plots.plot_confusion(
        metadata["reference_metrics"],
        figures / "confusion_matrix_reference.png",
        f"Matriz de confusão - referência (limiar {metadata['threshold']})",
    )
    plots.plot_coefficients(coefficient_table(pipeline), figures / "model_coefficients.png")
    (ctx.paths.reports_dir / "training_metrics.json").write_text(
        json.dumps(
            {
                k: metadata[k]
                for k in (
                    "model_version",
                    "threshold",
                    "candidates",
                    "validation_metrics",
                    "reference_metrics",
                    "params",
                )
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_batch_inference(ctx: PipelineContext, batch_path: str | Path, batch_name: str) -> dict[str, Any]:
    """Gated inference of a batch file, tracked in MLflow as FINISHED or FAILED."""
    frame = read_batch(batch_path)
    configure_mlflow(ctx.paths)
    ensure_experiment(ctx.config["mlflow"]["monitoring_experiment"], ctx.paths)
    nested = mlflow.active_run() is not None
    with ctx.tracker.stage(f"inference_{batch_name}", records_in=len(frame)) as record:
        run = mlflow.start_run(run_name=f"batch-{batch_name}", nested=nested)
        mlflow.set_tags({"pipeline_run_id": get_run_id(), "stage": "inference", "batch": batch_name})
        try:
            result = predict_batch(frame, batch_name, ctx.config, ctx.paths, ctx.alerts)
        except DataContractViolation as violation:
            res = violation.result
            record.status = "BLOCKED"
            record.records_out = 0
            record.details = {"expected_failure": True, "rules_violated": res.rules_violated}
            mlflow.set_tags({"batch_status": "rejected", "rules_violated": ",".join(res.rules_violated)})
            log_numeric_metrics(
                {
                    "rows": res.n_rows,
                    "contract_failures": res.n_failures,
                    "rules_violated": len(res.rules_violated),
                    "predictions_generated": 0,
                }
            )
            mlflow.log_artifact(str(ctx.paths.reports_dir / "validation" / f"{batch_name}_validation.json"))
            mlflow.log_artifact(str(ctx.paths.reports_dir / "validation" / f"{batch_name}_failure_cases.csv"))
            mlflow.end_run(status="FAILED")
            plots.plot_violations(res.summary, ctx.paths.reports_dir / FIGURES_DIR / "invalid_batch_violations.png")
            return _save_batch_result(
                {
                    "batch": batch_name,
                    "status": "rejected",
                    **res.to_dict(),
                    "predictions_generated": 0,
                    "mlflow_run_id": run.info.run_id,
                },
                ctx,
            )
        except Exception:
            mlflow.end_run(status="FAILED")
            raise
        record.records_out = len(result.scored)
        mlflow.set_tags({"batch_status": "accepted", "model_version": result.scored["model_version"].iloc[0]})
        log_numeric_metrics(
            {
                "rows": len(frame),
                "contract_failures": 0,
                "predictions_generated": len(result.scored),
                "latency_ms": result.latency_ms,
                "latency_ms_per_1k": result.latency_ms_per_1k,
                "predicted_positive_rate": float(result.scored["predicted_label"].mean()),
            }
        )
        mlflow.end_run(status="FINISHED")
    return _save_batch_result(
        {
            "batch": batch_name,
            "status": "accepted",
            **result.validation.to_dict(),
            "predictions_generated": len(result.scored),
            "latency_ms": round(result.latency_ms, 3),
            "predicted_positive_rate": round(float(result.scored["predicted_label"].mean()), 4),
            "mlflow_run_id": run.info.run_id,
        },
        ctx,
    )


def _save_batch_result(result: dict[str, Any], ctx: PipelineContext) -> dict[str, Any]:
    output = ctx.paths.reports_dir / "validation" / f"{result['batch']}_inference_result.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return result


def load_batch_results(ctx: PipelineContext) -> list[dict[str, Any]]:
    """Load the latest inference results of the demo batches (used by standalone commands)."""
    results = []
    for name in ("batch_valid", "batch_invalid"):
        path = ctx.paths.reports_dir / "validation" / f"{name}_inference_result.json"
        if path.exists():
            results.append(json.loads(path.read_text(encoding="utf-8")))
    return results


def stage_simulate(ctx: PipelineContext) -> dict[str, Any]:
    with ctx.tracker.stage("simulate_drift") as record:
        pool = load_partition(ctx.paths, "production")
        record.records_in = len(pool)
        manifest = write_scenarios(build_scenarios(pool, ctx.config), ctx.paths)
        record.records_out = sum(v["rows"] for v in manifest.values())
    return manifest


def stage_monitor(ctx: PipelineContext, scenarios: tuple[str, ...] = SCENARIOS) -> list[dict[str, Any]]:
    pipeline, metadata = load_model(ctx.paths)
    reference_scored = pd.read_csv(ctx.paths.reference_dir / REFERENCE_SCORED_FILE, dtype={"client_id": str})
    manifest_path = ctx.paths.production_dir / "scenarios" / "scenario_manifest.json"
    descriptions = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    configure_mlflow(ctx.paths)
    ensure_experiment(ctx.config["mlflow"]["monitoring_experiment"], ctx.paths)
    summaries = []
    for name in scenarios:
        features, outcomes = load_scenario(name, ctx.paths)
        with ctx.tracker.stage(f"monitor_{name}", records_in=len(features)) as record:
            summary = monitor_scenario(
                name,
                features,
                outcomes,
                reference_scored,
                pipeline,
                metadata,
                ctx.config,
                ctx.paths,
                ctx.alerts,
                descriptions.get(name, {}).get("description", ""),
            )
            record.records_out = summary["rows"]
            record.details = {"diagnosis": summary["diagnosis"]["diagnosis"]}
        summaries.append(summary)
    return summaries


def _scenario_figures(ctx: PipelineContext, summaries: list[dict], metadata: dict) -> None:
    figures = ctx.paths.reports_dir / FIGURES_DIR
    drift_dir = ctx.paths.reports_dir / "drift"
    psi = pd.DataFrame(
        {
            s["scenario"]: pd.read_csv(drift_dir / f"{s['scenario']}_feature_drift.csv").set_index("feature")["psi"]
            for s in summaries
        }
    )
    psi.loc["prediction (score)"] = [s["prediction_drift"]["prediction_psi"] for s in summaries]
    plots.plot_psi_heatmap(
        psi, figures / "psi_heatmap.png", ctx.config["drift"]["psi_moderate"], ctx.config["drift"]["psi_severe"]
    )
    labeled = [s for s in summaries if s["performance"]]
    if labeled:
        plots.plot_scenario_auc(labeled, metadata["reference_metrics"]["roc_auc"], figures / "scenario_auc.png")
    calib = {
        n: pd.read_csv(drift_dir / f"{n}_calibration_bins.csv")
        for n in ("data_drift", "concept_drift")
        if (drift_dir / f"{n}_calibration_bins.csv").exists()
    }
    if calib:
        plots.plot_calibration_bins(calib, figures / "calibration_by_score_decile.png")
    plots.plot_alert_counts(summaries, figures / "alerts_by_scenario.png")
    reference = pd.read_csv(ctx.paths.reference_dir / REFERENCE_SCORED_FILE, dtype={"client_id": str})
    shifted, _ = (
        load_scenario("data_drift", ctx.paths) if "data_drift" in [s["scenario"] for s in summaries] else (None, None)
    )
    if shifted is not None:
        plots.plot_distribution_shift(
            reference["LIMIT_BAL"],
            shifted["LIMIT_BAL"],
            figures / "shift_limit_bal.png",
            "LIMIT_BAL: referência vs produção (data drift)",
            "Limite de crédito (NT$)",
        )
        plots.plot_distribution_shift(
            reference["AGE"],
            shifted["AGE"],
            figures / "shift_age.png",
            "AGE: referência vs produção (data drift)",
            "Idade (anos)",
        )
        pipeline, _ = load_model(ctx.paths)
        from src.model.predict import score_frame

        scored = score_frame(shifted, pipeline, metadata, ctx.config)
        plots.plot_distribution_shift(
            reference["prediction"],
            scored["prediction"],
            figures / "shift_prediction.png",
            "Score do modelo: referência vs produção (prediction drift)",
            "Probabilidade prevista de inadimplência",
        )


def finalize(ctx: PipelineContext, batches: list[dict], summaries: list[dict], status: str) -> dict[str, Any]:
    """Write the consolidated summary, alerts file and HTML dashboard."""
    metadata_path = ctx.paths.model_dir / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    if summaries and metadata:
        with ctx.tracker.stage("scenario_figures"):
            _scenario_figures(ctx, summaries, metadata)
    fairness_path = ctx.paths.reports_dir / "fairness" / "fairness_findings.json"
    summary = {
        "run_id": ctx.run_id,
        "status": status,
        "generated_at": datetime.now(UTC).isoformat(),
        "model": {
            k: metadata.get(k)
            for k in (
                "model_name",
                "model_version",
                "registered_model_name",
                "registered_model_version",
                "threshold",
                "reference_metrics",
                "trained_at",
            )
        },
        "batches": batches,
        "scenarios": summaries,
        "alerts_total": ctx.alerts.counts(),
        "alerts": [a.to_dict() for a in ctx.alerts.alerts],
        "pipeline": ctx.tracker.summary(),
        "fairness": json.loads(fairness_path.read_text(encoding="utf-8")) if fairness_path.exists() else None,
        "resources": resource_snapshot(),
    }
    ctx.alerts.save(ctx.paths.reports_dir / "alerts.json")
    (ctx.paths.reports_dir / "monitoring_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    build_dashboard(summary, ctx.paths)
    return summary


def run_full_pipeline(ctx: PipelineContext, skip_download: bool = False, skip_train: bool = False) -> dict[str, Any]:
    """Run every stage; failures are logged, alerted and recorded before re-raising."""
    logger = get_logger()
    batches: list[dict] = []
    summaries: list[dict] = []
    status = "SUCCESS"
    try:
        if not skip_download:
            stage_download(ctx)
        stage_prepare(ctx)
        if not skip_train:
            stage_train(ctx)
        configure_mlflow(ctx.paths)
        ensure_experiment(ctx.config["mlflow"]["monitoring_experiment"], ctx.paths)
        with mlflow.start_run(run_name=f"pipeline-{ctx.run_id}") as parent:
            mlflow.set_tags({"pipeline_run_id": ctx.run_id, "stage": "pipeline"})
            batches.append(run_batch_inference(ctx, ctx.paths.production_dir / VALID_BATCH_FILE, "batch_valid"))
            batches.append(run_batch_inference(ctx, ctx.paths.invalid_dir / INVALID_BATCH_FILE, "batch_invalid"))
            stage_simulate(ctx)
            summaries = stage_monitor(ctx)
            stage_summary = ctx.tracker.summary()
            counts = ctx.alerts.counts()
            log_numeric_metrics(
                {
                    "batches_accepted": sum(b["status"] == "accepted" for b in batches),
                    "batches_rejected": sum(b["status"] == "rejected" for b in batches),
                    "stages_success": stage_summary["n_success"],
                    "stages_failed": stage_summary["n_failed"],
                    "stages_blocked": stage_summary["n_blocked"],
                    "alerts_info": counts["INFO"],
                    "alerts_warning": counts["WARNING"],
                    "alerts_critical": counts["CRITICAL"],
                    **{f"duration_s_{r['name']}": r["duration_s"] for r in stage_summary["stages"]},
                    **resource_snapshot(),
                }
            )
            parent_id = parent.info.run_id
        logger.info("pipeline finished", extra={"stage": "pipeline", "status": status, "mlflow_run_id": parent_id})
    except Exception as exc:
        status = "FAILED"
        logger.exception("pipeline failed", extra={"stage": "pipeline", "status": status, "error": str(exc)})
        finalize(ctx, batches, summaries, status)
        raise
    return finalize(ctx, batches, summaries, status)
