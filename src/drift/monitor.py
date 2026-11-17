"""Scenario monitoring: gated inference, drift statistics, concept evidence, alerts, tracking."""

from __future__ import annotations

import json
from typing import Any

import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import ProjectPaths, model_feature_columns
from src.drift.concept import calibration_shift_test, compare_performance, diagnose
from src.drift.evidently_reports import build_evidently_report
from src.drift.statistics import feature_drift_table, prediction_drift
from src.model.metrics import bootstrap_auc_ci, classification_metrics
from src.model.predict import predict_batch
from src.monitoring.alerts import (
    AlertManager,
    rule_concept_drift,
    rule_data_drift,
    rule_default_rate_shift,
    rule_latency,
    rule_performance,
    rule_prediction_drift,
)
from src.monitoring.logger import get_logger, get_run_id
from src.monitoring.tracker import log_numeric_metrics
from src.validation.gate import enforce_contract

DRIFT_DIR = "drift"
EVIDENTLY_DIR = "evidently"


def split_feature_types(config: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return (numerical, categorical) model features for drift tooling."""
    feats = config["features"]
    categorical = [*feats["categorical"], *feats["ordinal_payment_status"]]
    numerical = [c for c in model_feature_columns(config) if c not in categorical]
    return numerical, categorical


def monitor_scenario(
    name: str,
    features: pd.DataFrame,
    outcomes: pd.DataFrame | None,
    reference_scored: pd.DataFrame,
    pipeline: Pipeline,
    metadata: dict[str, Any],
    config: dict[str, Any],
    paths: ProjectPaths,
    alert_manager: AlertManager,
    description: str = "",
) -> dict[str, Any]:
    """Full monitoring cycle for one production batch. Returns a JSON-serializable summary."""
    logger = get_logger()
    drift_cfg, alert_cfg = config["drift"], config["alerts"]
    target = config["features"]["target"]
    model_features = model_feature_columns(config)
    numerical, categorical = split_feature_types(config)
    drift_dir = paths.reports_dir / DRIFT_DIR
    drift_dir.mkdir(parents=True, exist_ok=True)
    scope = f"scenario:{name}"

    with mlflow.start_run(run_name=f"monitor-{name}", nested=mlflow.active_run() is not None):
        mlflow.set_tags(
            {
                "pipeline_run_id": get_run_id(),
                "stage": "monitoring",
                "scenario": name,
                "model_version": metadata["model_version"],
            }
        )

        # 1) Inference time: contract gate + scoring (no labels yet).
        prediction = predict_batch(features, f"scenario_{name}", config, paths, alert_manager, pipeline, metadata)
        scored = prediction.scored
        drift_table = feature_drift_table(reference_scored, scored, model_features, drift_cfg)
        drift_table.to_csv(drift_dir / f"{name}_feature_drift.csv", index=False)
        drifted = drift_table[drift_table["drift_detected"]]
        n_severe = int((drifted["severity"] == "severe").sum())
        n_moderate = int((drifted["severity"] == "moderate").sum())
        pred_drift = prediction_drift(
            reference_scored["prediction"].to_numpy(), scored["prediction"].to_numpy(), drift_cfg["psi_bins"]
        )

        severe_features = drifted.loc[drifted["severity"] == "severe", "feature"].tolist()
        alerts = [rule_data_drift(scope, n_moderate, n_severe, drifted["feature"].tolist(), drift_cfg, severe_features)]
        alerts.append(rule_prediction_drift(scope, pred_drift["prediction_psi"], alert_cfg))
        alerts.append(rule_latency(scope, prediction.latency_ms_per_1k, alert_cfg))

        # 2) Outcome maturation: delayed labels arrive and are validated before use.
        performance: dict[str, Any] = {}
        calibration: dict[str, Any] = {"calibration_pvalue": 1.0}
        labeled = None
        if outcomes is not None:
            enforce_contract(outcomes, "outcome", f"scenario_{name}_outcomes", paths, alert_manager)
            labeled = scored.merge(outcomes, on=config["features"]["id_column"], how="inner")
        if labeled is not None and len(labeled) >= drift_cfg["min_labeled_rows"]:
            ref_metrics = metadata["reference_metrics"]
            cur_metrics = classification_metrics(labeled[target], labeled["prediction"], metadata["threshold"])
            ci_low, ci_high = bootstrap_auc_ci(labeled[target], labeled["prediction"], seed=config["project"]["seed"])
            calibration = calibration_shift_test(
                reference_scored["prediction"].to_numpy(),
                reference_scored[target].to_numpy(),
                labeled["prediction"].to_numpy(),
                labeled[target].to_numpy(),
                drift_cfg["calibration_bins"],
            )
            calibration["table"].to_csv(drift_dir / f"{name}_calibration_bins.csv", index=False)
            performance = {
                "current": cur_metrics,
                "reference": ref_metrics,
                "deltas": compare_performance(ref_metrics, cur_metrics),
                "roc_auc_ci95": [round(ci_low, 4), round(ci_high, 4)],
                "auc_drop": round(ref_metrics["roc_auc"] - cur_metrics["roc_auc"], 5),
                "labeled_rows": len(labeled),
            }
            perf_alert = rule_performance(scope, ref_metrics["roc_auc"], cur_metrics["roc_auc"], alert_cfg)
            alerts.append(perf_alert)
            alerts.append(
                rule_default_rate_shift(
                    scope, ref_metrics["observed_default_rate"], cur_metrics["observed_default_rate"], alert_cfg
                )
            )
            alerts.append(
                rule_concept_drift(scope, calibration["calibration_pvalue"], perf_alert is not None, drift_cfg)
            )

        diagnosis = diagnose(
            len(drifted),
            pred_drift["prediction_psi"],
            performance.get("auc_drop", 0.0),
            calibration["calibration_pvalue"],
            drift_cfg,
            alert_cfg,
        )
        alert_manager.extend([a for a in alerts if a is not None])
        alert_counts = alert_manager.counts(scope)

        # 3) Evidently visual report (reference vs current, with labels when available).
        evidently_path = paths.reports_dir / EVIDENTLY_DIR / f"{name}_report.html"
        current_for_report = labeled if labeled is not None else scored
        evidently_summary = build_evidently_report(
            reference_scored,
            current_for_report,
            numerical,
            categorical,
            evidently_path,
            config["evidently"],
            metadata["threshold"],
            target=target if labeled is not None else None,
            title=f"Credit scoring monitoring - {name}",
        )

        summary = {
            "scenario": name,
            "description": description,
            "run_id": get_run_id(),
            "model_version": metadata["model_version"],
            "rows": len(scored),
            "latency_ms": round(prediction.latency_ms, 3),
            "latency_ms_per_1k": round(prediction.latency_ms_per_1k, 3),
            "predicted_positive_rate": round(float(scored["predicted_label"].mean()), 4),
            "n_features_monitored": len(model_features),
            "n_drifted_features": len(drifted),
            "n_moderate_drift": n_moderate,
            "n_severe_drift": n_severe,
            "drifted_features": drifted["feature"].tolist(),
            "prediction_drift": pred_drift,
            "performance": performance,
            "calibration": {k: v for k, v in calibration.items() if k != "table"},
            "diagnosis": diagnosis,
            "alerts": alert_counts,
            "evidently": {**evidently_summary, "html": evidently_path.relative_to(paths.root).as_posix()},
        }
        (drift_dir / f"{name}_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        metrics = {
            "rows": summary["rows"],
            "latency_ms": summary["latency_ms"],
            "latency_ms_per_1k": summary["latency_ms_per_1k"],
            "predicted_positive_rate": summary["predicted_positive_rate"],
            "n_drifted_features": summary["n_drifted_features"],
            "n_moderate_drift": n_moderate,
            "n_severe_drift": n_severe,
            "share_drifted_features": len(drifted) / len(model_features),
            "evidently_drifted_columns": evidently_summary["drifted_columns_count"] or 0,
            "alerts_info": alert_counts["INFO"],
            "alerts_warning": alert_counts["WARNING"],
            "alerts_critical": alert_counts["CRITICAL"],
            **pred_drift,
            "calibration_pvalue": calibration["calibration_pvalue"],
            "calibration_mean_abs_gap_pp": calibration.get("calibration_mean_abs_gap_pp", 0.0),
        }
        if performance:
            metrics["auc_drop"] = performance["auc_drop"]
            log_numeric_metrics(performance["current"], prefix="current_")
        for _, row in drift_table.iterrows():
            metrics[f"psi_{row['feature']}"] = row["psi"]
        log_numeric_metrics(metrics)
        mlflow.set_tags({"diagnosis": diagnosis["diagnosis"], "batch_status": "accepted"})
        mlflow.log_artifact(str(drift_dir / f"{name}_feature_drift.csv"), artifact_path="drift")
        mlflow.log_artifact(str(drift_dir / f"{name}_summary.json"), artifact_path="drift")
        mlflow.log_artifact(str(evidently_path), artifact_path="evidently")

    logger.info(
        f"scenario '{name}' monitored",
        extra={
            "stage": "monitoring",
            "scenario": name,
            "diagnosis": diagnosis["diagnosis"],
            "n_drifted_features": summary["n_drifted_features"],
            "prediction_psi": pred_drift["prediction_psi"],
            "auc_drop": performance.get("auc_drop"),
            "alerts": alert_counts,
        },
    )
    return summary
