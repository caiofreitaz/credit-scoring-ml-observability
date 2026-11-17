"""Evidently AI reports (data drift + classification quality) per scenario."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from evidently import BinaryClassification, DataDefinition, Dataset, Report
from evidently.presets import ClassificationPreset, DataDriftPreset

_VALUE_DRIFT = re.compile(r"ValueDrift\(column=(?P<column>[^,]+),method=(?P<method>[^,]+),threshold=(?P<thr>[^)]+)\)")


def _definition(numerical: list[str], categorical: list[str], target: str | None, prediction: str) -> DataDefinition:
    classification = None
    cats = list(categorical)
    if target is not None:
        classification = [BinaryClassification(target=target, prediction_probas=prediction)]
        cats.append(target)
    return DataDefinition(
        numerical_columns=[*numerical, prediction],
        categorical_columns=cats,
        classification=classification,
    )


def build_evidently_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numerical: list[str],
    categorical: list[str],
    output_html: str | Path,
    cfg: dict[str, Any],
    threshold: float,
    target: str | None = None,
    prediction: str = "prediction",
    title: str | None = None,
) -> dict[str, Any]:
    """Run DataDriftPreset (+ ClassificationPreset when labels exist) and save the HTML.

    Returns a compact JSON-serializable summary extracted from the snapshot.
    """
    columns = [*numerical, *categorical, prediction] + ([target] if target else [])
    definition = _definition(numerical, categorical, target, prediction)
    metrics: list[Any] = [
        DataDriftPreset(method=cfg["method"], threshold=cfg["threshold"], drift_share=cfg["drift_share"])
    ]
    if target is not None:
        metrics.append(ClassificationPreset(probas_threshold=threshold))
    report = Report(metrics, metadata={"title": title or "credit-scoring monitoring"})
    snapshot = report.run(
        Dataset.from_pandas(current[columns].copy(), data_definition=definition),
        Dataset.from_pandas(reference[columns].copy(), data_definition=definition),
        name=title,
    )
    output = Path(output_html)
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(output))
    return summarize_snapshot(snapshot.dict(), cfg["threshold"])


def summarize_snapshot(snapshot: dict[str, Any], threshold: float) -> dict[str, Any]:
    """Extract per-column drift scores and headline metrics from an Evidently snapshot."""
    column_scores: dict[str, float] = {}
    drifted_count, drifted_share, method = None, None, None
    headline: dict[str, float] = {}
    for metric in snapshot.get("metrics", []):
        name = metric.get("metric_name", "")
        value = metric.get("value")
        match = _VALUE_DRIFT.match(name)
        if match:
            column_scores[match.group("column")] = float(value)
            method = match.group("method")
        elif name.startswith("DriftedColumnsCount") and isinstance(value, dict):
            drifted_count, drifted_share = value.get("count"), value.get("share")
        else:
            short = name.split("(")[0]
            if short in {"Accuracy", "Precision", "Recall", "F1Score", "RocAuc", "LogLoss"} and not isinstance(
                value, dict
            ):
                headline[short] = float(value)
    drifted_columns = sorted(c for c, score in column_scores.items() if score >= threshold)
    return {
        "method": method,
        "threshold": threshold,
        "drifted_columns_count": drifted_count,
        "drifted_columns_share": drifted_share,
        "drifted_columns": drifted_columns,
        "column_scores": {k: round(v, 5) for k, v in sorted(column_scores.items())},
        "classification": {k: round(v, 5) for k, v in headline.items()},
    }


def save_summary(summary: dict[str, Any], path: str | Path) -> None:
    """Persist an Evidently summary as JSON."""
    Path(path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
