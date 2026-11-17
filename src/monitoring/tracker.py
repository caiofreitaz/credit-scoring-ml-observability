"""MLflow tracking helpers (local SQLite backend + local artifact store)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from src.config import ProjectPaths


def tracking_uri(paths: ProjectPaths) -> str:
    """SQLite tracking URI that works on Windows and Linux (forward slashes)."""
    return "sqlite:///" + paths.mlflow_db.resolve().as_posix()


def configure_mlflow(paths: ProjectPaths) -> str:
    """Point MLflow to the project's local tracking database."""
    paths.mlflow_artifacts.mkdir(parents=True, exist_ok=True)
    uri = tracking_uri(paths)
    mlflow.set_tracking_uri(uri)
    return uri


def ensure_experiment(name: str, paths: ProjectPaths) -> str:
    """Create (if needed) and activate an experiment with a project-local artifact root."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(name)
    if experiment is None:
        location = (paths.mlflow_artifacts.resolve() / name).as_uri()
        experiment_id = client.create_experiment(
            name,
            artifact_location=location,
            # Opens the experiment in MLflow's "Model training" view (runs table) instead of GenAI.
            tags={"mlflow.experimentKind": "custom_model_development"},
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


def log_numeric_metrics(metrics: Mapping[str, Any], prefix: str = "") -> int:
    """Log only finite numeric values; returns how many metrics were logged."""
    clean = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            clean[f"{prefix}{key}"] = float(value)
    if clean:
        mlflow.log_metrics(clean)
    return len(clean)
