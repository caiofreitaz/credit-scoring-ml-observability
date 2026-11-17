"""Model loading and gated batch inference."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import ProjectPaths, model_feature_columns
from src.model.train import METADATA_FILE, MODEL_FILE
from src.monitoring.alerts import AlertManager
from src.monitoring.logger import get_logger
from src.validation.gate import ValidationResult, enforce_contract

PREDICTIONS_DIR = "predictions"


@dataclass
class PredictionResult:
    """Scored batch plus operational measurements."""

    batch_name: str
    scored: pd.DataFrame
    validation: ValidationResult
    latency_ms: float
    latency_ms_per_1k: float


def load_model(paths: ProjectPaths) -> tuple[Pipeline, dict[str, Any]]:
    """Load the persisted pipeline and its metadata."""
    model_path = paths.model_dir / MODEL_FILE
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run: python -m src.cli train")
    metadata = json.loads((paths.model_dir / METADATA_FILE).read_text(encoding="utf-8"))
    return joblib.load(model_path), metadata


def score_frame(frame: pd.DataFrame, pipeline: Pipeline, metadata: dict[str, Any], config: dict) -> pd.DataFrame:
    """Add probability and decision columns to an already-validated frame."""
    features = model_feature_columns(config)
    scored = frame.copy()
    scored["prediction"] = pipeline.predict_proba(frame[features])[:, 1]
    scored["predicted_label"] = (scored["prediction"] >= metadata["threshold"]).astype(int)
    scored["model_version"] = metadata["model_version"]
    return scored


def predict_batch(
    frame: pd.DataFrame,
    batch_name: str,
    config: dict[str, Any],
    paths: ProjectPaths,
    alert_manager: AlertManager | None = None,
    pipeline: Pipeline | None = None,
    metadata: dict[str, Any] | None = None,
) -> PredictionResult:
    """Validate the batch with the inference contract and score it.

    Raises ``DataContractViolation`` (and therefore produces NO predictions) when
    the batch is invalid.
    """
    validation = enforce_contract(frame, "inference", batch_name, paths, alert_manager)
    if pipeline is None or metadata is None:
        pipeline, metadata = load_model(paths)
    start = time.perf_counter()
    scored = score_frame(frame, pipeline, metadata, config)
    latency_ms = (time.perf_counter() - start) * 1000
    per_1k = latency_ms / max(len(frame), 1) * 1000

    output_dir = paths.production_dir / PREDICTIONS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    scored[["client_id", "prediction", "predicted_label", "model_version"]].to_csv(
        output_dir / f"{batch_name}_predictions.csv", index=False, lineterminator="\n"
    )
    get_logger().info(
        f"batch '{batch_name}' scored",
        extra={
            "stage": "inference",
            "batch": batch_name,
            "rows": len(frame),
            "latency_ms": round(latency_ms, 3),
            "latency_ms_per_1k": round(per_1k, 3),
            "model_version": metadata["model_version"],
            "predicted_positive_rate": round(float(scored["predicted_label"].mean()), 4),
        },
    )
    return PredictionResult(batch_name, scored, validation, latency_ms, per_1k)
