"""Baseline credit-scoring model: training, selection, evaluation, persistence and tracking."""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import ProjectPaths, model_feature_columns
from src.data.prepare import load_partition
from src.model.features import ENGINEERED, CreditFeatureEngineer
from src.model.metrics import best_f1_threshold, classification_metrics
from src.monitoring.logger import get_logger, get_run_id
from src.monitoring.tracker import configure_mlflow, ensure_experiment, log_numeric_metrics
from src.validation.gate import enforce_contract

MODEL_FILE = "model.joblib"
METADATA_FILE = "model_metadata.json"
REFERENCE_SCORED_FILE = "reference_scored.csv"
TRUSTED_TYPES = ["src.model.features.CreditFeatureEngineer"]


@dataclass
class TrainingResult:
    """Artifacts and metrics produced by a training run."""

    pipeline: Pipeline
    metadata: dict[str, Any]
    reference_scored: pd.DataFrame


def build_pipeline(config: dict[str, Any], c_value: float) -> Pipeline:
    """Build the preprocessing + logistic regression pipeline."""
    categorical = config["features"]["categorical"]
    numeric = [c for c in model_feature_columns(config) if c not in categorical] + ENGINEERED
    preprocess = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ("numeric", StandardScaler(), numeric),
        ],
        verbose_feature_names_out=False,
    )
    model = LogisticRegression(
        C=c_value,
        class_weight=config["model"]["class_weight"],
        max_iter=config["model"]["max_iter"],
        random_state=config["project"]["seed"],
    )
    return Pipeline([("features", CreditFeatureEngineer()), ("preprocess", preprocess), ("model", model)])


def data_fingerprint(frame: pd.DataFrame) -> str:
    """Short SHA-256 fingerprint of a DataFrame (used in the model version)."""
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def coefficient_table(pipeline: Pipeline) -> pd.DataFrame:
    """Return the logistic-regression coefficients ordered by absolute magnitude."""
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    coefs = pipeline.named_steps["model"].coef_.ravel()
    table = pd.DataFrame({"feature": names, "coefficient": coefs})
    table["abs_coefficient"] = table["coefficient"].abs()
    table["odds_ratio_per_std"] = np.exp(table["coefficient"])
    return table.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def train_baseline(config: dict[str, Any], paths: ProjectPaths) -> TrainingResult:
    """Train, select and evaluate the baseline; persist artifacts and log to MLflow."""
    logger = get_logger()
    target = config["features"]["target"]
    features = model_feature_columns(config)
    seed = config["project"]["seed"]
    np.random.seed(seed)

    partitions = {name: load_partition(paths, name) for name in ("train", "validation", "reference")}
    for name, frame in partitions.items():
        enforce_contract(frame, "training", f"{name}_partition", paths)
    train, validation, reference = partitions["train"], partitions["validation"], partitions["reference"]

    configure_mlflow(paths)
    ensure_experiment(config["mlflow"]["training_experiment"], paths)
    model_cfg = config["model"]
    version = f"{model_cfg['version']}+{data_fingerprint(train)}"

    with mlflow.start_run(run_name=f"train-{model_cfg['name']}") as parent:
        mlflow.set_tags({"pipeline_run_id": get_run_id(), "stage": "training", "model_version": version})
        candidates: list[tuple[float, float]] = []
        for c_value in model_cfg["candidate_c"]:
            with mlflow.start_run(run_name=f"candidate-C={c_value}", nested=True):
                candidate = build_pipeline(config, c_value).fit(train[features], train[target])
                val_auc = float(roc_auc_score(validation[target], candidate.predict_proba(validation[features])[:, 1]))
                mlflow.log_params({"C": c_value, "class_weight": model_cfg["class_weight"]})
                mlflow.log_metric("validation_roc_auc", val_auc)
                candidates.append((c_value, val_auc))
                logger.info("candidate evaluated", extra={"stage": "training", "C": c_value, "val_auc": val_auc})

        best_c, best_val_auc = max(candidates, key=lambda item: (item[1], -item[0]))
        pipeline = build_pipeline(config, best_c).fit(train[features], train[target])
        val_proba = pipeline.predict_proba(validation[features])[:, 1]
        threshold, val_f1 = best_f1_threshold(
            validation[target],
            val_proba,
            model_cfg["threshold_grid_start"],
            model_cfg["threshold_grid_stop"],
            model_cfg["threshold_grid_step"],
        )
        val_metrics = classification_metrics(validation[target], val_proba, threshold)
        ref_proba = pipeline.predict_proba(reference[features])[:, 1]
        ref_metrics = classification_metrics(reference[target], ref_proba, threshold)

        params = {
            "model_type": "LogisticRegression",
            "C": best_c,
            "class_weight": model_cfg["class_weight"],
            "max_iter": model_cfg["max_iter"],
            "threshold": threshold,
            "seed": seed,
            "n_features_raw": len(features),
            "n_train": len(train),
            "n_validation": len(validation),
            "n_reference": len(reference),
            "sklearn_version": sklearn.__version__,
        }
        mlflow.log_params(params)
        log_numeric_metrics(val_metrics, prefix="validation_")
        log_numeric_metrics(ref_metrics, prefix="reference_")

        paths.model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, paths.model_dir / MODEL_FILE)
        coefs = coefficient_table(pipeline)
        coefs.to_csv(paths.model_dir / "coefficients.csv", index=False)

        reference_scored = reference.copy()
        reference_scored["prediction"] = ref_proba
        reference_scored["predicted_label"] = (ref_proba >= threshold).astype(int)
        reference_scored.to_csv(paths.reference_dir / REFERENCE_SCORED_FILE, index=False, lineterminator="\n")

        with warnings.catch_warnings():
            # Integer-column hint is expected: nulls are blocked by the data contract before inference.
            warnings.filterwarnings("ignore", message=".*Integer columns.*", category=UserWarning)
            signature = infer_signature(reference[features].head(50), ref_proba[:50])
        model_info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=signature,
            registered_model_name=config["mlflow"]["registered_model_name"],
            skops_trusted_types=TRUSTED_TYPES,
        )
        metadata = {
            "model_name": model_cfg["name"],
            "model_version": version,
            "registered_model_name": config["mlflow"]["registered_model_name"],
            "registered_model_version": getattr(model_info, "registered_model_version", None),
            "trained_at": datetime.now(UTC).isoformat(),
            "pipeline_run_id": get_run_id(),
            "mlflow_run_id": parent.info.run_id,
            "params": params,
            "candidates": [{"C": c, "validation_roc_auc": round(a, 5)} for c, a in candidates],
            "best_validation_roc_auc": round(best_val_auc, 5),
            "validation_best_f1": round(val_f1, 5),
            "features": features,
            "engineered_features": ENGINEERED,
            "excluded_attributes": config["features"]["protected_attributes"],
            "threshold": threshold,
            "validation_metrics": val_metrics,
            "reference_metrics": ref_metrics,
            "train_data_fingerprint": data_fingerprint(train),
        }
        (paths.model_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(paths.model_dir / METADATA_FILE))
        mlflow.log_artifact(str(paths.model_dir / "coefficients.csv"))

    logger.info(
        "baseline trained",
        extra={
            "stage": "training",
            "model_version": version,
            "C": best_c,
            "threshold": threshold,
            "reference_roc_auc": round(ref_metrics["roc_auc"], 4),
        },
    )
    return TrainingResult(pipeline=pipeline, metadata=metadata, reference_scored=reference_scored)
