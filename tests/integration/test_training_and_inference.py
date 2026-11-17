"""Integration: training + persistence + MLflow, and the inference blocking flow."""

import json

import numpy as np
import pytest
from mlflow.tracking import MlflowClient

from src.config import model_feature_columns
from src.data.prepare import load_partition, prepare_data
from src.model.predict import load_model, predict_batch
from src.model.train import MODEL_FILE, build_pipeline, train_baseline
from src.monitoring.alerts import AlertManager
from src.monitoring.tracker import configure_mlflow
from src.validation.gate import DataContractViolation, read_batch


@pytest.fixture()
def trained(project, synthetic_raw):
    prepare_data(synthetic_raw, project["config"], project["paths"])
    result = train_baseline(project["config"], project["paths"])
    return {**project, "result": result}


def test_training_persists_model_metadata_and_reference_scores(trained):
    paths, metadata = trained["paths"], trained["result"].metadata
    assert (paths.model_dir / MODEL_FILE).exists()
    saved = json.loads((paths.model_dir / "model_metadata.json").read_text())
    assert saved["model_version"] == metadata["model_version"]
    assert set(saved["reference_metrics"]) >= {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "tn",
        "fp",
        "fn",
        "tp",
    }
    assert saved["reference_metrics"]["roc_auc"] > 0.6
    assert {"SEX", "MARRIAGE"}.isdisjoint(saved["features"])
    assert (paths.reference_dir / "reference_scored.csv").exists()


def test_training_is_logged_and_registered_in_mlflow(trained):
    configure_mlflow(trained["paths"])
    client = MlflowClient()
    experiment = client.get_experiment_by_name(trained["config"]["mlflow"]["training_experiment"])
    runs = client.search_runs([experiment.experiment_id])
    parent = next(r for r in runs if r.info.run_id == trained["result"].metadata["mlflow_run_id"])
    assert "reference_roc_auc" in parent.data.metrics and "C" in parent.data.params
    assert len(runs) == 1 + len(trained["config"]["model"]["candidate_c"])
    versions = client.search_model_versions(f"name='{trained['config']['mlflow']['registered_model_name']}'")
    assert len(versions) == 1


def test_persisted_model_loads_and_predicts_identically(trained):
    pipeline, _ = load_model(trained["paths"])
    reference = load_partition(trained["paths"], "reference")
    features = model_feature_columns(trained["config"])
    loaded = pipeline.predict_proba(reference[features])[:, 1]
    original = trained["result"].pipeline.predict_proba(reference[features])[:, 1]
    assert np.allclose(loaded, original)
    assert ((loaded >= 0) & (loaded <= 1)).all()


def test_training_is_reproducible(project, synthetic_raw):
    prepare_data(synthetic_raw, project["config"], project["paths"])
    train = load_partition(project["paths"], "train")
    features, target = model_feature_columns(project["config"]), project["config"]["features"]["target"]
    first = build_pipeline(project["config"], 0.1).fit(train[features], train[target])
    second = build_pipeline(project["config"], 0.1).fit(train[features], train[target])
    assert np.array_equal(first.named_steps["model"].coef_, second.named_steps["model"].coef_)


def test_valid_batch_generates_predictions(trained):
    from src.data.batches import build_valid_batch

    pool = load_partition(trained["paths"], "production")
    batch = build_valid_batch(pool, trained["config"], size=100)
    result = predict_batch(batch, "integration_valid", trained["config"], trained["paths"])
    assert len(result.scored) == 100 and result.latency_ms > 0
    assert (trained["paths"].production_dir / "predictions" / "integration_valid_predictions.csv").exists()


def test_invalid_batch_blocks_inference_and_produces_no_predictions(trained):
    from src.data.batches import build_invalid_batch

    paths = trained["paths"]
    pool = load_partition(paths, "production")
    alerts = AlertManager()
    with pytest.raises(DataContractViolation):
        predict_batch(
            build_invalid_batch(pool, trained["config"], size=80),
            "integration_invalid",
            trained["config"],
            paths,
            alerts,
        )
    assert not (paths.production_dir / "predictions" / "integration_invalid_predictions.csv").exists()
    assert alerts.counts()["CRITICAL"] == 1
    quarantined = list(paths.quarantine_dir.glob("integration_invalid__*.csv"))
    assert len(quarantined) == 1 and len(read_batch(quarantined[0])) == 80
