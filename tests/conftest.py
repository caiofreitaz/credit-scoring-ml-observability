"""Shared fixtures. The test suite never downloads data: it uses a small synthetic
dataset with the same schema and value domains as the UCI dataset."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.config import load_config, resolve_paths
from src.data.ingest import CANONICAL_COLUMNS
from src.monitoring.logger import configure_logging, set_run_id

SYNTHETIC_ROWS = 6000


def make_synthetic_raw(n: int = SYNTHETIC_ROWS, seed: int = 7) -> pd.DataFrame:
    """Synthetic credit data in the canonical raw schema (including undocumented codes)."""
    rng = np.random.default_rng(seed)
    limit = (rng.lognormal(mean=11.7, sigma=0.75, size=n) // 10_000 * 10_000).clip(10_000, 1_000_000)
    age = rng.integers(21, 76, size=n)
    pay0 = rng.choice(np.arange(-2, 9), size=n, p=[0.09, 0.19, 0.49, 0.12, 0.08, 0.015, 0.01, 0.005, 0.0, 0.0, 0.0])
    pay_hist = np.column_stack([np.clip(pay0 + rng.integers(-1, 2, size=n), -2, 8) for _ in range(5)])
    utilization = rng.beta(2, 3, size=n) * 1.1
    bills = np.column_stack([(limit * utilization * rng.uniform(0.8, 1.2, size=n)).round() for _ in range(6)])
    bills[:5, 2] = -1_500  # negative bills (credit balance) exist in the real data
    payments = np.column_stack([(bills[:, i] * rng.uniform(0.0, 0.25, size=n)).clip(0).round() for i in range(6)])
    logit = -1.9 + 1.5 * np.clip(pay0, 0, None) + 0.3 * utilization - 0.000002 * limit + 0.01 * (age - 35)
    target = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    frame = pd.DataFrame(
        {
            "ID": np.arange(1, n + 1),
            "LIMIT_BAL": limit,
            "SEX": rng.integers(1, 3, size=n),
            "EDUCATION": rng.choice([0, 1, 2, 3, 4, 5, 6], size=n, p=[0.01, 0.35, 0.45, 0.15, 0.02, 0.01, 0.01]),
            "MARRIAGE": rng.choice([0, 1, 2, 3], size=n, p=[0.01, 0.45, 0.52, 0.02]),
            "AGE": age,
            "PAY_0": pay0,
            **{f"PAY_{i}": pay_hist[:, i - 2] for i in range(2, 7)},
            **{f"BILL_AMT{i}": bills[:, i - 1] for i in range(1, 7)},
            **{f"PAY_AMT{i}": payments[:, i - 1] for i in range(1, 7)},
            "default_next_month": target,
        }
    )
    return frame[CANONICAL_COLUMNS].astype("int64")


def make_test_config(base: dict) -> dict:
    """Config adapted to the small synthetic dataset."""
    cfg = copy.deepcopy(base)
    cfg["data"]["canonical_sha256"] = None
    cfg["data"]["expected_rows"] = SYNTHETIC_ROWS
    cfg["simulation"]["batch_size"] = 700
    cfg["drift"]["min_labeled_rows"] = 200
    cfg["model"]["candidate_c"] = [0.1, 1.0]
    return cfg


@pytest.fixture(scope="session")
def base_config() -> dict:
    return load_config()


@pytest.fixture(scope="session")
def synthetic_raw() -> pd.DataFrame:
    return make_synthetic_raw()


def _write_project(root: Path, cfg: dict, raw: pd.DataFrame) -> tuple[Path, dict]:
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    paths = resolve_paths(cfg, root)
    paths.ensure()
    raw.to_csv(paths.raw_data, index=False, lineterminator="\n")
    paths.raw_manifest.write_text(
        json.dumps(
            {"source_name": "synthetic", "rows": len(raw), "columns": raw.shape[1], "canonical_sha256": "synthetic"}
        ),
        encoding="utf-8",
    )
    return config_path, cfg


@pytest.fixture()
def project(tmp_path: Path, base_config: dict, synthetic_raw: pd.DataFrame):
    """Fresh isolated project folder with the synthetic raw dataset (function scope)."""
    config_path, cfg = _write_project(tmp_path, make_test_config(base_config), synthetic_raw)
    paths = resolve_paths(cfg, tmp_path)
    configure_logging(paths.logs_dir)
    set_run_id("test-run")
    return {"config_path": config_path, "config": cfg, "paths": paths, "root": tmp_path}


@pytest.fixture(scope="session")
def pipeline_run(tmp_path_factory, base_config: dict, synthetic_raw: pd.DataFrame):
    """Run the whole pipeline once on synthetic data and share the outputs."""
    from src.pipeline.run_pipeline import create_context, run_full_pipeline

    root = tmp_path_factory.mktemp("pipeline")
    config_path, cfg = _write_project(root, make_test_config(base_config), synthetic_raw)
    ctx = create_context(str(config_path), root)
    summary = run_full_pipeline(ctx, skip_download=True)
    return {"ctx": ctx, "summary": summary, "paths": ctx.paths, "config": cfg, "config_path": config_path, "root": root}
