"""Central configuration loading and path resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved absolute paths used across the project."""

    root: Path
    raw_data: Path
    raw_manifest: Path
    reference_dir: Path
    production_dir: Path
    invalid_dir: Path
    quarantine_dir: Path
    artifacts_dir: Path
    model_dir: Path
    reports_dir: Path
    logs_dir: Path
    mlflow_db: Path
    mlflow_artifacts: Path

    def ensure(self) -> None:
        """Create every output directory if it does not exist yet."""
        for directory in (
            self.raw_data.parent,
            self.reference_dir,
            self.production_dir,
            self.invalid_dir,
            self.quarantine_dir,
            self.artifacts_dir,
            self.model_dir,
            self.reports_dir,
            self.logs_dir,
            self.mlflow_artifacts,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML configuration file."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_paths(config: dict[str, Any], root: str | Path | None = None) -> ProjectPaths:
    """Resolve configured relative paths against the project (or a custom) root."""
    base = Path(root) if root else PROJECT_ROOT
    raw = config["paths"]
    return ProjectPaths(
        root=base,
        raw_data=base / raw["raw_data"],
        raw_manifest=base / raw["raw_manifest"],
        reference_dir=base / raw["reference_dir"],
        production_dir=base / raw["production_dir"],
        invalid_dir=base / raw["invalid_dir"],
        quarantine_dir=base / raw["quarantine_dir"],
        artifacts_dir=base / raw["artifacts_dir"],
        model_dir=base / raw["model_dir"],
        reports_dir=base / raw["reports_dir"],
        logs_dir=base / raw["logs_dir"],
        mlflow_db=base / raw["mlflow_db"],
        mlflow_artifacts=base / raw["mlflow_artifacts"],
    )


def model_feature_columns(config: dict[str, Any]) -> list[str]:
    """Return the ordered list of raw columns consumed by the model."""
    feats = config["features"]
    return [
        "LIMIT_BAL",
        *feats["categorical"],
        "AGE",
        *feats["ordinal_payment_status"],
        *[c for c in feats["numeric"] if c not in ("LIMIT_BAL", "AGE")],
    ]
