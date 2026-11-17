"""Data preparation: cleaning, pseudonymization and leakage-free splits."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import ProjectPaths
from src.monitoring.logger import get_logger

EDUCATION_RECODE = {0: 4, 5: 4, 6: 4}  # undocumented codes -> "others" (4)
MARRIAGE_RECODE = {0: 3}  # undocumented code -> "others" (3)
SPLIT_FILES = {
    "train": "train.csv",
    "validation": "validation.csv",
    "reference": "reference.csv",
}
PRODUCTION_POOL_FILE = "production_pool.csv"


@dataclass(frozen=True)
class DataSplits:
    """Container for the four disjoint data partitions."""

    train: pd.DataFrame
    validation: pd.DataFrame
    reference: pd.DataFrame
    production: pd.DataFrame

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {
            "train": self.train,
            "validation": self.validation,
            "reference": self.reference,
            "production": self.production,
        }


def resolve_salt(config: dict[str, Any]) -> str:
    """Read the pseudonymization salt from the environment, falling back to the academic default."""
    data_cfg = config["data"]
    return os.environ.get(data_cfg["pseudonymization_salt_env"], data_cfg["pseudonymization_default_salt"])


def pseudonymize_id(raw_id: int | str, salt: str) -> str:
    """Return a keyed hash (HMAC-SHA256, 16 hex chars) of the original identifier."""
    digest = hmac.new(salt.encode("utf-8"), str(raw_id).encode("utf-8"), hashlib.sha256)
    return digest.hexdigest()[:16]


def clean_dataset(raw: pd.DataFrame, salt: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Recode undocumented categories and replace the raw ID by a pseudonymous key."""
    frame = raw.copy()
    stats = {
        "education_recoded": int(frame["EDUCATION"].isin(EDUCATION_RECODE).sum()),
        "marriage_recoded": int(frame["MARRIAGE"].isin(MARRIAGE_RECODE).sum()),
    }
    frame["EDUCATION"] = frame["EDUCATION"].replace(EDUCATION_RECODE).astype("int64")
    frame["MARRIAGE"] = frame["MARRIAGE"].replace(MARRIAGE_RECODE).astype("int64")
    frame.insert(0, "client_id", frame["ID"].map(lambda value: pseudonymize_id(value, salt)))
    frame = frame.drop(columns=["ID"])
    if frame["client_id"].duplicated().any():
        raise ValueError("Pseudonymization produced colliding identifiers")
    return frame, stats


def split_dataset(frame: pd.DataFrame, config: dict[str, Any]) -> DataSplits:
    """Create stratified, disjoint train/validation/reference/production partitions."""
    split_cfg = config["data"]["split"]
    seed = config["project"]["seed"]
    target = config["features"]["target"]
    total = sum(split_cfg.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")

    development, production = train_test_split(
        frame, test_size=split_cfg["production"], stratify=frame[target], random_state=seed
    )
    remaining = 1.0 - split_cfg["production"]
    train_val, reference = train_test_split(
        development,
        test_size=split_cfg["reference"] / remaining,
        stratify=development[target],
        random_state=seed,
    )
    train, validation = train_test_split(
        train_val,
        test_size=split_cfg["validation"] / (split_cfg["train"] + split_cfg["validation"]),
        stratify=train_val[target],
        random_state=seed,
    )
    return DataSplits(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        reference=reference.reset_index(drop=True),
        production=production.reset_index(drop=True),
    )


def check_no_leakage(splits: DataSplits, id_column: str) -> dict[str, int]:
    """Count shared identifiers between every pair of partitions (must all be zero)."""
    parts = splits.as_dict()
    names = list(parts)
    overlaps: dict[str, int] = {}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            shared = set(parts[left][id_column]) & set(parts[right][id_column])
            overlaps[f"{left}__{right}"] = len(shared)
    return overlaps


def prepare_data(raw: pd.DataFrame, config: dict[str, Any], paths: ProjectPaths) -> dict[str, Any]:
    """Run cleaning + splitting and persist partitions and a manifest. Returns the manifest."""
    logger = get_logger()
    target = config["features"]["target"]
    id_column = config["features"]["id_column"]
    cleaned, recode_stats = clean_dataset(raw, resolve_salt(config))
    splits = split_dataset(cleaned, config)
    overlaps = check_no_leakage(splits, id_column)
    if any(overlaps.values()):
        raise ValueError(f"Data leakage detected between partitions: {overlaps}")

    paths.reference_dir.mkdir(parents=True, exist_ok=True)
    paths.production_dir.mkdir(parents=True, exist_ok=True)
    for name, file_name in SPLIT_FILES.items():
        getattr(splits, name).to_csv(paths.reference_dir / file_name, index=False, lineterminator="\n")
    splits.production.to_csv(paths.production_dir / PRODUCTION_POOL_FILE, index=False, lineterminator="\n")

    manifest = {
        "seed": config["project"]["seed"],
        "rows_total": int(len(cleaned)),
        "recoding": recode_stats,
        "partitions": {
            name: {
                "rows": int(len(part)),
                "default_rate": round(float(part[target].mean()), 4),
            }
            for name, part in splits.as_dict().items()
        },
        "id_overlaps": overlaps,
        "pseudonymization": "HMAC-SHA256(salt, ID)[:16]; salt from env var or academic default",
    }
    paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (paths.artifacts_dir / "data_split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    logger.info(
        "data prepared",
        extra={
            "stage": "prepare",
            "rows": manifest["rows_total"],
            "partitions": {k: v["rows"] for k, v in manifest["partitions"].items()},
        },
    )
    return manifest


def load_partition(paths: ProjectPaths, name: str) -> pd.DataFrame:
    """Load a persisted partition by name (train, validation, reference, production)."""
    if name == "production":
        path = paths.production_dir / PRODUCTION_POOL_FILE
    else:
        path = paths.reference_dir / SPLIT_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"Partition '{name}' not found at {path}. Run: python -m src.cli prepare-data")
    return pd.read_csv(path, dtype={"client_id": str})
