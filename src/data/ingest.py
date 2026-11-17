"""Dataset acquisition: download, normalize and integrity-check the UCI dataset.

Dataset: "Default of Credit Card Clients" (Yeh, I. C., 2009), UCI Machine Learning
Repository, DOI 10.24432/C55S3H, licensed under CC BY 4.0.
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
import zipfile
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import ProjectPaths
from src.monitoring.logger import get_logger

TARGET_RAW_NAMES = ("default payment next month", "default.payment.next.month")
CANONICAL_COLUMNS = [
    "ID",
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    *[f"BILL_AMT{i}" for i in range(1, 7)],
    *[f"PAY_AMT{i}" for i in range(1, 7)],
    "default_next_month",
]
DATASET_CITATION = (
    "Yeh, I. (2009). Default of Credit Card Clients [Dataset]. "
    "UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H"
)
DATASET_LICENSE = "CC BY 4.0"

Fetcher = Callable[[str], bytes]


class DatasetIntegrityError(RuntimeError):
    """Raised when downloaded content does not match the expected dataset."""


def http_fetch(url: str, timeout: int = 60) -> bytes:
    """Download a URL and return its raw bytes."""
    request = urllib.request.Request(url, headers={"User-Agent": "credit-scoring-ml-observability"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (fixed https URLs)
        return response.read()


def parse_source(content: bytes, fmt: str) -> pd.DataFrame:
    """Parse raw bytes of a supported source format into a DataFrame."""
    if fmt == "csv":
        return pd.read_csv(io.BytesIO(content))
    if fmt == "xls":
        # Original UCI spreadsheet: first row holds X1..X23 aliases, second row the names.
        return pd.read_excel(io.BytesIO(content), header=1, engine="xlrd")
    if fmt == "zip_xls":
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            names = [n for n in archive.namelist() if n.lower().endswith(".xls")]
            if not names:
                raise DatasetIntegrityError("ZIP file does not contain an .xls spreadsheet")
            return parse_source(archive.read(names[0]), "xls")
    raise ValueError(f"Unsupported source format: {fmt}")


def normalize_raw_frame(frame: pd.DataFrame, expected_rows: int | None = None) -> pd.DataFrame:
    """Standardize column names and dtypes so every source yields identical content."""
    renamed = frame.rename(columns={name: "default_next_month" for name in TARGET_RAW_NAMES})
    missing = [c for c in CANONICAL_COLUMNS if c not in renamed.columns]
    if missing:
        raise DatasetIntegrityError(f"Missing expected columns: {missing}")
    normalized = renamed[CANONICAL_COLUMNS].astype("int64").reset_index(drop=True)
    if expected_rows is not None and len(normalized) != expected_rows:
        raise DatasetIntegrityError(f"Expected {expected_rows} rows, found {len(normalized)}")
    return normalized


def canonical_sha256(frame: pd.DataFrame) -> str:
    """Compute a content hash independent of the source file format."""
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def download_dataset(
    config: dict[str, Any],
    paths: ProjectPaths,
    fetcher: Fetcher | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Download the dataset trying the official source first and mirrors as fallback.

    Returns the download manifest (source used, hash, row count, timestamp).
    """
    logger = get_logger()
    data_cfg = config["data"]
    fetch = fetcher or http_fetch
    if paths.raw_data.exists() and paths.raw_manifest.exists() and not force:
        manifest = json.loads(paths.raw_manifest.read_text(encoding="utf-8"))
        logger.info("raw dataset already present, skipping download", extra={"stage": "download"})
        return manifest

    errors: list[str] = []
    for source in data_cfg["sources"]:
        try:
            logger.info(
                "downloading dataset",
                extra={"stage": "download", "source": source["name"], "url": source["url"]},
            )
            frame = normalize_raw_frame(
                parse_source(fetch(source["url"]), source["format"]), data_cfg["expected_rows"]
            )
        except Exception as exc:  # noqa: BLE001 - try next mirror, keep the reason
            errors.append(f"{source['name']}: {type(exc).__name__}: {exc}")
            logger.warning(
                "dataset source failed",
                extra={"stage": "download", "source": source["name"], "error": str(exc)},
            )
            continue

        digest = canonical_sha256(frame)
        expected = data_cfg.get("canonical_sha256")
        if expected and digest != expected:
            errors.append(f"{source['name']}: hash mismatch {digest}")
            logger.error(
                "dataset hash mismatch",
                extra={"stage": "download", "source": source["name"], "sha256": digest},
            )
            continue

        paths.raw_data.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(paths.raw_data, index=False, lineterminator="\n")
        manifest = {
            "source_name": source["name"],
            "source_url": source["url"],
            "canonical_sha256": digest,
            "rows": int(len(frame)),
            "columns": int(frame.shape[1]),
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "citation": DATASET_CITATION,
            "license": DATASET_LICENSE,
            "failed_sources": errors,
        }
        paths.raw_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info(
            "dataset downloaded and verified",
            extra={"stage": "download", "source": source["name"], "rows": len(frame), "sha256": digest},
        )
        return manifest

    raise DatasetIntegrityError("All dataset sources failed: " + " | ".join(errors))


def load_raw_dataset(paths: ProjectPaths) -> pd.DataFrame:
    """Load the normalized raw dataset from disk."""
    if not paths.raw_data.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {paths.raw_data}. Run: python -m src.cli download-data"
        )
    return pd.read_csv(paths.raw_data)
