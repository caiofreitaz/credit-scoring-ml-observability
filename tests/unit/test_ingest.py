"""Data loading: normalization, integrity hash and source fallback (no network)."""

import io
import json
import zipfile

import pandas as pd
import pytest

from src.data import ingest
from src.data.ingest import (
    DatasetIntegrityError,
    canonical_sha256,
    download_dataset,
    load_raw_dataset,
    normalize_raw_frame,
    parse_source,
)


def _kaggle_style(raw: pd.DataFrame) -> pd.DataFrame:
    return raw.rename(columns={"default_next_month": "default.payment.next.month"})


def test_normalize_renames_target_and_casts_int(synthetic_raw):
    frame = _kaggle_style(synthetic_raw).astype({"LIMIT_BAL": "float64"})
    normalized = normalize_raw_frame(frame, expected_rows=len(frame))
    assert list(normalized.columns) == ingest.CANONICAL_COLUMNS
    assert (normalized.dtypes == "int64").all()


def test_normalize_rejects_missing_columns(synthetic_raw):
    with pytest.raises(DatasetIntegrityError, match="Missing expected columns"):
        normalize_raw_frame(synthetic_raw.drop(columns=["AGE"]))


def test_normalize_rejects_unexpected_row_count(synthetic_raw):
    with pytest.raises(DatasetIntegrityError, match="Expected 10 rows"):
        normalize_raw_frame(synthetic_raw, expected_rows=10)


def test_canonical_hash_is_format_independent(synthetic_raw):
    as_float = normalize_raw_frame(_kaggle_style(synthetic_raw).astype("float64"))
    assert canonical_sha256(as_float) == canonical_sha256(normalize_raw_frame(synthetic_raw))


def test_parse_csv_and_unsupported_format(synthetic_raw):
    content = synthetic_raw.to_csv(index=False).encode()
    assert parse_source(content, "csv").shape == synthetic_raw.shape
    with pytest.raises(ValueError, match="Unsupported"):
        parse_source(content, "parquet")


def test_parse_xls_uses_second_header_row(monkeypatch, synthetic_raw):
    captured = {}

    def fake_read_excel(buffer, header, engine):
        captured.update(header=header, engine=engine)
        return synthetic_raw

    monkeypatch.setattr(ingest.pd, "read_excel", fake_read_excel)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("default of credit card clients.xls", b"fake-xls")
    result = parse_source(buffer.getvalue(), "zip_xls")
    assert captured == {"header": 1, "engine": "xlrd"}
    assert result.shape == synthetic_raw.shape


def test_zip_without_xls_is_rejected():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("readme.txt", "nothing")
    with pytest.raises(DatasetIntegrityError, match="does not contain"):
        parse_source(buffer.getvalue(), "zip_xls")


def _download_config(base_config, rows, digest=None):
    cfg = json.loads(json.dumps(base_config))
    cfg["data"]["expected_rows"] = rows
    cfg["data"]["canonical_sha256"] = digest
    return cfg


def test_download_falls_back_to_mirror_and_writes_manifest(project, base_config, synthetic_raw):
    paths = project["paths"]
    paths.raw_data.unlink()
    paths.raw_manifest.unlink()
    cfg = _download_config(base_config, len(synthetic_raw), canonical_sha256(synthetic_raw))
    csv_bytes = _kaggle_style(synthetic_raw).to_csv(index=False).encode()

    def fetcher(url):
        if "archive.ics.uci.edu" in url:
            raise ConnectionError("blocked")
        if url.endswith(".xls"):
            raise ConnectionError("mirror down")
        return csv_bytes

    manifest = download_dataset(cfg, paths, fetcher=fetcher)
    assert manifest["source_name"] == "github_mirror_csv"
    assert len(manifest["failed_sources"]) == 2
    assert load_raw_dataset(paths).shape == synthetic_raw.shape
    assert json.loads(paths.raw_manifest.read_text())["license"] == "CC BY 4.0"


def test_download_rejects_hash_mismatch(project, base_config, synthetic_raw):
    paths = project["paths"]
    cfg = _download_config(base_config, len(synthetic_raw), "0" * 64)
    with pytest.raises(DatasetIntegrityError, match="All dataset sources failed"):
        download_dataset(cfg, paths, fetcher=lambda url: synthetic_raw.to_csv(index=False).encode(), force=True)


def test_download_skips_when_dataset_present(project, base_config):
    def fail(url):
        raise AssertionError("should not download")

    manifest = download_dataset(base_config, project["paths"], fetcher=fail)
    assert manifest["source_name"] == "synthetic"


def test_load_raw_dataset_missing_file(project):
    project["paths"].raw_data.unlink()
    with pytest.raises(FileNotFoundError, match="download-data"):
        load_raw_dataset(project["paths"])
