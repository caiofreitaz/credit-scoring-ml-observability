"""Preparation, pseudonymization, splitting and leakage checks."""

import json

import pytest

from src.data.prepare import (
    DataSplits,
    check_no_leakage,
    clean_dataset,
    load_partition,
    prepare_data,
    pseudonymize_id,
    resolve_salt,
    split_dataset,
)


def test_clean_recodes_undocumented_categories(synthetic_raw):
    cleaned, stats = clean_dataset(synthetic_raw, "salt")
    assert set(cleaned["EDUCATION"].unique()) <= {1, 2, 3, 4}
    assert set(cleaned["MARRIAGE"].unique()) <= {1, 2, 3}
    assert stats["education_recoded"] == int(synthetic_raw["EDUCATION"].isin([0, 5, 6]).sum())
    assert stats["marriage_recoded"] == int((synthetic_raw["MARRIAGE"] == 0).sum())


def test_pseudonymization_removes_raw_id_and_is_keyed(synthetic_raw):
    cleaned, _ = clean_dataset(synthetic_raw, "salt")
    assert "ID" not in cleaned.columns
    assert cleaned["client_id"].str.fullmatch(r"[0-9a-f]{16}").all()
    assert pseudonymize_id(1, "salt") == pseudonymize_id(1, "salt")
    assert pseudonymize_id(1, "salt") != pseudonymize_id(1, "other-salt")


def test_salt_comes_from_environment(monkeypatch, base_config):
    monkeypatch.setenv(base_config["data"]["pseudonymization_salt_env"], "from-env")
    assert resolve_salt(base_config) == "from-env"


def test_split_sizes_and_stratification(synthetic_raw, project):
    cleaned, _ = clean_dataset(synthetic_raw, "salt")
    splits = split_dataset(cleaned, project["config"])
    sizes = {k: len(v) for k, v in splits.as_dict().items()}
    assert sum(sizes.values()) == len(cleaned)
    assert sizes["production"] == pytest.approx(0.25 * len(cleaned), abs=2)
    assert sizes["train"] == pytest.approx(0.50 * len(cleaned), abs=2)
    overall = cleaned["default_next_month"].mean()
    for part in splits.as_dict().values():
        assert part["default_next_month"].mean() == pytest.approx(overall, abs=0.02)


def test_split_is_reproducible(synthetic_raw, project):
    cleaned, _ = clean_dataset(synthetic_raw, "salt")
    first = split_dataset(cleaned, project["config"])
    second = split_dataset(cleaned, project["config"])
    assert first.production["client_id"].tolist() == second.production["client_id"].tolist()


def test_no_leakage_between_partitions(synthetic_raw, project):
    cleaned, _ = clean_dataset(synthetic_raw, "salt")
    overlaps = check_no_leakage(split_dataset(cleaned, project["config"]), "client_id")
    assert len(overlaps) == 6
    assert all(v == 0 for v in overlaps.values())


def test_leakage_is_detected(synthetic_raw, project):
    cleaned, _ = clean_dataset(synthetic_raw, "salt")
    splits = split_dataset(cleaned, project["config"])
    leaked = DataSplits(splits.train, splits.validation, splits.reference, splits.train.head(5))
    assert check_no_leakage(leaked, "client_id")["train__production"] == 5


def test_invalid_split_fractions_raise(synthetic_raw, project):
    cfg = json.loads(json.dumps(project["config"]))
    cfg["data"]["split"]["train"] = 0.9
    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        split_dataset(synthetic_raw, cfg)


def test_prepare_data_persists_partitions_and_manifest(synthetic_raw, project):
    paths = project["paths"]
    manifest = prepare_data(synthetic_raw, project["config"], paths)
    assert manifest["rows_total"] == len(synthetic_raw)
    assert all(v == 0 for v in manifest["id_overlaps"].values())
    for name in ("train", "validation", "reference", "production"):
        part = load_partition(paths, name)
        assert part["client_id"].dtype == object
        assert len(part) == manifest["partitions"][name]["rows"]
    assert (paths.artifacts_dir / "data_split_manifest.json").exists()


def test_load_partition_missing(project):
    with pytest.raises(FileNotFoundError, match="prepare-data"):
        load_partition(project["paths"], "train")
