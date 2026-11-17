"""One test per data-quality rule (R01-R09) plus contract-mode differences."""

import numpy as np
import pandas as pd
import pytest

from src.data.batches import to_inference_frame
from src.data.prepare import clean_dataset
from src.validation.contracts import RULE_DESCRIPTIONS, classify_check, get_schema, summarize_failures
from src.validation.gate import validate_batch


@pytest.fixture()
def labeled(synthetic_raw):
    cleaned, _ = clean_dataset(synthetic_raw.head(50), "salt")
    return cleaned


@pytest.fixture()
def inference(labeled, base_config):
    return to_inference_frame(labeled, base_config)


def _rules(frame, mode="inference"):
    return validate_batch(frame, mode, "t").rules_violated


def test_valid_batches_pass_both_contracts(labeled, inference):
    assert validate_batch(labeled, "training", "train").is_valid
    assert validate_batch(inference, "inference", "inf").is_valid


def test_r01_required_column_missing(inference):
    assert "R01_REQUIRED_COLUMNS" in _rules(inference.drop(columns=["PAY_AMT3"]))


def test_r02_unexpected_columns_blocked_in_inference(inference):
    leaked = inference.assign(SEX=1)
    assert _rules(leaked) == ["R02_NO_UNEXPECTED_COLUMNS"]


def test_r02_target_not_required_and_not_accepted_in_inference(labeled, inference):
    assert "default_next_month" not in inference.columns
    assert "R02_NO_UNEXPECTED_COLUMNS" in _rules(inference.assign(default_next_month=0))
    assert "R01_REQUIRED_COLUMNS" in _rules(labeled.drop(columns=["default_next_month"]), "training")


def test_r03_wrong_dtype(inference):
    broken = inference.assign(AGE=inference["AGE"].astype(str))
    assert "R03_DTYPES" in _rules(broken)


def test_r04_null_values(inference):
    broken = inference.copy()
    broken["LIMIT_BAL"] = broken["LIMIT_BAL"].astype("float64")
    broken.loc[0, "LIMIT_BAL"] = np.nan
    assert "R04_NOT_NULL" in _rules(broken)


@pytest.mark.parametrize(
    ("column", "value"),
    [("AGE", 17), ("AGE", 101), ("LIMIT_BAL", 5_000), ("PAY_AMT1", -1), ("BILL_AMT2", 2_000_001)],
)
def test_r05_numeric_ranges(inference, column, value):
    broken = inference.copy()
    broken.loc[0, column] = value
    assert _rules(broken) == ["R05_NUMERIC_RANGES"]


@pytest.mark.parametrize(("column", "value"), [("EDUCATION", 0), ("EDUCATION", 9), ("PAY_0", 9), ("PAY_6", -3)])
def test_r06_allowed_categories(inference, column, value):
    broken = inference.copy()
    broken.loc[0, column] = value
    assert _rules(broken) == ["R06_ALLOWED_CATEGORIES"]


def test_r06_training_only_categories(labeled):
    broken = labeled.copy()
    broken.loc[0, "SEX"] = 3
    broken.loc[1, "default_next_month"] = 2
    assert _rules(broken, "training") == ["R06_ALLOWED_CATEGORIES"]


def test_r07_duplicate_identifier(inference):
    broken = inference.copy()
    broken.loc[1, "client_id"] = broken.loc[0, "client_id"]
    assert _rules(broken) == ["R07_UNIQUE_ID"]


def test_r08_identifier_must_be_pseudonymous(inference):
    broken = inference.copy()
    broken.loc[0, "client_id"] = "123.456.789-00"
    assert _rules(broken) == ["R08_ID_FORMAT"]


def test_r09_empty_batch(inference):
    assert "R09_NON_EMPTY_BATCH" in _rules(inference.head(0))


def test_outcome_contract(labeled):
    outcomes = labeled[["client_id", "default_next_month"]]
    assert validate_batch(outcomes, "outcome", "o").is_valid
    assert not validate_batch(outcomes.assign(default_next_month=5), "outcome", "o").is_valid


def test_unknown_mode():
    with pytest.raises(ValueError, match="Unknown validation mode"):
        get_schema("scoring")


def test_classify_check_mapping():
    assert classify_check("column_in_dataframe") == "R01_REQUIRED_COLUMNS"
    assert classify_check("greater_than(0)") == "R05_NUMERIC_RANGES"
    assert classify_check("something_else") == "R99_OTHER"
    assert len(RULE_DESCRIPTIONS) == 9


def test_summarize_failures_empty():
    assert summarize_failures(pd.DataFrame()).empty
