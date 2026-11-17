"""Data contracts (Pandera) for training and inference batches.

Rules (IDs are referenced in logs, alerts, tests and documentation):

* R01_REQUIRED_COLUMNS   - every contract column must be present.
* R02_NO_UNEXPECTED_COLUMNS - strict schema; for inference this also blocks
  protected attributes (SEX, MARRIAGE) and the target (data minimization).
* R03_DTYPES             - integer dtypes for numeric/categorical codes, string id.
* R04_NOT_NULL           - no missing values in any column.
* R05_NUMERIC_RANGES     - business ranges (age, credit limit, bills, payments).
* R06_ALLOWED_CATEGORIES - documented codes for EDUCATION, PAY_x, SEX, MARRIAGE, target.
* R07_UNIQUE_ID          - no duplicated client identifiers inside a batch.
* R08_ID_FORMAT          - identifier must be a pseudonymous 16-hex-char key.
* R09_NON_EMPTY_BATCH    - a batch must contain at least one row.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pandera.pandas as pa

ID_PATTERN = r"^[0-9a-f]{16}$"
AGE_RANGE = (18, 100)
LIMIT_BAL_RANGE = (10_000, 1_000_000)
BILL_AMT_RANGE = (-500_000, 2_000_000)
PAY_AMT_RANGE = (0, 2_000_000)
PAY_STATUS_CODES = list(range(-2, 9))
EDUCATION_CODES = [1, 2, 3, 4]
SEX_CODES = [1, 2]
MARRIAGE_CODES = [1, 2, 3]
TARGET_CODES = [0, 1]

RULE_DESCRIPTIONS = {
    "R01_REQUIRED_COLUMNS": "Todas as colunas obrigatórias do contrato devem existir",
    "R02_NO_UNEXPECTED_COLUMNS": "Colunas fora do contrato são proibidas (minimização LGPD)",
    "R03_DTYPES": "Tipos de dados devem ser inteiros (códigos/valores) e string (id)",
    "R04_NOT_NULL": "Nenhuma coluna pode conter valores nulos",
    "R05_NUMERIC_RANGES": "Valores numéricos dentro dos limites de negócio",
    "R06_ALLOWED_CATEGORIES": "Códigos categóricos dentro do domínio documentado",
    "R07_UNIQUE_ID": "Identificador de cliente não pode se repetir no lote",
    "R08_ID_FORMAT": "Identificador deve ser pseudônimo (16 caracteres hexadecimais)",
    "R09_NON_EMPTY_BATCH": "Lote deve conter pelo menos um registro",
}


def _int_column(checks: list[pa.Check]) -> pa.Column:
    return pa.Column("int64", checks=checks, nullable=False, coerce=False)


def _feature_columns() -> dict[str, pa.Column]:
    columns: dict[str, pa.Column] = {
        "client_id": pa.Column(
            str,
            checks=pa.Check.str_matches(ID_PATTERN),
            nullable=False,
            unique=True,
        ),
        "LIMIT_BAL": _int_column([pa.Check.in_range(*LIMIT_BAL_RANGE)]),
        "EDUCATION": _int_column([pa.Check.isin(EDUCATION_CODES)]),
        "AGE": _int_column([pa.Check.in_range(*AGE_RANGE)]),
    }
    for col in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]:
        columns[col] = _int_column([pa.Check.isin(PAY_STATUS_CODES)])
    for i in range(1, 7):
        columns[f"BILL_AMT{i}"] = _int_column([pa.Check.in_range(*BILL_AMT_RANGE)])
    for i in range(1, 7):
        columns[f"PAY_AMT{i}"] = _int_column([pa.Check.in_range(*PAY_AMT_RANGE)])
    return columns


def _non_empty_check() -> pa.Check:
    return pa.Check(lambda frame: len(frame) > 0, element_wise=False, name="non_empty_batch")


def build_inference_schema() -> pa.DataFrameSchema:
    """Contract for production batches: features only, no target, no protected attributes."""
    return pa.DataFrameSchema(
        _feature_columns(),
        strict=True,
        ordered=False,
        checks=[_non_empty_check()],
        name="inference_contract",
    )


def build_training_schema() -> pa.DataFrameSchema:
    """Contract for labeled data: features + target + protected attributes for auditing."""
    columns = _feature_columns()
    columns["SEX"] = _int_column([pa.Check.isin(SEX_CODES)])
    columns["MARRIAGE"] = _int_column([pa.Check.isin(MARRIAGE_CODES)])
    columns["default_next_month"] = _int_column([pa.Check.isin(TARGET_CODES)])
    return pa.DataFrameSchema(
        columns,
        strict=True,
        ordered=False,
        checks=[_non_empty_check()],
        name="training_contract",
    )


def build_outcome_schema() -> pa.DataFrameSchema:
    """Contract for delayed outcome (label) files joined later to predictions."""
    return pa.DataFrameSchema(
        {
            "client_id": pa.Column(str, checks=pa.Check.str_matches(ID_PATTERN), unique=True),
            "default_next_month": _int_column([pa.Check.isin(TARGET_CODES)]),
        },
        strict=True,
        checks=[_non_empty_check()],
        name="outcome_contract",
    )


SCHEMAS = {
    "training": build_training_schema,
    "inference": build_inference_schema,
    "outcome": build_outcome_schema,
}


def get_schema(mode: str) -> pa.DataFrameSchema:
    """Return the contract for a validation mode (training, inference or outcome)."""
    if mode not in SCHEMAS:
        raise ValueError(f"Unknown validation mode '{mode}'. Use one of {sorted(SCHEMAS)}")
    return SCHEMAS[mode]()


def classify_check(check: Any) -> str:
    """Map a Pandera check identifier to the project's rule ID."""
    name = str(check)
    if name == "column_in_dataframe":
        return "R01_REQUIRED_COLUMNS"
    if name == "column_in_schema":
        return "R02_NO_UNEXPECTED_COLUMNS"
    if name.startswith("dtype("):
        return "R03_DTYPES"
    if name == "not_nullable":
        return "R04_NOT_NULL"
    if name.startswith(("in_range", "greater_than", "less_than")):
        return "R05_NUMERIC_RANGES"
    if name.startswith("isin"):
        return "R06_ALLOWED_CATEGORIES"
    if name in ("field_uniqueness", "multiple_fields_uniqueness"):
        return "R07_UNIQUE_ID"
    if name.startswith("str_matches"):
        return "R08_ID_FORMAT"
    if name == "non_empty_batch":
        return "R09_NON_EMPTY_BATCH"
    return "R99_OTHER"


def summarize_failures(failure_cases: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Pandera failure cases into one row per (rule, column, check)."""
    if failure_cases is None or failure_cases.empty:
        return pd.DataFrame(columns=["rule_id", "column", "check", "n_failures", "examples"])
    frame = failure_cases.copy()
    frame["rule_id"] = frame["check"].map(classify_check)
    schema_level = frame["check"].astype(str).isin(["column_in_schema", "column_in_dataframe"])
    frame["column"] = frame["column"].astype(object)
    frame.loc[schema_level, "column"] = frame.loc[schema_level, "failure_case"]
    frame["column"] = frame["column"].fillna(frame["failure_case"]).astype(str)
    grouped = (
        frame.groupby(["rule_id", "column", "check"], dropna=False)["failure_case"]
        .agg(n_failures="size", examples=lambda s: ", ".join(map(str, list(dict.fromkeys(s))[:5])))
        .reset_index()
        .sort_values(["rule_id", "column"])
        .reset_index(drop=True)
    )
    return grouped
