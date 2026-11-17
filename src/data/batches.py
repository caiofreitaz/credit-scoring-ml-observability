"""Construction of demo batches: a valid inference batch and an intentionally invalid one."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import ProjectPaths, model_feature_columns

VALID_BATCH_FILE = "batch_valid.csv"
INVALID_BATCH_FILE = "batch_invalid.csv"

# Documented list of injected violations (row position -> description).
INJECTED_VIOLATIONS = [
    ("AGE", 15, "R05_NUMERIC_RANGES: idade abaixo de 18 anos"),
    ("AGE", 130, "R05_NUMERIC_RANGES: idade acima de 100 anos"),
    ("LIMIT_BAL", None, "R04_NOT_NULL + R03_DTYPES: limite de crédito ausente"),
    ("EDUCATION", 9, "R06_ALLOWED_CATEGORIES: código de escolaridade inexistente"),
    ("PAY_0", 12, "R06_ALLOWED_CATEGORIES: status de pagamento fora do domínio"),
    ("PAY_AMT1", -500, "R05_NUMERIC_RANGES: pagamento negativo"),
    ("BILL_AMT1", 9_999_999, "R05_NUMERIC_RANGES: fatura acima do limite de negócio"),
    ("client_id", "DUPLICATE", "R07_UNIQUE_ID: identificador duplicado"),
    ("client_id", "CLIENTE-123", "R08_ID_FORMAT: identificador não pseudonimizado"),
    ("SEX", "EXTRA_COLUMN", "R02_NO_UNEXPECTED_COLUMNS: atributo protegido enviado na inferência"),
]


def to_inference_frame(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Keep only the identifier and model features (drops target and protected attributes)."""
    return frame[[config["features"]["id_column"], *model_feature_columns(config)]].copy()


def build_valid_batch(production_pool: pd.DataFrame, config: dict[str, Any], size: int = 500) -> pd.DataFrame:
    """Sample a clean inference batch from the production pool."""
    sample = production_pool.sample(n=size, random_state=config["project"]["seed"])
    return to_inference_frame(sample, config).reset_index(drop=True)


def build_invalid_batch(production_pool: pd.DataFrame, config: dict[str, Any], size: int = 200) -> pd.DataFrame:
    """Create a batch with one injected violation per documented rule."""
    seed = config["project"]["seed"] + 1
    batch = to_inference_frame(production_pool.sample(n=size, random_state=seed), config).reset_index(drop=True)
    batch["LIMIT_BAL"] = batch["LIMIT_BAL"].astype("float64")
    row = 0
    for column, value, _ in INJECTED_VIOLATIONS:
        if column == "SEX":
            batch["SEX"] = np.where(np.arange(len(batch)) % 2 == 0, 1, 2)
            continue
        if value == "DUPLICATE":
            batch.loc[row, column] = batch.loc[row + 1, column]
        elif value is None:
            batch.loc[row, column] = np.nan
        else:
            if isinstance(value, str):
                batch[column] = batch[column].astype(object)
            batch.loc[row, column] = value
        row += 2
    return batch


def write_demo_batches(production_pool: pd.DataFrame, config: dict[str, Any], paths: ProjectPaths) -> dict[str, str]:
    """Persist the valid and invalid demo batches; return their paths."""
    paths.production_dir.mkdir(parents=True, exist_ok=True)
    paths.invalid_dir.mkdir(parents=True, exist_ok=True)
    valid_path = paths.production_dir / VALID_BATCH_FILE
    invalid_path = paths.invalid_dir / INVALID_BATCH_FILE
    build_valid_batch(production_pool, config).to_csv(valid_path, index=False, lineterminator="\n")
    build_invalid_batch(production_pool, config).to_csv(invalid_path, index=False, lineterminator="\n")
    return {"valid": str(valid_path), "invalid": str(invalid_path)}
