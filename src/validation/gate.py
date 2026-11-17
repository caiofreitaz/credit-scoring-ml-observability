"""Validation gate: accepts valid batches and blocks (quarantines) invalid ones."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from src.config import ProjectPaths
from src.monitoring.alerts import AlertManager, rule_contract_violation
from src.monitoring.logger import get_logger, get_run_id
from src.validation.contracts import RULE_DESCRIPTIONS, get_schema, summarize_failures


class DataContractViolation(Exception):  # noqa: N818 - domain name used in docs
    """Raised when a batch violates its data contract; carries the validation result."""

    def __init__(self, result: ValidationResult) -> None:
        self.result = result
        super().__init__(
            f"Batch '{result.batch_name}' violated the {result.mode} contract "
            f"({result.n_failures} failures: {', '.join(result.rules_violated)})"
        )


@dataclass
class ValidationResult:
    """Outcome of validating one batch against a contract."""

    batch_name: str
    mode: str
    is_valid: bool
    n_rows: int
    n_failures: int
    rules_violated: list[str]
    duration_s: float
    summary: pd.DataFrame = field(repr=False)
    failure_cases: pd.DataFrame = field(repr=False)

    def to_dict(self) -> dict:
        return {
            "run_id": get_run_id(),
            "batch_name": self.batch_name,
            "mode": self.mode,
            "is_valid": self.is_valid,
            "n_rows": self.n_rows,
            "n_failures": self.n_failures,
            "rules_violated": self.rules_violated,
            "rule_descriptions": {r: RULE_DESCRIPTIONS.get(r, "") for r in self.rules_violated},
            "duration_s": round(self.duration_s, 4),
            "violations": self.summary.to_dict(orient="records"),
        }


def validate_batch(frame: pd.DataFrame, mode: str, batch_name: str) -> ValidationResult:
    """Validate a DataFrame against the contract for ``mode`` without side effects."""
    schema = get_schema(mode)
    start = time.perf_counter()
    try:
        schema.validate(frame, lazy=True)
        failure_cases = pd.DataFrame()
    except pa.errors.SchemaErrors as errors:
        failure_cases = errors.failure_cases
    except pa.errors.SchemaError as error:  # non-lazy errors (e.g. empty frame edge cases)
        failure_cases = pd.DataFrame(
            [
                {
                    "schema_context": "DataFrameSchema",
                    "column": None,
                    "check": str(error.check),
                    "check_number": None,
                    "failure_case": str(error),
                    "index": None,
                }
            ]
        )
    duration = time.perf_counter() - start
    summary = summarize_failures(failure_cases)
    rules = sorted(summary["rule_id"].unique().tolist()) if not summary.empty else []
    return ValidationResult(
        batch_name=batch_name,
        mode=mode,
        is_valid=summary.empty,
        n_rows=len(frame),
        n_failures=len(failure_cases),
        rules_violated=rules,
        duration_s=duration,
        summary=summary,
        failure_cases=failure_cases,
    )


def enforce_contract(
    frame: pd.DataFrame,
    mode: str,
    batch_name: str,
    paths: ProjectPaths,
    alert_manager: AlertManager | None = None,
) -> ValidationResult:
    """Validate and enforce: persist evidence; on failure quarantine, alert and raise."""
    logger = get_logger()
    result = validate_batch(frame, mode, batch_name)
    report_dir = paths.reports_dir / "validation"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{batch_name}_validation.json").write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if result.is_valid:
        logger.info(
            f"batch '{batch_name}' accepted by {mode} contract",
            extra={
                "stage": "validation",
                "batch": batch_name,
                "rows": result.n_rows,
                "status": "accepted",
                "duration_s": round(result.duration_s, 4),
            },
        )
        return result

    result.failure_cases.to_csv(report_dir / f"{batch_name}_failure_cases.csv", index=False)
    quarantine_path = _quarantine(frame, batch_name, paths.quarantine_dir)
    logger.error(
        f"batch '{batch_name}' REJECTED by {mode} contract",
        extra={
            "stage": "validation",
            "batch": batch_name,
            "rows": result.n_rows,
            "status": "rejected",
            "n_failures": result.n_failures,
            "rules_violated": result.rules_violated,
            "quarantine_path": str(quarantine_path),
        },
    )
    if alert_manager is not None:
        alert_manager.emit(rule_contract_violation(batch_name, result.n_failures, result.rules_violated))
    raise DataContractViolation(result)


def _quarantine(frame: pd.DataFrame, batch_name: str, quarantine_dir: Path) -> Path:
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    target = quarantine_dir / f"{batch_name}__{get_run_id()}.csv"
    frame.to_csv(target, index=False)
    return target


def read_batch(path: str | Path) -> pd.DataFrame:
    """Read a CSV batch keeping the identifier as string."""
    return pd.read_csv(path, dtype={"client_id": str})
