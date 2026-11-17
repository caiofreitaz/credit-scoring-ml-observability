"""Stage-level observability: status, timing, volumes and basic resource usage."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import psutil

from src.monitoring.alerts import AlertManager, rule_stage_failure
from src.monitoring.logger import get_logger


def resource_snapshot() -> dict[str, float]:
    """Process-level resource usage of the local Python process (not infrastructure)."""
    process = psutil.Process(os.getpid())
    with process.oneshot():
        memory = process.memory_info()
        cpu = process.cpu_times()
    return {
        "process_rss_mb": round(memory.rss / 1024**2, 2),
        "process_cpu_time_s": round(cpu.user + cpu.system, 3),
    }


@dataclass
class StageRecord:
    """Execution record of one pipeline stage."""

    name: str
    status: str = "RUNNING"
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    finished_at: str | None = None
    duration_s: float | None = None
    records_in: int | None = None
    records_out: int | None = None
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    resources: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StageTracker:
    """Context-manager based tracker for pipeline stages."""

    def __init__(self, alert_manager: AlertManager | None = None) -> None:
        self.records: list[StageRecord] = []
        self.alert_manager = alert_manager

    @contextmanager
    def stage(self, name: str, records_in: int | None = None) -> Iterator[StageRecord]:
        logger = get_logger()
        record = StageRecord(name=name, records_in=records_in)
        self.records.append(record)
        logger.info(f"stage '{name}' started", extra={"stage": name, "status": "RUNNING"})
        start = time.perf_counter()
        try:
            yield record
        except Exception as exc:
            record.status = "FAILED"
            record.error = f"{type(exc).__name__}: {exc}"
            raise
        else:
            if record.status == "RUNNING":
                record.status = "SUCCESS"
        finally:
            record.duration_s = round(time.perf_counter() - start, 4)
            record.finished_at = datetime.now(UTC).isoformat()
            record.resources = resource_snapshot()
            level = 40 if record.status == "FAILED" else 20
            logger.log(
                level,
                f"stage '{name}' finished with status {record.status}",
                extra={
                    "stage": name,
                    "status": record.status,
                    "duration_s": record.duration_s,
                    "records_in": record.records_in,
                    "records_out": record.records_out,
                    "error": record.error,
                    **record.resources,
                },
            )
            if (
                record.status == "FAILED"
                and self.alert_manager is not None
                and not record.details.get("expected_failure")
            ):
                self.alert_manager.emit(rule_stage_failure(name, record.error or "unknown"))

    def summary(self) -> dict[str, Any]:
        statuses = [r.status for r in self.records]
        return {
            "n_stages": len(self.records),
            "n_success": statuses.count("SUCCESS"),
            "n_failed": statuses.count("FAILED"),
            "n_blocked": statuses.count("BLOCKED"),
            "total_duration_s": round(sum(r.duration_s or 0 for r in self.records), 3),
            "stages": [r.to_dict() for r in self.records],
        }
