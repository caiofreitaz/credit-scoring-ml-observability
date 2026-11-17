"""Structured JSON logging correlated by a pipeline run identifier."""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_RUN_ID: contextvars.ContextVar[str] = contextvars.ContextVar("run_id", default="no-run")
LOGGER_NAME = "credit_monitoring"
_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


def new_run_id() -> str:
    """Create a sortable, unique run identifier (UTC timestamp + random suffix)."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def set_run_id(run_id: str) -> None:
    """Bind the run identifier to the current execution context."""
    _RUN_ID.set(run_id)


def get_run_id() -> str:
    """Return the run identifier bound to the current execution context."""
    return _RUN_ID.get()


class JsonFormatter(logging.Formatter):
    """Render log records as one JSON object per line (JSON Lines)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "run_id": get_run_id(),
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class ConsoleFormatter(logging.Formatter):
    """Compact human-readable console output."""

    def format(self, record: logging.LogRecord) -> str:
        stamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        return f"{stamp} | {record.levelname:<8} | {record.getMessage()}"


def configure_logging(log_dir: str | Path, level: int = logging.INFO) -> logging.Logger:
    """Configure the project logger with a JSONL file handler and a console handler."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    file_handler = logging.FileHandler(log_path / "pipeline.jsonl", encoding="utf-8")
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(ConsoleFormatter())
    logger.addHandler(console)
    return logger


def get_logger() -> logging.Logger:
    """Return the project logger (configure_logging must be called by entry points)."""
    return logging.getLogger(LOGGER_NAME)


def read_log_events(log_file: str | Path, run_id: str | None = None) -> list[dict[str, Any]]:
    """Read JSONL log events, optionally filtered by run identifier."""
    path = Path(log_file)
    if not path.exists():
        return []
    events = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            if run_id is None or event.get("run_id") == run_id:
                events.append(event)
    return events
