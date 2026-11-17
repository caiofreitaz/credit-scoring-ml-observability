"""Alert model, severity levels and objective alerting rules."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from src.monitoring.logger import get_logger, get_run_id


class Severity(StrEnum):
    """Alert severity levels."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


_LOG_LEVEL = {
    Severity.INFO: logging.INFO,
    Severity.WARNING: logging.WARNING,
    Severity.CRITICAL: logging.ERROR,
}


@dataclass
class Alert:
    """A single operational alert."""

    severity: Severity
    code: str
    message: str
    scope: str
    metric: str | None = None
    value: float | None = None
    threshold: float | None = None
    run_id: str = field(default_factory=get_run_id)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["severity"] = self.severity.value
        return payload


class AlertManager:
    """Collects alerts during a run, logs them and persists them to JSON."""

    def __init__(self) -> None:
        self.alerts: list[Alert] = []

    def emit(self, alert: Alert) -> Alert:
        self.alerts.append(alert)
        get_logger().log(
            _LOG_LEVEL[alert.severity],
            f"[ALERT {alert.severity.value}] {alert.code}: {alert.message}",
            extra={"alert": alert.to_dict(), "stage": "alerting"},
        )
        return alert

    def extend(self, alerts: list[Alert]) -> None:
        for alert in alerts:
            self.emit(alert)

    def counts(self, scope: str | None = None) -> dict[str, int]:
        selected = [a for a in self.alerts if scope is None or a.scope == scope]
        return {sev.value: sum(1 for a in selected if a.severity == sev) for sev in Severity}

    def for_scope(self, scope: str) -> list[Alert]:
        return [a for a in self.alerts if a.scope == scope]

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps([a.to_dict() for a in self.alerts], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return target


# ---------------------------------------------------------------------------
# Objective alert rules
# ---------------------------------------------------------------------------


def rule_contract_violation(batch_name: str, n_failures: int, rules: list[str]) -> Alert:
    """Any contract violation blocks the batch -> CRITICAL."""
    return Alert(
        severity=Severity.CRITICAL,
        code="DATA_CONTRACT_VIOLATION",
        message=(
            f"Lote '{batch_name}' rejeitado: {n_failures} violações em {len(rules)} regras "
            f"({', '.join(rules)}). Inferência bloqueada."
        ),
        scope=batch_name,
        metric="contract_failures",
        value=float(n_failures),
        threshold=0.0,
    )


def rule_data_drift(
    scope: str, n_moderate: int, n_severe: int, features: list[str], cfg: dict, severe_features: list[str] | None = None
) -> Alert:
    """Severe drift in >=1 feature -> CRITICAL; moderate only -> WARNING; none -> INFO."""
    severe_features = severe_features or []
    moderate_features = [f for f in features if f not in severe_features]
    if n_severe > 0:
        severity, code = Severity.CRITICAL, "DATA_DRIFT_SEVERE"
        message = (
            f"Drift severo (PSI >= {cfg['psi_severe']}) em {n_severe} feature(s): {', '.join(severe_features)}; "
            f"drift moderado em {n_moderate}: {', '.join(moderate_features) or '-'}"
        )
    elif n_moderate > 0:
        severity, code = Severity.WARNING, "DATA_DRIFT_MODERATE"
        message = f"Drift moderado (PSI >= {cfg['psi_moderate']}) em {n_moderate} feature(s): {', '.join(features)}"
    else:
        severity, code = Severity.INFO, "DATA_DRIFT_NONE"
        message = "Nenhuma feature com drift relevante (PSI < limiar ou teste não significativo)"
    return Alert(severity, code, message, scope, "n_drifted_features", float(n_moderate + n_severe), 0.0)


def rule_prediction_drift(scope: str, prediction_psi: float, cfg: dict) -> Alert | None:
    """PSI of the score distribution against reference."""
    if prediction_psi >= cfg["prediction_psi_critical"]:
        severity, threshold = Severity.CRITICAL, cfg["prediction_psi_critical"]
    elif prediction_psi >= cfg["prediction_psi_warning"]:
        severity, threshold = Severity.WARNING, cfg["prediction_psi_warning"]
    else:
        return None
    return Alert(
        severity,
        "PREDICTION_DRIFT",
        f"Distribuição dos scores mudou (PSI={prediction_psi:.3f})",
        scope,
        "prediction_psi",
        round(prediction_psi, 4),
        threshold,
    )


def rule_performance(scope: str, auc_reference: float, auc_current: float, cfg: dict) -> Alert | None:
    """ROC-AUC drop against reference once outcomes are available."""
    drop = auc_reference - auc_current
    if drop >= cfg["auc_drop_critical"]:
        severity, threshold = Severity.CRITICAL, cfg["auc_drop_critical"]
    elif drop >= cfg["auc_drop_warning"]:
        severity, threshold = Severity.WARNING, cfg["auc_drop_warning"]
    else:
        return None
    return Alert(
        severity,
        "PERFORMANCE_DEGRADATION",
        f"ROC-AUC caiu de {auc_reference:.4f} para {auc_current:.4f} (queda {drop:.4f})",
        scope,
        "roc_auc_drop",
        round(drop, 4),
        threshold,
    )


def rule_default_rate_shift(scope: str, reference_rate: float, current_rate: float, cfg: dict) -> Alert | None:
    """Observed default-rate change in percentage points."""
    shift_pp = (current_rate - reference_rate) * 100
    if abs(shift_pp) < cfg["default_rate_shift_warning_pp"]:
        return None
    return Alert(
        Severity.WARNING,
        "DEFAULT_RATE_SHIFT",
        f"Taxa de inadimplência observada mudou {shift_pp:+.1f} p.p. ({reference_rate:.1%} -> {current_rate:.1%})",
        scope,
        "default_rate_shift_pp",
        round(shift_pp, 2),
        cfg["default_rate_shift_warning_pp"],
    )


def rule_concept_drift(scope: str, calibration_pvalue: float, performance_degraded: bool, cfg: dict) -> Alert | None:
    """P(Y|score) changed -> suspected; plus performance degradation -> confirmed (CRITICAL)."""
    if calibration_pvalue >= cfg["calibration_pvalue_threshold"]:
        return None
    if performance_degraded:
        return Alert(
            Severity.CRITICAL,
            "CONCEPT_DRIFT_CONFIRMED",
            "Relação score -> inadimplência mudou (teste de calibração significativo) "
            "e houve degradação de performance com outcomes observados",
            scope,
            "calibration_pvalue",
            float(calibration_pvalue),
            cfg["calibration_pvalue_threshold"],
        )
    return Alert(
        Severity.WARNING,
        "CONCEPT_DRIFT_SUSPECTED",
        "Relação score -> inadimplência mudou (teste de calibração significativo), sem queda relevante de AUC",
        scope,
        "calibration_pvalue",
        float(calibration_pvalue),
        cfg["calibration_pvalue_threshold"],
    )


def rule_latency(scope: str, latency_ms_per_1k: float, cfg: dict) -> Alert | None:
    """Inference latency per 1,000 records above the operational threshold."""
    if latency_ms_per_1k <= cfg["latency_ms_per_1k_warning"]:
        return None
    return Alert(
        Severity.WARNING,
        "HIGH_INFERENCE_LATENCY",
        f"Latência de inferência {latency_ms_per_1k:.1f} ms por 1.000 registros",
        scope,
        "latency_ms_per_1k",
        round(latency_ms_per_1k, 2),
        cfg["latency_ms_per_1k_warning"],
    )


def rule_stage_failure(stage: str, error: str) -> Alert:
    """Any pipeline stage exception -> CRITICAL."""
    return Alert(Severity.CRITICAL, "PIPELINE_STAGE_FAILED", f"Etapa '{stage}' falhou: {error}", stage)
