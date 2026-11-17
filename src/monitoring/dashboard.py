"""Consolidated, self-contained HTML monitoring dashboard (no external assets)."""

from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Any

from src.config import ProjectPaths

DASHBOARD_FILE = "monitoring_dashboard.html"
SCENARIO_LABELS = {
    "normal": "Normal",
    "data_drift": "Data Drift",
    "concept_drift": "Concept Drift",
    "data_and_concept_drift": "Data + Concept Drift",
}
FIGURES = [
    ("psi_heatmap.png", "PSI por feature e cenário", "Onde a distribuição mudou (P(X)) e o score (prediction drift)."),
    ("scenario_auc.png", "ROC-AUC por cenário", "Desempenho com outcomes maduros versus a referência."),
    (
        "calibration_by_score_decile.png",
        "P(Y | score) por decil",
        "Data Drift preserva a curva; Concept Drift a desloca para cima.",
    ),
    ("alerts_by_scenario.png", "Alertas por severidade", "Regras objetivas INFO / WARNING / CRITICAL."),
    ("shift_limit_bal.png", "LIMIT_BAL (data drift)", "Mais clientes com limite baixo."),
    ("shift_age.png", "AGE (data drift)", "Mais clientes jovens."),
    ("shift_prediction.png", "Distribuição do score", "Prediction drift decorrente do Data Drift."),
    ("invalid_batch_violations.png", "Lote inválido bloqueado", "Violações por regra do contrato Pandera."),
    ("roc_reference.png", "ROC do baseline", "Conjunto de referência (holdout)."),
    ("confusion_matrix_reference.png", "Matriz de confusão do baseline", "Conjunto de referência (holdout)."),
    ("model_coefficients.png", "Coeficientes do modelo", "Interpretabilidade do baseline."),
    ("fairness_approval_rate.png", "Aprovação por grupo", "Auditoria de vieses (atributos fora do modelo)."),
]
SEVERITY_ICON = {"INFO": "&#9679;", "WARNING": "&#9650;", "CRITICAL": "&#9632;"}

CSS = """
:root{--surface:#fcfcfb;--card:#ffffff;--text:#0b0b0b;--muted:#52514e;--line:#e4e3df;--blue:#2a78d6;
--info:#0ca30c;--warning:#b37800;--critical:#d03b3b}
*{box-sizing:border-box}body{margin:0;font-family:Segoe UI,system-ui,-apple-system,Roboto,Arial,sans-serif;
background:var(--surface);color:var(--text);font-size:14px;line-height:1.45}
header{padding:28px 36px 18px;border-bottom:1px solid var(--line);background:var(--card)}
h1{margin:0 0 6px;font-size:24px}h2{font-size:18px;margin:34px 0 12px}.muted{color:var(--muted)}
main{padding:8px 36px 48px;max-width:1400px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-top:18px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.tile .label{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.tile .value{font-size:26px;font-weight:600;margin-top:4px}.tile .sub{font-size:12px;color:var(--muted)}
table{border-collapse:collapse;width:100%;background:var(--card);border:1px solid var(--line);border-radius:10px;
overflow:hidden}th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.03em;background:#f6f5f2}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.badge{display:inline-block;white-space:nowrap;padding:2px 8px;border-radius:999px;font-size:12px;
font-weight:600;border:1px solid}
.INFO{color:var(--info);border-color:var(--info)}.WARNING{color:var(--warning);border-color:#fab219}
.CRITICAL{color:var(--critical);border-color:var(--critical)}.SUCCESS{color:var(--info);border-color:var(--info)}
.FAILED{color:var(--critical);border-color:var(--critical)}.BLOCKED{color:var(--warning);border-color:#fab219}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px}
figure{margin:0;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
figure img{width:100%;height:auto}figcaption{font-size:12px;color:var(--muted);margin-top:6px}
.note{background:#f6f5f2;border-left:4px solid var(--blue);padding:10px 14px;border-radius:6px}
code{background:#f0efec;padding:1px 5px;border-radius:4px}a{color:#1c5cab}
"""


def _img(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img alt="{html.escape(path.stem)}" src="data:image/png;base64,{data}">'


def _badge(text: str, css: str | None = None) -> str:
    css = css or text
    icon = SEVERITY_ICON.get(text, "")
    return f'<span class="badge {html.escape(css)}">{icon} {html.escape(text)}</span>'


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if value.is_integer() and abs(value) >= 1:
            return str(int(value))
        if abs(value) < 1e-3 and value != 0:
            return f"{value:.2e}"
        return f"{value:.{digits}f}"
    return html.escape(str(value))


def _tile(label: str, value: Any, sub: str = "") -> str:
    return (
        f'<div class="tile"><div class="label">{html.escape(label)}</div><div class="value">{value}</div>'
        f'<div class="sub">{sub}</div></div>'
    )


def _scenario_rows(scenarios: list[dict]) -> str:
    rows = []
    for s in scenarios:
        perf = s.get("performance") or {}
        current = perf.get("current", {})
        ci = perf.get("roc_auc_ci95") or [None, None]
        alerts = s["alerts"]
        alert_html = " ".join(_badge(f"{k} {v}", k) for k, v in alerts.items() if v)
        report = s["evidently"]["html"]
        rows.append(
            "<tr>"
            f"<td><strong>{SCENARIO_LABELS.get(s['scenario'], s['scenario'])}</strong><br>"
            f"<span class='muted'>{html.escape(s.get('description', ''))}</span></td>"
            f"<td class='num'>{s['rows']}</td>"
            f"<td class='num'>{s['n_drifted_features']} / {s['n_features_monitored']}<br>"
            f"<span class='muted'>{s['n_severe_drift']} severas</span></td>"
            f"<td class='num'>{_fmt(s['prediction_drift']['prediction_psi'], 3)}</td>"
            f"<td class='num'>{_fmt(current.get('roc_auc'), 4)}<br><span class='muted'>IC95% "
            f"{_fmt(ci[0], 3)}-{_fmt(ci[1], 3)}</span></td>"
            f"<td class='num'>{_fmt(perf.get('auc_drop'), 4)}</td>"
            f"<td class='num'>{_fmt(s['calibration'].get('calibration_pvalue'))}</td>"
            f"<td class='num'>{_fmt(current.get('observed_default_rate'), 3)}</td>"
            f"<td>{html.escape(s['diagnosis']['diagnosis'])}</td>"
            f"<td>{alert_html}</td>"
            f"<td><a href='../{html.escape(report)}'>Evidently</a></td>"
            "</tr>"
        )
    return "\n".join(rows)


def _alert_rows(alerts: list[dict]) -> str:
    order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
    rows = []
    for a in sorted(alerts, key=lambda item: (order[item["severity"]], item["scope"])):
        rows.append(
            f"<tr><td>{_badge(a['severity'])}</td><td><code>{html.escape(a['code'])}</code></td>"
            f"<td>{html.escape(a['scope'])}</td><td>{html.escape(a['message'])}</td>"
            f"<td class='num'>{_fmt(a.get('value'))}</td><td class='num'>{_fmt(a.get('threshold'))}</td></tr>"
        )
    return "\n".join(rows)


def _stage_rows(stages: list[dict]) -> str:
    rows = []
    for st in stages:
        rows.append(
            f"<tr><td>{html.escape(st['name'])}</td><td>{_badge(st['status'])}</td>"
            f"<td class='num'>{_fmt(st.get('duration_s'), 3)}</td><td class='num'>{_fmt(st.get('records_in'))}</td>"
            f"<td class='num'>{_fmt(st.get('records_out'))}</td>"
            f"<td class='num'>{_fmt(st.get('resources', {}).get('process_rss_mb'), 1)}</td>"
            f"<td>{html.escape(st.get('started_at') or '')}</td></tr>"
        )
    return "\n".join(rows)


def _batch_rows(batches: list[dict]) -> str:
    rows = []
    for b in batches:
        status = "SUCCESS" if b["status"] == "accepted" else "BLOCKED"
        rules = ", ".join(b.get("rules_violated") or []) or "-"
        rows.append(
            f"<tr><td>{html.escape(b['batch'])}</td><td>{_badge(b['status'].upper(), status)}</td>"
            f"<td class='num'>{b['n_rows']}</td><td class='num'>{b['n_failures']}</td><td>{html.escape(rules)}</td>"
            f"<td class='num'>{b['predictions_generated']}</td></tr>"
        )
    return "\n".join(rows)


def build_dashboard(summary: dict[str, Any], paths: ProjectPaths) -> Path:
    """Render the dashboard to reports/monitoring_dashboard.html."""
    figures_dir = paths.reports_dir / "figures"
    model = summary.get("model") or {}
    ref = model.get("reference_metrics") or {}
    pipeline = summary["pipeline"]
    alerts = summary["alerts_total"]
    batches = summary.get("batches", [])
    accepted = sum(b["status"] == "accepted" for b in batches)
    rejected = sum(b["status"] == "rejected" for b in batches)

    tiles = "".join(
        [
            _tile("Status do pipeline", _badge(summary["status"]), f"run_id {html.escape(summary['run_id'])}"),
            _tile(
                "Etapas",
                f"{pipeline['n_success']}/{pipeline['n_stages']}",
                f"{pipeline['n_failed']} falhas · {pipeline['n_blocked']} bloqueios intencionais",
            ),
            _tile("Lotes", f"{accepted} aceitos", f"{rejected} rejeitados pelo contrato"),
            _tile("Alertas CRITICAL", alerts["CRITICAL"], f"{alerts['WARNING']} WARNING · {alerts['INFO']} INFO"),
            _tile(
                "ROC-AUC referência",
                _fmt(ref.get("roc_auc"), 4),
                f"F1 {_fmt(ref.get('f1'), 3)} · limiar {_fmt(model.get('threshold'), 2)}",
            ),
            _tile(
                "Modelo",
                f"v{html.escape(str(model.get('registered_model_version') or '-'))}",
                html.escape(str(model.get("model_version") or "-")),
            ),
        ]
    )
    figure_html = "".join(
        f"<figure>{_img(figures_dir / name)}<figcaption><strong>{html.escape(title)}</strong> - "
        f"{html.escape(caption)}</figcaption></figure>"
        for name, title, caption in FIGURES
        if (figures_dir / name).exists()
    )
    page = f"""<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Credit Scoring - Monitoramento</title><style>{CSS}</style></head>
<body><header><h1>Credit Scoring · Painel de Confiabilidade do Modelo</h1>
<div class="muted">Execução <code>{html.escape(summary["run_id"])}</code> · gerado em
{html.escape(summary["generated_at"])} · modelo <code>{html.escape(str(model.get("model_version")))}</code></div>
<div class="tiles">{tiles}</div></header>
<main>
<h2>Comparação entre cenários de produção</h2>
<table><thead><tr><th>Cenário</th><th>Registros</th><th>Features com drift</th><th>PSI do score</th>
<th>ROC-AUC</th><th>Queda AUC</th><th>p-valor calibração</th><th>Inadimplência obs.</th><th>Diagnóstico</th>
<th>Alertas</th><th>Relatório</th></tr></thead><tbody>{_scenario_rows(summary.get("scenarios", []))}</tbody></table>
<p class="note"><strong>Como ler:</strong> <em>Data Drift</em> = mudança em P(X) (PSI &ge; 0,10 e teste KS/qui-quadrado
significativo). <em>Prediction Drift</em> = mudança na distribuição do score. <em>Concept Drift</em> = mudança em
P(Y|X), evidenciada pela mudança da inadimplência observada dentro dos mesmos decis de score (teste de calibração),
o que exige outcomes observados (que chegam com atraso). <em>Degradação</em> = queda de ROC-AUC &ge; 0,02 (WARNING) ou
&ge; 0,05 (CRITICAL).</p>

<h2>Evidências visuais</h2><div class="grid">{figure_html}</div>

<h2>Contratos de dados (gate de ingestão/inferência)</h2>
<table><thead><tr><th>Lote</th><th>Status</th><th>Registros</th><th>Falhas</th><th>Regras violadas</th>
<th>Predições geradas</th></tr></thead><tbody>{_batch_rows(batches)}</tbody></table>

<h2>Alertas emitidos</h2>
<table><thead><tr><th>Severidade</th><th>Código</th><th>Escopo</th><th>Mensagem</th><th>Valor</th><th>Limiar</th>
</tr></thead><tbody>{_alert_rows(summary.get("alerts", []))}</tbody></table>

<h2>Saúde operacional das etapas</h2>
<table><thead><tr><th>Etapa</th><th>Status</th><th>Duração (s)</th><th>Entrada</th><th>Saída</th><th>RSS (MB)</th>
<th>Início (UTC)</th></tr></thead><tbody>{_stage_rows(pipeline["stages"])}</tbody></table>

<h2>Escopo e limitações</h2>
<p class="note">As métricas operacionais (status, duração, volumes, latência, memória do processo) são medidas
<strong>localmente</strong> na execução do pipeline Python. Não existe neste projeto monitoramento de servidores,
APIs ou infraestrutura de produção. Detalhes de cada execução estão no MLflow (<code>mlflow ui --backend-store-uri
sqlite:///mlflow.db</code>) e nos logs estruturados em <code>logs/pipeline.jsonl</code>, correlacionados pelo
<code>run_id</code>.</p>
</main></body></html>"""
    target = paths.reports_dir / DASHBOARD_FILE
    target.write_text(page, encoding="utf-8")
    return target
