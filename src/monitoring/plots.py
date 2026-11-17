"""Static figures (PNG) for reports, README and the consolidated dashboard."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve

REFERENCE_COLOR = "#2a78d6"  # categorical slot 1 (blue)
CURRENT_COLOR = "#eb6834"  # categorical slot 2 (orange)
THIRD_COLOR = "#1baf7a"  # categorical slot 3 (aqua)
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
STATUS = {"INFO": "#0ca30c", "WARNING": "#fab219", "CRITICAL": "#d03b3b"}
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"]
)
SCENARIO_LABELS = {
    "normal": "Normal",
    "data_drift": "Data Drift",
    "concept_drift": "Concept Drift",
    "data_and_concept_drift": "Data + Concept",
}


def _style(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, loc="left", fontsize=12, color=TEXT_PRIMARY, fontweight="bold")
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig: plt.Figure, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(target, dpi=110, facecolor="white")
    plt.close(fig)
    return target


def plot_class_balance(target: pd.Series, path: str | Path) -> Path:
    counts = target.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    bars = ax.bar(["Adimplente (0)", "Inadimplente (1)"], counts.values, color=REFERENCE_COLOR, width=0.55)
    for bar, value in zip(bars, counts.values, strict=True):
        ax.annotate(
            f"{value:,} ({value / counts.sum():.1%})".replace(",", "."),
            (bar.get_x() + bar.get_width() / 2, value),
            ha="center",
            va="bottom",
            fontsize=9,
            color=TEXT_PRIMARY,
        )
    _style(ax, "Distribuição da variável target", ylabel="Clientes")
    return _save(fig, path)


def plot_default_rate_by(frame: pd.DataFrame, column: str, target: str, path: str | Path, title: str) -> Path:
    rates = frame.groupby(column)[target].agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(rates.index.astype(str), rates["mean"] * 100, color=REFERENCE_COLOR, width=0.6)
    _style(ax, title, xlabel=column, ylabel="Taxa de inadimplência (%)")
    return _save(fig, path)


def plot_roc(y_true, proba, path: str | Path, auc: float) -> Path:
    fpr, tpr, _ = roc_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.plot(fpr, tpr, color=REFERENCE_COLOR, linewidth=2, label=f"Baseline (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=TEXT_SECONDARY, linewidth=1, linestyle="--", label="Aleatório")
    ax.legend(frameon=False, fontsize=9)
    _style(ax, "Curva ROC - conjunto de referência", "Taxa de falsos positivos", "Taxa de verdadeiros positivos")
    return _save(fig, path)


def plot_confusion(metrics: dict, path: str | Path, title: str) -> Path:
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.imshow(matrix, cmap=SEQUENTIAL)
    for (i, j), value in np.ndenumerate(matrix):
        color = "white" if value > matrix.max() * 0.55 else TEXT_PRIMARY
        ax.text(j, i, f"{value:,}".replace(",", "."), ha="center", va="center", fontsize=12, color=color)
    ax.set_xticks([0, 1], ["Prev. 0", "Prev. 1"])
    ax.set_yticks([0, 1], ["Real 0", "Real 1"])
    ax.set_title(title, loc="left", fontsize=12, color=TEXT_PRIMARY, fontweight="bold")
    ax.tick_params(colors=TEXT_SECONDARY)
    return _save(fig, path)


def plot_coefficients(coefs: pd.DataFrame, path: str | Path, top: int = 12) -> Path:
    data = coefs.head(top).iloc[::-1]
    colors = [CURRENT_COLOR if c > 0 else REFERENCE_COLOR for c in data["coefficient"]]
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.barh(data["feature"], data["coefficient"], color=colors, height=0.6)
    ax.axvline(0, color=TEXT_SECONDARY, linewidth=0.8)
    ax.grid(axis="x", color=GRID)
    _style(ax, "Coeficientes (laranja aumenta risco, azul reduz)", xlabel="Coeficiente padronizado")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def plot_psi_heatmap(psi_table: pd.DataFrame, path: str | Path, moderate: float, severe: float) -> Path:
    """psi_table: index = feature, columns = scenarios."""
    data = psi_table.copy()
    data.columns = [SCENARIO_LABELS.get(c, c) for c in data.columns]
    fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(data) + 1.6))
    image = ax.imshow(np.clip(data.values, 0, 0.5), cmap=SEQUENTIAL, aspect="auto", vmin=0, vmax=0.5)
    ax.set_xticks(range(len(data.columns)), data.columns, fontsize=9)
    ax.set_yticks(range(len(data.index)), data.index, fontsize=8)
    for (i, j), value in np.ndenumerate(data.values):
        marker = " !!" if value >= severe else (" !" if value >= moderate else "")
        color = "white" if value > 0.3 else TEXT_PRIMARY
        ax.text(j, i, f"{value:.3f}{marker}", ha="center", va="center", fontsize=7.5, color=color)
    cbar = fig.colorbar(image, ax=ax, fraction=0.04)
    cbar.set_label("PSI (escala truncada em 0,5)", color=TEXT_SECONDARY, fontsize=8)
    ax.set_title(
        f"PSI por feature e cenário  (! >= {moderate}, !! >= {severe})",
        loc="left",
        fontsize=11,
        fontweight="bold",
        color=TEXT_PRIMARY,
    )
    ax.tick_params(colors=TEXT_SECONDARY)
    return _save(fig, path)


def plot_distribution_shift(
    reference: pd.Series,
    current: pd.Series,
    path: str | Path,
    title: str,
    xlabel: str,
    current_label: str = "Produção (data drift)",
) -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    bins = np.histogram_bin_edges(np.concatenate([reference, current]), bins=30)
    ax.hist(reference, bins=bins, density=True, alpha=0.55, color=REFERENCE_COLOR, label="Referência")
    ax.hist(current, bins=bins, density=True, alpha=0.55, color=CURRENT_COLOR, label=current_label)
    ax.legend(frameon=False, fontsize=9)
    _style(ax, title, xlabel=xlabel, ylabel="Densidade")
    return _save(fig, path)


def plot_scenario_auc(summaries: list[dict], reference_auc: float, path: str | Path) -> Path:
    """Dot plot with bootstrap CI (a truncated axis would distort bar lengths)."""
    names = [SCENARIO_LABELS.get(s["scenario"], s["scenario"]) for s in summaries]
    aucs = [s["performance"]["current"]["roc_auc"] for s in summaries]
    lows = [a - s["performance"]["roc_auc_ci95"][0] for a, s in zip(aucs, summaries, strict=True)]
    highs = [s["performance"]["roc_auc_ci95"][1] - a for a, s in zip(aucs, summaries, strict=True)]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.axhline(
        reference_auc, color=REFERENCE_COLOR, linewidth=2, linestyle="--", label=f"Referência ({reference_auc:.3f})"
    )
    ax.errorbar(
        names,
        aucs,
        yerr=[lows, highs],
        fmt="o",
        color=CURRENT_COLOR,
        ecolor=TEXT_SECONDARY,
        elinewidth=1.2,
        capsize=5,
        markersize=9,
        label="Produção (IC 95%)",
    )
    for i, value in enumerate(aucs):
        ax.annotate(
            f"{value:.3f}", (i, value), xytext=(10, -4), textcoords="offset points", fontsize=9, color=TEXT_PRIMARY
        )
    ax.set_ylim(0.5, 0.85)
    ax.set_xlim(-0.5, len(names) - 0.5)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    _style(ax, "ROC-AUC por cenário (IC 95% bootstrap)", ylabel="ROC-AUC")
    return _save(fig, path)


def plot_calibration_bins(tables: dict[str, pd.DataFrame], path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    first = next(iter(tables.values()))
    ax.plot(
        first["bin"] + 1,
        first["reference_default_rate"] * 100,
        color=REFERENCE_COLOR,
        marker="o",
        linewidth=2,
        label="Referência",
    )
    palette = [CURRENT_COLOR, THIRD_COLOR]
    for (name, table), color in zip(tables.items(), palette, strict=False):
        ax.plot(
            table["bin"] + 1,
            table["current_default_rate"] * 100,
            color=color,
            marker="o",
            linewidth=2,
            label=SCENARIO_LABELS.get(name, name),
        )
    ax.legend(frameon=False, fontsize=9)
    _style(
        ax,
        "Inadimplência observada por decil de score: P(Y | score)",
        xlabel="Decil de score do modelo (bins da referência)",
        ylabel="Inadimplência (%)",
    )
    return _save(fig, path)


def plot_alert_counts(summaries: list[dict], path: str | Path) -> Path:
    names = [SCENARIO_LABELS.get(s["scenario"], s["scenario"]) for s in summaries]
    x = np.arange(len(names))
    width = 0.26
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for offset, severity in zip((-width, 0, width), ("INFO", "WARNING", "CRITICAL"), strict=True):
        values = [s["alerts"][severity] for s in summaries]
        ax.bar(x + offset, values, width=width - 0.03, color=STATUS[severity], label=severity)
    ax.set_xticks(x, names)
    ax.legend(frameon=False, fontsize=9)
    _style(ax, "Alertas por cenário e severidade", ylabel="Quantidade")
    return _save(fig, path)


def plot_violations(summary: pd.DataFrame, path: str | Path) -> Path:
    grouped = summary.groupby("rule_id")["n_failures"].sum().sort_values()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.barh(grouped.index, grouped.values, color=STATUS["CRITICAL"], height=0.55)
    ax.grid(axis="x", color=GRID)
    _style(ax, "Lote inválido: violações por regra do contrato", xlabel="Casos de falha")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def plot_fairness(groups: pd.DataFrame, path: str | Path) -> Path:
    data = groups[~groups["suppressed"]].copy()
    data["label"] = data["attribute"] + ": " + data["group"]
    data = data.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.6, 0.3 * len(data) + 1.4))
    ax.barh(data["label"], data["approval_rate"] * 100, color=REFERENCE_COLOR, height=0.6)
    ax.grid(axis="x", color=GRID)
    _style(ax, "Taxa de aprovação por grupo (referência)", xlabel="Aprovação (%)")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)
