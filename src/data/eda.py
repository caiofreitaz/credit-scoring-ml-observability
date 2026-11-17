"""Objective exploratory analysis computed on the TRAIN partition only (no leakage)."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from src.config import ProjectPaths, model_feature_columns
from src.monitoring import plots

EDA_DIR = "eda"


def eda_summary(train: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Key descriptive statistics used to justify modeling and contract decisions."""
    target = config["features"]["target"]
    features = model_feature_columns(config)
    describe = train[features].describe().T[["mean", "std", "min", "50%", "max"]].round(2)
    correlations = (
        train[features + [target]].corr(numeric_only=True)[target].drop(target).sort_values(key=abs, ascending=False)
    )
    by_pay0 = train.groupby("PAY_0")[target].agg(["size", "mean"]).round(4)
    return {
        "partition": "train",
        "rows": int(len(train)),
        "columns": int(train.shape[1]),
        "missing_values": int(train.isna().sum().sum()),
        "class_distribution": {str(k): int(v) for k, v in train[target].value_counts().sort_index().items()},
        "default_rate": round(float(train[target].mean()), 4),
        "feature_statistics": describe.to_dict(orient="index"),
        "top_correlations_with_target": correlations.head(8).round(4).to_dict(),
        "default_rate_by_PAY_0": {str(k): {"n": int(r["size"]), "rate": float(r["mean"])} for k, r in
                                  by_pay0.iterrows()},
    }


def run_eda(train: pd.DataFrame, config: dict[str, Any], paths: ProjectPaths) -> dict[str, Any]:
    """Persist the EDA summary and figures under reports/eda/."""
    output = paths.reports_dir / EDA_DIR
    output.mkdir(parents=True, exist_ok=True)
    target = config["features"]["target"]
    summary = eda_summary(train, config)
    (output / "eda_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    plots.plot_class_balance(train[target], output / "class_balance.png")
    plots.plot_default_rate_by(train, "PAY_0", target, output / "default_rate_by_pay0.png",
                               "Inadimplência por status de pagamento mais recente (PAY_0)")
    banded = train.assign(LIMIT_BAND=pd.qcut(train["LIMIT_BAL"], 5, duplicates="drop").astype(str))
    plots.plot_default_rate_by(banded, "LIMIT_BAND", target, output / "default_rate_by_limit.png",
                               "Inadimplência por faixa de limite de crédito (quintis)")
    return summary
