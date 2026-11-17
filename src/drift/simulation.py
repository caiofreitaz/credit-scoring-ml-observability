"""Reproducible simulation of production scenarios (Data Drift and Concept Drift).

Design principles
-----------------
* All scenarios are drawn ONLY from the production partition (never used in training).
* Data Drift (change in P(X)) is produced by a *customer-mix shift*: weighted
  sampling without replacement that favours clients with low credit limits and
  young clients. Every row is a real record with its real outcome, therefore
  P(Y|X) is preserved by construction - only the population changes.
* Concept Drift (change in P(Y|X)) keeps the exact same feature rows as the
  baseline sample and alters outcomes through an explicit, seeded mechanism
  (liquidity shock + renegotiation programme). P(X) is identical by construction.
* The combined scenario applies the concept mechanism on top of the shifted mix.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.config import ProjectPaths
from src.data.batches import to_inference_frame

SCENARIOS = ("normal", "data_drift", "concept_drift", "data_and_concept_drift")
SCENARIO_DESCRIPTIONS = {
    "normal": "Produção estável: amostra aleatória do período de produção, sem alterações.",
    "data_drift": "Mudança no perfil de clientes (P(X)): mais clientes com limite baixo e mais jovens.",
    "concept_drift": "Mesmos clientes do cenário normal; choque de liquidez e renegociação mudam P(Y|X).",
    "data_and_concept_drift": "Mudança de perfil de clientes combinada com a mudança de comportamento.",
}
SCENARIO_DIR = "scenarios"


@dataclass
class ScenarioBatch:
    """Inference features plus delayed outcomes for one simulated scenario."""

    name: str
    features: pd.DataFrame
    outcomes: pd.DataFrame
    description: str
    mechanism: dict[str, Any] = field(default_factory=dict)


def uniform_sample(pool: pd.DataFrame, size: int, rng: np.random.Generator) -> pd.DataFrame:
    """Simple random sample without replacement (no distribution change)."""
    index = rng.choice(len(pool), size=size, replace=False)
    return pool.iloc[np.sort(index)].reset_index(drop=True)


def customer_mix_weights(pool: pd.DataFrame, cfg: dict[str, Any]) -> np.ndarray:
    """Sampling weights favouring low credit limits and young clients."""
    limit_weight = np.exp(-pool["LIMIT_BAL"].to_numpy(dtype="float64") / cfg["limit_bal_scale"])
    age_weight = np.where(pool["AGE"].to_numpy() < cfg["young_age_cutoff"], cfg["young_boost"], 1.0)
    weights = limit_weight * age_weight
    return weights / weights.sum()


def shifted_sample(pool: pd.DataFrame, size: int, cfg: dict[str, Any], rng: np.random.Generator) -> pd.DataFrame:
    """Weighted sample without replacement producing a covariate (P(X)) shift."""
    probabilities = customer_mix_weights(pool, cfg)
    index = rng.choice(len(pool), size=size, replace=False, p=probabilities)
    return pool.iloc[np.sort(index)].reset_index(drop=True)


def apply_concept_shift(
    frame: pd.DataFrame, target: str, cfg: dict[str, Any], rng: np.random.Generator
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Change outcomes for fixed features, altering P(Y|X).

    1. Liquidity shock: non-defaulting clients that are *current* on payments
       (PAY_0 <= 0) but highly utilized (BILL_AMT1 / LIMIT_BAL >= threshold)
       start defaulting with probability ``p_new_default_high_utilization``.
    2. Renegotiation programme: defaulting clients with severe delays
       (PAY_0 >= ``late_payment_status_threshold``) are cured with probability
       ``p_cure_late_payers``.
    """
    shifted = frame.copy()
    y = shifted[target].to_numpy().copy()
    utilization = shifted["BILL_AMT1"].to_numpy(dtype="float64") / shifted["LIMIT_BAL"].to_numpy(dtype="float64")
    current = shifted["PAY_0"].to_numpy() <= 0
    late = shifted["PAY_0"].to_numpy() >= cfg["late_payment_status_threshold"]

    shock_eligible = (y == 0) & current & (utilization >= cfg["utilization_threshold"])
    cure_eligible = (y == 1) & late
    draws = rng.random(len(y))
    new_defaults = shock_eligible & (draws < cfg["p_new_default_high_utilization"])
    cures = cure_eligible & (draws < cfg["p_cure_late_payers"])
    y[new_defaults] = 1
    y[cures] = 0
    shifted[target] = y.astype("int64")
    stats = {
        "shock_eligible": int(shock_eligible.sum()),
        "new_defaults": int(new_defaults.sum()),
        "cure_eligible": int(cure_eligible.sum()),
        "cures": int(cures.sum()),
        "default_rate_before": round(float(frame[target].mean()), 4),
        "default_rate_after": round(float(shifted[target].mean()), 4),
    }
    return shifted, stats


def _to_batch(name: str, frame: pd.DataFrame, config: dict[str, Any], mechanism: dict[str, Any]) -> ScenarioBatch:
    id_col, target = config["features"]["id_column"], config["features"]["target"]
    return ScenarioBatch(
        name=name,
        features=to_inference_frame(frame, config).reset_index(drop=True),
        outcomes=frame[[id_col, target]].reset_index(drop=True),
        description=SCENARIO_DESCRIPTIONS[name],
        mechanism=mechanism,
    )


def build_scenarios(pool: pd.DataFrame, config: dict[str, Any]) -> dict[str, ScenarioBatch]:
    """Build the four independent, seeded production scenarios."""
    sim = config["simulation"]
    seed = config["project"]["seed"]
    target = config["features"]["target"]
    size = min(sim["batch_size"], len(pool))

    base = uniform_sample(pool, size, np.random.default_rng(seed + 100))
    shifted = shifted_sample(pool, size, sim["data_drift"], np.random.default_rng(seed + 200))
    concept, concept_stats = apply_concept_shift(base, target, sim["concept_drift"], np.random.default_rng(seed + 300))
    both, both_stats = apply_concept_shift(shifted, target, sim["concept_drift"], np.random.default_rng(seed + 400))

    return {
        "normal": _to_batch("normal", base, config, {"sampling": "uniform", "seed": seed + 100}),
        "data_drift": _to_batch(
            "data_drift", shifted, config, {"sampling": "customer_mix_shift", "seed": seed + 200, **sim["data_drift"]}
        ),
        "concept_drift": _to_batch(
            "concept_drift",
            concept,
            config,
            {"sampling": "uniform (same rows as normal)", "label_mechanism": sim["concept_drift"], **concept_stats},
        ),
        "data_and_concept_drift": _to_batch(
            "data_and_concept_drift",
            both,
            config,
            {
                "sampling": "customer_mix_shift (same rows as data_drift)",
                "label_mechanism": sim["concept_drift"],
                **both_stats,
            },
        ),
    }


def write_scenarios(scenarios: dict[str, ScenarioBatch], paths: ProjectPaths) -> dict[str, Any]:
    """Persist features and outcomes (separate files, simulating delayed labels)."""
    output = paths.production_dir / SCENARIO_DIR
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {}
    for name, batch in scenarios.items():
        batch.features.to_csv(output / f"{name}_features.csv", index=False, lineterminator="\n")
        batch.outcomes.to_csv(output / f"{name}_outcomes.csv", index=False, lineterminator="\n")
        manifest[name] = {
            "description": batch.description,
            "rows": len(batch.features),
            "observed_default_rate": round(float(batch.outcomes.iloc[:, 1].mean()), 4),
            "mechanism": batch.mechanism,
        }
    (output / "scenario_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def load_scenario(name: str, paths: ProjectPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the features and outcomes files of a persisted scenario."""
    folder = paths.production_dir / SCENARIO_DIR
    features_path = folder / f"{name}_features.csv"
    if not features_path.exists():
        raise FileNotFoundError(f"Scenario '{name}' not found. Run: python -m src.cli simulate-drift")
    features = pd.read_csv(features_path, dtype={"client_id": str})
    outcomes = pd.read_csv(folder / f"{name}_outcomes.csv", dtype={"client_id": str})
    return features, outcomes
