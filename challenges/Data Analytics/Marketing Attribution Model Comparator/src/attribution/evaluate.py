"""Scoring models against simulated truth, across scenarios and random seeds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from attribution.compare import LABELS, MODELS, Settings, compare
from attribution.simulate import SCENARIOS, simulate


@dataclass
class ScenarioResult:
    name: str
    description: str
    channels: list[str]
    truth: np.ndarray  # (K,) mean true incremental share over seeds
    shares: np.ndarray  # (M, K) mean model share over seeds
    error: (
        np.ndarray
    )  # (M,) mean L1 distance to the truth, half of which is the share misplaced
    error_sd: np.ndarray  # (M,)
    tau: (
        np.ndarray
    )  # (M,) mean Kendall tau of the model's channel ranking with the truth
    seeds: int
    customers: int


def evaluate(
    scenario: str,
    customers: int = 100_000,
    seeds: int = 5,
    models: tuple[str, ...] = MODELS,
    half_life: float = 7.0,
) -> ScenarioResult:
    errors, taus, shares, truths = [], [], [], []
    channels: list[str] = []
    for seed in range(seeds):
        world = simulate(scenario, customers, seed=seed)
        channels = world.journeys.channels
        c = compare(world.journeys, Settings(half_life=half_life), models)
        truth = world.truth_share
        errors.append([0.5 * float(np.abs(row - truth).sum()) for row in c.share])
        taus.append([_tau(row, truth) for row in c.share])
        shares.append(c.share)
        truths.append(truth)
    err = np.array(errors)
    sc = SCENARIOS[scenario]
    return ScenarioResult(
        scenario, sc.description, channels, np.mean(truths, axis=0), np.mean(shares, axis=0),
        err.mean(axis=0), err.std(axis=0, ddof=1) if seeds > 1 else np.zeros(len(models)),
        np.mean(taus, axis=0), seeds, customers,
    )  # fmt: skip


def _tau(a: np.ndarray, b: np.ndarray) -> float:
    t = stats.kendalltau(a, b)[0]
    return 0.0 if np.isnan(t) else float(t)


def best_model(r: ScenarioResult, models: tuple[str, ...] = MODELS) -> str:
    return LABELS[models[int(np.argmin(r.error))]]
