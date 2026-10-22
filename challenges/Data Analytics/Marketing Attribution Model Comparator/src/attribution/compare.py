"""Running every model on the same journeys and comparing them, with bootstrap uncertainty."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from attribution import logistic, markov, rules, shapley
from attribution.journeys import AttributionError, Journeys

MODELS = (*rules.RULES, "markov", "shapley", "logistic")
LABELS = {
    "first_touch": "First touch",
    "last_touch": "Last touch",
    "linear": "Linear",
    "time_decay": "Time decay",
    "position_based": "Position based",
    "last_click": "Last click",
    "markov": "Markov chain",
    "shapley": "Shapley value",
    "logistic": "Logistic regression",
}


@dataclass(frozen=True)
class Settings:
    half_life: float  # in the journeys' time unit
    shapley_prior: float = 5.0


def run_model(
    j: Journeys, model: str, s: Settings, weights: np.ndarray | None = None
) -> np.ndarray:
    """Credit per channel (conversions, or conversion value) for one model."""
    if model in rules.RULES:
        return rules.credit(j, model, weights, half_life=s.half_life)
    if model == "markov":
        return markov.credit(j, weights)
    if model == "shapley":
        return shapley.credit(j, weights)
    if model == "logistic":
        return logistic.credit(j, weights)
    raise AttributionError(f"unknown model {model!r}; choose from {', '.join(MODELS)}")


def credit_table(
    j: Journeys, s: Settings, models: tuple[str, ...] = MODELS
) -> np.ndarray:
    """(models, channels) matrix of credit; every row is on the same total."""
    return np.vstack([run_model(j, m, s) for m in models])


def shares(table: np.ndarray) -> np.ndarray:
    total = table.sum(axis=1, keepdims=True)
    return np.divide(table, total, out=np.zeros_like(table), where=total != 0)


def resampled(j: Journeys, rng: np.random.Generator) -> tuple[Journeys, np.ndarray]:
    """One bootstrap replicate: converted journeys redrawn, null aggregates redrawn as Poisson."""
    w = rng.multinomial(
        j.n_converted, np.full(j.n_converted, 1.0 / j.n_converted)
    ).astype(float)
    r = copy(j)
    r.null_start = rng.poisson(j.null_start).astype(float)
    r.null_end = rng.poisson(j.null_end).astype(float)
    r.null_trans = rng.poisson(j.null_trans).astype(float)
    r.null_sets = {m: float(rng.poisson(c)) for m, c in j.null_sets.items()}
    r.null_count = float(sum(r.null_sets.values()))
    return r, w


@dataclass
class Comparison:
    journeys: Journeys
    settings: Settings
    models: tuple[str, ...]
    credit: np.ndarray  # (M, K)
    share: np.ndarray  # (M, K)
    low: np.ndarray | None = None  # (M, K) 2.5th percentile of the bootstrap share
    high: np.ndarray | None = None
    draws: int = 0
    extras: dict = field(default_factory=dict)

    def model_index(self, name: str) -> int:
        return self.models.index(name)

    def spend_share(self) -> np.ndarray:
        spend = self.journeys.touch_spend()
        return spend / spend.sum() if spend.sum() > 0 else np.zeros(self.journeys.k)

    def cost_per_conversion(self) -> np.ndarray:
        """(M, K) spend of each channel divided by the conversions a model credits it with."""
        spend = self.journeys.touch_spend()
        return np.divide(
            spend[None, :],
            self.credit,
            out=np.full_like(self.credit, np.nan),
            where=self.credit > 0,
        )

    def rank_correlation(self) -> np.ndarray:
        """Kendall tau between every pair of models' channel rankings."""
        m = len(self.models)
        out = np.eye(m)
        for a in range(m):
            for b in range(a + 1, m):
                tau = stats.kendalltau(self.share[a], self.share[b])[0]
                out[a, b] = out[b, a] = 0.0 if np.isnan(tau) else tau
        return out

    def distance(self) -> np.ndarray:
        """Total variation distance between every pair of models' shares: the share that moves."""
        m = len(self.models)
        d = np.zeros((m, m))
        for a in range(m):
            for b in range(m):
                d[a, b] = 0.5 * float(np.abs(self.share[a] - self.share[b]).sum())
        return d

    def rank_matrix(self) -> np.ndarray:
        """(M, K) rank of each channel under each model; 1 = most credit."""
        return np.vstack(
            [stats.rankdata(-row, method="min") for row in self.share]
        ).astype(int)


def compare(
    j: Journeys,
    settings: Settings,
    models: tuple[str, ...] = MODELS,
    *,
    draws: int = 0,
    seed: int = 0,
) -> Comparison:
    j.validate_names()
    credit = credit_table(j, settings, models)
    result = Comparison(j, settings, models, credit, shares(credit))
    if draws > 0:
        rng = np.random.default_rng(seed)
        boot = np.empty((draws, len(models), j.k))
        for b in range(draws):
            r, w = resampled(j, rng)
            boot[b] = shares(np.vstack([run_model(r, m, settings, w) for m in models]))
        result.low = np.percentile(boot, 2.5, axis=0)
        result.high = np.percentile(boot, 97.5, axis=0)
        result.draws = draws
        result.extras["boot"] = boot
    return result
