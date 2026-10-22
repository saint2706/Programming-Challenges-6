"""A world where the truth is known: simulated customers, with each channel's real effect on conversion.

Real marketing data never says what each channel *caused*. Here every customer has a hidden
intent ``u``; each channel delivers a random number of exposures over a 30-day horizon; and the
chance of converting is a logistic function of intent plus the lingering effect of each exposure
(``beta`` of log-odds, fading with time constant ``tau``). Because the chance is known exactly, a
channel's **true incremental conversions** are the sum, over customers, of the chance with all its
exposures minus the chance with none of them, holding everything else (including the intent that
steered who got targeted) fixed. Models are then scored on how far their credit is from that.

Channels: display (early awareness), social, search (people who are already looking), email, and
retargeting (shown to people who already intend to buy).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from attribution.build import from_touches
from attribution.journeys import Journeys

CHANNELS = ["display", "email", "retargeting", "search", "social"]
HORIZON = 30.0  # days


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    rate: dict[str, float]  # mean exposures per customer over the horizon
    beta: dict[str, float]  # log-odds added by one fresh exposure
    tau: dict[str, float]  # days for that effect to fade by a factor e
    gamma: dict[str, float]  # how strongly exposure rate rises with intent (targeting)
    timing: dict[str, str]  # early, uniform or late within the horizon
    base_logit: float = -3.0
    intent_weight: float = 1.0


def _full(default: float, **over: float) -> dict[str, float]:
    return {c: over.get(c, default) for c in CHANNELS}


SCENARIOS: dict[str, Scenario] = {
    "independent": Scenario(
        "independent",
        "Channels reach people at random and each really works a little: the easy case.",
        rate=_full(
            1.2, display=1.5, email=1.0, retargeting=0.8, search=1.2, social=1.3
        ),
        beta=_full(
            0.35, display=0.30, email=0.40, retargeting=0.35, search=0.40, social=0.30
        ),
        tau=_full(10.0),
        gamma=_full(0.0),
        timing=_full("uniform"),  # type: ignore[arg-type]
    ),
    "retargeting": Scenario(
        "retargeting",
        "Retargeting is aimed at people who were already going to buy and adds nothing; it also "
        "sits at the end of the journey. Search and email do the work.",
        rate=_full(
            1.2, display=1.5, email=1.0, retargeting=2.5, search=1.2, social=1.3
        ),
        beta=_full(
            0.35, display=0.30, email=0.55, retargeting=0.0, search=0.55, social=0.30
        ),
        tau=_full(10.0),
        gamma=_full(0.0, retargeting=1.6, search=0.6),
        timing={**_full("uniform"), "retargeting": "late"},  # type: ignore[dict-item]
    ),
    "awareness": Scenario(
        "awareness",
        "Display plants the idea early and its effect lasts weeks; search and email only collect "
        "the demand it created and have a short effect.",
        rate=_full(
            1.2, display=1.5, email=1.5, retargeting=0.8, search=1.8, social=1.0
        ),
        beta=_full(
            0.30, display=0.70, email=0.10, retargeting=0.05, search=0.12, social=0.30
        ),
        tau={**_full(3.0), "display": 25.0, "social": 12.0},
        gamma=_full(0.0, search=0.5, email=0.4),
        timing={**_full("late"), "display": "early", "social": "uniform"},  # type: ignore[dict-item]
    ),
    "closing": Scenario(
        "closing",
        "Display is seen by everyone early and does nothing; search and email at the end are what "
        "tips a customer over the line, with short-lived effects.",
        rate=_full(
            1.0, display=3.0, email=1.2, retargeting=0.6, search=1.5, social=1.0
        ),
        beta=_full(
            0.2, display=0.0, email=0.60, retargeting=0.1, search=0.80, social=0.15
        ),
        tau=_full(4.0),
        gamma=_full(0.0, search=0.4),
        timing={
            **_full("uniform"),
            "display": "early",
            "email": "late",
            "search": "late",
        },  # type: ignore[dict-item]
    ),
}


def _times(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "early":
        return HORIZON * rng.beta(1.0, 4.0, n)
    if kind == "late":
        return HORIZON * (1.0 - rng.beta(1.0, 5.0, n))
    return HORIZON * rng.uniform(size=n)


def _fade(t: np.ndarray, tau: float) -> np.ndarray:
    return np.exp(-(HORIZON - t) / tau)


@dataclass
class World:
    scenario: Scenario
    journeys: Journeys
    truth: (
        np.ndarray
    )  # (K,) incremental conversions per channel, in the order of journeys.channels
    n_exposed: int
    n_converted: int

    @property
    def truth_share(self) -> np.ndarray:
        return self.truth / self.truth.sum()


def simulate(
    scenario: Scenario | str, n_customers: int = 100_000, seed: int = 0
) -> World:
    sc = SCENARIOS[scenario] if isinstance(scenario, str) else scenario
    rng = np.random.default_rng(seed)
    u = rng.normal(size=n_customers)
    users, times, chans = [], [], []
    effect = {c: np.zeros(n_customers) for c in CHANNELS}
    for ci, c in enumerate(CHANNELS):
        lam = sc.rate[c] * np.exp(sc.gamma[c] * u - sc.gamma[c] ** 2 / 2)
        counts = rng.poisson(lam)
        who = np.repeat(np.arange(n_customers), counts)
        t = _times(sc.timing[c], who.size, rng)
        np.add.at(effect[c], who, sc.beta[c] * _fade(t, sc.tau[c]))
        users.append(who)
        times.append(t)
        chans.append(np.full(who.size, ci, dtype=np.int32))
    user, time, channel = (
        np.concatenate(users),
        np.concatenate(times),
        np.concatenate(chans),
    )
    logit = sc.base_logit + sc.intent_weight * u + sum(effect.values())
    p = 1.0 / (1.0 + np.exp(-logit))
    converted_user = rng.uniform(size=n_customers) < p
    exposed = np.zeros(n_customers, bool)
    exposed[user] = True
    truth = np.empty(len(CHANNELS))
    for ci, c in enumerate(CHANNELS):
        without = 1.0 / (1.0 + np.exp(-(logit - effect[c])))
        truth[ci] = float(np.sum((p - without)[exposed]))
    journeys = from_touches(
        user, time, channel, converted_user[user], CHANNELS, conv_time=np.full(time.size, HORIZON),
        unit="days",
    )  # fmt: skip
    return World(sc, journeys, truth, int(exposed.sum()), journeys.n_converted)
