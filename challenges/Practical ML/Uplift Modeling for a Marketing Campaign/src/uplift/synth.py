"""Semi-synthetic outcomes with a known individual effect.

Real Criteo features, simulated outcomes. The real data has no ground truth for the
individual treatment effect (nobody is observed both contacted and not), so this is
the only place "did the learner recover the effect" has an answer.

Outcome model on the logit scale: ``p0 = sigmoid(base + Z a)`` for an untreated customer
and ``p1 = sigmoid(base + Z a + b(x))`` for a treated one, ``Z`` the standardized
features. The true individual effect is ``tau = p1 - p0``. ``treatment`` is taken as
given (randomized), so the observed data is a valid RCT.
"""

from __future__ import annotations

import numpy as np

SCENARIOS = ("heterogeneous", "constant", "none")


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def simulate(
    X, t, seed, scenario="heterogeneous", base_rate=0.05, strength=1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(y, tau)``: simulated 0/1 outcomes and the true effect on the probability scale."""
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}, got {scenario!r}")
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    Z = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    base = np.log(base_rate / (1 - base_rate)) + Z @ (
        rng.normal(size=Z.shape[1]) * (0.4 / np.sqrt(Z.shape[1]))
    )
    if scenario == "heterogeneous":
        # Winners (z0 high), sleeping dogs (z0 low), plus an interaction.
        b = strength * (0.8 * np.tanh(Z[:, 0]) + 0.5 * np.sign(Z[:, 1]) * (Z[:, 2] > 0))
    elif scenario == "constant":
        b = np.full(len(Z), 0.5 * strength)
    else:
        b = np.zeros(len(Z))
    p0, p1 = _sigmoid(base), _sigmoid(base + b)
    t = np.asarray(t)
    y = (rng.random(len(Z)) < np.where(t == 1, p1, p0)).astype(np.int8)
    return y, p1 - p0
