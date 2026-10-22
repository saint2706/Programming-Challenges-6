"""Shared test helpers: the committed R reference numbers, small datasets and a naive Kaplan-Meier."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np
import polars as pl
from survival_toolkit.data import read_table
from survival_toolkit.types import Survival

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "tests" / "reference" / "r_survival.json"


@cache
def r_cases() -> dict:
    return json.loads(REFERENCE.read_text(encoding="utf-8"))["cases"]


@cache
def lung() -> pl.DataFrame:
    return read_table(ROOT / "sample_data" / "lung.csv")


@cache
def gbsg() -> pl.DataFrame:
    return read_table(ROOT / "sample_data" / "gbsg.csv")


def arr(values) -> np.ndarray:
    """JSON list (null for NA) to a float array with NaN."""
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def rng_data(
    seed: int, n: int = 60, *, entry: bool = False, ties: bool = False
) -> Survival:
    """A small random right-censored sample, optionally with late entry or heavy ties."""
    rng = np.random.default_rng(seed)
    time = rng.exponential(10, n)
    cens = rng.exponential(14, n)
    event = time <= cens
    obs = np.minimum(time, cens)
    if ties:
        obs = np.ceil(
            obs
        )  # integer times: many events and censorings at the same value
    obs = np.maximum(obs, 1.0 if ties else 1e-3)
    ent = None
    if entry:
        ent = obs * rng.uniform(0, 0.7, n)
    return Survival(obs, event, ent)


def naive_km(data: Survival) -> tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier written as the textbook loop, one event time at a time."""
    times = np.unique(data.time[data.event])
    entry = np.zeros(len(data)) if data.entry is None else data.entry
    s, out = 1.0, []
    for t in times:
        at_risk = np.sum((entry < t) & (data.time >= t))
        died = np.sum((data.time == t) & data.event)
        s *= 1 - died / at_risk
        out.append(s)
    return times, np.array(out)
