"""Benchmark the detectors on streams with known drift.

A stream is ``n_windows`` windows resampled (i.i.d.) from the reference split,
with a drift injected from window ``drift_window`` onward. For every detector the
benchmark records whether it alarmed within ``DETECT_WITHIN`` windows of the true
start (and how many windows later), and how often it alarmed *before* the drift,
which is the false-alarm rate the calibration targets (per window, not per stream:
at 1% per window a 40-window stream raises some alarm about a third of the time).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl

from drift_monitor import inject
from drift_monitor.data import NUMERIC
from drift_monitor.monitor import WINDOW, MonitorResult, monitor
from drift_monitor.stats import Baseline

DETECT_WITHIN = 10
RULE_OF_THUMB_PSI = 0.2
SCORE_STATS = ("psi", "ks", "js", "wasserstein")

# detector name -> how to read its alarm windows from a MonitorResult
Detectors = dict[str, set[int]]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """``(rate, lo, hi)``: the proportion k/n with a Wilson score interval."""
    if n == 0:
        return math.nan, math.nan, math.nan
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def alarm_windows(
    res: MonitorResult, thresholds: dict[tuple[str, str], float]
) -> Detectors:
    """Windows in which each named detector alarmed."""
    out: Detectors = {}
    a = res.alerts
    for stat in SCORE_STATS:
        out[f"score_{stat}"] = set(
            a.filter((pl.col("signal") == "score") & (pl.col("stat") == stat))["window"]
        )
    out["features_any"] = set(a.filter(pl.col("signal") != "score")["window"])
    out["psi_rule_of_thumb"] = set(
        res.table.filter(pl.col("score_psi") > RULE_OF_THUMB_PSI)["window"]
    )
    for det in ("page_hinkley_score", "adwin_score", "adwin_error"):
        out[det] = set(res.sequential.filter(pl.col("detector") == det)["window"])
    return out


@dataclass
class Scenario:
    name: str
    magnitude: float
    apply: Callable  # (df, start_row, seed, clf) -> (df, clf)


def default_scenarios(clf) -> list[Scenario]:
    """Covariate shift on the model's most and least important feature, plus prior, concept, score noise."""
    gain = dict(
        zip(["day", *NUMERIC], clf.booster_.feature_importance("gain"), strict=True)
    )
    numeric = sorted(NUMERIC, key=lambda f: gain[f])
    least, most = numeric[0], numeric[-1]
    out = [Scenario("none", 0.0, lambda df, s, seed, c: (df, c))]
    for feature, label in ((most, "important"), (least, "unimportant")):
        for sigmas in (0.25, 0.5, 1.0, 2.0):
            out.append(
                Scenario(
                    f"covariate_{label}:{feature}",
                    sigmas,
                    lambda df, s, seed, c, f=feature, g=sigmas: (
                        inject.covariate(df, f, g, s),
                        c,
                    ),
                )
            )
    for rate in (0.5, 0.6, 0.75):
        out.append(
            Scenario(
                "prior",
                rate,
                lambda df, s, seed, c, r=rate: (inject.prior(df, r, s, seed), c),
            )
        )
    for strength in (0.1, 0.2, 0.4):
        out.append(
            Scenario(
                "concept",
                strength,
                lambda df, s, seed, c, g=strength: (inject.concept(df, g, s, seed), c),
            )
        )
    for sd in (0.05, 0.1, 0.2):
        out.append(
            Scenario(
                "score_noise",
                sd,
                lambda df, s, seed, c, g=sd: (
                    df,
                    inject.NoisyClassifier(c, g, s, seed),
                ),
            )
        )
    return out


def _stream(reference: pl.DataFrame, n_rows: int, seed: int) -> pl.DataFrame:
    idx = np.random.default_rng(seed).integers(0, reference.height, n_rows)
    return reference[idx.tolist()]


def benchmark(
    clf,
    baseline: Baseline,
    thresholds: dict[tuple[str, str], float],
    seq: dict,
    reference: pl.DataFrame,
    scenarios: list[Scenario],
    n_seeds: int = 20,
    n_windows: int = 40,
    drift_window: int = 15,
    window: int = WINDOW,
    delay: int = 4,
) -> pl.DataFrame:
    """One row per (scenario, magnitude, detector, seed)."""
    rows = []
    start = drift_window * window
    for sc in scenarios:
        for seed in range(n_seeds):
            df, model_clf = sc.apply(
                _stream(reference, n_windows * window, 1000 + seed), start, seed, clf
            )
            res = monitor(
                model_clf, baseline, thresholds, df, seq, window=window, delay=delay
            )
            for det, windows_hit in alarm_windows(res, thresholds).items():
                after = sorted(
                    w
                    for w in windows_hit
                    if drift_window <= w < drift_window + DETECT_WITHIN
                )
                rows.append(
                    {
                        "scenario": sc.name,
                        "magnitude": sc.magnitude,
                        "detector": det,
                        "seed": seed,
                        "detected": bool(after),
                        "delay_windows": (after[0] - drift_window) if after else None,
                        "pre_alarm_windows": sum(
                            1 for w in windows_hit if w < drift_window
                        ),
                        "pre_windows": drift_window,
                        "all_alarm_windows": len(windows_hit),
                        "all_windows": n_windows,
                    }
                )
    return pl.DataFrame(rows, schema_overrides={"delay_windows": pl.Int64})


def summarize(runs: pl.DataFrame) -> pl.DataFrame:
    """Per scenario x magnitude x detector.

    ``detection_rate`` (Wilson CI) is the share of streams alarmed within the
    horizon after the drift start; ``chance_rate`` is the same quantity on the
    no-drift control, i.e. what false alarms alone produce over that horizon, so
    a detector with a high false-alarm rate cannot look good by firing often.
    ``false_alarm_per_window`` is the per-window rate before the drift.
    """
    chance = {}
    for (detector,), g in runs.filter(pl.col("scenario") == "none").group_by(
        ["detector"]
    ):
        chance[detector] = float(g["detected"].mean())
    out = []
    for (scenario, magnitude, detector), g in runs.group_by(
        ["scenario", "magnitude", "detector"], maintain_order=True
    ):
        k, n = int(g["detected"].sum()), g.height
        rate, lo, hi = wilson(k, n)
        if scenario == "none":  # no drift anywhere: every alarm window is a false alarm
            fa = g["all_alarm_windows"].sum() / g["all_windows"].sum()
        else:
            fa = g["pre_alarm_windows"].sum() / g["pre_windows"].sum()
        delays = g["delay_windows"].drop_nulls()
        out.append(
            {
                "scenario": scenario,
                "magnitude": magnitude,
                "detector": detector,
                "n": n,
                "detection_rate": rate,
                "detection_lo": lo,
                "detection_hi": hi,
                "chance_rate": chance.get(detector),
                "mean_delay_windows": float(delays.mean()) if len(delays) else None,
                "false_alarm_per_window": float(fa),
            }
        )
    return pl.DataFrame(
        out,
        schema_overrides={"mean_delay_windows": pl.Float64, "chance_rate": pl.Float64},
    )
