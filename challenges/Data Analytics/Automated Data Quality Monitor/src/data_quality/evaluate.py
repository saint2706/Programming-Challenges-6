"""False-positive rate on clean batches vs detection rate on planted corruptions.

    uv run python -m data_quality.evaluate            # 300 clean + 30 trials per corruption, seed 0

Baseline: taxi days 1-21. Clean batches: the 10 real held-out days, plus simulated ones
drawn without replacement from the held-out rows (150-260 rows each, like a real day).
Every batch is checked with an as-of time six hours after its newest trip.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import polars as pl

from data_quality.corruptions import (
    CORRUPTIONS,
    Corruption,
    mix_sweep,
    null_sweep,
    scale_sweep,
)
from data_quality.dq_monitor import (
    CheckResult,
    Config,
    build_baseline,
    check_batch,
    default_config,
)
from data_quality.taxi import load_taxi, split


@dataclass
class FalsePositives:
    real_batches: int
    real_flagged: int
    simulated_batches: int
    simulated_flagged: int
    by_check: dict[str, int]

    @property
    def total(self) -> int:
        return self.real_batches + self.simulated_batches

    @property
    def flagged(self) -> int:
        return self.real_flagged + self.simulated_flagged

    @property
    def rate(self) -> float:
        return self.flagged / self.total


def _as_of(df: pl.DataFrame):
    return df["pickup"].max() + timedelta(hours=6)


def simulated_clean(
    pool: pl.DataFrame, n_batches: int, rng: np.random.Generator
) -> list[pl.DataFrame]:
    """Random row subsets of the held-out days, re-dated onto a single held-out day.

    Re-dating keeps each simulated batch a one-day slice like a real daily delivery
    (otherwise a batch mixing ten days would rightly trip the time-span check), while
    the rows themselves stay real and the mix of days makes the batches i.i.d.-ish.
    """
    days = pool["pickup"].dt.truncate("1d").unique().sort().to_list()
    out = []
    for _ in range(n_batches):
        n = int(rng.integers(150, 261))
        idx = np.sort(rng.choice(pool.height, size=min(n, pool.height), replace=False))
        target = days[int(rng.integers(0, len(days)))]
        batch = pool[idx]
        shift = pl.col("pickup").dt.truncate("1d") - pl.lit(target)
        out.append(
            batch.with_columns(
                (pl.col("pickup") - shift).alias("pickup"),
                (pl.col("dropoff") - shift).alias("dropoff"),
            ).sort("pickup")
        )
    return out


def setup(cfg: Config | None = None):
    df = load_taxi()
    train, held = split(df)
    baseline = build_baseline(train, cfg)
    pool = pl.concat([b for _, b in held])
    return baseline, held, pool


def false_positive_rate(
    baseline,
    held,
    pool,
    cfg: Config | None = None,
    n_simulated: int = 300,
    seed: int = 0,
    min_severity: str = "warn",
) -> FalsePositives:
    rng = np.random.default_rng(seed)
    by_check: dict[str, int] = {}

    def flagged(res: CheckResult) -> bool:
        bad = res.fired(min_severity)
        for c in bad:
            by_check[c] = by_check.get(c, 0) + 1
        return bool(bad)

    real = [flagged(check_batch(baseline, b, cfg, _as_of(b), n)) for n, b in held]
    sim = [
        flagged(check_batch(baseline, b, cfg, _as_of(b)))
        for b in simulated_clean(pool, n_simulated, rng)
    ]
    return FalsePositives(len(real), sum(real), len(sim), sum(sim), by_check)


def detection_rate(
    baseline,
    pool,
    corruption: Corruption,
    cfg: Config | None = None,
    trials: int = 30,
    seed: int = 1,
) -> float:
    rng = np.random.default_rng(seed)
    hits = 0
    for clean in simulated_clean(pool, trials, rng):
        res = check_batch(baseline, corruption.apply(clean, rng), cfg, _as_of(clean))
        hits += bool(res.fired(corruption.min_severity) & corruption.expect)
    return hits / trials


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--simulated", type=int, default=300)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = default_config()
    baseline, held, pool = setup(cfg)
    fp = false_positive_rate(baseline, held, pool, cfg, args.simulated, args.seed)
    print(
        f"baseline: {len(baseline['batches'])} real daily batches, {sum(baseline['row_counts'])} rows"
    )
    print(
        f"clean batches flagged at warn+: {fp.real_flagged}/{fp.real_batches} real, "
        f"{fp.simulated_flagged}/{fp.simulated_batches} simulated -> {fp.rate:.2%} overall"
    )
    if fp.by_check:
        print("  by check:", dict(sorted(fp.by_check.items(), key=lambda kv: -kv[1])))

    print(f"\ndetection over {args.trials} planted trials each:")
    for c in CORRUPTIONS:
        print(
            f"  {c.name:<20} {detection_rate(baseline, pool, c, cfg, args.trials, args.seed + 1):6.0%}  {c.description}"
        )
    for title, sweep in (
        ("fare scale", scale_sweep()),
        ("fare null rate", null_sweep()),
        ("cash->card share", mix_sweep()),
    ):
        print(f"\n{title} sweep:")
        for c in sweep:
            print(
                f"  {c.name:<20} {detection_rate(baseline, pool, c, cfg, args.trials, args.seed + 1):6.0%}"
            )


if __name__ == "__main__":
    main()
