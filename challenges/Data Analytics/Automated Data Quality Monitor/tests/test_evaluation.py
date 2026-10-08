"""The claims in the README, tested on real taxi data with planted corruptions."""

from __future__ import annotations

import itertools

import numpy as np
import polars as pl
import pytest
from data_quality import dq_monitor as dq
from data_quality import evaluate as ev
from data_quality.corruptions import CORRUPTIONS, mix_sweep, null_sweep, scale_sweep
from data_quality.taxi import daily_batches, load_taxi, split


@pytest.fixture(scope="module")
def env():
    return ev.setup()


def test_taxi_fixture_shape():
    df = load_taxi()
    assert df.height == 6433 and df["trip_id"].n_unique() == df.height
    days = daily_batches(df)
    assert (
        len(days) == 31 and sum(b.height for b in days.values()) == df.height
    )  # Feb-28 trip folded in
    train, held = split(df)
    assert (len(train), len(held)) == (21, 10)


def test_real_held_out_days_raise_no_warnings(env):
    baseline, held, _ = env
    for name, batch in held:
        res = dq.check_batch(baseline, batch, as_of=ev._as_of(batch), name=name)
        assert not res.fired("warn"), (name, [a.message for a in res.alerts])


def test_false_positive_rate_on_simulated_clean_batches(env):
    baseline, held, pool = env
    fp = ev.false_positive_rate(baseline, held, pool, n_simulated=300, seed=0)
    assert fp.real_flagged == 0
    # ~40 tests per batch at alpha = 0.001 allows a few percent; the measured rate is 1.7%
    assert fp.rate <= 0.04, fp


@pytest.mark.parametrize("corruption", CORRUPTIONS, ids=lambda c: c.name)
def test_every_planted_corruption_is_detected(env, corruption):
    baseline, _, pool = env
    assert ev.detection_rate(baseline, pool, corruption, trials=20) >= 0.95


@pytest.mark.parametrize(
    "sweep,tiny,large",
    [(scale_sweep(), 0, 3), (null_sweep(), 0, 3), (mix_sweep(), 0, 3)],
    ids=["fare-scale", "fare-nulls", "cash-mix"],
)
def test_detection_grows_with_magnitude_and_has_a_floor(env, sweep, tiny, large):
    baseline, _, pool = env
    rates = [ev.detection_rate(baseline, pool, c, trials=20) for c in sweep]
    assert rates[tiny] <= 0.2, (
        "the smallest change is inside normal noise and must not alert"
    )
    assert rates[large] >= 0.95 and rates[-1] >= 0.95
    assert all(b >= a - 0.2 for a, b in itertools.pairwise(rates)), rates


def test_significant_but_small_shift_does_not_alert(env):
    """+10% on `fare` over ~2000 rows: KS says p ~ 1e-15, PSI and the effect size say who cares."""
    baseline, _, pool = env
    spec = {c["name"]: c for c in baseline["columns"]}["fare"]
    shifted = pool.with_columns((pl.col("fare") * 1.1).alias("fare"))
    _, p = dq.ks_against_reference(
        shifted["fare"].to_numpy(), np.asarray(spec["quantiles"]), spec["n_finite"]
    )
    assert p < 1e-6
    assert not dq.check_batch(baseline, shifted).by_check("drift.numeric")


def test_csv_round_trip_gives_the_same_baseline(tmp_path):
    train, _ = split(load_taxi())
    frames = []
    for name, df in train:
        path = tmp_path / name
        df.write_csv(path, datetime_format="%Y-%m-%d %H:%M:%S")
        frames.append((name, dq.read_batch(path)))
    from_csv = dq.build_baseline(frames)
    from_frames = dq.build_baseline(train)
    strip = lambda b: [
        (c["name"], c["type"], c["kind"], c["unique"], c["n_null"])
        for c in b["columns"]
    ]
    assert strip(from_csv) == strip(from_frames)
    assert from_csv["row_counts"] == from_frames["row_counts"]
