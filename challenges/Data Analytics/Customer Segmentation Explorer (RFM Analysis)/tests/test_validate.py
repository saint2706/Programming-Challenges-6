from datetime import date

import numpy as np
import polars as pl
import pytest
from helpers import lines
from rfm_explorer import validate as v
from rfm_explorer.rfm import build_rfm

SNAP = date(2011, 6, 1)
POP_SNAP = date(2021, 3, 1)  # the synthetic population runs 2020-01-01 for 540 days


def test_top_share_when_ordering_is_perfect():
    spend = [100, 10, 1, 0, 0, 0, 0, 0, 0, 0]
    assert v.top_share(spend, spend, 0.2) == pytest.approx(110 / 111)


def test_top_share_of_a_useless_ordering_matches_the_fraction():
    assert v.top_share(np.zeros(50), np.arange(50), 0.2) == pytest.approx(
        0.2
    )  # all tied


def test_top_share_splits_a_tied_group_across_the_cut_proportionally():
    # top 2 of 4: priority 2 has one customer, priority 1 has three tied (spend 3, 6, 9)
    assert v.top_share([2, 1, 1, 1], [10, 3, 6, 9], 0.5) == pytest.approx(
        (10 + 18 / 3) / 28
    )


def test_top_share_is_nan_without_outcome_or_customers():
    assert np.isnan(v.top_share([1, 2, 3], [0, 0, 0]))
    assert np.isnan(v.top_share([], []))


def test_future_window_is_start_inclusive_end_exclusive_and_ignores_refunds():
    ln = lines(
        (1, "2011-05-31", "A", 999.0),  # before the snapshot
        (1, "2011-06-01", "B", 10.0),  # first day, counts
        (1, "2011-06-20", "C9", -4.0),  # a refund is not spend
        (2, "2011-08-30", "D", 5.0),  # snapshot + 90 days is excluded
        (3, "2011-08-29", "E", 7.0),
    )
    out = v.future_outcomes(ln, pl.Series("customer_id", [1, 2, 3, 4]), SNAP, 90).sort(
        "customer_id"
    )
    assert out["future_spend"].to_list() == [10.0, 0.0, 7.0, 0.0]
    assert out["bought"].to_list() == [True, False, True, False]


def test_scoring_ignores_what_happens_after_the_snapshot(pop):
    before = pop.filter(
        pl.col("invoice_date") < pl.lit(POP_SNAP).cast(pl.Datetime("ms"))
    )
    assert build_rfm(pop, POP_SNAP).equals(build_rfm(before, POP_SNAP))


def test_segments_separate_future_behaviour(pop):
    res = v.validate(pop, POP_SNAP, horizon_days=90, resamples=60)
    q = res.segments["quintile"].sort("priority", descending=True)
    top, bottom = q.row(0, named=True), q.row(-1, named=True)
    assert top["repeat_rate"] > bottom["repeat_rate"] + 0.2
    assert top["mean_future_spend"] > 3 * bottom["mean_future_spend"]
    assert q["customers"].sum() == res.customers


def test_orderings_beat_random_and_the_reference_differs_from_itself_by_nothing(pop):
    res = v.validate(pop, POP_SNAP, resamples=60)
    o = {r["ordering"]: r for r in res.orderings.iter_rows(named=True)}
    ref = o[v.REFERENCE]
    assert ref["top_share"] > 0.3  # random is 0.2
    assert ref["top_share_lo"] <= ref["top_share"] <= ref["top_share_hi"]
    assert ref["vs_rfm_lo"] == ref["vs_rfm_hi"] == 0.0
    assert set(o) >= {
        "recency only",
        "frequency only",
        "monetary only",
        "kmeans: cluster rank",
        "fixed: R+F+M",
    }


def test_validation_is_reproducible(pop):
    a = v.validate(pop, POP_SNAP, resamples=40, seed=3)
    b = v.validate(pop, POP_SNAP, resamples=40, seed=3)
    assert a.orderings.equals(b.orderings)


def test_validate_rejects_a_snapshot_with_no_customers(pop):
    with pytest.raises(ValueError, match="no customers"):
        v.validate(pop, date(2019, 1, 1))
