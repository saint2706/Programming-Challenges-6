from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from fraud_flagging import policy
from fraud_flagging.data import RAW_COLUMNS


@pytest.fixture(scope="module")
def scored(out, frame):
    raw = frame.select([c for c in frame.columns if c in RAW_COLUMNS or c == "y"]).head(
        5_000
    )
    return policy.score_frame(out["booster"], out["meta"], raw)


def test_score_frame_adds_hour_and_calibrated_probabilities(scored):
    assert {"hour", "p_raw", "p"} <= set(scored.columns)
    assert scored["p"].is_between(0, 1).all() and scored["p"].n_unique() > 100


def test_score_frame_validates_its_input(out, frame):
    raw = frame.select(RAW_COLUMNS).head(10)
    with pytest.raises(ValueError, match="missing columns"):
        policy.score_frame(out["booster"], out["meta"], raw.drop("V9"))


def test_precision_and_recall_policies_apply_the_stored_threshold(out, scored):
    meta = out["meta"]
    thr = meta["thresholds"]["precision"]["0.9"]
    flags = policy.decide(scored, meta, "precision", precision=0.9)
    assert np.array_equal(flags, scored["p"].to_numpy() >= thr)
    rec = policy.decide(scored, meta, "recall", recall=0.8)
    assert len(rec) == scored.height and rec.dtype == bool


def test_a_target_outside_the_stored_grid_is_rejected_with_the_options(out, scored):
    with pytest.raises(ValueError, match="one of"):
        policy.decide(scored, out["meta"], "precision", precision=0.42)


def test_an_unreachable_target_says_so(out, scored):
    meta = {**out["meta"], "thresholds": {"precision": {"0.9": None}, "recall": {}}}
    with pytest.raises(ValueError, match="not reachable"):
        policy.decide(scored, meta, "precision", precision=0.9)


def test_cost_policy_flags_more_as_reviews_get_cheaper(out, scored):
    counts = [
        policy.decide(scored, out["meta"], "cost", review_cost=c).sum()
        for c in (200, 25, 3, 0.5)
    ]
    assert counts == sorted(counts) and counts[-1] > counts[0]


def test_cost_policy_weighs_the_amount(out, scored):
    big = scored.with_columns((pl.col("Amount") * 10).alias("Amount"))
    assert (
        policy.decide(big, out["meta"], "cost", review_cost=25).sum()
        >= policy.decide(scored, out["meta"], "cost", review_cost=25).sum()
    )


def test_capacity_policy_flags_at_most_k_per_hour(out, scored):
    flags = policy.decide(scored, out["meta"], "capacity", per_hour=2)
    per_hour = (
        scored.with_columns(pl.Series("f", flags))
        .group_by("hour")
        .agg(pl.col("f").sum())
    )
    assert per_hour["f"].max() == 2 and flags.sum() == 2 * scored["hour"].n_unique()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"policy": "magic"}, "policy must be"),
        ({"policy": "cost", "review_cost": 0}, "positive"),
        ({"policy": "cost", "friction": -1}, "not negative"),
        ({"policy": "capacity", "per_hour": 0}, "at least 1"),
    ],
)
def test_bad_policy_parameters_are_rejected(out, scored, kwargs, message):
    with pytest.raises(ValueError, match=message):
        policy.decide(scored, out["meta"], **kwargs)
