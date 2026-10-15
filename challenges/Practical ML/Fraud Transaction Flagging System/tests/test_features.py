from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from fraud_flagging import features
from fraud_flagging.data import RAW_COLUMNS
from fraud_flagging.features import FEATURES, GROUPS


def one(**over) -> pl.DataFrame:
    row = {c: 0.0 for c in RAW_COLUMNS} | {"Time": 0.0, "Amount": 10.0}
    return pl.DataFrame([row | over])


def test_amount_features_by_hand():
    f = features.build(
        pl.concat([one(Amount=0.0), one(Amount=12.0), one(Amount=19.99)])
    )
    assert f["log_amount"].to_list() == pytest.approx(
        [0.0, np.log1p(12.0), np.log1p(19.99)]
    )
    assert f["amount_frac"].to_list() == pytest.approx([0.0, 0.0, 0.99])
    assert f["amount_zero"].to_list() == [1.0, 0.0, 0.0]


def test_time_of_day_is_a_circle_and_hour_counts_from_the_start():
    f = features.build(
        pl.concat([one(Time=t) for t in (0.0, 21_600.0, 43_200.0, 86_400.0, 90_000.0)])
    )
    assert f["tod_sin"].to_list() == pytest.approx(
        [0, 1, 0, 0, np.sin(2 * np.pi / 24)], abs=1e-9
    )
    assert f["tod_cos"].to_list() == pytest.approx(
        [1, 0, -1, 1, np.cos(2 * np.pi / 24)], abs=1e-9
    )
    assert f["hour"].to_list() == [0, 6, 12, 24, 25]


def test_features_of_a_row_do_not_depend_on_the_other_rows(frame):
    """Scoring is stateless: a row alone, or among others in another order, gets the same features."""
    cols = [c for c in frame.columns if c in RAW_COLUMNS]
    pick = frame.sample(50, seed=1)
    alone = features.build(pick.select(cols).head(1))
    among = features.build(pick.select(cols).reverse())
    row = pick.row(0, named=True)
    match = among.filter(pl.col("Time") == row["Time"]).filter(
        pl.col("V1") == row["V1"]
    )
    assert match.select(FEATURES).row(0) == pytest.approx(alone.select(FEATURES).row(0))


def test_groups_cover_every_feature_exactly_once():
    flat = [f for g in GROUPS.values() for f in g]
    assert len(flat) == len(set(flat)) == len(FEATURES)
    for group, cols in GROUPS.items():
        assert set(FEATURES) - set(features.without(group)) == set(cols)


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda: one().drop("V7"), "missing columns"),
        (lambda: one(Amount=-1.0), "negative"),
        (lambda: one(Time=-5.0), "negative"),
        (lambda: one(V3=float("nan")), "non-finite"),
        (lambda: one(V3=float("inf")), "non-finite"),
        (lambda: one().with_columns(pl.lit("x").alias("V2")), "numeric"),
        (
            lambda: one().with_columns(pl.lit(None, dtype=pl.Float64).alias("V2")),
            "non-finite",
        ),
    ],
)
def test_validate_rejects_bad_input(make, message):
    with pytest.raises(ValueError, match=message):
        features.build(make())


def test_build_keeps_the_labels_and_ids_it_was_given(frame):
    assert {"y", "tx_id", "hour"} <= set(frame.columns)
    assert frame.select(pl.col(FEATURES).is_infinite().any()).to_numpy().sum() == 0
