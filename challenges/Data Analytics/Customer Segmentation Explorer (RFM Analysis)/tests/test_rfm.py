from datetime import date

import polars as pl
import pytest
from helpers import lines
from rfm_explorer.rfm import build_rfm, last_day, window

SNAP = date(2011, 1, 31)


def one(frame, cid):
    return frame.filter(frame["customer_id"] == cid).row(0, named=True)


def test_recency_frequency_and_money_by_hand():
    ln = lines(
        (1, "2011-01-10", "A", 100.0),
        (1, "2011-01-10", "A", 50.0),  # same invoice: still one purchase
        (1, "2011-01-20", "B", 30.0),
        (2, "2011-01-30", "C", 10.0),
    )
    out = build_rfm(ln, SNAP, lookback_days=None)
    a = one(out, 1)
    assert (a["recency_days"], a["frequency"], a["monetary"]) == (11, 2, 180.0)
    b = one(out, 2)
    assert (b["recency_days"], b["frequency"], b["monetary"]) == (1, 1, 10.0)


def test_snapshot_day_itself_and_later_activity_are_invisible():
    ln = lines(
        (1, "2011-01-10", "A", 10.0),
        (1, "2011-01-31", "B", 99.0),
        (1, "2011-02-15", "C", 99.0),
    )
    a = one(build_rfm(ln, SNAP, None), 1)
    assert (a["recency_days"], a["frequency"], a["monetary"]) == (21, 1, 10.0)


def test_future_rows_cannot_change_the_result():
    past = lines((1, "2011-01-10", "A", 10.0), (2, "2011-01-05", "B", 5.0))
    with_future = lines(
        (1, "2011-01-10", "A", 10.0),
        (2, "2011-01-05", "B", 5.0),
        (1, "2011-03-01", "Z", 1000.0),
        (3, "2011-03-01", "Y", 1.0),
    )
    assert build_rfm(past, SNAP, 365).equals(build_rfm(with_future, SNAP, 365))


def test_a_refund_changes_net_money_but_is_not_activity():
    ln = lines((1, "2011-01-10", "A", 100.0), (1, "2011-01-25", "C9", -40.0))
    net = one(build_rfm(ln, SNAP, None, "net"), 1)
    gross = one(build_rfm(ln, SNAP, None, "gross"), 1)
    assert net["monetary"] == 60.0
    assert gross["monetary"] == 100.0
    assert (
        net["recency_days"] == gross["recency_days"] == 21
    )  # the refund on the 25th is not a visit
    assert net["frequency"] == 1


def test_customer_with_only_a_refund_in_the_window_is_not_scored():
    ln = lines((1, "2010-06-01", "A", 100.0), (1, "2011-01-25", "C9", -100.0))
    assert build_rfm(ln, SNAP, lookback_days=90).is_empty()


def test_net_money_can_be_negative():
    ln = lines((1, "2011-01-10", "A", 20.0), (1, "2011-01-25", "C9", -50.0))
    assert one(build_rfm(ln, SNAP, None), 1)["monetary"] == -30.0


def test_lookback_includes_its_first_day_and_excludes_the_day_before():
    ln = lines((1, "2011-01-01", "A", 1.0), (2, "2010-12-31", "B", 1.0))
    out = build_rfm(ln, SNAP, lookback_days=30)  # window starts 2011-01-01
    assert out["customer_id"].to_list() == [1]
    assert window(ln, SNAP, 30).height == 1


def test_country_is_the_one_on_the_latest_purchase():
    ln = lines((1, "2011-01-10", "A", 1.0), (1, "2011-01-20", "B", 1.0)).with_columns(
        country=pl.Series(["France", "Spain"])
    )
    assert one(build_rfm(ln, SNAP, None), 1)["country"] == "Spain"


def test_bad_monetary_mode_is_rejected():
    with pytest.raises(ValueError, match="monetary"):
        build_rfm(lines((1, "2011-01-10", "A", 1.0)), SNAP, None, "weird")  # type: ignore[arg-type]


def test_empty_window_gives_an_empty_table_with_the_right_columns():
    out = build_rfm(lines((1, "2011-06-01", "A", 1.0)), SNAP, None)
    assert out.is_empty()
    assert out.columns == [
        "customer_id",
        "country",
        "recency_days",
        "frequency",
        "monetary",
    ]


def test_last_day():
    assert last_day(
        lines((1, "2011-01-10", "A", 1.0), (1, "2011-03-02", "B", 1.0))
    ) == date(2011, 3, 2)
