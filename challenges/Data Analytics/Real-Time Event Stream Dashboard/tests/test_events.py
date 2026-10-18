import numpy as np
from conftest import SAMPLE
from stream_dashboard.events import Disorder, arrival_order, load_events


def test_sample_loads_sorted_with_expected_columns():
    df = load_events(SAMPLE)
    assert df.columns == [
        "ts_ms",
        "type",
        "product_id",
        "brand",
        "price_cents",
        "user_id",
    ]
    assert df.height == 100_000
    assert df["ts_ms"].is_sorted()
    assert set(df["type"].unique()) <= {"view", "cart", "remove_from_cart", "purchase"}
    assert df["brand"].null_count() == 0  # missing brands become ""
    assert df["price_cents"].min() >= 0


def test_sample_is_one_contiguous_stretch_of_real_traffic():
    df = load_events(SAMPLE)
    span_h = (df["ts_ms"].max() - df["ts_ms"].min()) / 3_600_000
    assert 1.0 < span_h < 3.0
    gaps = np.diff(df["ts_ms"].to_numpy())
    assert gaps.max() < 120_000  # no missing minutes


def test_limit():
    assert load_events(SAMPLE, limit=123).height == 123


def test_no_disorder_keeps_event_order(events):
    out = arrival_order(events)
    assert out["ts_ms"].to_list() == events["ts_ms"].to_list()
    assert (out["arrival_ms"] == out["ts_ms"]).all()


def test_disorder_is_seeded_and_bounded(events):
    d = Disorder(fraction=0.1, max_delay_ms=20_000, seed=3)
    a, b = arrival_order(events, d), arrival_order(events, d)
    assert a.equals(b)
    delay = a["arrival_ms"] - a["ts_ms"]
    assert delay.min() == 0 and delay.max() <= 20_000
    share = (delay > 0).mean()
    assert 0.07 < share < 0.13
    assert a["arrival_ms"].is_sorted()
    assert not a["ts_ms"].is_sorted()


def test_stragglers_exceed_max_delay(events):
    d = Disorder(
        fraction=0.0, max_delay_ms=20_000, straggler_fraction=0.05, straggler_ms=200_000
    )
    arrived = arrival_order(events, d)
    delay = arrived["arrival_ms"] - arrived["ts_ms"]
    late = delay.filter(delay > 0)
    assert late.min() > 20_000 and late.max() <= 200_000


def test_different_seed_changes_the_order(events):
    a = arrival_order(events, Disorder(0.1, seed=1))
    b = arrival_order(events, Disorder(0.1, seed=2))
    assert not a.equals(b)
