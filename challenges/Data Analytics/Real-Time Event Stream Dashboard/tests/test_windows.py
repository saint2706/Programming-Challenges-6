import random

import pytest
from stream_dashboard.windows import Layout, WindowSpec, parse_duration, watermark


def test_parse_duration_units():
    assert parse_duration("250ms") == 250
    assert parse_duration("30s") == 30_000
    assert parse_duration("5m") == 300_000
    assert parse_duration("2h") == 7_200_000


@pytest.mark.parametrize("bad", ["", "0s", "-5s", "1.5m", "10", "5d", "m5", "1 m s"])
def test_parse_duration_rejects(bad):
    with pytest.raises(ValueError):
        parse_duration(bad)


def test_tumbling_has_one_window_aligned_to_epoch():
    spec = WindowSpec.tumbling("1m")
    assert spec.kind == "tumbling"
    assert spec.starts_containing(0) == [0]
    assert spec.starts_containing(59_999) == [0]
    assert spec.starts_containing(60_000) == [60_000]


def test_sliding_event_lands_in_size_over_hop_windows():
    spec = WindowSpec.sliding("5m", "1m")
    assert spec.kind == "sliding"
    starts = spec.starts_containing(10 * 60_000 + 5)
    assert starts == [6 * 60_000, 7 * 60_000, 8 * 60_000, 9 * 60_000, 10 * 60_000]


def test_starts_containing_property_random():
    rng = random.Random(1)
    for _ in range(500):
        hop = rng.choice([1_000, 7_000, 60_000])
        size = hop * rng.randint(1, 8)
        spec = WindowSpec("x", size, hop)
        ts = rng.randint(0, 10**12)
        starts = spec.starts_containing(ts)
        assert len(starts) == size // hop
        assert all(s % hop == 0 and s <= ts < s + size for s in starts)
        assert starts == sorted(starts)
        assert not (
            starts[0] - hop <= ts < starts[0] - hop + size
        )  # none missed before
        assert not (starts[-1] + hop <= ts)  # nor after


def test_window_not_containing_neighbours():
    spec = WindowSpec.sliding("3m", "1m")
    ts = 10 * 60_000
    contained = set(spec.starts_containing(ts))
    for s in range(0, 20 * 60_000, 60_000):
        assert (s in contained) == (s <= ts < s + spec.size_ms)


def test_closed_exactly_at_the_boundary():
    spec = WindowSpec.tumbling("1m")
    assert not spec.is_closed(0, 59_999)
    assert spec.is_closed(0, 60_000)


def test_open_starts_drops_only_closed_windows():
    spec = WindowSpec.sliding("5m", "1m")
    ts = 10 * 60_000
    all_starts = spec.starts_containing(ts)  # 6m..10m, ending 11m..15m
    assert spec.open_starts(ts, 8 * 60_000) == all_starts  # all still end after 8m
    assert (
        spec.open_starts(ts, 11 * 60_000) == all_starts[1:]
    )  # the 6m window ends at 11m: closed
    assert spec.open_starts(ts, 13 * 60_000 + 1) == all_starts[3:]


def test_open_starts_empty_when_everything_closed():
    spec = WindowSpec.tumbling("1m")
    assert spec.open_starts(30_000, 120_000) == []


def test_spec_validation():
    with pytest.raises(ValueError):
        WindowSpec("x", 60_000, 120_000)
    with pytest.raises(ValueError):
        WindowSpec("x", 100_000, 60_000)
    with pytest.raises(ValueError):
        WindowSpec("x", 0, 0)


def test_parse_specs():
    assert WindowSpec.parse("1m") == WindowSpec.tumbling("1m")
    assert WindowSpec.parse("5m/1m") == WindowSpec.sliding("5m", "1m")
    with pytest.raises(ValueError):
        WindowSpec.parse("1m/5m")


def test_watermark():
    assert watermark(100_000, 30_000) == 70_000
    assert watermark(None, 30_000) < -(10**15)


def test_layout_keys_share_the_prefix():
    L = Layout(prefix="p:")
    spec = WindowSpec.tumbling("1m")
    keys = [
        L.stream,
        L.done,
        L.max_ts,
        L.telemetry,
        L.meta,
        L.late,
        L.index(spec),
        L.agg(spec, 0),
    ]
    assert all(k.startswith("p:") for k in keys) and len(set(keys)) == len(keys)
