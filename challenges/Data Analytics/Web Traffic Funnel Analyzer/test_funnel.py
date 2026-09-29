"""Tests for funnel.py.

Correctness is checked against brute-force oracles that enumerate every possible
event combination (itertools), never against the two-pointer/greedy code under
test. The end-to-end oracle also does its own grouping, sorting and
sessionization in plain Python.
"""

from __future__ import annotations

import gzip
import itertools
import json
import math
import random
from pathlib import Path

import funnel
import polars as pl
import pytest
from funnel import (
    Columns,
    Config,
    DataError,
    analyze,
    assign_entities,
    benjamini_hochberg,
    build_report,
    load_events,
    parse_duration,
    parse_steps,
    reach_any_order,
    reach_ordered,
    reach_strict,
    wilson_interval,
)
from scipy import stats

SAMPLE = Path(__file__).parent / "sample_data" / "rees46_sample.csv.gz"
MIN = 60_000
HOUR = 3_600_000


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def make_cfg(steps: str = "view,cart,purchase", **kw) -> Config:
    sets, labels = parse_steps(steps)
    return Config(steps=sets, labels=labels, **kw)


def make_events(rows: list[tuple]) -> pl.DataFrame:
    """rows: (user, t_ms, event[, session[, segment]]) -> the cleaned-events frame."""
    padded = [(*r, *([None] * (5 - len(r)))) for r in rows]
    return pl.DataFrame(
        {
            "seq": pl.Series(range(len(padded)), dtype=pl.UInt32),
            "user": [r[0] for r in padded],
            "ts": pl.Series([r[1] for r in padded], dtype=pl.Int64),
            "event": [r[2] for r in padded],
            "session": pl.Series([r[3] for r in padded], dtype=pl.String),
            "segment": pl.Series([r[4] for r in padded], dtype=pl.String),
        }
    )


def counts(result: funnel.FunnelResult) -> list[int]:
    return [s.count for s in result.steps]


def run_rows(rows, steps="view,cart,purchase", cols=None, **kw):
    return analyze(make_events(rows), make_cfg(steps, **kw), cols)


def csv_bytes(header: str, rows: list[str]) -> bytes:
    return ("\n".join([header, *rows]) + "\n").encode()


# --------------------------------------------------------------------------- #
# brute-force oracles
# --------------------------------------------------------------------------- #


def oracle_ordered(times, masks, k_steps, window):
    """Lexicographically smallest strictly increasing index tuple of maximal depth."""
    best = (0, [])
    for d in range(1, k_steps + 1):
        found = None
        for idxs in itertools.combinations(range(len(times)), d):
            if not all(masks[i] >> k & 1 for k, i in enumerate(idxs)):
                continue
            if window is not None and times[idxs[-1]] - times[idxs[0]] > window:
                continue
            found = idxs
            break
        if found is None:
            break
        best = (d, [times[i] for i in found])
    return best


def oracle_strict(times, masks, k_steps, window):
    best = (0, [])
    for d in range(1, k_steps + 1):
        found = None
        for start in range(len(times) - d + 1):
            idxs = range(start, start + d)
            if not all(masks[i] >> k & 1 for k, i in enumerate(idxs)):
                continue
            if window is not None and times[idxs[-1]] - times[idxs[0]] > window:
                continue
            found = idxs
            break
        if found is None:
            break
        best = (d, [times[i] for i in found])
    return best


def oracle_any(times, masks, k_steps, window):
    spans = []
    for n in range(1, k_steps + 1):
        pools = [[t for t, m in zip(times, masks) if m >> k & 1] for k in range(n)]
        if any(not pool for pool in pools):
            break
        best = min(max(c) - min(c) for c in itertools.product(*pools))
        if window is not None and best > window:
            break
        spans.append(best)
    return spans


def random_entity(rng: random.Random, k_steps: int, *, single_bit: bool):
    n = rng.randint(0, 9)
    times = sorted(rng.randint(0, 40) for _ in range(n))
    if single_bit:
        masks = [rng.choice([0, *(1 << k for k in range(k_steps))]) for _ in range(n)]
    else:
        masks = [rng.randint(0, (1 << k_steps) - 1) for _ in range(n)]
    return times, masks, rng.choice([None, 0, 3, 8, 15, 100])


# --------------------------------------------------------------------------- #
# configuration parsing
# --------------------------------------------------------------------------- #


def test_parse_duration():
    assert parse_duration("90s") == 90_000
    assert parse_duration("30m") == 30 * MIN
    assert parse_duration("2h") == 2 * HOUR
    assert parse_duration("7d") == 7 * 86_400_000
    assert parse_duration("1.5h") == 90 * MIN


@pytest.mark.parametrize("bad", ["", "30", "m", "-5m", "0m", "5x", "5 minutes"])
def test_parse_duration_rejects(bad):
    with pytest.raises(DataError):
        parse_duration(bad)


def test_parse_steps_and_or_sets():
    sets, labels = parse_steps("view, cart|wishlist ,purchase")
    assert sets == (
        frozenset({"view"}),
        frozenset({"cart", "wishlist"}),
        frozenset({"purchase"}),
    )
    assert labels == ("view", "cart or wishlist", "purchase")


@pytest.mark.parametrize("bad", ["view", "", "view,,cart", "view,|"])
def test_parse_steps_rejects(bad):
    with pytest.raises(DataError):
        parse_steps(bad)


def test_config_validation():
    with pytest.raises(DataError):
        make_cfg(mode="nope").validate()
    with pytest.raises(DataError):
        make_cfg(unit="nope").validate()
    with pytest.raises(DataError):  # overlapping names are ambiguous for any-order
        make_cfg("a|b,b", mode="any-order").validate()
    make_cfg("a|b,b", mode="ordered").validate()  # fine when order disambiguates
    with pytest.raises(DataError):
        make_cfg(outcome_step=1).validate()
    with pytest.raises(DataError):
        make_cfg(outcome_step=4).validate()


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("k", "n"), [(0, 10), (10, 10), (1, 2), (5, 10), (81, 263), (3455, 119850)]
)
@pytest.mark.parametrize("alpha", [0.05, 0.01])
def test_wilson_matches_scipy(k, n, alpha):
    ref = stats.binomtest(k, n).proportion_ci(1 - alpha, method="wilson")
    lo, hi = wilson_interval(k, n, alpha)
    assert lo == pytest.approx(ref.low, abs=1e-9)
    assert hi == pytest.approx(ref.high, abs=1e-9)


def test_wilson_edges():
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0
    assert 0 < hi < 1  # unlike the Wald interval, never collapses to a point
    lo, hi = wilson_interval(0, 0)
    assert math.isnan(lo)
    assert math.isnan(hi)


def test_benjamini_hochberg_known_example():
    # Textbook example: m=5, q_i = min over j>=i of p_j * m / j
    q = benjamini_hochberg([0.01, 0.04, 0.03, 0.005, 0.5])
    assert q == pytest.approx([0.025, 0.05, 0.05, 0.025, 0.5])


def test_benjamini_hochberg_skips_none_and_is_bounded():
    q = benjamini_hochberg([None, 0.2, None, 0.001])
    assert q[0] is None
    assert q[2] is None
    assert q[3] == pytest.approx(0.002)
    assert q[1] == pytest.approx(0.2)
    assert all(v is None or 0 <= v <= 1 for v in benjamini_hochberg([0.9, 0.99, 1.0]))


# --------------------------------------------------------------------------- #
# per-entity matching against oracles
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("k_steps", [2, 3, 4])
def test_reach_ordered_matches_oracle(k_steps):
    rng = random.Random(100 + k_steps)
    for _ in range(600):
        times, masks, window = random_entity(rng, k_steps, single_bit=False)
        got = reach_ordered(times, masks, k_steps, window)
        depth, chain = oracle_ordered(times, masks, k_steps, window)
        assert (got.depth, got.times) == (depth, chain), (times, masks, window)


@pytest.mark.parametrize("k_steps", [2, 3, 4])
def test_reach_strict_matches_oracle(k_steps):
    rng = random.Random(200 + k_steps)
    for _ in range(600):
        times, masks, window = random_entity(rng, k_steps, single_bit=False)
        got = reach_strict(times, masks, k_steps, window)
        depth, chain = oracle_strict(times, masks, k_steps, window)
        assert (got.depth, got.times) == (depth, chain), (times, masks, window)


@pytest.mark.parametrize("k_steps", [2, 3, 4])
def test_reach_any_order_matches_oracle(k_steps):
    rng = random.Random(300 + k_steps)
    for _ in range(600):
        times, masks, window = random_entity(rng, k_steps, single_bit=True)
        got = reach_any_order(times, masks, k_steps, window)
        spans = oracle_any(times, masks, k_steps, window)
        assert (got.depth, got.spans) == (len(spans), spans), (times, masks, window)


def test_mode_containment_strict_within_ordered_within_any_order():
    rng = random.Random(7)
    for _ in range(500):
        times, masks, window = random_entity(rng, 3, single_bit=True)
        s = reach_strict(times, masks, 3, window).depth
        o = reach_ordered(times, masks, 3, window).depth
        a = reach_any_order(times, masks, 3, window).depth
        assert s <= o <= a


def test_ordered_window_needs_a_later_attempt():
    # The first view is too old by the time the cart happens; the second isn't.
    times, masks = [0, 100, 110], [1, 1, 2]
    assert reach_ordered(times, masks, 2, None).times == [0, 110]
    assert reach_ordered(times, masks, 2, 20).times == [100, 110]
    assert reach_ordered(times, masks, 2, 5).depth == 1


def test_window_boundary_is_inclusive():
    times, masks = [0, 10], [1, 2]
    assert reach_ordered(times, masks, 2, 10).depth == 2
    assert reach_ordered(times, masks, 2, 9).depth == 1
    assert reach_strict(times, masks, 2, 10).depth == 2
    assert reach_any_order(times, [1, 2], 2, 10).depth == 2
    assert reach_any_order(times, [1, 2], 2, 9).depth == 1


def test_ordered_allows_events_between_strict_does_not():
    # view, other, cart
    times, masks = [0, 1, 2], [1, 0, 2]
    assert reach_ordered(times, masks, 2, None).depth == 2
    assert reach_strict(times, masks, 2, None).depth == 1


def test_strict_tolerates_repeats_only_when_consecutive_match():
    # view, view, cart: the second view is directly followed by the cart
    assert reach_strict([0, 1, 2], [1, 1, 2], 2, None).depth == 2
    # view, cart, view, purchase with steps view->cart->purchase: cart is followed by view
    assert reach_strict([0, 1, 2, 3], [1, 2, 1, 4], 3, None).depth == 2


def test_one_event_cannot_serve_two_steps():
    # A single event matching both step 1 and step 2 is one event, not two
    assert reach_ordered([0], [0b11], 2, None).depth == 1
    assert reach_ordered([0, 1], [0b11, 0b10], 2, None).depth == 2


def test_any_order_accepts_reversed_sequence_ordered_does_not():
    times, masks = [0, 5], [2, 1]  # cart then view
    assert reach_ordered(times, masks, 2, None).depth == 1
    assert reach_any_order(times, masks, 2, None).spans == [0, 5]


# --------------------------------------------------------------------------- #
# loading and cleaning
# --------------------------------------------------------------------------- #


def load(
    rows: list[str], header="user_id,event_time,event_type,user_session", cols=None
):
    raw = pl.read_csv(csv_bytes(header, rows), infer_schema=False)
    return load_events(raw, cols or Columns())


def test_timestamp_formats_all_land_on_the_same_instant():
    stamps = [
        "2019-11-01T00:00:00.000Z",
        "2019-11-01T02:00:00+02:00",
        "2019-11-01 00:00:00 UTC",
        "2019-11-01 00:00:00",
        "2019-11-01T00:00:00",
        "1572566400",  # epoch seconds
        "1572566400000",  # epoch milliseconds
        "2019-11-01",
    ]
    rows = [f"u{i},{s},view,s{i}" for i, s in enumerate(stamps)]
    df, q = load(rows)
    assert q.unparseable_time == 0
    assert set(df["ts"].to_list()) == {1572566400000}


def test_quality_counts_add_up():
    rows = [
        ",2024-01-01T00:00:00Z,view,s1",  # missing user
        "u1,,view,s1",  # missing time
        "u1,garbage,view,s1",  # unparseable
        "u1,2024-01-01T00:00:00Z,,s1",  # missing event
        "u1,2024-01-01T00:00:00Z,view,",  # missing session
        "u1,2024-01-01T00:00:01Z,view,s1",
        "u1,2024-01-01T00:00:01Z,view,s1",  # exact duplicate
        "u1,2024-01-01T00:00:02Z,cart,s1",
    ]
    df, q = load(rows, cols=Columns(session="user_session"))
    assert (q.null_user, q.null_time, q.unparseable_time) == (1, 1, 1)
    assert (q.null_event, q.null_session, q.duplicate_events) == (1, 1, 1)
    assert q.rows_read == 8
    assert q.rows_used == df.height == 2
    dropped = (
        q.null_user
        + q.null_time
        + q.unparseable_time
        + q.null_event
        + q.null_session
        + q.duplicate_events
    )
    assert q.rows_read - dropped == q.rows_used


def test_out_of_order_rows_are_counted_kept_and_resorted():
    rows = [
        "u1,2024-01-01T00:00:10Z,cart,",
        "u1,2024-01-01T00:00:05Z,view,",  # arrives after a later event
        "u2,2024-01-01T00:00:00Z,view,",
    ]
    df, q = load(rows)
    assert q.out_of_order_rows == 1
    assert df.height == 3
    result = analyze(df, make_cfg("view,cart", unit="user"))
    assert counts(result) == [2, 1]  # sorted by time, the view precedes the cart


def test_missing_column_and_empty_inputs():
    raw = pl.read_csv(csv_bytes("a,b", ["1,2"]), infer_schema=False)
    with pytest.raises(DataError, match="missing required column"):
        load_events(raw, Columns())
    df, q = load_events(pl.DataFrame(), Columns())
    assert df.height == 0
    assert q.rows_read == 0
    df, _ = load([])  # header only
    result = analyze(df, make_cfg())
    assert result.entrants == 0
    assert counts(result) == [0, 0, 0]
    assert "nothing to chart" in build_report(result)


def test_segment_level_keeps_leading_parts():
    rows = [
        "u1,2024-01-01T00:00:00Z,view,s1,electronics.smartphone",
        "u2,2024-01-01T00:00:00Z,view,s2,electronics",
        "u3,2024-01-01T00:00:00Z,view,s3,",
    ]
    hdr = "user_id,event_time,event_type,user_session,category_code"
    df, _ = load(rows, hdr, Columns(segment="category_code", segment_level=1))
    assert df.sort("user")["segment"].to_list() == ["electronics", "electronics", None]
    df2, _ = load(rows, hdr, Columns(segment="category_code"))
    assert "electronics.smartphone" in df2["segment"].to_list()


def test_gzip_input_is_read(tmp_path):
    path = tmp_path / "e.csv.gz"
    path.write_bytes(
        gzip.compress(
            csv_bytes(
                "user_id,event_time,event_type",
                ["u1,2024-01-01T00:00:00Z,view", "u1,2024-01-01T00:00:05Z,cart"],
            )
        )
    )
    df, q = load_events(funnel.read_raw(path))
    assert q.rows_used == 2
    assert df.height == 2


# --------------------------------------------------------------------------- #
# sessionization
# --------------------------------------------------------------------------- #


def sessions_of(rows, gap=30 * MIN, cols=None):
    cfg = make_cfg(unit="session", session_gap_ms=gap)
    frame = assign_entities(make_events(rows), cfg, cols or Columns())
    return frame.group_by("entity").len()["len"].sort().to_list()


def test_gap_of_exactly_thirty_minutes_stays_in_the_session():
    assert sessions_of([("u", 0, "view"), ("u", 30 * MIN, "view")]) == [2]
    assert sessions_of([("u", 0, "view"), ("u", 30 * MIN + 1, "view")]) == [1, 1]


def test_sessions_are_per_user_and_gap_is_measured_between_neighbours():
    rows = [
        ("a", 0, "view"),
        ("b", 5 * MIN, "view"),
        ("a", 20 * MIN, "view"),  # a: 20m after a's previous
        ("a", 45 * MIN, "view"),  # 25m after a's previous -> same session
        ("a", 200 * MIN, "view"),  # new session
    ]
    assert sessions_of(rows) == [1, 1, 3]


def test_native_session_column_overrides_the_gap_rule():
    rows = [("u", 0, "view", "s1"), ("u", 10 * HOUR, "cart", "s1")]
    assert sessions_of(rows, cols=Columns(session="user_session")) == [2]
    assert sessions_of(rows) == [1, 1]


def reference_sessions(rows, gap):
    by_user: dict[str, list] = {}
    for seq, r in enumerate(rows):
        by_user.setdefault(r[0], []).append((r[1], seq))
    out = 0
    for events in by_user.values():
        events.sort()
        out += 1 + sum(1 for a, b in itertools.pairwise(events) if b[0] - a[0] > gap)
    return out


def test_sessionization_matches_python_reference():
    rng = random.Random(11)
    for _ in range(40):
        rows = [
            (f"u{rng.randrange(6)}", rng.randrange(0, 6 * HOUR), "view")
            for _ in range(rng.randrange(1, 60))
        ]
        assert len(sessions_of(rows)) == reference_sessions(rows, 30 * MIN)


# --------------------------------------------------------------------------- #
# end-to-end analysis vs an independent oracle
# --------------------------------------------------------------------------- #

_ORACLES = {"ordered": oracle_ordered, "strict": oracle_strict}


def reference_counts(rows, steps, mode, unit, window, gap=30 * MIN):
    """Group, sort and sessionize in plain Python, then match by brute force."""
    sets, _ = parse_steps(steps)
    k_steps = len(sets)
    per_user: dict[str, list] = {}
    for seq, r in enumerate(rows):
        per_user.setdefault(r[0], []).append((r[1], seq, r[2]))
    entities: list[list] = []
    for events in per_user.values():
        events.sort()
        if unit == "user":
            entities.append(events)
            continue
        current = [events[0]]
        for prev, ev in itertools.pairwise(events):
            if ev[0] - prev[0] > gap:
                entities.append(current)
                current = []
            current.append(ev)
        entities.append(current)
    totals = [0] * k_steps
    for events in entities:
        times = [e[0] for e in events]
        masks = [sum(1 << k for k, s in enumerate(sets) if e[2] in s) for e in events]
        if mode == "any-order":
            depth = len(oracle_any(times, masks, k_steps, window))
        else:
            depth = _ORACLES[mode](times, masks, k_steps, window)[0]
        for k in range(depth):
            totals[k] += 1
    return totals


@pytest.mark.parametrize("mode", ["ordered", "strict", "any-order"])
@pytest.mark.parametrize("unit", ["session", "user"])
def test_analyze_matches_reference_on_random_logs(mode, unit):
    rng = random.Random(f"{mode}-{unit}")
    for _ in range(25):
        rows = [
            (
                f"u{rng.randrange(5)}",
                rng.randrange(0, 4 * HOUR),
                rng.choice(["view", "view", "cart", "purchase", "other"]),
            )
            for _ in range(rng.randrange(1, 40))
        ]
        window = rng.choice([None, 10 * MIN, HOUR])
        got = counts(run_rows(rows, mode=mode, unit=unit, window_ms=window))
        want = reference_counts(rows, "view,cart,purchase", mode, unit, window)
        assert got == want, (rows, mode, unit, window)


def test_same_timestamp_ties_follow_file_order():
    rows = [("u", 0, "cart"), ("u", 0, "view")]
    assert counts(run_rows(rows, "view,cart", mode="ordered")) == [1, 0]
    assert counts(run_rows(rows, "view,cart", mode="any-order")) == [1, 1]
    rows = [("u", 0, "view"), ("u", 0, "cart")]
    assert counts(run_rows(rows, "view,cart", mode="ordered")) == [1, 1]


def test_repeated_events_never_inflate_counts():
    once = [("u", 0, "view"), ("u", 1000, "cart")]
    spam = [("u", i * 1000, "view") for i in range(50)] + [("u", 60_000, "cart")] * 5
    assert counts(run_rows(once, "view,cart")) == [1, 1]
    assert counts(run_rows(spam, "view,cart")) == [1, 1]


def test_entities_that_never_view_are_not_entrants():
    rows = [("a", 0, "cart"), ("a", 1000, "purchase"), ("b", 0, "view")]
    result = run_rows(rows)
    assert result.entrants == 1
    assert counts(result) == [1, 0, 0]


def test_unmatched_step_is_reported():
    result = run_rows([("a", 0, "view")], "view,carrt")
    assert result.quality.unmatched_steps == ["carrt"]
    assert "matches no event" in build_report(result)


def test_or_step_accepts_either_event():
    rows = [("a", 0, "view"), ("a", 1, "wishlist"), ("b", 0, "view"), ("b", 1, "cart")]
    assert counts(run_rows(rows, "view,cart|wishlist")) == [2, 2]


# --------------------------------------------------------------------------- #
# step statistics
# --------------------------------------------------------------------------- #


def test_step_rates_intervals_and_times():
    # Four sessions: all view; three cart (after 10s, 20s, 30s); one purchases 60s after the cart.
    rows = []
    for i, cart_delay in enumerate([10, 20, 30]):
        rows += [
            (f"u{i}", 0, "view", f"s{i}"),
            (f"u{i}", cart_delay * 1000, "cart", f"s{i}"),
        ]
    rows += [("u0", 70_000, "purchase", "s0"), ("u3", 0, "view", "s3")]
    result = run_rows(rows, cols=Columns(session="user_session"))
    v, c, p = result.steps
    assert (v.count, c.count, p.count) == (4, 3, 1)
    assert c.of_entry == pytest.approx(0.75)
    assert c.of_prev == pytest.approx(0.75)
    assert p.of_prev == pytest.approx(1 / 3)
    assert p.dropped == 2
    assert c.of_prev_ci == pytest.approx(wilson_interval(3, 4))
    assert c.median_from_prev_s == pytest.approx(20)
    assert (c.q1_from_prev_s, c.q3_from_prev_s) == pytest.approx((15, 25))
    assert p.median_from_prev_s == pytest.approx(60)
    assert p.median_from_entry_s == pytest.approx(70)
    assert result.biggest_leak["from"] == "cart"
    assert result.biggest_leak["dropped"] == 2
    assert result.biggest_leak["share_of_all_drops"] == pytest.approx(2 / 3)


def test_chosen_chain_is_the_earliest_attempt_reaching_max_depth():
    # With a 20s window the first view times out, so the timing comes from the second attempt.
    rows = [("u", 0, "view"), ("u", 100_000, "view"), ("u", 110_000, "cart")]
    result = run_rows(rows, "view,cart", unit="user", window_ms=20_000)
    assert counts(result) == [1, 1]
    assert result.steps[1].median_from_prev_s == pytest.approx(10)


def test_any_order_reports_span_instead_of_step_gaps():
    rows = [("u", 0, "cart"), ("u", 7000, "view")]
    result = run_rows(rows, "view,cart", mode="any-order")
    assert counts(result) == [1, 1]
    assert result.steps[1].median_from_prev_s is None
    assert result.steps[1].median_from_entry_s == pytest.approx(7)


def test_window_none_vs_set_changes_user_level_conversion():
    rows = [("u", 0, "view"), ("u", 3 * HOUR, "cart")]
    assert counts(run_rows(rows, "view,cart", unit="user")) == [1, 1]
    assert counts(run_rows(rows, "view,cart", unit="user", window_ms=HOUR)) == [1, 0]


# --------------------------------------------------------------------------- #
# right-censoring
# --------------------------------------------------------------------------- #


def test_drop_censored_removes_entrants_whose_window_is_unfinished():
    rows = [
        ("early_done", 0, "view"),
        ("early_done", 10 * MIN, "cart"),
        ("early_lost", 0, "view"),
        ("late", 9 * HOUR + 30 * MIN, "view"),  # window (1h) runs past the log end
        ("end", 10 * HOUR, "view"),  # pins the log end at 10h
    ]
    kw = {"unit": "user", "window_ms": HOUR}
    keep = run_rows(rows, "view,cart", **kw)
    assert counts(keep) == [4, 1]
    assert keep.censored_dropped == 0
    drop = run_rows(rows, "view,cart", drop_censored=True, **kw)
    assert counts(drop) == [2, 1]
    assert drop.censored_dropped == 2
    assert "excluded" in build_report(drop)


def test_drop_censored_is_a_noop_without_a_window():
    rows = [("a", 0, "view"), ("b", HOUR, "view")]
    result = run_rows(rows, "view,cart", unit="user", drop_censored=True)
    assert result.censored_dropped == 0
    assert result.entrants == 2


# --------------------------------------------------------------------------- #
# segments
# --------------------------------------------------------------------------- #


def segmented_rows(spec: dict[str, tuple[int, int]]):
    """spec: segment -> (entrants, converters). Returns rows, one session each."""
    rows = []
    n = 0
    for seg, (total, conv) in spec.items():
        for i in range(total):
            sid = f"s{n}"
            n += 1
            rows.append((sid, 0, "view", sid, seg))
            if i < conv:
                rows.append((sid, 1000, "cart", sid, seg))
    return rows


SEG_COLS = Columns(session="user_session", segment="seg")


def test_segment_tests_match_scipy():
    spec = {"a": (200, 60), "b": (150, 20), "c": (100, 30)}
    result = run_rows(
        segmented_rows(spec), "view,cart", cols=SEG_COLS, min_segment_size=1
    )
    by_name = {s.name: s for s in result.segments}
    assert [s.name for s in result.segments] == ["a", "b", "c"]  # largest first
    total_n, total_c = 450, 110
    ps = []
    for name, (n, c) in spec.items():
        s = by_name[name]
        assert (s.entrants, s.counts) == (n, [n, c])
        assert s.conversion == pytest.approx(c / n)
        assert s.ci == pytest.approx(wilson_interval(c, n))
        _, p = stats.fisher_exact(
            [[c, n - c], [total_c - c, total_n - n - (total_c - c)]]
        )
        assert s.p_value == pytest.approx(p)
        rest_rate = (total_c - c) / (total_n - n)
        assert s.lift_vs_rest == pytest.approx((c / n) / rest_rate)
        ps.append(p)
    for s, q in zip(result.segments, benjamini_hochberg(ps)):
        assert s.q_value == pytest.approx(q)
    table = [[c, n - c] for n, c in spec.values()]
    assert result.omnibus_p == pytest.approx(
        stats.chi2_contingency(table, correction=False).pvalue
    )
    assert by_name["b"].significant
    assert by_name["b"].lift_vs_rest < 1


def test_identical_segments_are_not_flagged():
    spec = {"a": (300, 60), "b": (300, 60), "c": (300, 60)}
    result = run_rows(
        segmented_rows(spec), "view,cart", cols=SEG_COLS, min_segment_size=1
    )
    assert not any(s.significant for s in result.segments)
    assert result.omnibus_p == pytest.approx(1.0)


def test_small_and_excess_segments_are_merged_into_other():
    spec = {"big1": (100, 10), "big2": (90, 9), "tiny": (5, 1), "big3": (80, 8)}
    result = run_rows(
        segmented_rows(spec),
        "view,cart",
        cols=SEG_COLS,
        top_segments=2,
        min_segment_size=30,
    )
    names = [s.name for s in result.segments]
    assert names == ["big1", "big2", "(other)"]  # other is always last
    other = result.segments[-1]
    assert other.entrants == 85  # big3 + tiny
    assert other.counts == [85, 9]


def test_missing_segment_value_is_unknown_and_single_segment_has_no_test():
    rows = [(f"s{i}", 0, "view", f"s{i}", None) for i in range(40)]
    result = run_rows(rows, "view,cart", cols=SEG_COLS, min_segment_size=1)
    assert [s.name for s in result.segments] == ["(unknown)"]
    assert result.segments[0].p_value is None
    assert result.omnibus_p is None


def test_segment_is_taken_from_the_entry_event():
    rows = [
        ("u", 0, "view", "s", "first"),
        ("u", 1000, "view", "s", "second"),
        ("u", 2000, "cart", "s", "third"),
    ]
    result = run_rows(rows, "view,cart", cols=SEG_COLS, min_segment_size=1)
    assert [s.name for s in result.segments] == ["first"]


def test_outcome_step_selects_the_compared_step():
    rows = []
    for i in range(60):
        sid = f"s{i}"
        seg = "a" if i < 30 else "b"
        rows.append((sid, 0, "view", sid, seg))
        rows.append((sid, 1, "cart", sid, seg))  # everyone carts
        if seg == "a" and i < 20:
            rows.append((sid, 2, "purchase", sid, seg))
    result = run_rows(rows, cols=SEG_COLS, min_segment_size=1, outcome_step=2)
    assert all(s.conversion == 1.0 for s in result.segments)
    result = run_rows(rows, cols=SEG_COLS, min_segment_size=1)
    conv = {s.name: s.conversion for s in result.segments}
    assert conv == pytest.approx({"a": 20 / 30, "b": 0.0})


# --------------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------------- #

XSS = "<script>alert(1)</script>"
IMG = '"><img src=x onerror=alert(2)>'


def test_report_escapes_everything_data_derived():
    spec = {XSS: (60, 12), IMG: (60, 30), "plain": (60, 6)}
    # the third step never occurs, so its (hostile) name lands in the unmatched-step warning
    result = run_rows(
        segmented_rows(spec),
        f"view,cart,{XSS}",
        cols=SEG_COLS,
        min_segment_size=1,
    )
    assert {s.name for s in result.segments} >= {XSS, IMG}
    page = build_report(result, source_name=f"{XSS}{IMG}.csv")
    assert "<script>alert" not in page
    assert "<img src=x" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "&lt;img src=x onerror=alert(2)&gt;" in page
    # only Plotly's own bundle and figure scripts remain
    assert page.count("<script") == page.count("</script>")


def test_report_sections_and_self_contained(tmp_path):
    spec = {"electronics": (300, 90), "apparel": (200, 10)}
    result = run_rows(
        segmented_rows(spec), "view,cart", cols=SEG_COLS, min_segment_size=1
    )
    page = build_report(result, "demo.csv")
    for text in ("Funnel", "Data quality", "Biggest leak", "electronics", "Fisher"):
        assert text in page
    assert "<script src=" not in page  # the Plotly bundle is inlined, not fetched
    assert 'src="http' not in page


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def write_csv(
    tmp_path: Path, rows: list[str], header="user_id,event_time,event_type"
) -> Path:
    path = tmp_path / "events.csv"
    path.write_bytes(csv_bytes(header, rows))
    return path


def test_cli_end_to_end(tmp_path, capsys):
    path = write_csv(
        tmp_path,
        [
            "a,2024-01-01T00:00:00Z,view",
            "a,2024-01-01T00:00:10Z,cart",
            "b,2024-01-01T00:00:00Z,view",
        ],
    )
    out, js = tmp_path / "r.html", tmp_path / "r.json"
    code = funnel.main(
        [str(path), "--steps", "view,cart", "-o", str(out), "--json", str(js)]
    )
    assert code == 0
    assert out.read_text(encoding="utf-8").startswith("<!doctype html>")
    data = json.loads(js.read_text(encoding="utf-8"))
    assert [s["count"] for s in data["steps"]] == [2, 1]
    assert "2 sessions entered" in capsys.readouterr().out


@pytest.mark.parametrize(
    "args",
    [
        ["--steps", "view"],
        ["--window", "soon"],
        ["--user-col", "nope"],
        ["--mode", "any-order", "--steps", "a|b,b"],
        ["--outcome-step", "9"],
    ],
)
def test_cli_bad_input_exits_2(tmp_path, capsys, args):
    path = write_csv(tmp_path, ["a,2024-01-01T00:00:00Z,view"])
    assert funnel.main([str(path), "-o", str(tmp_path / "o.html"), *args]) == 2
    assert "error:" in capsys.readouterr().err


def test_cli_missing_file_exits_2(tmp_path, capsys):
    assert (
        funnel.main([str(tmp_path / "missing.csv"), "-o", str(tmp_path / "o.html")])
        == 2
    )
    assert "error:" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# the committed real-data sample (REES46, see README)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def real_events():
    if not SAMPLE.exists():
        pytest.skip("sample_data/rees46_sample.csv.gz not present (run fetch_data.py)")
    try:
        return load_events(
            funnel.read_raw(SAMPLE),
            Columns(session="user_session", segment="category_code", segment_level=1),
        )
    except Exception as exc:  # noqa: BLE001 -- any read failure means "skip", never "fail"
        pytest.skip(f"could not read the sample: {exc}")


def test_real_sample_invariants(real_events):
    df, q = real_events
    assert q.rows_read > 100_000
    assert q.unparseable_time == 0
    cols = Columns(session="user_session", segment="category_code", segment_level=1)
    result = analyze(df, make_cfg(), cols, q)
    c = counts(result)
    assert c[0] > c[1] > c[2] > 0
    assert all(0 < s.of_prev <= 1 for s in result.steps[1:])
    for s in result.steps:
        assert s.of_entry_ci[0] <= s.of_entry <= s.of_entry_ci[1]
    assert sum(s.entrants for s in result.segments) == result.entrants


def test_real_sample_mode_containment(real_events):
    df, q = real_events
    cols = Columns(session="user_session")
    by_mode = {m: counts(analyze(df, make_cfg(mode=m), cols, q)) for m in funnel.MODES}
    for k in range(3):
        assert by_mode["strict"][k] <= by_mode["ordered"][k] <= by_mode["any-order"][k]
    assert by_mode["strict"][2] < by_mode["any-order"][2]  # the modes genuinely differ


def test_real_sample_matches_reference_on_a_slice(real_events):
    """Cross-check the fast path against the brute-force reference on real sessions."""
    df, _ = real_events
    sessions = df["session"].unique().sort().head(400).to_list()
    part = df.filter(pl.col("session").is_in(sessions))
    rows = [
        (r["session"], r["ts"], r["event"], r["session"])
        for r in part.sort(["ts", "seq"]).iter_rows(named=True)
    ]
    for mode in ("ordered", "strict"):
        got = counts(
            analyze(part, make_cfg(mode=mode), Columns(session="user_session"))
        )
        assert got == reference_counts(
            [(r[0], r[1], r[2]) for r in rows], "view,cart,purchase", mode, "user", None
        )
