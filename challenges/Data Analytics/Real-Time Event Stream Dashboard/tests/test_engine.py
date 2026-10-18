import polars as pl
from helpers import HUGE_LATENESS_MS, ev, events_in_order, messy
from stream_dashboard.engine import MemorySink, plan_event, simulate
from stream_dashboard.oracle import expected_windows
from stream_dashboard.windows import DEFAULT_SPECS, Layout, WindowSpec

L = Layout(prefix="t:")
T1 = WindowSpec.tumbling("1m")
S5 = WindowSpec.sliding("5m", "1m")


def run(events, specs, lateness):
    return simulate(events, specs, lateness, L)


def test_plan_counts_a_purchase_everywhere():
    sink = MemorySink()
    plan = plan_event(
        ev(90_000, "purchase", product=7, brand="b", cents=250, user=9), [T1], 0, L
    )
    sink.apply(plan.ops)
    assert dict(sink.hashes[L.agg(T1, 60_000)]) == {
        "n": 1,
        "purchase": 1,
        "revenue_cents": 250,
    }
    assert sink.sets[L.users(T1, 60_000)] == {"9"}
    assert sink.zsets[L.products(T1, 60_000)]["7"] == 1
    assert sink.zsets[L.brands(T1, 60_000)]["b"] == 250
    assert plan.late_specs == []


def test_views_add_no_revenue_and_blank_brand_is_not_ranked():
    sink = MemorySink()
    sink.apply(plan_event(ev(1_000, "view", brand=""), [T1], 0, L).ops)
    sink.apply(plan_event(ev(2_000, "purchase", brand=""), [T1], 0, L).ops)
    assert "revenue_cents" not in {
        k for k in sink.hashes[L.agg(T1, 0)] if sink.hashes[L.agg(T1, 0)][k] == 0
    }
    assert sink.zsets[L.brands(T1, 0)] == {}


def test_late_event_goes_to_late_not_to_a_window():
    # the watermark (4 minutes) is past the end of the event's only tumbling window
    plan = plan_event(ev(30_000), [T1], 4 * 60_000, L)
    assert plan.late_specs == ["t1m"]
    assert plan.ops == [["HINCRBY", L.late, "t1m", "1"]]


def test_sliding_event_is_partly_late_but_not_dropped_while_a_window_is_open():
    ts = 10 * 60_000 + 1
    plan = plan_event(
        ev(ts), [S5], 12 * 60_000, L
    )  # windows ending 11m, 12m closed; 13m..15m open
    assert plan.late_specs == []
    keys = {op[1] for op in plan.ops if op[0] == "HINCRBY"}
    assert {L.agg(S5, s) for s in (8 * 60_000, 9 * 60_000, 10 * 60_000)} <= keys
    assert L.agg(S5, 6 * 60_000) not in keys


def test_ordered_stream_matches_polars_oracle_exactly(events):
    sink = run(events_in_order(events), DEFAULT_SPECS, HUGE_LATENESS_MS)
    for spec in DEFAULT_SPECS:
        exp = expected_windows(events, spec)
        for row in exp.iter_rows(named=True):
            agg = sink.hashes[L.agg(spec, row["start_ms"])]
            assert agg["n"] == row["n"]
            assert agg.get("revenue_cents", 0) == round(row["revenue"] * 100)
            assert len(sink.sets[L.users(spec, row["start_ms"])]) == row["users"]
        assert len(sink.zsets[L.index(spec)]) == exp.height


def test_disorder_within_lateness_loses_nothing(events):
    shuffled = messy(events, fraction=0.2)
    assert shuffled["ts_ms"].to_list() != events["ts_ms"].to_list()
    sink = run(events_in_order(shuffled), DEFAULT_SPECS, 30_000)
    assert sum(sink.hashes[L.late].values()) == 0
    exp = expected_windows(events, T1)
    got = {
        int(k.split(":")[3]): v["n"]
        for k, v in sink.hashes.items()
        if k.startswith("t:w:t1m:") and k.endswith(":agg")
    }
    assert got == dict(zip(exp["start_ms"], exp["n"], strict=True))


def test_stragglers_are_dropped_and_every_event_is_accounted_for(events):
    shuffled = messy(events, fraction=0.05, stragglers=0.02)
    sink = run(events_in_order(shuffled), [T1], 30_000)
    in_windows = sum(v["n"] for k, v in sink.hashes.items() if k.startswith("t:w:t1m:"))
    late = sink.hashes[L.late]["t1m"]
    assert late > 0
    assert in_windows + late == events.height


def test_zero_lateness_drops_more_than_thirty_seconds(events):
    shuffled = messy(events, fraction=0.2)
    strict = run(events_in_order(shuffled), [T1], 0).hashes[L.late]["t1m"]
    loose = (
        run(events_in_order(shuffled), [T1], 30_000)
        .hashes.get(L.late, {})
        .get("t1m", 0)
    )
    assert strict > loose


def test_watermark_never_moves_backwards():
    # a late-arriving old event must not reopen windows the watermark already closed
    events = [ev(10 * 60_000), ev(1_000), ev(10 * 60_000 + 1)]
    sink = run(events, [T1], 30_000)
    assert sink.hashes[L.late]["t1m"] == 1
    assert L.agg(T1, 0) not in sink.hashes


def test_unique_users_counts_distinct_users_once():
    events = [ev(1_000, user=1), ev(2_000, user=1), ev(3_000, user=2)]
    sink = run(events, [T1], HUGE_LATENESS_MS)
    assert len(sink.sets[L.users(T1, 0)]) == 2 and sink.hashes[L.agg(T1, 0)]["n"] == 3


def test_empty_input():
    sink = run([], DEFAULT_SPECS, 30_000)
    assert not sink.hashes and not sink.zsets and pl.DataFrame().is_empty()
