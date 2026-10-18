import plotly.graph_objects as go
import polars as pl
import pytest
from helpers import HUGE_LATENESS_MS
from stream_dashboard import charts
from stream_dashboard.producer import ProducerConfig, produce
from stream_dashboard.snapshot import snapshot, throughput, top_table
from stream_dashboard.windows import DEFAULT_SPECS
from stream_dashboard.worker import Worker, WorkerConfig


@pytest.fixture
def populated(broker, events):
    head = events.head(3000)
    produce(
        broker,
        head.with_columns(arrival_ms=head["ts_ms"]),
        ProducerConfig(speed=0, backpressure="none"),
    )
    Worker(broker, WorkerConfig(name="w1", lateness_ms=60_000, block_ms=50)).run()
    return broker


def test_snapshot_of_a_fresh_prefix_is_empty_not_an_error(broker):
    snap = snapshot(broker.r, broker.layout)
    assert all(df.is_empty() for df in snap.windows.values())
    assert snap.health.stream_len == 0 and snap.health.max_event_ms is None
    assert snap.health.watermark_ms is None and snap.health.backlog == 0
    assert snap.telemetry.is_empty() and snap.throughput.is_empty()


def test_lag_and_snapshot_before_any_stream_exists(redis_url, layout):
    from stream_dashboard.broker import Broker, connect

    b = Broker(
        connect(redis_url), layout
    )  # no ensure_group: the stream key does not exist
    assert b.lag().backlog == 0 and b.consumers() == []
    assert snapshot(b.r, layout).health.stream_len == 0


def test_snapshot_reports_windows_health_and_workers(populated):
    snap = snapshot(populated.r, populated.layout, DEFAULT_SPECS, 60_000)
    t1, s5 = snap.windows["t1m"], snap.windows["s5m-1m"]
    assert t1["n"].sum() == 3000
    assert t1["start_ms"].is_sorted() and (t1["start_ms"] % 60_000 == 0).all()
    assert (
        t1["view"] + t1["cart"] + t1["remove_from_cart"] + t1["purchase"] == t1["n"]
    ).all()
    assert (
        s5["n"].sum() > t1["n"].sum()
    )  # events counted in several overlapping windows
    h = snap.health
    assert h.producer_done and h.sent == h.total == 3000 and h.backlog == 0
    assert h.watermark_ms == h.max_event_ms - 60_000
    assert snap.consumers["name"].to_list() == ["w1"]
    assert snap.consumers["pending"].sum() == 0


def test_only_windows_before_the_watermark_are_closed(populated):
    snap = snapshot(populated.r, populated.layout, DEFAULT_SPECS, 60_000)
    t1 = snap.windows["t1m"]
    wm = snap.health.watermark_ms
    assert (t1["closed"] == (t1["start_ms"] + 60_000 <= wm)).all()
    assert t1["closed"].any() and not t1["closed"].all()


def test_last_n_limits_the_windows_returned(populated):
    snap = snapshot(populated.r, populated.layout, DEFAULT_SPECS, 60_000, last_n=3)
    assert snap.windows["t1m"].height == 3


def test_top_tables_are_sorted_descending(populated):
    spec = DEFAULT_SPECS[0]
    start = int(populated.r.zrange(populated.layout.index(spec), 0, 0)[0])
    prod = top_table(populated.r, populated.layout, spec, start, "products", k=5)
    assert prod.height <= 5 and prod["events"].is_sorted(descending=True)
    brands = top_table(populated.r, populated.layout, spec, start, "brands")
    assert brands["revenue"].is_sorted(descending=True) or brands.is_empty()


def test_top_table_of_an_unknown_window_is_empty(broker):
    assert top_table(
        broker.r, broker.layout, DEFAULT_SPECS[0], 123, "products"
    ).is_empty()
    assert top_table(
        broker.r, broker.layout, DEFAULT_SPECS[0], 123, "brands"
    ).is_empty()


def test_throughput_fills_empty_seconds_with_zero():
    tel = pl.DataFrame(
        {"t_ms": [1_000_500, 1_000_900, 1_003_100], "applied": [10, 5, 7]},
    )
    tp = throughput(tel)
    assert tp["t_s"].to_list() == [1000, 1001, 1002, 1003]
    assert tp["per_s"].to_list() == [15, 0, 0, 7]


def test_late_counts_show_up_in_health(broker, events):
    from helpers import messy

    produce(
        broker,
        messy(events, 0.05, stragglers=0.05),
        ProducerConfig(speed=0, backpressure="none"),
    )
    Worker(broker, WorkerConfig(name="w1", lateness_ms=30_000, block_ms=50)).run()
    snap = snapshot(broker.r, broker.layout, DEFAULT_SPECS, 30_000)
    assert snap.health.late["t1m"] > 0


# -- charts -------------------------------------------------------------------------------------
def test_event_chart_stacks_four_types_in_fixed_colours(populated):
    df = snapshot(
        populated.r, populated.layout, DEFAULT_SPECS, HUGE_LATENESS_MS
    ).windows["t1m"]
    fig = charts.events_by_type(df, "t", watermark_ms=df["start_ms"][-1])
    assert [t.name for t in fig.data] == [
        "view",
        "add to cart",
        "remove from cart",
        "purchase",
    ]
    assert [t.marker.color for t in fig.data] == list(charts.TYPE_COLOR.values())
    assert fig.layout.barmode == "stack" and len(fig.layout.shapes) == 1


def test_colour_follows_the_entity_not_the_rank(populated):
    df = snapshot(
        populated.r, populated.layout, DEFAULT_SPECS, HUGE_LATENESS_MS
    ).windows["t1m"]
    full = {t.name: t.marker.color for t in charts.events_by_type(df, "t").data}
    zeroed = df.with_columns(view=pl.lit(0))
    again = {t.name: t.marker.color for t in charts.events_by_type(zeroed, "t").data}
    assert full == again


@pytest.mark.parametrize(
    "build",
    [
        lambda: charts.events_by_type(pl.DataFrame(), "t"),
        lambda: charts.line(pl.DataFrame(), "users", "t", "users"),
        lambda: charts.top_bar(pl.DataFrame(), "a", "b", "t", "b"),
        lambda: charts.throughput_chart(pl.DataFrame()),
        lambda: charts.latency_chart(
            pl.DataFrame(schema={"t_ms": pl.Int64, "applied": pl.Int64})
        ),
        lambda: charts.backlog_chart(pl.DataFrame(schema={"t_ms": pl.Int64})),
    ],
)
def test_every_chart_has_a_designed_empty_state(build):
    fig = build()
    assert isinstance(fig, go.Figure) and len(fig.data) == 0 and fig.layout.annotations


def test_other_charts_draw_data(populated):
    snap = snapshot(populated.r, populated.layout, DEFAULT_SPECS, HUGE_LATENESS_MS)
    assert len(charts.line(snap.windows["t1m"], "users", "t", "users").data) == 1
    assert len(charts.throughput_chart(snap.throughput).data) == 1
    assert len(charts.latency_chart(snap.telemetry).data) == 2
    hist = pl.DataFrame(
        {"t_ms": [1000, 2000, 3000], "undelivered": [5, 3, 0], "pending": [1, 1, 0]}
    )
    assert len(charts.backlog_chart(hist).data) == 2
    top = pl.DataFrame({"brand": ["a", "b"], "revenue": [3.0, 9.0]})
    assert charts.top_bar(top, "brand", "revenue", "t", "$").data[0].y[-1] == "b"
