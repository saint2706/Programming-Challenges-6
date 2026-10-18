"""The live dashboard: ``uv run streamlit run src/stream_dashboard/app.py``.

It only reads Redis. Start the pipeline with ``uv run stream-dash demo`` (or the ``produce`` and
``worker`` commands) and the charts fill in; refresh interval and window choice are in the sidebar.
"""

from __future__ import annotations

import os
import time

import polars as pl
import redis
import streamlit as st

from stream_dashboard import charts
from stream_dashboard.snapshot import snapshot, top_table
from stream_dashboard.windows import (
    DEFAULT_LATENESS_MS,
    DEFAULT_SPECS,
    Layout,
    WindowSpec,
)

st.set_page_config(page_title="Real-Time Event Stream Dashboard", layout="wide")

DEFAULT_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
HISTORY_KEEP = 300


@st.cache_resource
def _client(url: str) -> redis.Redis:
    return redis.Redis.from_url(url, decode_responses=True)


def _fmt_clock(ms: int | None) -> str:
    return "none yet" if ms is None else time.strftime("%H:%M", time.gmtime(ms / 1000))


def _sample(history: list[dict], h) -> pl.DataFrame:
    """Keep a short in-session history of backlog; Redis only knows the current value."""
    now = int(time.time() * 1000)
    if not history or now - history[-1]["t_ms"] >= 500:
        history.append(
            {"t_ms": now, "undelivered": h.undelivered, "pending": h.pending}
        )
        del history[:-HISTORY_KEEP]
    return pl.DataFrame(
        history, schema={"t_ms": pl.Int64, "undelivered": pl.Int64, "pending": pl.Int64}
    )


def main() -> None:
    st.title("Real-Time Event Stream Dashboard")
    with st.sidebar:
        st.header("Source")
        url = st.text_input("Redis URL", DEFAULT_URL)
        prefix = st.text_input("Key prefix", "sd:")
        st.header("Windows")
        spec_text = st.text_input(
            "Window specs",
            ", ".join(w.name.replace("-", "/")[1:] for w in DEFAULT_SPECS),
            help="Comma separated. `1m` is tumbling, `5m/1m` is a 5 minute window hopping every minute.",
        )
        lateness_s = st.number_input(
            "Allowed lateness (s)", 0, 3600, DEFAULT_LATENESS_MS // 1000
        )
        last_n = st.slider("Windows shown", 10, 200, 60)
        refresh = st.select_slider("Refresh every (s)", options=[1, 2, 5, 10], value=2)
        paused = st.toggle("Pause", value=False)
    try:
        specs = tuple(WindowSpec.parse(t) for t in spec_text.split(",") if t.strip())
    except ValueError as exc:
        st.error(str(exc))
        return
    if not specs:
        st.error("Give at least one window spec.")
        return
    layout = Layout(prefix=prefix)
    client = _client(url)
    history = st.session_state.setdefault("backlog_history", [])

    @st.fragment(run_every=None if paused else refresh)
    def live() -> None:
        try:
            snap = snapshot(client, layout, specs, lateness_s * 1000, last_n)
        except redis.RedisError as exc:
            st.error(f"Cannot read Redis at {url}: {exc}")
            return
        h = snap.health
        if h.stream_len == 0 and h.max_event_ms is None:
            st.info(
                "The stream is empty. Run `uv run stream-dash demo` in another terminal."
            )
        cols = st.columns(6)
        state = "done" if h.producer_done else "replaying"
        of = f"of {h.total:,}, {state}" if h.total else state
        cols[0].metric("Produced", f"{h.sent:,}", of, delta_color="off")
        last_5s = snap.telemetry.filter(pl.col("t_ms") > snap.taken_ms - 5_000)
        recent = last_5s["applied"].sum() / 5
        cols[1].metric("Events/s (last 5 s)", f"{recent:,.0f}")
        cols[2].metric(
            "Backlog", f"{h.backlog:,}", f"{h.pending:,} unacked", delta_color="off"
        )
        p95 = (
            snap.telemetry.tail(20)["lat_p95"].mean()
            if not snap.telemetry.is_empty()
            else 0
        )
        cols[3].metric("Latency p95", f"{p95 or 0:,.0f} ms")
        cols[4].metric("Watermark (UTC)", _fmt_clock(h.watermark_ms))
        # An event late for a sliding spec is late for the tumbling one too, so the largest is the
        # number of events that found no open window anywhere.
        late = max(h.late.values(), default=0)
        share = f"{late / h.sent:.2%} of events" if h.sent else "none yet"
        cols[5].metric("Late (dropped)", f"{late:,}", share, delta_color="off")

        tab_win, tab_top, tab_health = st.tabs(
            ["Windows", "Top products and brands", "Pipeline health"]
        )
        with tab_win:
            for spec in specs:
                df = snap.windows[spec.name]
                label = f"{spec.kind} {spec.name[1:].replace('-', ' every ')}"
                st.plotly_chart(
                    charts.events_by_type(
                        df, f"Events per {label} window", h.watermark_ms
                    ),
                    width="stretch",
                    key=f"ev-{spec.name}",
                )
                c1, c2 = st.columns(2)
                c1.plotly_chart(
                    charts.line(
                        df, "users", f"Unique users per {label} window", "users"
                    ),
                    width="stretch",
                    key=f"us-{spec.name}",
                )
                c2.plotly_chart(
                    charts.line(
                        df, "revenue", f"Revenue per {label} window", "revenue ($)"
                    ),
                    width="stretch",
                    key=f"rv-{spec.name}",
                )
        with tab_top:
            spec = specs[-1]
            df = snap.windows[spec.name]
            if df.is_empty():
                st.info("No window yet.")
            else:
                closed = df.filter(pl.col("closed"))
                pick = closed if not closed.is_empty() else df
                start = int(pick["start_ms"][-1])
                st.caption(
                    f"Latest {'closed ' if not closed.is_empty() else ''}{spec.name} window starting {_fmt_clock(start)}"
                )
                c1, c2 = st.columns(2)
                c1.plotly_chart(
                    charts.top_bar(
                        top_table(client, layout, spec, start, "products"),
                        "product_id",
                        "events",
                        "Top products by events",
                        "events",
                    ),
                    width="stretch",
                    key="top-p",
                )
                c2.plotly_chart(
                    charts.top_bar(
                        top_table(client, layout, spec, start, "brands"),
                        "brand",
                        "revenue",
                        "Top brands by revenue ($)",
                        "revenue ($)",
                    ),
                    width="stretch",
                    key="top-b",
                )
        with tab_health:
            hist = _sample(history, h)
            c1, c2 = st.columns(2)
            c1.plotly_chart(
                charts.backlog_chart(hist), width="stretch", key="h-backlog"
            )
            c2.plotly_chart(
                charts.throughput_chart(snap.throughput),
                width="stretch",
                key="h-tp",
            )
            c3, c4 = st.columns(2)
            c3.plotly_chart(
                charts.latency_chart(snap.telemetry),
                width="stretch",
                key="h-lat",
            )
            with c4:
                st.subheader("Consumers")
                st.dataframe(snap.consumers, hide_index=True, width="stretch")
                st.caption(
                    f"Producer: {h.speed:g}x replay, backpressure `{h.backpressure or 'n/a'}`, "
                    f"blocked {h.blocked_ms / 1000:,.1f} s. "
                    "A worker that stops acknowledging shows up as pending with a growing idle time; "
                    "another worker reclaims it after the idle threshold."
                )
            st.caption(
                f"Reclaimed after a worker stalled: {h.reclaimed:,} entries. "
                f"Redelivered but already applied (skipped): {h.duplicates:,}."
            )

    live()


main()
