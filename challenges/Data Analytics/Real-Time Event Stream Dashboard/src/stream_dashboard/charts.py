"""Plotly figures for the dashboard. Pure functions of polars frames; no Redis, no Streamlit.

Colour follows the entity: each event type keeps its own categorical slot whatever else is shown.
One y-axis per chart (throughput and latency are two charts, not one dual-axis chart). Surfaces are
transparent so Streamlit's light and dark themes show through.
"""

from __future__ import annotations

from datetime import UTC, datetime

import plotly.graph_objects as go
import polars as pl

TYPE_COLOR = {
    "view": "#2a78d6",
    "cart": "#eb6834",
    "remove_from_cart": "#1baf7a",
    "purchase": "#eda100",
}
TYPE_LABEL = {
    "view": "view",
    "cart": "add to cart",
    "remove_from_cart": "remove from cart",
    "purchase": "purchase",
}
SERIES = "#2a78d6"
ALERT = "#e34948"
GRID = "rgba(128,128,128,0.25)"


def _base(title: str, height: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title={"text": title, "x": 0, "font": {"size": 15}},
        height=height,
        margin={"l": 8, "r": 8, "t": 44, "b": 64},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "yanchor": "top", "y": -0.22, "x": 0},
        hovermode="x unified",
        bargap=0.15,
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def _empty(fig: go.Figure, text: str) -> go.Figure:
    fig.add_annotation(text=text, showarrow=False, font={"size": 14}, opacity=0.7)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def events_by_type(
    df: pl.DataFrame, title: str, watermark_ms: int | None = None
) -> go.Figure:
    fig = _base(title)
    if df.is_empty():
        return _empty(fig, "waiting for the first window")
    for t, color in TYPE_COLOR.items():
        fig.add_bar(
            x=df["start"],
            y=df[t],
            name=TYPE_LABEL[t],
            marker={
                "color": color,
                "line": {"width": 1, "color": "rgba(128,128,128,0.0)"},
            },
        )
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="events in window")
    if watermark_ms is not None:
        fig.add_vline(
            x=datetime.fromtimestamp(watermark_ms / 1000, tz=UTC),
            line={"width": 1, "dash": "dot", "color": ALERT},
            annotation_text="watermark",
            annotation_position="top left",
        )
    return fig


def line(
    df: pl.DataFrame, y: str, title: str, ytitle: str, x: str = "start"
) -> go.Figure:
    fig = _base(title, height=260)
    if df.is_empty():
        return _empty(fig, "waiting for data")
    fig.add_scatter(
        x=df[x], y=df[y], mode="lines", line={"width": 2, "color": SERIES}, name=ytitle
    )
    fig.update_yaxes(title=ytitle, rangemode="tozero")
    fig.update_layout(showlegend=False)
    return fig


def top_bar(
    df: pl.DataFrame, label: str, value: str, title: str, vtitle: str
) -> go.Figure:
    fig = _base(title, height=320)
    if df.is_empty():
        return _empty(fig, "no data in this window yet")
    d = df.sort(value)
    fig.add_bar(
        y=d[label],
        x=d[value],
        orientation="h",
        marker={"color": SERIES},
        text=d[value].round(0).cast(pl.Int64),
        textposition="outside",
        cliponaxis=False,
        name=vtitle,
    )
    fig.update_yaxes(type="category", gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(title=vtitle, showgrid=True, gridcolor=GRID)
    fig.update_layout(showlegend=False, hovermode="closest")
    return fig


def throughput_chart(tp: pl.DataFrame) -> go.Figure:
    fig = _base("Throughput: events applied per second (wall clock)", height=260)
    if tp.is_empty():
        return _empty(fig, "no worker has reported yet")
    t = pl.from_epoch(tp["t_s"], time_unit="s")
    fig.add_scatter(
        x=t,
        y=tp["per_s"],
        mode="lines",
        line={"width": 2, "color": SERIES, "shape": "hv"},
    )
    fig.update_yaxes(title="events/s", rangemode="tozero")
    fig.update_layout(showlegend=False)
    return fig


def latency_chart(tel: pl.DataFrame) -> go.Figure:
    fig = _base("End-to-end latency: producer XADD to applied (ms)", height=260)
    d = tel.filter(pl.col("applied") > 0)
    if d.is_empty():
        return _empty(fig, "no worker has reported yet")
    t = pl.from_epoch(d["t_ms"], time_unit="ms")
    fig.add_scatter(
        x=t,
        y=d["lat_p50"],
        mode="lines",
        name="p50",
        line={"width": 2, "color": SERIES},
    )
    fig.add_scatter(
        x=t,
        y=d["lat_p95"],
        mode="lines",
        name="p95",
        line={"width": 2, "color": "#eb6834"},
    )
    fig.update_yaxes(title="ms", rangemode="tozero")
    return fig


def backlog_chart(history: pl.DataFrame) -> go.Figure:
    """``history``: t_ms, undelivered, pending (sampled by the dashboard on every refresh)."""
    fig = _base("Consumer backlog: waiting + read but unacknowledged", height=260)
    if history.height < 2:
        return _empty(fig, "collecting samples")
    t = pl.from_epoch(history["t_ms"], time_unit="ms")
    fig.add_scatter(
        x=t,
        y=history["undelivered"],
        mode="lines",
        name="undelivered (lag)",
        line={"width": 2, "color": SERIES},
        stackgroup="b",
    )
    fig.add_scatter(
        x=t,
        y=history["pending"],
        mode="lines",
        name="pending (unacked)",
        line={"width": 2, "color": "#eb6834"},
        stackgroup="b",
    )
    fig.update_yaxes(title="entries", rangemode="tozero")
    return fig
