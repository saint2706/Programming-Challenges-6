"""Plotly figures shared by the dashboard and the command line."""

from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl

from rfm_explorer.scoring import SEGMENT_GRID, SEGMENT_ORDER, fm_score

PALETTE = px.colors.qualitative.Safe


def _order(names: list[str]) -> list[str]:
    known = [s for s in SEGMENT_ORDER if s in names]
    return known + sorted(n for n in names if n not in known)


def segment_bars(summary: pl.DataFrame) -> go.Figure:
    """Share of customers next to share of money, per segment: the gap is the point."""
    order = _order(summary["segment"].to_list())[::-1]
    by = {r["segment"]: r for r in summary.iter_rows(named=True)}
    fig = go.Figure()
    for key, label, colour in (
        ("customer_share", "share of customers", PALETTE[0]),
        ("money_share", "share of money", PALETTE[1]),
    ):
        fig.add_bar(
            y=order,
            x=[by[s][key] for s in order],
            name=label,
            orientation="h",
            marker_color=colour,
            hovertemplate="%{y}: %{x:.1%}<extra>" + label + "</extra>",
        )
    fig.update_layout(
        barmode="group",
        xaxis_tickformat=".0%",
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        legend_orientation="h",
    )
    return fig


def rfm_grid(scored: pl.DataFrame) -> go.Figure:
    """Customers per (recency score, frequency/monetary score) cell, labelled with the segment."""
    fm = fm_score(scored["f"].to_numpy(), scored["m"].to_numpy())
    counts = np.zeros(
        (5, 5), dtype=int
    )  # row 0 is R=5 so the best customers sit top right
    for r, c in zip(scored["r"].to_numpy(), fm):
        counts[5 - r, c - 1] += 1
    labels = [[SEGMENT_GRID[(5 - i, j + 1)] for j in range(5)] for i in range(5)]
    text = [[f"{labels[i][j]}<br>{counts[i, j]:,}" for j in range(5)] for i in range(5)]
    fig = go.Figure(
        go.Heatmap(
            z=counts,
            x=[1, 2, 3, 4, 5],
            y=[5, 4, 3, 2, 1],
            text=text,
            texttemplate="%{text}",
            # a narrow light range so one dark label colour reads on every cell, in either theme
            colorscale=[[0, "#eaf2fb"], [1, "#6fa8dc"]],
            textfont={"color": "#0b1f33"},
            showscale=False,
            hovertemplate="R=%{y}, FM=%{x}: %{z:,} customers<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="frequency / monetary score (higher = more)",
        yaxis_title="recency score (5 = bought most recently)",
        yaxis={
            "type": "category",
            "categoryorder": "array",
            "categoryarray": [1, 2, 3, 4, 5],
        },
        xaxis={"type": "category"},
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
    )
    return fig


def fm_scatter(scored: pl.DataFrame, limit: int = 4000, seed: int = 0) -> go.Figure:
    """Frequency against money on log axes, one dot per customer, coloured by segment."""
    data = scored.filter(pl.col("monetary") > 0)
    if data.height > limit:
        data = data.sample(limit, seed=seed)
    fig = px.scatter(
        data.to_pandas(),
        x="frequency",
        y="monetary",
        color="segment",
        category_orders={"segment": _order(data["segment"].unique().to_list())},
        color_discrete_sequence=px.colors.qualitative.Bold,
        log_x=True,
        log_y=True,
        hover_data=["customer_id", "recency_days"],
        opacity=0.7,
    )
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        xaxis_title="purchases (log)",
        yaxis_title="net spend, £ (log)",
    )
    return fig


def pareto(scored: pl.DataFrame) -> go.Figure:
    """Cumulative share of money against share of customers, best customers first."""
    money = np.sort(np.clip(scored["monetary"].to_numpy(), 0, None))[::-1]
    x = np.arange(1, len(money) + 1) / len(money)
    y = np.cumsum(money) / money.sum()
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines", name="customers"))
    fig.add_shape(
        type="line", x0=0, y0=0, x1=1, y1=1, line={"dash": "dot", "color": "grey"}
    )
    fig.update_layout(
        xaxis_title="share of customers, best first",
        yaxis_title="share of money",
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        showlegend=False,
    )
    return fig


def future_bars(table: pl.DataFrame) -> go.Figure:
    """Per segment, how many customers bought again and how much they spent per head."""
    order = _order(table["segment"].to_list())
    by = {r["segment"]: r for r in table.iter_rows(named=True)}
    fig = go.Figure()
    fig.add_bar(
        x=order,
        y=[by[s]["repeat_rate"] for s in order],
        name="bought again",
        marker_color=PALETTE[0],
        yaxis="y",
    )
    fig.add_scatter(
        x=order,
        y=[by[s]["mean_future_spend"] for s in order],
        name="mean spend per customer (£)",
        mode="markers",
        marker={"size": 12, "color": PALETTE[1], "symbol": "diamond"},
        yaxis="y2",
    )
    fig.update_layout(
        yaxis={"title": "bought again", "tickformat": ".0%", "rangemode": "tozero"},
        yaxis2={
            "title": "£ per customer",
            "overlaying": "y",
            "side": "right",
            "rangemode": "tozero",
        },
        legend_orientation="h",
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
    )
    return fig


def ordering_chart(orderings: pl.DataFrame, fraction: float = 0.2) -> go.Figure:
    """Share of future spend earned by the top ``fraction`` of customers, with 95% bootstrap bars."""
    o = orderings.sort("top_share")
    fig = go.Figure(
        go.Bar(
            y=o["ordering"],
            x=o["top_share"],
            orientation="h",
            error_x={
                "type": "data",
                "symmetric": False,
                "array": (o["top_share_hi"] - o["top_share"]).to_list(),
                "arrayminus": (o["top_share"] - o["top_share_lo"]).to_list(),
            },
            marker_color=PALETTE[0],
        )
    )
    fig.add_vline(x=fraction, line_dash="dot", annotation_text="random")
    fig.update_layout(
        xaxis_tickformat=".0%",
        xaxis_title=f"share of future spend from the top {fraction:.0%}",
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
    )
    return fig
