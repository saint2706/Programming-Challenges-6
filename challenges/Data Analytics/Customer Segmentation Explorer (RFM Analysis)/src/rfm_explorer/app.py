"""Streamlit dashboard: rescore the customer base live and check the segments against what came next.

Run with:  uv run streamlit run src/rfm_explorer/app.py
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import polars as pl
import streamlit as st

from rfm_explorer import charts, data, scoring, validate
from rfm_explorer.pipeline import segment_customers
from rfm_explorer.rfm import last_day

LOOKBACKS = {
    "90 days": 90,
    "180 days": 180,
    "1 year": 365,
    "2 years": 730,
    "all history": None,
}
EDGE_LABELS = {
    "recency_days": "Recency days (score 5 = fastest)",
    "frequency": "Frequency (purchases)",
    "monetary": "Monetary, £",
}


@st.cache_data(show_spinner="Loading and cleaning transactions...")
def load(path: str, mtime: float) -> tuple[pl.DataFrame, pl.DataFrame]:
    return data.clean(data.load_raw(Path(path)))


@st.cache_data(show_spinner="Scoring customers...")
def run_segments(
    path, mtime, snapshot, lookback, monetary, method, k, edges, countries
):
    lines, _ = load(path, mtime)
    return segment_customers(
        lines, snapshot, lookback, monetary, method, k, edges, list(countries)
    )


@st.cache_data(show_spinner="Checking segments against the next period...")
def run_validation(path, mtime, horizon, lookback, monetary, k):
    lines, _ = load(path, mtime)
    snapshot = last_day(lines) + timedelta(days=1) - timedelta(days=horizon)
    return validate.validate(
        lines, snapshot, horizon, lookback, monetary, k, resamples=200
    )


def parse_edges(text: str, name: str) -> list[float] | None:
    try:
        values = [float(x) for x in text.split(",") if x.strip()]
    except ValueError:
        st.sidebar.error(f"{name}: numbers separated by commas, please")
        return None
    if len(values) != 4 or values != sorted(values):
        st.sidebar.error(f"{name}: four ascending numbers, please")
        return None
    return values


def main() -> None:
    st.set_page_config(page_title="RFM Customer Segments", layout="wide")
    st.title("Customer segments (RFM)")
    path = data.find_data_file()
    mtime = path.stat().st_mtime
    lines, ledger = load(str(path), mtime)
    newest = last_day(lines)

    with st.sidebar:
        st.caption(
            f"Data: `{path.name}`, {lines.height:,} cleaned lines up to {newest}"
        )
        snapshot = st.date_input(
            "Snapshot date (only earlier activity counts)",
            value=newest + timedelta(days=1),
            min_value=lines["invoice_date"].min().date() + timedelta(days=1),
            max_value=newest + timedelta(days=1),
        )
        lookback = LOOKBACKS[st.selectbox("Look-back window", list(LOOKBACKS), index=2)]
        monetary = st.radio(
            "Monetary value",
            ["net", "gross"],
            help="net subtracts refunds; gross counts purchases only",
            horizontal=True,
        )
        method = st.radio("Scoring", scoring.METHODS, horizontal=True)
        k = st.slider("Clusters (k-means)", 2, 8, 5) if method == "kmeans" else 5
        edges = None
        if method == "fixed":
            with st.expander("Thresholds (upper edge of scores 1-4)", expanded=True):
                parsed = {
                    name: parse_edges(
                        st.text_input(
                            EDGE_LABELS[name], ", ".join(f"{e:g}" for e in default)
                        ),
                        EDGE_LABELS[name],
                    )
                    for name, default in scoring.DEFAULT_EDGES.items()
                }
                if any(v is None for v in parsed.values()):
                    st.stop()
                edges = parsed
        countries = tuple(
            st.multiselect(
                "Countries (none = all)", sorted(lines["country"].unique().to_list())
            )
        )

    scored = run_segments(
        str(path), mtime, snapshot, lookback, monetary, method, k, edges, countries
    )
    if scored.is_empty():
        st.warning(
            "No customer bought in that window. Widen the look-back or move the snapshot later."
        )
        st.stop()
    summary = scoring.summarize(scored)

    a, b, c = st.columns(3)
    a.metric("Customers", f"{scored.height:,}")
    b.metric("Money in window", f"£{scored['monetary'].sum():,.0f}")
    top = summary.row(0, named=True)
    c.metric(
        f"Best segment: {top['segment']}",
        f"{top['customer_share']:.0%} of customers",
        f"{top['money_share']:.0%} of money",
        delta_color="off",
    )

    tab_seg, tab_map, tab_cust, tab_val, tab_data = st.tabs(
        ["Segments", "RFM map", "Customers", "Does it hold up?", "Data"]
    )
    with tab_seg:
        st.plotly_chart(charts.segment_bars(summary), use_container_width=True)
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.plotly_chart(charts.pareto(scored), use_container_width=True)
    with tab_map:
        if method == "kmeans":
            st.caption(
                "Clusters have no 1 to 5 grid; this grid shows the quintile scores for reference."
            )
        st.plotly_chart(charts.rfm_grid(scored), use_container_width=True)
        st.plotly_chart(charts.fm_scatter(scored), use_container_width=True)
    with tab_cust:
        names = summary["segment"].to_list()
        chosen = st.multiselect("Segments", names, default=names[:1])
        view = (
            scored.filter(pl.col("segment").is_in(chosen)).sort(
                "monetary", descending=True
            )
            if chosen
            else scored.head(0)
        )
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.download_button(
            "Download shown customers (CSV)",
            view.write_csv(),
            "rfm_customers.csv",
            "text/csv",
        )
    with tab_val:
        horizon = st.slider("Days ahead to check", 30, 180, 90, step=30)
        result = run_validation(str(path), mtime, horizon, lookback, monetary, k)
        st.write(
            f"Scored as of **{result.snapshot}** using only earlier data, then looked at the next "
            f"{horizon} days. {result.customers:,} customers; {result.repeat_rate:.0%} bought again."
        )
        which = st.radio(
            "Segmentation",
            scoring.METHODS,
            index=scoring.METHODS.index(method),
            horizontal=True,
            key="val_method",
        )
        st.plotly_chart(
            charts.future_bars(result.segments[which]), use_container_width=True
        )
        st.dataframe(result.segments[which], use_container_width=True, hide_index=True)
        st.subheader("Which ordering finds the future spenders?")
        st.plotly_chart(
            charts.ordering_chart(result.orderings), use_container_width=True
        )
        st.dataframe(result.orderings, use_container_width=True, hide_index=True)
        st.caption(
            "Bars are 95% bootstrap intervals. `vs_rfm_lo`/`hi` is the paired interval of the "
            "difference from quintile R+F+M; if it spans 0 the two are not distinguishable."
        )
    with tab_data:
        st.write("Rows removed before scoring (in this order):")
        st.dataframe(ledger, use_container_width=True, hide_index=True)


main()
