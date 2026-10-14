"""Streamlit app: pick a sale (or describe a property), see its comps on a map and why each was chosen.

Run with ``uv run streamlit run src/comp_finder/app.py``.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date

import numpy as np
import polars as pl
import pydeck as pdk
import streamlit as st

from comp_finder import data, explain, pipeline
from comp_finder.engine import METHODS
from comp_finder.paths import project_root

st.set_page_config(page_title="Comparable Listings Finder", layout="wide")


@st.cache_resource
def load():
    return pipeline.build_engine()


@st.cache_resource
def extended(record_json: str, sold_iso: str):
    engine, _ = load()
    frame = data.subject_frame(
        json.loads(record_json), date.fromisoformat(sold_iso), engine.sales
    )
    return engine.extended(frame)


engine, base = load()
last_sale = engine.sales["sold"].max()

st.title("Comparable Listings Finder")
st.caption(
    f"{engine.sales.height:,} King County sales, {engine.sales['sold'].min()} to {last_sale}. "
    "Comps come only from sales strictly before the subject's sale date; the models were fit "
    f"on the first {engine.n_fit:,} sales."
)

mode = st.sidebar.radio("Subject", ["A sale from the test window", "A custom property"])
method = st.sidebar.selectbox(
    "Similarity", list(METHODS), index=3, format_func=METHODS.get
)
k = st.sidebar.slider("Comps", 3, 12, 5)
bracket = st.sidebar.checkbox(
    "Bracket the subject (a better and a worse comp)", value=True
)

actual = None
if mode.startswith("A sale"):
    sales = engine.sales
    pool = np.flatnonzero(
        (np.arange(sales.height) >= engine.n_fit) & (sales["n_prior"].to_numpy() == 0)
    )
    q = st.sidebar.selectbox(
        "Sale",
        pool.tolist(),
        format_func=lambda i: (
            f"{sales['id'][i]} | {sales['sold'][i]} | ${sales['price'][i]:,.0f} | {sales['zipcode'][i]}"
        ),
    )
    eng = engine
    actual = float(engine.price[q])
else:
    st.sidebar.caption("Defaults sit in a typical Seattle neighbourhood.")
    record = {
        "lat": st.sidebar.number_input(
            "Latitude", 47.15, 47.78, 47.6, 0.001, format="%.4f"
        ),
        "long": st.sidebar.number_input(
            "Longitude", -122.52, -121.3, -122.3, 0.001, format="%.4f"
        ),
        "sqft_living": st.sidebar.number_input(
            "Living area (sq ft)", 400, 8000, 1900, 50
        ),
        "sqft_lot": st.sidebar.number_input("Lot (sq ft)", 600, 100_000, 7500, 100),
        "bedrooms": st.sidebar.number_input("Bedrooms", 1, 10, 3),
        "bathrooms": st.sidebar.number_input("Bathrooms", 0.5, 8.0, 2.0, 0.25),
        "floors": st.sidebar.number_input("Floors", 1.0, 3.5, 1.0, 0.5),
        "grade": st.sidebar.number_input("Build grade (3-13)", 3, 13, 7),
        "condition": st.sidebar.number_input("Condition (1-5)", 1, 5, 3),
        "view": st.sidebar.number_input("View (0-4)", 0, 4, 0),
        "waterfront": int(st.sidebar.checkbox("Waterfront")),
        "yr_built": st.sidebar.number_input("Year built", 1900, 2015, 1975),
    }
    eng = extended(json.dumps(record), str(last_sale))
    q = eng.sales.height - 1

settings = replace(base, k=k, selection="bracket" if bracket else "top")
rows = np.array([q])
comps = eng.rank(method, rows, k * 3 if bracket else k, settings.pool)
est = eng.estimate(rows, comps, settings)
details = explain.explain_comps(eng, method, q, est.comps[0], settings)
avm = float(eng.avm(rows, settings.extrapolate)[0])
estimate = float(est.price[0])

cols = st.columns(4)
cols[0].metric("Comps estimate", f"${estimate:,.0f}")
cols[1].metric("GBM AVM (no comps)", f"${avm:,.0f}")
if actual is not None:
    cols[2].metric(
        "Actual sale price",
        f"${actual:,.0f}",
        f"{estimate / actual - 1:+.1%} comps error",
    )
cols[3].metric("Bracketed", "yes" if est.bracketed[0] else "no")

if details:
    sales = eng.sales
    pts = [
        {
            "name": "Subject",
            "lat": float(sales["lat"][q]),
            "lon": float(sales["long"][q]),
            "color": [220, 60, 60, 220],
            "radius": 90,
            "detail": "subject",
        }
    ] + [
        {
            "name": f"Comp {i}",
            "lat": float(sales["lat"][d["row"]]),
            "lon": float(sales["long"][d["row"]]),
            "color": [40, 110, 200, 200],
            "radius": 70,
            "detail": f"sold {d['sold']} for ${d['price']:,.0f}; adjusted ${d['adjusted_price']:,.0f}",
        }
        for i, d in enumerate(details, 1)
    ]
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=pts,
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius="radius",
        pickable=True,
    )
    view = pdk.ViewState(latitude=pts[0]["lat"], longitude=pts[0]["lon"], zoom=11.5)
    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view,
            tooltip={"text": "{name}\n{detail}"},
        )
    )

    table = pl.DataFrame(
        {
            "Comp": [f"#{i}" for i in range(1, len(details) + 1)],
            "Sold": [d["sold"] for d in details],
            "Sale price": [round(d["price"]) for d in details],
            "km": [round(d["km"], 2) for d in details],
            "Market %": [round(d["time_pct"], 1) for d in details],
            "Attribute %": [round(d["net_attribute_pct"], 1) for d in details],
            "Gross attr. %": [round(d["gross_attribute_pct"], 1) for d in details],
            "Adjusted price": [round(d["adjusted_price"]) for d in details],
        }
    )
    st.dataframe(table, hide_index=True, width="stretch")
    for i, d in enumerate(details, 1):
        with st.expander(f"Comp #{i}: why it was chosen and how it was adjusted"):
            for r in d["reasons"]:
                st.markdown(f"- **{r['label']}**: {r['text']}")
            grid = [a for a in d["attribute_adjustments"] if abs(a["pct"]) >= 0.05]
            if grid:
                st.dataframe(
                    pl.DataFrame(
                        {
                            "Attribute": [a["label"] for a in grid],
                            "Adjustment %": [round(a["pct"], 2) for a in grid],
                        }
                    ),
                    hide_index=True,
                )
            else:
                st.caption(
                    "No attribute adjustment: the comp matches the subject on every attribute."
                )
else:
    st.warning("No earlier sales to compare with.")

summary = project_root() / "results" / "metrics.json"
if summary.exists():
    saved = json.loads(summary.read_text(encoding="utf-8"))
    with st.expander(
        "How accurate is each method? (last `comps evaluate` run, test window)"
    ):
        st.dataframe(
            pl.DataFrame(
                {
                    "Method": [
                        METHODS.get(m, "GBM AVM (no comps)") for m in saved["main"]
                    ],
                    "MdAPE": [f"{v['mdape']:.1%}" for v in saved["main"].values()],
                    "Within 10%": [
                        f"{v['within10']:.1%}" for v in saved["main"].values()
                    ],
                    "Within 20%": [
                        f"{v['within20']:.1%}" for v in saved["main"].values()
                    ],
                }
            ),
            hide_index=True,
        )
