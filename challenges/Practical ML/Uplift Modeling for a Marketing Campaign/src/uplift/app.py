"""Streamlit dashboard: who to contact, and what it is worth.

Run:  uv run streamlit run src/uplift/app.py
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from uplift import metrics, pipeline, policy

st.set_page_config(page_title="Uplift Modeling", layout="wide")


@st.cache_resource(show_spinner="Loading the benchmark...")
def get_state():
    art = pipeline.load_artifacts(pipeline.RESULTS_DIR)
    return art, {o: pipeline.ranked_scores(art, o) for o in art.scores}


art, ranked_by_outcome = get_state()

st.title("Uplift Modeling for a Marketing Campaign")
st.caption(
    "Criteo Uplift v2.1 (a randomized ad campaign). A learner scores each customer by the "
    "*extra* chance they act because they were contacted; the curve shows what contacting "
    "the top share is worth, against contacting at random. Treatment is not perfectly random "
    "here, so every estimate is inverse-propensity weighted. Intervals are a bootstrap on the "
    "held-out test split."
)

with st.sidebar:
    outcome = st.selectbox("Outcome", list(art.scores), key="outcome")
    names = [n for n in ranked_by_outcome[outcome] if n != "random"]
    learner = st.selectbox("Learner", names, key="learner")
    budget = st.slider(
        "Contact the top share of customers (%)", 1, 100, 20, key="budget"
    )
    value = st.number_input(
        "Value of one incremental outcome", min_value=0.0, value=0.0, key="value"
    )
    cost = st.number_input("Cost of one contact", min_value=0.0, value=0.0, key="cost")

ranked = ranked_by_outcome[outcome]
r = ranked[learner]
summary = art.report["outcomes"][outcome]
n = len(r.t)
frac = budget / 100
ate = metrics.ate(r)

boot = metrics.bootstrap(
    {learner: r},
    n_boot=100,
    seed=0,
    stat=lambda rr, w=None: {"inc": policy.incremental_at(rr, frac, w)},
)
inc = metrics.interval(boot, learner, "inc")

left, mid, right = st.columns(3)
left.metric(
    "Incremental outcomes per 1,000 customers",
    f"{1000 * inc['est']:.2f}",
    delta=f"{1000 * (inc['est'] - frac * ate):+.2f} vs random targeting",
)
left.caption(f"95% CI [{1000 * inc['lo']:.2f}, {1000 * inc['hi']:.2f}]")
mid.metric("Random targeting, same share", f"{1000 * frac * ate:.2f}")
right.metric("Treat everyone", f"{1000 * ate:.2f}")
raw = summary["ate_unadjusted"]
right.caption(
    f"unadjusted difference in means: {1000 * raw['est']:.2f} (not propensity-corrected)"
)

if value > 0:
    be = policy.break_even_cost(r, frac, value)
    best, prof = policy.best_fraction(r, value, cost, n)
    st.markdown(
        f"At value **{value:g}** per incremental outcome, contacting this share breaks even "
        f"at a cost of **{be:.4f}** per contact. **Best share** at your cost of {cost:g}: "
        f"**{best:.0%}** (profit {prof:,.1f} per {n:,} customers)."
    )

grid = np.linspace(0, 1, 201)
rows = []
for name, rk in ranked.items():
    x, _, u = metrics.curve(rk)
    rows += [
        {"share": s, "per 1,000": 1000 * v, "scorer": name}
        for s, v in zip(grid, np.interp(grid, x, u), strict=True)
    ]
rows += [
    {"share": s, "per 1,000": 1000 * s * ate, "scorer": "random line"} for s in grid
]
curves = (
    alt.Chart(pd.DataFrame(rows))
    .mark_line()
    .encode(
        x=alt.X("share:Q", title="share of customers contacted"),
        y=alt.Y("per 1,000:Q", title="incremental outcomes per 1,000 customers"),
        color="scorer:N",
    )
)
st.subheader("Uplift curves")
st.altair_chart(curves, width="stretch")

st.subheader("Benchmark (test split)")


def _fmt(c, spec=".5f"):
    if c is None:
        return "-"
    return f"{c['est']:{spec}} [{c['lo']:{spec}}, {c['hi']:{spec}}]"


table = pd.DataFrame(
    {
        name: {
            "Qini": _fmt(e["qini"]),
            "AUUC": _fmt(e["auuc"]),
            "Qini minus T": _fmt(e["qini_vs_T"], "+.5f"),
        }
        for name, e in summary["learners"].items()
    }
).T
st.dataframe(table, width="stretch")

st.subheader(f"Decile calibration: {learner}")
st.caption(
    "Customers sorted by predicted uplift into ten equal groups; each bar is the "
    "*observed* uplift (treated minus control rate) with a 95% CI."
)
cal = pd.DataFrame(summary["learners"][learner]["calibration"])
cal["lo"] = cal["observed"] - 1.96 * cal["se"]
cal["hi"] = cal["observed"] + 1.96 * cal["se"]
bars = (
    alt.Chart(cal)
    .mark_bar()
    .encode(
        x=alt.X("decile:O", title="predicted-uplift decile (1 = highest)"),
        y=alt.Y("observed:Q", title="observed uplift"),
    )
)
err = alt.Chart(cal).mark_errorbar().encode(x="decile:O", y="lo:Q", y2="hi:Q")
st.altair_chart(bars + err, width="stretch")
