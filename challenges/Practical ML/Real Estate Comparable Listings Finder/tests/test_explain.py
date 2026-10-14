"""Explanations are recomputed from the table and models, so each claim can be checked by hand."""

from __future__ import annotations

import numpy as np
import pytest
from comp_finder import explain, features
from comp_finder.engine import METHODS, Settings
from test_features import haversine_km

S = Settings(k=5)


def explained(engine, method, q):
    comps = engine.rank(method, np.array([q]), 5)[0]
    return explain.explain_comps(engine, method, q, comps, S)


def subject(engine) -> int:
    return engine.sales.height - 11


def test_facts_match_the_raw_table(engine):
    q = subject(engine)
    sales = engine.sales
    for d in explained(engine, "leaf", q):
        c = d["row"]
        assert d["km"] == pytest.approx(
            haversine_km(
                sales["lat"][q], sales["long"][q], sales["lat"][c], sales["long"][c]
            ),
            rel=0.01,
        )
        assert d["days_before"] == (sales["sold"][q] - sales["sold"][c]).days > 0
        assert d["same_zip"] == (sales["zipcode"][q] == sales["zipcode"][c])
        assert d["price"] == sales["price"][c] and d["sale_id"] == sales["id"][c]


def test_every_reason_sentence_is_true_of_the_raw_values(engine):
    q = subject(engine)
    sales = engine.sales
    s = sales.row(q, named=True)
    seen = set()
    for method in METHODS:
        for d in explained(engine, method, q):
            p = sales.row(d["row"], named=True)
            for r in d["reasons"]:
                seen.add(r["group"])
                g, text = r["group"], r["text"]
                if g == "grade":
                    assert text == f"build grade {p['grade']} (subject {s['grade']})"
                elif g == "condition":
                    assert (
                        text == f"condition {p['condition']} (subject {s['condition']})"
                    )
                elif g == "bedrooms":
                    assert (
                        text
                        == f"bedrooms {p['bedrooms']:.0f} (subject {s['bedrooms']:.0f})"
                    )
                elif g == "log_living":
                    pct = (p["sqft_living"] / s["sqft_living"] - 1) * 100
                    assert (
                        text
                        == f"{p['sqft_living']:,} sq ft living area (subject {s['sqft_living']:,}, {pct:+.0f}%)"
                    )
                elif g == "location":
                    assert text.startswith(f"{d['km']:.1f} km away")
                    assert text.endswith(f"both in {s['zipcode']}") == d["same_zip"]
                elif g == "eff_year":
                    year = lambda x: max(x["yr_built"], x["yr_renovated"])
                    assert text == f"built/renovated {year(p)} (subject {year(s)})"
                elif g == "has_basement":
                    both, neither = (
                        s["sqft_basement"] > 0 and p["sqft_basement"] > 0,
                        not (s["sqft_basement"] > 0 or p["sqft_basement"] > 0),
                    )
                    assert text == (
                        "both have a basement"
                        if both
                        else "neither has a basement"
                        if neither
                        else "basement differs"
                    )
    assert {"location", "log_living", "grade"} <= seen


def test_euclid_similarity_splits_exactly_over_features(engine):
    q = subject(engine)
    for method in ("knn", "ridge", "hedonic"):
        for d in explained(engine, method, q):
            assert sum(d["contributions"].values()) == pytest.approx(d["similarity"])
            assert list(d["contributions"]) == features.COLUMNS


def test_leaf_attribution_splits_the_proximity(engine):
    for d in explained(engine, "leaf", subject(engine)):
        assert sum(d["contributions"].values()) == pytest.approx(
            d["similarity"], rel=1e-6
        )


def test_neutralising_a_feature_removes_exactly_its_contribution(engine):
    """Faithfulness: copy the subject's value into the comp and the distance falls by that share."""
    q = subject(engine)
    metric = engine.metrics["hedonic"]
    for d in explained(engine, "hedonic", q):
        c = d["row"]
        for f in ("grade", "log_living", "east_km"):
            j = features.COLUMNS.index(f)
            z = engine.z.copy()
            z[c, j] = z[q, j]
            from comp_finder.similarity import WeightedEuclid

            after = WeightedEuclid(z, metric.weights)
            drop = (
                -metric.sim(np.array([q]), np.array([c]))[0, 0]
                + after.sim(np.array([q]), np.array([c]))[0, 0]
            )
            assert drop == pytest.approx(d["contributions"][f])


def test_leaf_explanation_credits_the_real_drivers_not_the_noise(engine):
    """Synthetic price depends on size, grade, view and location only."""
    share = {f: 0.0 for f in features.COLUMNS}
    for q in np.linspace(engine.n_fit, engine.sales.height - 1, 25).astype(int):
        for d in explained(engine, "leaf", q):
            total = sum(d["contributions"].values())
            for f, v in d["contributions"].items():
                share[f] += v / total
    norm = sum(share.values())
    share = {f: v / norm for f, v in share.items()}
    drivers = (
        share["log_living"]
        + share["grade"]
        + share["east_km"]
        + share["north_km"]
        + share["view"]
    )
    noise = (
        share["condition"]
        + share["floors"]
        + share["has_basement"]
        + share["bedrooms"]
        + share["waterfront"]
    )
    assert drivers > 0.75
    assert noise < 0.08


def test_reasons_beat_the_typical_pair_and_are_sorted(engine):
    for method in METHODS:
        for d in explained(engine, method, subject(engine)):
            scores = [r["beyond_typical"] for r in d["reasons"]]
            assert all(v > 0 for v in scores)
            assert scores == sorted(scores, reverse=True)
            assert len(d["reasons"]) <= 3


def test_the_price_grid_adds_up_to_the_adjusted_price(engine):
    q = subject(engine)
    for d in explained(engine, "hedonic", q):
        line_items = sum(a["log_adjustment"] for a in d["attribute_adjustments"])
        assert np.expm1(line_items) * 100 == pytest.approx(d["net_attribute_pct"])
        assert d["gross_attribute_pct"] == pytest.approx(
            100 * sum(abs(a["log_adjustment"]) for a in d["attribute_adjustments"])
        )
        market = (
            engine.hedonic.index(engine.month[[q]], S.extrapolate)[0]
            - engine.hedonic.index(engine.month[[d["row"]]], S.extrapolate)[0]
        )
        assert d["adjusted_price"] == pytest.approx(
            d["price"] * np.exp(market + line_items), rel=1e-9
        )


def test_short_comp_lists_are_explained_without_padding(engine):
    out = explain.explain_comps(
        engine, "knn", subject(engine), np.array([1, 2, -1, -1, -1]), S
    )
    assert [d["row"] for d in out] == [1, 2]
    assert explain.explain_comps(engine, "knn", 0, np.array([-1, -1]), S) == []
