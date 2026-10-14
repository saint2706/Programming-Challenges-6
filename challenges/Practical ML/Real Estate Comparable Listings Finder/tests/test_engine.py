from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from comp_finder import features
from comp_finder.engine import Engine, Settings

pytestmark = pytest.mark.filterwarnings("ignore")


def rows(engine: Engine, n: int = 40) -> np.ndarray:
    return np.linspace(engine.n_fit, engine.sales.height - 1, n).astype(int)


def test_settings_are_validated():
    for bad in (
        {"k": 0},
        {"combine": "mode"},
        {"adjust": "magic"},
        {"pool": "all"},
        {"selection": "best"},
    ):
        with pytest.raises(ValueError):
            Settings(**bad)


def test_engine_rejects_unsorted_sales_and_tiny_fit_windows(sales):
    with pytest.raises(ValueError, match="sorted"):
        Engine(sales.reverse(), 500)
    with pytest.raises(ValueError, match="fit rows"):
        Engine(sales, 10)


def test_the_adjusted_price_is_reproduced_from_first_principles(engine):
    """price * market factor * per-attribute factor, computed here without the engine's helpers."""
    s = Settings(k=3, extrapolate="flat", combine="median", adjust="full")
    r = rows(engine, 5)
    est = engine.predict("knn", r, s)
    h = engine.hedonic
    for i, subject in enumerate(r):
        for j in range(3):
            c = est.comps[i, j]
            market = np.interp(
                engine.month[subject], h.months, h.month_effect
            ) - np.interp(engine.month[c], h.months, h.month_effect)
            attrs = sum(
                h.beta[features.COLUMNS.index(f)]
                * (
                    engine.z[subject, features.COLUMNS.index(f)]
                    - engine.z[c, features.COLUMNS.index(f)]
                )
                for f in features.ATTRIBUTES
            )
            assert est.adjusted[i, j] == pytest.approx(
                engine.price[c] * np.exp(market + attrs)
            )
        assert est.price[i] == pytest.approx(np.median(est.adjusted[i]))


def test_adjust_none_is_the_median_of_raw_comp_prices(engine):
    r = rows(engine, 10)
    est = engine.predict("leaf", r, Settings(adjust="none"))
    for i in range(len(r)):
        assert est.price[i] == pytest.approx(np.median(engine.price[est.comps[i]]))


def test_mean_is_a_geometric_mean(engine):
    r = rows(engine, 5)
    est = engine.predict("knn", r, Settings(combine="mean"))
    assert est.price == pytest.approx(np.exp(np.log(est.adjusted).mean(axis=1)))


def test_gross_adjustment_is_the_sum_of_absolute_line_items(engine):
    est = engine.predict("hedonic", rows(engine, 10), Settings())
    assert np.allclose(est.gross, np.abs(est.attr_log).sum(axis=2))
    assert np.allclose(est.net, est.attr_log.sum(axis=2))
    assert (est.gross >= np.abs(est.net) - 1e-12).all()


def test_pool_fit_keeps_comps_inside_the_fit_window(engine):
    comps = engine.rank("knn", rows(engine), 5, pool="fit")
    assert (comps[comps >= 0] < engine.n_fit).all()
    wide = engine.rank("knn", rows(engine), 5, pool="prior")
    assert (wide >= engine.n_fit).any()  # the default pool grows with time


def test_rank_returns_best_first_and_unique_comps(engine):
    r = rows(engine, 30)
    comps = engine.rank("leaf", r, 10)
    sim = engine.metrics["leaf"]
    for q, row in zip(r, comps, strict=True):
        assert len(set(row.tolist())) == 10
        s = sim.sim(np.array([q]), row)[0]
        assert (np.diff(s) <= 1e-12).all()


def test_rank_matches_a_brute_force_search(engine):
    q = int(engine.sales.height - 3)
    pool = np.flatnonzero(
        (engine.sold_days < engine.sold_days[q]) & (engine.ids != engine.ids[q])
    )
    d = ((engine.z[pool] - engine.z[q]) ** 2).sum(axis=1)
    assert (
        engine.rank("knn", np.array([q]), 5)[0].tolist()
        == pool[np.argsort(d)[:5]].tolist()
    )


def test_the_earliest_subjects_get_nan_not_a_crash(engine):
    est = engine.predict("knn", np.array([0]), Settings())
    assert np.isnan(est.price[0])
    assert (est.comps == -1).all()


def test_bracket_selection_brackets_more_often_and_keeps_k_unique_comps(engine):
    r = rows(engine, 120)
    top = engine.predict("knn", r, Settings(k=5))
    br = engine.predict("knn", r, Settings(k=5, selection="bracket"))
    assert br.comps.shape == top.comps.shape == (120, 5)
    assert br.bracketed.mean() >= top.bracketed.mean()
    assert all(len(set(row[row >= 0].tolist())) == (row >= 0).sum() for row in br.comps)
    keep = ~top.bracketed
    # subjects that were already bracketed keep their top-k exactly
    assert np.array_equal(br.comps[top.bracketed], top.comps[top.bracketed])
    assert keep.any()


def test_bracketed_means_a_better_and_a_worse_comp(engine):
    est = engine.predict("knn", rows(engine, 60), Settings())
    for i in range(60):
        has_both = (est.net[i] > 0).any() and (est.net[i] < 0).any()
        assert bool(est.bracketed[i]) == bool(has_both)


def test_comps_beat_a_naive_guess_on_synthetic_data(engine):
    r = rows(engine, 200)
    actual = engine.price[r]
    naive = np.median(engine.price[: engine.n_fit])
    err_naive = np.median(np.abs(naive - actual) / actual)
    for method in ("knn", "hedonic", "leaf"):
        est = engine.predict(method, r, Settings())
        assert np.median(np.abs(est.price - actual) / actual) < 0.5 * err_naive


def test_avm_is_the_gbm_rolled_to_the_sale_month(engine):
    r = rows(engine, 10)
    expected = np.exp(
        engine.gbm.predict(engine.z[r]) + engine.hedonic.index(engine.month[r], "trend")
    )
    assert engine.avm(r, "trend") == pytest.approx(expected)


def test_extended_engine_reuses_the_fitted_models_and_adds_the_subject(engine):
    from datetime import timedelta

    from comp_finder import data

    rec = {
        "lat": 47.6,
        "long": -122.3,
        "sqft_living": 1800,
        "sqft_lot": 5000,
        "bedrooms": 3,
        "bathrooms": 2,
        "floors": 1,
        "grade": 8,
        "condition": 3,
    }
    last = engine.sales["sold"].max()
    ext = engine.extended(
        data.subject_frame(rec, last + timedelta(days=1), engine.sales)
    )
    assert ext.sales.height == engine.sales.height + 1
    assert ext.hedonic is engine.hedonic and ext.gbm is engine.gbm
    q = ext.sales.height - 1
    est = ext.predict("leaf", np.array([q]), Settings())
    assert (est.comps >= 0).all() and np.isfinite(est.price[0])
    assert (ext.sold_days[est.comps[0]] < ext.sold_days[q]).all()
    assert ext.before[q] == engine.sales.height  # the whole history is available


def test_real_data_comps_respect_the_date_rule(real_sales):
    from comp_finder import data

    n_fit, n_val, _ = data.window_counts(real_sales, data.Windows())
    eng = Engine(real_sales, n_fit + n_val)
    r = np.linspace(n_fit + n_val, real_sales.height - 1, 200).astype(int)
    for method in ("knn", "leaf"):
        comps = eng.rank(method, r, 5)
        for q, row in zip(r, comps, strict=True):
            got = row[row >= 0]
            assert len(got) == 5 and (eng.sold_days[got] < eng.sold_days[q]).all()
            assert (eng.ids[got] != eng.ids[q]).all()
    assert pl.Series(eng.price).min() > 0
