"""The comps rule and the fit-window rule, each with a canary that proves the test can fail."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from comp_finder.engine import METHODS, Engine, Settings

pytestmark = pytest.mark.filterwarnings("ignore")


def subjects(engine: Engine, n: int = 150) -> np.ndarray:
    return np.linspace(engine.n_fit, engine.sales.height - 1, n).astype(int)


@pytest.mark.parametrize("method", list(METHODS))
@pytest.mark.parametrize("pool", ["prior", "fit"])
def test_every_comp_was_sold_strictly_before_the_subject(engine, method, pool):
    rows = subjects(engine)
    comps = engine.rank(method, rows, 8, pool)
    assert (comps >= 0).any()
    for r, row in zip(rows, comps, strict=True):
        got = row[row >= 0]
        assert (engine.sold_days[got] < engine.sold_days[r]).all()
        if pool == "fit":
            assert (got < engine.n_fit).all()


def test_same_day_sales_are_never_comps(engine):
    rows = subjects(engine, 300)
    for method in METHODS:
        comps = engine.rank(method, rows, 6)
        for r, row in zip(rows, comps, strict=True):
            same_day = engine.sold_days[row[row >= 0]] == engine.sold_days[r]
            assert not same_day.any()


def test_a_subjects_own_earlier_sale_is_only_a_comp_when_asked(engine):
    sales = engine.sales
    resale = int(np.flatnonzero(sales["n_prior"].to_numpy() > 0)[0])
    prior = int(np.flatnonzero(engine.ids[:resale] == engine.ids[resale])[0])
    rows = np.array([resale])
    assert prior not in engine.rank("knn", rows, 8)[0]
    assert (
        engine.rank("knn", rows, 1, include_own=True)[0][0] == prior
    )  # identical attributes


def future_canary(engine: Engine) -> tuple[Engine, int, int]:
    """A copy of the subject (identical features, absurd price) sold 30 days after the subject."""
    subject_id = int(engine.ids[engine.sales.height // 2 + 7])
    original = engine.sales.filter(pl.col("id") == subject_id).tail(1)
    twin = original.with_columns(
        pl.lit(1e9).alias("price"),
        pl.lit(-5, dtype=pl.Int64).alias("id"),
        (pl.col("sold") + pl.duration(days=30)).alias("sold"),
    ).with_columns(
        (pl.col("sold").dt.year() * 12 + pl.col("sold").dt.month()).alias("month")
    )
    table = pl.concat([engine.sales, twin]).sort("sold", maintain_order=True)
    ids = table["id"].to_numpy()
    subject = int(np.flatnonzero(ids == subject_id)[-1])
    return Engine(table, engine.n_fit), subject, int(np.flatnonzero(ids == -5)[0])


def test_a_future_twin_with_an_absurd_price_never_reaches_the_estimate(engine):
    eng, subject, canary = future_canary(engine)
    for method in METHODS:
        comps = eng.rank(method, np.array([subject]), 12)[0]
        assert canary not in comps
        est = eng.estimate(np.array([subject]), comps[None, :], Settings(k=5))
        assert est.price[0] < 1e7


def test_the_canary_test_can_fail(engine, monkeypatch):
    """With the date rule switched off the twin is the best comp, so the test above has teeth."""
    eng, subject, canary = future_canary(engine)
    monkeypatch.setattr(
        Engine,
        "pool_limit",
        lambda self, rows, pool="prior": np.full(len(rows), self.sales.height),
    )
    comps = eng.rank("knn", np.array([subject]), 3, include_own=False)[0]
    assert canary in comps


def test_estimate_ignores_the_subjects_own_price_and_everything_after_it(engine):
    sales, n_fit = engine.sales, engine.n_fit
    sold = sales["sold"].to_numpy()
    first_of_day = np.flatnonzero(sold != np.roll(sold, 1))
    i = int(first_of_day[first_of_day > n_fit + 50][0])
    later = pl.int_range(pl.len()) >= i
    altered = sales.with_columns(
        pl.when(later)
        .then(pl.col("price") * 7)
        .otherwise(pl.col("price"))
        .alias("price")
    )
    other = Engine(altered, n_fit)
    s = Settings(k=5)
    for method in METHODS:
        a = engine.predict(method, np.array([i]), s)
        b = other.predict(method, np.array([i]), s)
        assert a.price[0] == b.price[0]
        assert np.array_equal(a.comps, b.comps)
    assert engine.avm(np.array([i]))[0] == other.avm(np.array([i]))[0]


def test_nothing_fitted_sees_prices_or_features_after_the_fit_window(engine):
    sales, n_fit = engine.sales, engine.n_fit
    after = pl.int_range(pl.len()) >= n_fit
    altered = sales.with_columns(
        pl.when(after)
        .then(pl.col("price") * 3)
        .otherwise(pl.col("price"))
        .alias("price"),
        pl.when(after)
        .then(pl.col("sqft_living") * 2)
        .otherwise(pl.col("sqft_living"))
        .alias("sqft_living"),
    )
    other = Engine(altered, n_fit)
    assert np.array_equal(engine.scaler.mean, other.scaler.mean)
    assert np.array_equal(engine.hedonic.beta, other.hedonic.beta)
    assert np.array_equal(engine.hedonic.month_effect, other.hedonic.month_effect)
    assert np.array_equal(engine.gbm.shap_weight, other.gbm.shap_weight)
    assert np.array_equal(engine.gbm.tree_weight, other.gbm.tree_weight)
    assert np.array_equal(engine.gbm_pred[:n_fit], other.gbm_pred[:n_fit])


def test_the_fit_window_test_can_fail(engine):
    """A model fit on more rows differs, so equality above is a real constraint."""
    wider = Engine(engine.sales, engine.n_fit + 300)
    assert not np.array_equal(engine.hedonic.beta, wider.hedonic.beta)
    assert not np.array_equal(engine.gbm.shap_weight, wider.gbm.shap_weight)
