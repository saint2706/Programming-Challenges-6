from __future__ import annotations

import numpy as np
import pytest
from comp_finder import features
from comp_finder.hedonic import LinearHedonic


def test_recovers_the_known_price_drivers(engine):
    beta = dict(zip(features.COLUMNS, engine.hedonic.beta, strict=True))
    std = dict(zip(features.COLUMNS, engine.scaler.std, strict=True))
    assert beta["log_living"] / std["log_living"] == pytest.approx(
        0.8, abs=0.15
    )  # elasticity
    assert beta["grade"] / std["grade"] == pytest.approx(0.12, abs=0.02)
    for noise in ("condition", "floors", "has_basement"):  # not in the price function
        assert abs(beta[noise]) < 0.02


def test_market_index_recovers_the_monthly_trend(engine):
    months = engine.hedonic.months
    slope = np.polyfit(months, engine.hedonic.month_effect, 1)[0]
    assert slope == pytest.approx(0.01, abs=0.004)
    assert engine.hedonic.month_effect[0] == 0.0


def test_index_holds_flat_or_continues_the_trend_beyond_the_fit_window(engine):
    h = engine.hedonic
    last = int(h.months[-1])
    ahead = np.array([last, last + 1, last + 3])
    flat = h.index(ahead, "flat")
    trend = h.index(ahead, "trend")
    assert flat[0] == flat[1] == flat[2] == h.month_effect[-1]
    assert trend[0] == flat[0]
    assert trend[2] > trend[1] > trend[0]  # the synthetic market rises
    assert h.index(np.array([int(h.months[0]) - 5]), "flat")[0] == h.month_effect[0]


def test_index_rejects_an_unknown_extrapolation(engine):
    with pytest.raises(ValueError, match="extrapolate"):
        engine.hedonic.index(np.array([24180]), "spline")


def test_adjustments_are_beta_times_the_difference_and_skip_location(engine):
    z = engine.z
    adj = engine.hedonic.adjustments(z[10], z[3])
    assert adj.shape == (len(features.ATTRIBUTES),)
    manual = [
        (z[10, features.COLUMNS.index(f)] - z[3, features.COLUMNS.index(f)])
        * engine.hedonic.beta[features.COLUMNS.index(f)]
        for f in features.ATTRIBUTES
    ]
    assert np.allclose(adj, manual)
    assert np.allclose(engine.hedonic.adjustments(z[10], z[10]), 0)


def test_predict_log_explains_most_of_the_price(engine):
    n = engine.n_fit
    pred = engine.hedonic.predict_log(
        engine.z[:n], engine.zip[:n], engine.month[:n], "flat"
    )
    resid = engine.log_price[:n] - pred
    assert resid.std() < 0.3 * engine.log_price[:n].std()


def test_unseen_zip_codes_get_no_location_effect(engine):
    assert engine.hedonic.location_effect(np.array(["00000"]))[0] == 0.0


def test_fit_is_deterministic(engine):
    again = LinearHedonic.fit(
        engine.z[: engine.n_fit],
        engine.zip[: engine.n_fit],
        engine.month[: engine.n_fit],
        engine.log_price[: engine.n_fit],
    )
    assert np.array_equal(again.beta, engine.hedonic.beta)
