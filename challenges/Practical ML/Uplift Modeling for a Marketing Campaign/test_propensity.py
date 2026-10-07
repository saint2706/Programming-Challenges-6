import data
import numpy as np
import propensity
import pytest
from helpers import make_frame


def test_hanley_mcneil_matches_a_hand_computed_value():
    # A = 0.5, n1 = n0 = 100: SE^2 = (0.25 + 2 * 99 * (1/3 - 0.25)) / 100^2
    lo, hi = propensity.auc_ci(0.5, 100, 100)
    se = np.sqrt((0.25 + 2 * 99 * (1 / 3 - 0.25)) / 100**2)
    assert lo == pytest.approx(0.5 - 1.96 * se)
    assert hi == pytest.approx(0.5 + 1.96 * se)


def fit_diagnose(df):
    train, val, _ = data.split(df, seed=0)
    X, t, _ = data.xy(train)
    Xv, tv, _ = data.xy(val)
    model = propensity.PropensityModel(seed=0).fit(X, t)
    return model, propensity.diagnose(model, Xv, tv)


def test_a_randomized_frame_looks_randomized():
    _, diag = fit_diagnose(make_frame(60000))
    assert diag["auc_lo"] <= 0.5 <= diag["auc_hi"] + 0.01
    shares = [b["treated_share"] for b in diag["bins"]]
    assert max(shares) - min(shares) < 0.04


def test_a_confounded_frame_is_detected():
    _, diag = fit_diagnose(make_frame(60000, confounded=True))
    assert diag["auc_lo"] > 0.6
    shares = [b["treated_share"] for b in diag["bins"]]
    assert shares[-1] - shares[0] > 0.1


def test_predictions_are_clipped_away_from_zero_and_one():
    df = make_frame(20000, confounded=True)
    X, t, _ = data.xy(df)
    e = propensity.PropensityModel(seed=0).fit(X, t).predict(X)
    lo, hi = propensity.CLIP
    assert e.min() >= lo and e.max() <= hi


def test_the_model_recovers_the_true_propensity_on_confounded_data():
    df = make_frame(60000, confounded=True)
    X, t, _ = data.xy(df)
    e = propensity.PropensityModel(seed=0).fit(X, t).predict(X)
    true = 1 / (1 + np.exp(-(1.7 + 1.5 * X[:, 0])))
    assert np.corrcoef(e, true)[0, 1] > 0.9


def test_diagnose_reports_the_propensity_range_and_bins():
    _, diag = fit_diagnose(make_frame(20000))
    assert len(diag["bins"]) == 5
    assert diag["e_min"] <= diag["e_max"]
    assert sum(b["n"] for b in diag["bins"]) == 4000
