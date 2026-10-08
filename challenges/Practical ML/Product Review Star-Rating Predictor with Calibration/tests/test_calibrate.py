import numpy as np
import pytest
from helpers import (
    make_classification_preds,
    make_ordinal_preds,
    make_regression_preds,
)
from review_stars import calibrate, metrics
from review_stars.probs import Pred, Predictions
from sklearn.isotonic import IsotonicRegression


def ece(P, y):
    return metrics.top_label_ece(P, y)


def fit_on_cal_score_on_test(kind, make, **kw):
    """Fit on one draw, score on a fresh one: a calibrator must never see the data it is scored on."""
    cal_preds, cal_y = make(seed=1, **kw)
    test_preds, test_y = make(seed=2, **kw)
    cal = calibrate.make(kind).fit(cal_preds, cal_y)
    return cal, test_preds, test_y


# ---------------------------------------------------------------- temperature, all framings


@pytest.mark.parametrize("sharpen", [0.6, 1.0, 2.0, 3.0])
def test_temperature_recovers_a_known_classifier_sharpness(sharpen):
    cal, preds, y = fit_on_cal_score_on_test(
        "temperature", make_classification_preds, sharpen=sharpen
    )
    assert cal.T == pytest.approx(sharpen, rel=0.04)
    assert ece(cal.proba(preds), y) < 0.015


def test_temperature_fixes_overconfidence_by_a_wide_margin():
    cal, preds, y = fit_on_cal_score_on_test(
        "temperature", make_classification_preds, sharpen=2.5
    )
    before, after = ece(preds.proba(), y), ece(cal.proba(preds), y)
    assert before > 0.1 and after < 0.015
    assert metrics.nll(cal.proba(preds), y) < metrics.nll(preds.proba(), y)


@pytest.mark.parametrize("factor", [0.5, 1.0, 1.6])
def test_variance_scaling_recovers_a_known_regression_sigma_error(factor):
    cal, preds, y = fit_on_cal_score_on_test(
        "temperature", make_regression_preds, factor=factor
    )
    assert cal.T == pytest.approx(1.0 / factor, rel=0.06)
    assert ece(cal.proba(preds), y) < 0.02


@pytest.mark.parametrize("scale", [1.0, 1.5, 2.2])
def test_ordinal_temperature_recovers_the_latent_noise_scale(scale):
    cal, preds, y = fit_on_cal_score_on_test(
        "temperature", make_ordinal_preds, scale_true=scale
    )
    assert cal.T == pytest.approx(scale, rel=0.05)
    assert ece(cal.proba(preds), y) < 0.02


def test_an_ensemble_is_calibrated_with_one_temperature_applied_to_every_member():
    cal_p, cal_y = make_classification_preds(seed=1, sharpen=2.0)
    test_p, test_y = make_classification_preds(seed=2, sharpen=2.0)
    # a second, differently over-sharp member: the ensemble is over-sharp too
    cal_p2, _ = make_classification_preds(seed=1, sharpen=2.6)
    test_p2, _ = make_classification_preds(seed=2, sharpen=2.6)
    ens_cal = Predictions([cal_p.members[0], cal_p2.members[0]])
    ens_test = Predictions([test_p.members[0], test_p2.members[0]])
    cal = calibrate.make("temperature").fit(ens_cal, cal_y)
    assert cal.T > 1.5
    assert ece(cal.proba(ens_test), test_y) < ece(ens_test.proba(), test_y)


def test_temperature_stays_in_bounds_and_does_not_crash_on_a_single_star_calibration_set():
    preds, _ = make_classification_preds(n=500, seed=3)
    only_fives = np.full(500, 5, dtype=np.int64)
    cal = calibrate.make("temperature").fit(preds, only_fives)
    assert calibrate.T_BOUNDS[0] <= cal.T <= calibrate.T_BOUNDS[1]
    assert np.isfinite(cal.proba(preds)).all()


def test_an_empty_calibration_set_is_an_error():
    preds, y = make_classification_preds(n=100)
    with pytest.raises(ValueError, match="empty"):
        calibrate.make("temperature").fit(preds.take(np.array([], dtype=int)), y[:0])


def test_nothing_is_learned_from_the_scored_split():
    # fitting on cal and scoring on test must give the same numbers however often it is applied
    cal, preds, _ = fit_on_cal_score_on_test(
        "temperature", make_classification_preds, sharpen=2.0
    )
    first = cal.proba(preds)
    assert np.array_equal(first, cal.proba(preds)) and cal.T == cal.T


# ---------------------------------------------------------------- vector scaling


def test_vector_scaling_calibrates_an_overconfident_classifier():
    cal, preds, y = fit_on_cal_score_on_test(
        "vector", make_classification_preds, sharpen=2.2
    )
    assert ece(cal.proba(preds), y) < 0.015
    assert np.allclose(np.sort(cal.a), np.sort(np.full(5, 1 / 2.2)), atol=0.08)


def test_vector_scaling_only_applies_to_a_single_classification_model():
    cls, _ = make_classification_preds(n=200)
    reg, _ = make_regression_preds(n=200)
    ens = Predictions([cls.members[0], cls.members[0]])
    assert calibrate.applicable("vector", cls)
    assert not calibrate.applicable("vector", reg) and not calibrate.applicable(
        "vector", ens
    )
    for kind in ("none", "temperature", "isotonic"):
        assert all(calibrate.applicable(kind, p) for p in (cls, reg, ens))
    with pytest.raises(ValueError, match="single classification"):
        calibrate.make("vector").fit(reg, np.ones(200, dtype=np.int64))


# ---------------------------------------------------------------- isotonic


def test_isotonic_calibrates_an_overconfident_model_of_any_framing():
    for make, kw in (
        (make_classification_preds, {"sharpen": 2.0}),
        (make_regression_preds, {"factor": 0.5}),
        (make_ordinal_preds, {"scale_true": 1.8}),
    ):
        cal, preds, y = fit_on_cal_score_on_test("isotonic", make, **kw)
        P = cal.proba(preds)
        assert np.allclose(P.sum(axis=1), 1.0) and (P > 0).all()
        assert ece(P, y) < 0.03 < ece(preds.proba(), y)


def test_isotonic_maps_are_monotone_and_match_scikit_learns_transform():
    preds, y = make_classification_preds(n=5000, sharpen=2.0)
    cal = calibrate.make("isotonic").fit(preds, y)
    P = preds.proba()
    for k in range(5):
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(
            P[:, k], (y == k + 1).astype(float)
        )
        grid = np.linspace(0, 1, 200)
        ours = np.interp(grid, cal.maps[k][0], cal.maps[k][1])
        assert np.all(np.diff(ours) >= -1e-12)
        assert np.allclose(ours, iso.transform(grid), atol=1e-9)


def test_a_calibration_split_missing_a_star_leaves_that_star_uncalibrated_not_zeroed():
    preds, y = make_classification_preds(n=6000, seed=4, sharpen=1.5)
    keep = y != 1  # no one-star reviews in the calibration split
    cal = calibrate.make("isotonic").fit(preds.take(np.flatnonzero(keep)), y[keep])
    assert cal.degenerate == [0]
    test_preds, test_y = make_classification_preds(n=6000, seed=5, sharpen=1.5)
    P = cal.proba(test_preds)
    assert np.allclose(P.sum(axis=1), 1.0) and (P[:, 0] > 1e-4).mean() > 0.3
    assert np.isfinite(metrics.nll(P, test_y)) and metrics.nll(P, test_y) < 3.0


def test_vector_scaling_with_a_missing_star_stays_finite():
    preds, y = make_classification_preds(n=6000, seed=4, sharpen=1.5)
    keep = y != 1
    cal = calibrate.make("vector").fit(preds.take(np.flatnonzero(keep)), y[keep])
    test_preds, test_y = make_classification_preds(n=6000, seed=5, sharpen=1.5)
    P = cal.proba(test_preds)
    assert np.isfinite(cal.a).all() and np.isfinite(cal.b).all()
    assert np.isfinite(metrics.nll(P, test_y)) and np.allclose(P.sum(axis=1), 1.0)


def test_isotonic_output_probabilities_have_a_floor_so_nll_cannot_blow_up():
    preds, y = make_classification_preds(n=3000, sharpen=3.0)
    P = calibrate.make("isotonic").fit(preds, y).proba(preds)
    assert P.min() >= calibrate.FLOOR / 2 and np.isfinite(metrics.nll(P, y))


# ---------------------------------------------------------------- bookkeeping


@pytest.mark.parametrize("kind", ["none", "temperature", "vector", "isotonic"])
def test_every_calibrator_round_trips_through_arrays_without_pickle(kind, tmp_path):
    preds, y = make_classification_preds(n=3000, sharpen=1.8)
    cal = calibrate.make(kind).fit(preds, y)
    path = tmp_path / "c.npz"
    np.savez(path, **cal.to_arrays("c/"))
    with np.load(path, allow_pickle=False) as z:
        again = calibrate.from_arrays(dict(z), "c/")
    assert again.name == kind and np.allclose(again.proba(preds), cal.proba(preds))


def test_none_is_the_uncalibrated_model_and_unknown_kinds_are_errors():
    preds, y = make_classification_preds(n=200)
    assert np.array_equal(
        calibrate.make("none").fit(preds, y).proba(preds), preds.proba()
    )
    with pytest.raises(ValueError, match="unknown calibrator"):
        calibrate.make("platt")


def test_fit_all_returns_only_the_applicable_calibrators():
    reg, y = make_regression_preds(n=1500)
    out = calibrate.fit_all(reg, y)
    assert list(out) == ["none", "temperature", "isotonic"]
    cls, y = make_classification_preds(n=1500)
    assert list(calibrate.fit_all(cls, y)) == list(calibrate.KINDS)


def test_a_calibrator_can_be_applied_to_a_pred_directly_for_a_single_review():
    preds, y = make_classification_preds(n=2000, sharpen=2.0)
    cal = calibrate.make("temperature").fit(preds, y)
    one = Predictions([preds.members[0].take(np.array([5]))])
    assert cal.proba(one).shape == (1, 5) and np.isclose(cal.proba(one).sum(), 1.0)
    assert isinstance(one.members[0], Pred)
