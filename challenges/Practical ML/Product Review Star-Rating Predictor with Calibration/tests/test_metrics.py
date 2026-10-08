import numpy as np
import pytest
from helpers import make_probs
from review_stars import metrics

ONE_HOT = np.eye(5)
UNIFORM = np.full((1, 5), 0.2)


# ---------------------------------------------------------------- point metrics by hand


def test_accuracy_and_star_errors_on_a_tiny_example():
    P = np.array(
        [
            [0.1, 0.1, 0.1, 0.1, 0.6],
            [0.7, 0.1, 0.1, 0.05, 0.05],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]
    )
    y = np.array([5, 3, 3])
    assert metrics.accuracy(P, y) == pytest.approx(2 / 3)
    assert metrics.mae_argmax(P, y) == pytest.approx((0 + 2 + 0) / 3)
    expected = P @ np.arange(1, 6)
    assert metrics.mae_expected(P, y) == pytest.approx(np.abs(expected - y).mean())
    assert metrics.rmse_expected(P, y) == pytest.approx(
        np.sqrt(((expected - y) ** 2).mean())
    )


def test_quadratic_weighted_kappa_is_one_for_perfect_and_near_zero_for_chance():
    y = np.repeat(np.arange(1, 6), 40)
    assert metrics.qwk(ONE_HOT[y - 1], y) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(y)
    assert abs(metrics.qwk(ONE_HOT[shuffled - 1], y)) < 0.15
    off_by_one = np.clip(y + 1, 1, 5)
    assert (
        0.5 < metrics.qwk(ONE_HOT[off_by_one - 1], y) < 1.0
    )  # near misses cost little


def test_qwk_is_undefined_not_a_crash_when_every_label_is_the_same_star():
    y = np.full(50, 5)
    assert np.isnan(metrics.qwk(ONE_HOT[y - 1], y))


def test_nll_brier_and_rps_of_the_uniform_forecast_and_of_certainty():
    y = np.array([1])
    assert metrics.nll(UNIFORM, y) == pytest.approx(np.log(5))
    assert metrics.brier(UNIFORM, y) == pytest.approx(0.8)
    assert metrics.rps(UNIFORM, y) == pytest.approx(0.3)  # (.64 + .36 + .16 + .04) / 4
    assert metrics.nll(ONE_HOT[:1], y) == pytest.approx(0.0, abs=1e-9)
    assert metrics.brier(ONE_HOT[:1], y) == 0.0 and metrics.rps(ONE_HOT[:1], y) == 0.0


def test_rps_charges_a_near_miss_less_than_a_far_one_but_brier_does_not():
    y = np.array([3])
    near, far = ONE_HOT[3:4], ONE_HOT[0:1]  # predicted star 4 vs star 1, truth is 3
    assert metrics.rps(near, y) < metrics.rps(far, y)
    assert metrics.brier(near, y) == metrics.brier(far, y) == 2.0


def test_nll_clips_zero_probabilities_instead_of_returning_infinity():
    assert np.isfinite(metrics.nll(ONE_HOT[:1], np.array([2])))


# ---------------------------------------------------------------- ECE variants


def test_equal_mass_ece_of_a_constant_overconfident_forecast_is_exact():
    conf = np.full(1000, 0.9)
    correct = np.arange(1000) < 600  # 60% right
    assert metrics.ece_equal_mass(conf, correct, bins=15) == pytest.approx(0.3)


def test_tied_confidences_share_a_bin_so_row_order_cannot_change_the_ece():
    # isotonic-calibrated outputs are piecewise constant: arbitrary row order must not matter
    rng = np.random.default_rng(0)
    conf = np.repeat([0.3, 0.55, 0.9], [400, 300, 300])
    correct = rng.uniform(size=1000) < conf - 0.1
    a = metrics.ece_equal_mass(conf, correct, bins=15)
    perm = rng.permutation(1000)
    assert metrics.ece_equal_mass(conf[perm], correct[perm], bins=15) == pytest.approx(
        a
    )
    sorted_first = np.argsort(~correct, kind="stable")  # all the correct rows first
    assert metrics.ece_equal_mass(
        conf[sorted_first], correct[sorted_first], bins=15
    ) == pytest.approx(a)
    curve = metrics.reliability_curve(conf, correct, bins=15)
    assert len(curve["n"]) == 3 and curve["n"].tolist() == [400, 300, 300]


def test_equal_mass_bins_hold_nearly_equal_counts():
    rng = np.random.default_rng(0)
    curve = metrics.reliability_curve(
        rng.uniform(size=1003), rng.uniform(size=1003) < 0.5, bins=15
    )
    assert curve["n"].sum() == 1003 and curve["n"].max() - curve["n"].min() <= 1
    assert np.all(np.diff(curve["conf"]) >= 0)


def test_ece_is_small_for_a_calibrated_model_and_grows_with_known_overconfidence():
    values = {}
    for sharpen in (0.6, 1.0, 1.5, 2.5):
        P, y = make_probs(sharpen=sharpen, seed=1)
        values[sharpen] = metrics.top_label_ece(P, y)
    assert values[1.0] < 0.01  # calibrated by construction
    assert values[1.5] > 0.04 and values[2.5] > values[1.5]
    assert values[0.6] > 0.04  # underconfidence is miscalibration too


def test_classwise_ece_catches_a_model_that_is_calibrated_only_on_its_top_label():
    P, y = make_probs(sharpen=1.0, seed=2)
    assert metrics.classwise_ece(P, y) < 0.01
    P2, y2 = make_probs(sharpen=2.0, seed=2)
    assert metrics.classwise_ece(P2, y2) > metrics.classwise_ece(P, y) + 0.02


def test_reliability_curve_of_a_calibrated_model_hugs_the_diagonal():
    P, y = make_probs(seed=3)
    conf, correct = metrics.top_label(P, y)
    curve = metrics.reliability_curve(conf, correct, bins=10)
    assert np.abs(curve["conf"] - curve["acc"]).max() < 0.04


# ---------------------------------------------------------------- smooth ECE


def test_smooth_ece_of_a_constant_overconfident_forecast_is_about_the_gap():
    conf = np.full(2000, 0.9)
    correct = np.arange(2000) < 1200
    assert metrics.smooth_ece(conf, correct) == pytest.approx(0.3, abs=0.01)


def test_smooth_ece_orders_models_like_the_binned_estimate_and_is_small_when_calibrated():
    got = {}
    for sharpen in (1.0, 1.5, 2.5):
        P, y = make_probs(sharpen=sharpen, seed=4)
        conf, correct = metrics.top_label(P, y)
        got[sharpen] = metrics.smooth_ece(conf, correct)
    assert got[1.0] < 0.01 and got[1.0] < got[1.5] < got[2.5]


@pytest.mark.parametrize("sharpen", [1.0, 1.8])
def test_smooth_ece_matches_apples_relplot_reference_implementation(sharpen):
    relplot = pytest.importorskip("relplot")
    P, y = make_probs(n=20_000, sharpen=sharpen, seed=5)
    conf, correct = metrics.top_label(P, y)
    ours = metrics.smooth_ece(conf, correct)
    theirs = float(relplot.smECE(conf, correct.astype(float)))
    # relplot finds its bandwidth by a 10-step bisection, so it is only good to about 1e-3
    assert abs(ours - theirs) < 0.003, (ours, theirs)


def test_smooth_ece_at_a_fixed_bandwidth_matches_relplot_closely():
    relplot = pytest.importorskip("relplot")
    P, y = make_probs(n=20_000, sharpen=1.8, seed=6)
    conf, correct = metrics.top_label(P, y)
    for sigma in (0.02, 0.05, 0.1):
        ours = metrics.smooth_ece_at(conf, correct, sigma)
        theirs = float(relplot.smECE_sigma(conf, correct.astype(float), sigma))
        assert abs(ours - theirs) < 0.002, (sigma, ours, theirs)


def test_smooth_ece_handles_degenerate_inputs():
    assert np.isnan(metrics.smooth_ece(np.array([]), np.array([], dtype=bool)))
    assert metrics.smooth_ece(np.array([0.5]), np.array([True])) >= 0.0
    one_conf = metrics.smooth_ece(np.full(500, 1.0), np.ones(500, dtype=bool))
    assert one_conf == pytest.approx(0.0, abs=1e-6)  # always sure, always right


# ---------------------------------------------------------------- coverage


def test_set_coverage_of_a_calibrated_model_is_at_least_nominal_and_grows_with_the_level():
    P, y = make_probs(seed=7)
    rows = metrics.coverage_curve(P, y, (0.5, 0.8, 0.9, 0.99))
    cov = [r["coverage"] for r in rows]
    size = [r["mean_size"] for r in rows]
    assert all(
        c >= lvl - 0.01 for c, lvl in zip(cov, (0.5, 0.8, 0.9, 0.99), strict=True)
    )
    assert cov == sorted(cov) and size == sorted(size)
    assert size[-1] <= 5.0


def test_an_overconfident_model_under_covers_at_high_levels():
    P, y = make_probs(sharpen=3.0, seed=8)
    row = metrics.coverage_curve(P, y, (0.9,))[0]
    assert row["coverage"] < 0.9 - 0.03


def test_coverage_curve_by_hand():
    P = np.array([[0.5, 0.3, 0.1, 0.05, 0.05], [0.1, 0.1, 0.1, 0.1, 0.6]])
    y = np.array([2, 5])
    low, high = metrics.coverage_curve(P, y, (0.5, 0.8))
    assert (low["coverage"], low["mean_size"]) == (
        0.5,
        1.0,
    )  # {1} and {5}: only the second hits
    assert (high["coverage"], high["mean_size"]) == (
        1.0,
        (2 + 3) / 2,
    )  # {1,2} and {5,1,2}: .6+.1 < .8


def test_gaussian_interval_coverage_matches_nominal_when_sigma_is_right_and_not_when_it_is_not():
    rng = np.random.default_rng(0)
    n = 50_000
    mu = rng.uniform(1, 5, n)
    y = mu + 0.7 * rng.normal(size=n)
    right = metrics.interval_coverage(mu, np.full(n, 0.7), y, (0.5, 0.9, 0.95))
    assert [round(r["coverage"], 2) for r in right] == [0.5, 0.9, 0.95]
    assert right[1]["width"] == pytest.approx(2 * 1.6449 * 0.7, rel=1e-3)
    wrong = metrics.interval_coverage(mu, np.full(n, 0.35), y, (0.9,))[0]
    assert wrong["coverage"] < 0.65


# ---------------------------------------------------------------- the bundle used by the bootstrap


def test_bundle_has_every_headline_metric_and_matches_the_individual_functions():
    P, y = make_probs(n=3000, sharpen=1.4, seed=9)
    b = metrics.bundle(P, y, bins=15)
    assert set(b) == set(metrics.BUNDLE_KEYS)
    assert b["acc"] == metrics.accuracy(P, y) and b["nll"] == metrics.nll(P, y)
    assert b["ece"] == metrics.top_label_ece(P, y, bins=15) and b[
        "brier"
    ] == metrics.brier(P, y)
    assert all(np.isfinite(v) for v in b.values())


def test_bundle_survives_a_split_where_every_review_has_the_same_star():
    y = np.full(200, 5)
    P = np.tile([0.02, 0.02, 0.03, 0.13, 0.8], (200, 1))
    b = metrics.bundle(P, y, bins=15)
    assert b["acc"] == 1.0 and np.isnan(b["qwk"])
    assert all(np.isfinite(v) for k, v in b.items() if k != "qwk")


def test_metrics_reject_probabilities_that_do_not_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        metrics.nll(np.full((2, 5), 0.3), np.array([1, 2]))
    with pytest.raises(ValueError, match="shape"):
        metrics.accuracy(np.ones((2, 4)) / 4, np.array([1, 2]))
    with pytest.raises(ValueError, match="1 and 5"):
        metrics.accuracy(np.full((1, 5), 0.2), np.array([6]))


def test_fixed_width_bins_count_the_confidence_histogram_with_the_same_edges_everywhere():
    conf = np.array([0.05, 0.15, 0.15, 0.95, 1.0, 0.0])
    correct = np.array([True, False, True, True, False, False])
    n, k = metrics.fixed_bin_counts(conf, correct, bins=10)
    assert n.tolist() == [
        2,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        2,
    ]  # 1.0 belongs to the last bin, not a 11th
    assert k.tolist() == [1, 1, 0, 0, 0, 0, 0, 0, 0, 1]
    assert (
        metrics.fixed_bin_counts(np.array([]), np.array([], dtype=bool), 4)[0].tolist()
        == [0] * 4
    )
