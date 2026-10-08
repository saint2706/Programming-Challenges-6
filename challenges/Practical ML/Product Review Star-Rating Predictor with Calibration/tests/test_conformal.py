import numpy as np
import pytest
from helpers import make_probs
from review_stars import conformal

ALPHA = 0.1


def draw(n, sharpen=1.0, seed=0):
    return make_probs(n=n, sharpen=sharpen, seed=seed)


def covered(mask, y):
    return float(mask[np.arange(len(y)), y - 1].mean())


# ---------------------------------------------------------------- the quantile


def test_conformal_quantile_is_the_finite_sample_corrected_order_statistic():
    scores = np.arange(1.0, 101.0)  # 1..100
    assert conformal.quantile(scores, 0.1) == 91.0  # ceil(101 * 0.9) = 91st smallest
    assert conformal.quantile(scores, 0.5) == 51.0


def test_a_calibration_set_too_small_for_alpha_gives_an_infinite_quantile_not_a_silent_undercover():
    assert (
        conformal.quantile(np.arange(5.0), 0.1) == np.inf
    )  # needs ceil(6 * .9) = 6 > 5
    P, y = draw(5, seed=3)
    model = conformal.ApsConformal(0.1).fit(P, y, np.full(5, 0.5))
    assert model.qhat == np.inf
    sets = model.sets(draw(50, seed=4)[0], np.full(50, 0.5))
    assert sets.all()  # every star: honest "I cannot say"


def test_quantile_rejects_empty_scores_and_bad_alpha():
    with pytest.raises(ValueError, match="empty"):
        conformal.quantile(np.array([]), 0.1)
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError, match="alpha"):
            conformal.quantile(np.arange(10.0), bad)


# ---------------------------------------------------------------- APS sets


@pytest.mark.parametrize("seed", range(4))
def test_randomized_aps_covers_close_to_exactly_the_target(seed):
    P_cal, y_cal = draw(8000, seed=10 + seed)
    P_test, y_test = draw(20_000, seed=20 + seed)
    rng = np.random.default_rng(seed)
    model = conformal.ApsConformal(ALPHA).fit(P_cal, y_cal, rng.uniform(size=8000))
    mask = model.sets(P_test, rng.uniform(size=20_000))
    assert 0.885 < covered(mask, y_test) < 0.925  # valid and tight, not just valid


def test_deterministic_aps_is_valid_but_conservative_and_its_sets_are_bigger():
    P_cal, y_cal = draw(8000, seed=1)
    P_test, y_test = draw(20_000, seed=2)
    rand = conformal.ApsConformal(ALPHA).fit(
        P_cal, y_cal, np.random.default_rng(0).uniform(size=8000)
    )
    det = conformal.ApsConformal(ALPHA, randomized=False).fit(P_cal, y_cal)
    m_rand = rand.sets(P_test, np.random.default_rng(1).uniform(size=20_000))
    m_det = det.sets(P_test)
    assert covered(m_det, y_test) >= ALPHA and covered(m_det, y_test) > 0.93
    assert m_det.sum(axis=1).mean() > m_rand.sum(axis=1).mean() + 0.3


def test_an_overconfident_model_still_covers_because_the_guarantee_needs_no_calibration():
    # the sets are not guaranteed to be larger or smaller (their size depends on how the ranking
    # and the cumulative mass interact), only that coverage holds whatever the model reports
    for sharpen in (0.5, 1.0, 3.0):
        P_cal, y_cal = draw(8000, sharpen=sharpen, seed=5)
        P_test, y_test = draw(20_000, sharpen=sharpen, seed=6)
        u = np.random.default_rng(0)
        model = conformal.ApsConformal(ALPHA).fit(P_cal, y_cal, u.uniform(size=8000))
        mask = model.sets(P_test, u.uniform(size=20_000))
        assert 0.885 < covered(mask, y_test) < 0.925, sharpen


def test_coverage_breaks_when_the_calibration_and_test_distributions_differ():
    P_cal, y_cal = draw(8000, seed=7)
    P_test, y_test = draw(20_000, seed=8)
    shifted = y_test.copy()
    flip = np.random.default_rng(0).uniform(size=len(y_test)) < 0.4
    shifted[flip] = np.random.default_rng(1).integers(
        1, 6, flip.sum()
    )  # labels no longer follow P
    u = np.random.default_rng(2)
    model = conformal.ApsConformal(ALPHA).fit(P_cal, y_cal, u.uniform(size=8000))
    mask = model.sets(P_test, u.uniform(size=20_000))
    assert (
        covered(mask, y_test) > 0.87 and covered(mask, shifted) < 0.82
    )  # exchangeability gone


def test_sets_are_never_empty_and_the_top_star_is_always_the_fallback():
    P_cal, y_cal = draw(4000, seed=9)
    P_test, _ = draw(5000, seed=10)
    rng = np.random.default_rng(3)
    model = conformal.ApsConformal(ALPHA).fit(P_cal, y_cal, rng.uniform(size=4000))
    raw = model.sets(P_test, rng.uniform(size=5000), nonempty=False)
    fixed = conformal.fill_empty(raw, P_test)
    assert (fixed.sum(axis=1) >= 1).all()
    empty = raw.sum(axis=1) == 0
    assert empty.any()  # randomization really does produce empties
    assert fixed[empty][np.arange(empty.sum()), P_test[empty].argmax(axis=1)].all()
    assert np.array_equal(fixed[~empty], raw[~empty])


def test_set_stats_report_coverage_size_and_the_size_distribution():
    mask = np.array([[1, 1, 0, 0, 0], [0, 0, 0, 0, 1], [0, 1, 1, 1, 0]], dtype=bool)
    stats = conformal.set_stats(mask, np.array([2, 4, 3]))
    assert stats["coverage"] == pytest.approx(2 / 3) and stats[
        "mean_size"
    ] == pytest.approx(2.0)
    assert stats["size_share"] == pytest.approx([1 / 3, 1 / 3, 1 / 3, 0, 0])
    assert (
        conformal.set_stats(mask[:0], np.array([], dtype=int))["coverage"]
        != conformal.set_stats(mask[:0], np.array([], dtype=int))["coverage"]
    )  # NaN


def test_random_draws_must_match_the_number_of_reviews():
    P, y = draw(100)
    with pytest.raises(ValueError, match="one uniform draw per review"):
        conformal.ApsConformal(ALPHA).fit(P, y, np.zeros(99))
    model = conformal.ApsConformal(ALPHA).fit(P, y, np.full(100, 0.5))
    with pytest.raises(ValueError, match="one uniform draw per review"):
        model.sets(P, np.zeros(3))


def test_aps_round_trips_through_arrays():
    P, y = draw(500)
    u = np.random.default_rng(0).uniform(size=500)
    for randomized in (True, False):
        model = conformal.ApsConformal(ALPHA, randomized=randomized).fit(
            P, y, u if randomized else None
        )
        arrays = model.to_arrays("c/")
        again = conformal.ApsConformal.from_arrays(arrays, "c/")
        assert (
            again.qhat == model.qhat
            and again.randomized == randomized
            and again.alpha == ALPHA
        )


# ---------------------------------------------------------------- cross-checks against MAPIE 1.5


@pytest.fixture(scope="module")
def mapie_parts():
    pytest.importorskip("mapie")
    from mapie.classification import SplitConformalClassifier
    from mapie.conformity_scores.sets.aps import APSConformityScore
    from sklearn.base import BaseEstimator, ClassifierMixin

    class Passthrough(ClassifierMixin, BaseEstimator):
        """A fitted classifier whose 'features' already are the probabilities."""

        classes_ = np.arange(1, 6)

        def fit(self, X, y=None):
            return self

        def predict_proba(self, X):
            return np.asarray(X)

        def predict(self, X):
            return self.classes_[np.asarray(X).argmax(axis=1)]

    class NoRandomAPS(APSConformityScore):
        """MAPIE's APS with its uniform randomization removed (it has no switch for that)."""

        def get_conformity_scores(self, y, y_pred, y_enc=None, **kwargs):
            scores, self.cutoff = self.get_true_label_cumsum_proba(
                y, y_pred, self.classes
            )
            return scores

    return SplitConformalClassifier, Passthrough, NoRandomAPS


def mapie_classifier(parts, score, seed=None):
    SplitConformalClassifier, Passthrough, _ = parts
    return SplitConformalClassifier(
        estimator=Passthrough(),
        confidence_level=1 - ALPHA,
        conformity_score=score,
        prefit=True,
        random_state=seed,
    )


@pytest.mark.filterwarnings("ignore:Estimator does not appear fitted")
def test_deterministic_aps_equals_mapie_exactly(mapie_parts):
    _, _, NoRandomAPS = mapie_parts
    P_cal, y_cal = draw(3000, sharpen=2.0, seed=11)
    P_test, _ = draw(3000, sharpen=2.0, seed=12)
    mc = mapie_classifier(mapie_parts, NoRandomAPS()).conformalize(P_cal, y_cal)
    _, theirs = mc.predict_set(
        P_test, conformity_score_params={"include_last_label": True}
    )
    ours = conformal.ApsConformal(ALPHA, randomized=False).fit(P_cal, y_cal)
    assert ours.qhat == pytest.approx(
        float(mc._mapie_classifier.quantiles_[0]), abs=1e-12
    )
    assert np.array_equal(ours.sets(P_test), theirs[:, :, 0])


@pytest.mark.filterwarnings("ignore:Estimator does not appear fitted")
def test_randomized_aps_equals_mapie_exactly_when_given_mapies_own_random_draws(
    mapie_parts,
):
    seed = 7
    P_cal, y_cal = draw(3000, sharpen=2.0, seed=13)
    P_test, _ = draw(3000, sharpen=2.0, seed=14)
    mc = mapie_classifier(mapie_parts, "aps", seed=seed).conformalize(P_cal, y_cal)
    _, theirs = mc.predict_set(
        P_test, conformity_score_params={"include_last_label": "randomized"}
    )
    # MAPIE re-seeds a RandomState(seed) for the calibration scores and again for the sets
    u_cal = np.random.RandomState(seed).uniform(size=3000)
    u_test = np.random.RandomState(seed).uniform(size=3000)
    ours = conformal.ApsConformal(ALPHA).fit(P_cal, y_cal, u_cal)
    assert ours.qhat == pytest.approx(
        float(mc._mapie_classifier.quantiles_[0]), abs=1e-12
    )
    mine = ours.sets(P_test, u_test, nonempty=False)
    assert (mine != theirs[:, :, 0]).any(
        axis=1
    ).mean() < 0.001  # identical up to float ties


@pytest.mark.filterwarnings("ignore:Estimator does not appear fitted")
def test_our_randomized_aps_and_mapies_stock_aps_agree_statistically(mapie_parts):
    P_cal, y_cal = draw(6000, seed=15)
    P_test, y_test = draw(20_000, seed=16)
    mc = mapie_classifier(mapie_parts, "aps", seed=0).conformalize(P_cal, y_cal)
    _, theirs = mc.predict_set(
        P_test, conformity_score_params={"include_last_label": "randomized"}
    )
    rng = np.random.default_rng(0)
    ours = conformal.ApsConformal(ALPHA).fit(P_cal, y_cal, rng.uniform(size=6000))
    mine = ours.sets(P_test, rng.uniform(size=20_000), nonempty=False)
    assert abs(covered(mine, y_test) - covered(theirs[:, :, 0], y_test)) < 0.01
    assert abs(mine.sum(axis=1).mean() - theirs[:, :, 0].sum(axis=1).mean()) < 0.05


# ---------------------------------------------------------------- regression intervals


def gaussian_data(n, seed, sigma_of=None):
    rng = np.random.default_rng(seed)
    mu = rng.uniform(1, 5, n)
    sigma = np.full(n, 0.6) if sigma_of is None else sigma_of(mu)
    return mu, sigma, mu + sigma * rng.normal(size=n)


def test_interval_conformal_hits_the_target_coverage_and_scales_with_sigma():
    sigma_of = lambda mu: 0.3 + 0.25 * mu
    mu_c, s_c, y_c = gaussian_data(8000, 1, sigma_of)
    mu_t, s_t, y_t = gaussian_data(20_000, 2, sigma_of)
    model = conformal.IntervalConformal(ALPHA).fit(mu_c, s_c, y_c)
    lo, hi = model.interval(mu_t, s_t)
    inside = (y_t >= lo) & (y_t <= hi)
    assert 0.885 < inside.mean() < 0.915
    low, high = mu_t < 2.5, mu_t > 3.5
    assert abs(inside[low].mean() - inside[high].mean()) < 0.03  # adapts to sigma
    assert (hi - lo)[high].mean() > (hi - lo)[low].mean() * 1.3


def test_interval_conformal_equals_mapie_with_a_constant_sigma():
    pytest.importorskip("mapie")
    from mapie.regression import SplitConformalRegressor
    from sklearn.base import BaseEstimator, RegressorMixin

    class Fixed(RegressorMixin, BaseEstimator):
        fitted_ = True  # prefit: MAPIE only checks that it looks fitted

        def fit(self, X, y=None):
            return self

        def predict(self, X):
            return np.asarray(X)[:, 0]

    mu_c, s_c, y_c = gaussian_data(3000, 3)
    mu_t, s_t, _ = gaussian_data(3000, 4)
    reg = SplitConformalRegressor(
        estimator=Fixed(), confidence_level=1 - ALPHA, prefit=True
    )
    reg.conformalize(mu_c[:, None], y_c)
    _, theirs = reg.predict_interval(mu_t[:, None])
    model = conformal.IntervalConformal(ALPHA).fit(
        mu_c, np.ones_like(s_c), y_c
    )  # sigma == 1
    lo, hi = model.interval(mu_t, np.ones_like(s_t))
    assert np.allclose(lo, theirs[:, 0, 0]) and np.allclose(hi, theirs[:, 1, 0])


def test_interval_conformal_with_too_few_calibration_reviews_returns_infinite_intervals():
    model = conformal.IntervalConformal(ALPHA).fit(
        np.array([3.0] * 4), np.ones(4), np.array([3.1] * 4)
    )
    lo, hi = model.interval(np.array([3.0]), np.array([1.0]))
    assert lo[0] == -np.inf and hi[0] == np.inf


def test_interval_conformal_rejects_non_positive_or_non_finite_sigma():
    with pytest.raises(ValueError, match="sigma"):
        conformal.IntervalConformal(ALPHA).fit(
            np.ones(30), np.array([1.0] * 29 + [np.nan]), np.ones(30)
        )
    with pytest.raises(ValueError, match="sigma"):
        conformal.IntervalConformal(ALPHA).fit(np.ones(30), np.zeros(30), np.ones(30))
