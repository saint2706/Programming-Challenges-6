import numpy as np
import pytest
from review_stars import probs
from review_stars.probs import Pred, Predictions
from scipy.stats import norm


def entropy(P):
    return float(-(P * np.log(np.clip(P, 1e-12, None))).sum(axis=1).mean())


def classification(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return Pred("classification", logits=rng.normal(size=(n, 5)) * 2)


def regression(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return Pred(
        "regression", mu=rng.uniform(0.5, 5.5, n), sigma=rng.uniform(0.2, 1.5, n)
    )


def ordinal(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return Pred(
        "ordinal", score=rng.normal(size=n) * 2, theta=np.array([-2.0, -0.5, 0.7, 2.0])
    )


ALL = [classification, regression, ordinal]


@pytest.mark.parametrize("make", ALL)
def test_every_framing_gives_a_valid_five_star_probability_vector(make):
    P = make().proba()
    assert P.shape == (40, 5) and (P >= 0).all()
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-9)


def test_gaussian_probs_integrate_the_normal_over_each_star_with_open_tails():
    P = probs.gaussian_probs(np.array([3.0]), np.array([1.0]))[0]
    cdf = norm.cdf([1.5, 2.5, 3.5, 4.5], loc=3.0, scale=1.0)
    assert np.allclose(
        P, [cdf[0], cdf[1] - cdf[0], cdf[2] - cdf[1], cdf[3] - cdf[2], 1 - cdf[3]]
    )
    far = probs.gaussian_probs(np.array([9.0, -4.0]), np.array([0.3, 0.3]))
    assert (
        far[0].argmax() == 4 and far[1].argmax() == 0
    )  # tails are assigned to 1 and 5
    assert np.allclose(far.sum(axis=1), 1.0)


def test_ordinal_probs_are_differenced_cumulative_logits():
    theta = np.array([-1.0, 0.0, 1.0, 2.0])
    P = probs.ordinal_probs(np.array([0.5]), theta)[0]
    cum = 1 / (1 + np.exp(-(theta - 0.5)))
    assert np.allclose(
        P, [cum[0], cum[1] - cum[0], cum[2] - cum[1], cum[3] - cum[2], 1 - cum[3]]
    )
    low, high = probs.ordinal_probs(np.array([-9.0, 9.0]), theta)
    assert (
        low.argmax() == 0 and high.argmax() == 4
    )  # a higher score pushes toward more stars


def test_a_larger_score_means_more_expected_stars_for_the_ordinal_head():
    theta = np.array([-1.0, 0.0, 1.0, 2.0])
    stars = probs.expected_star(probs.ordinal_probs(np.linspace(-3, 3, 7), theta))
    assert np.all(np.diff(stars) > 0)


@pytest.mark.parametrize("make", ALL)
def test_temperature_one_changes_nothing_and_a_larger_one_softens(make):
    pred = make()
    assert np.allclose(pred.proba(1.0), pred.proba())
    assert (
        entropy(pred.proba(1.5)) > entropy(pred.proba(1.0)) > entropy(pred.proba(0.5))
    )


def test_a_huge_ordinal_temperature_piles_mass_on_the_open_ended_extremes_not_on_uniform():
    # (theta - score) / T -> 0 makes every cumulative probability 0.5, so P = (.5, 0, 0, 0, .5):
    # a wide latent distribution against fixed thresholds lands in the open-ended stars 1 and 5
    P = ordinal().proba(1e6)
    assert np.allclose(P, [0.5, 0.0, 0.0, 0.0, 0.5], atol=1e-3)


@pytest.mark.parametrize("make", ALL)
def test_temperature_must_be_positive_and_finite(make):
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="temperature"):
            make().proba(bad)


def test_classification_temperature_never_changes_the_argmax():
    pred = classification()
    assert np.array_equal(
        pred.proba(0.2).argmax(axis=1), pred.proba(4.0).argmax(axis=1)
    )


def test_softmax_is_stable_for_huge_logits():
    P = probs.softmax(np.array([[1000.0, 0.0, -1000.0, 5.0, 5.0]]))
    assert np.isfinite(P).all() and np.isclose(P.sum(), 1.0) and P[0, 0] == 1.0


@pytest.mark.parametrize("bad", [np.nan, np.inf, 0.0, -1.0])
def test_a_non_finite_or_non_positive_sigma_is_an_error_not_a_silent_fix(bad):
    with pytest.raises(ValueError, match="sigma"):
        probs.gaussian_probs(np.array([3.0, 3.0]), np.array([1.0, bad]))


def test_a_non_finite_mean_is_an_error():
    with pytest.raises(ValueError, match="mu"):
        probs.gaussian_probs(np.array([np.nan]), np.array([1.0]))


def test_a_tiny_but_valid_sigma_gives_a_finite_near_one_hot_vector():
    P = probs.gaussian_probs(np.array([3.0]), np.array([1e-6]))[0]
    assert np.isfinite(P).all() and P[2] > 0.999999


def test_ordinal_thresholds_must_be_increasing():
    with pytest.raises(ValueError, match="theta"):
        probs.ordinal_probs(np.zeros(2), np.array([0.0, 1.0, 0.5, 2.0]))


# ---------------------------------------------------------------- containers


@pytest.mark.parametrize("make", ALL)
def test_take_selects_rows_and_keeps_shared_thresholds(make):
    pred = make()
    sub = pred.take(np.array([3, 1, 7]))
    assert len(sub) == 3 and np.allclose(sub.proba(), pred.proba()[[3, 1, 7]])


@pytest.mark.parametrize("make", ALL)
def test_arrays_round_trip_without_pickle(make, tmp_path):
    preds = Predictions([make(seed=0), make(seed=1)])
    path = tmp_path / "p.npz"
    np.savez(path, **preds.to_arrays("x/"))
    with np.load(path, allow_pickle=False) as z:
        again = Predictions.from_arrays(dict(z), "x/")
    assert len(again.members) == 2 and again.framing == preds.framing
    assert np.allclose(again.proba(), preds.proba()) and np.allclose(
        again.proba(2.0), preds.proba(2.0)
    )


def test_an_ensemble_averages_member_probabilities_and_applies_one_temperature_to_each():
    a, b = classification(seed=0), classification(seed=1)
    ens = Predictions([a, b])
    assert np.allclose(ens.proba(), (a.proba() + b.proba()) / 2)
    assert np.allclose(ens.proba(2.0), (a.proba(2.0) + b.proba(2.0)) / 2)
    assert len(ens) == 40 and not ens.is_single and Predictions([a]).is_single


def test_members_of_an_ensemble_must_share_a_framing_and_a_length():
    with pytest.raises(ValueError, match="framing"):
        Predictions([classification(), regression()])
    with pytest.raises(ValueError, match="rows"):
        Predictions([classification(n=10), classification(n=11)])
    with pytest.raises(ValueError, match="at least one"):
        Predictions([])


def test_a_pred_rejects_missing_or_wrong_shaped_fields():
    with pytest.raises(ValueError, match="logits"):
        Pred("classification", logits=np.zeros((3, 4)))
    with pytest.raises(ValueError, match="framing"):
        Pred("nope", logits=np.zeros((3, 5)))
    with pytest.raises(ValueError, match="mu"):
        Pred("regression", sigma=np.ones(3))


def test_expected_star_and_top_label_helpers():
    P = np.array([[0.0, 0.0, 0.0, 0.0, 1.0], [0.5, 0.0, 0.0, 0.0, 0.5]])
    assert np.allclose(probs.expected_star(P), [5.0, 3.0])
    assert probs.predicted_star(P).tolist() == [5, 1]
