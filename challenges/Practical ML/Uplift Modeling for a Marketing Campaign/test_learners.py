import learners
import numpy as np
import pytest
from helpers import confounded_rct
from scipy.stats import spearmanr

FAST = {"n_estimators": 80, "min_child_samples": 50, "num_leaves": 15}
NAMES = list(learners.LEARNERS)
SPLIT = 40_000


@pytest.fixture(scope="module")
def fitted(rct):
    X, t, y, tau = rct
    models = {
        name: cls(0.85, FAST, seed=0).fit(X[:SPLIT], t[:SPLIT], y[:SPLIT])
        for name, cls in {**learners.LEARNERS, **learners.BASELINES}.items()
    }
    return models, X[SPLIT:], tau[SPLIT:]


def test_t_is_the_headline_learner_and_listed_first():
    assert NAMES[0] == "T" and set(NAMES) == {"T", "S", "X", "TO", "DR"}


@pytest.mark.parametrize("name", NAMES)
def test_learner_ranks_customers_by_the_true_effect(fitted, name):
    models, Xte, tau = fitted
    pred = models[name].predict(Xte)
    assert pred.shape == (len(Xte),) and np.isfinite(pred).all()
    assert spearmanr(pred, tau).statistic > 0.2


def test_random_baseline_is_uncorrelated_with_the_effect(fitted):
    models, Xte, tau = fitted
    assert abs(spearmanr(models["random"].predict(Xte), tau).statistic) < 0.05


def test_random_baseline_is_seeded():
    a = learners.RandomScorer(seed=3).predict(np.zeros((10, 2)))
    b = learners.RandomScorer(seed=3).predict(np.zeros((10, 2)))
    assert np.array_equal(a, b)


def test_response_model_predicts_a_probability(fitted):
    models, Xte, _ = fitted
    p = models["response"].predict(Xte)
    assert ((p >= 0) & (p <= 1)).all()


def test_same_seed_same_predictions(rct):
    X, t, y, _ = rct
    a = learners.TLearner(0.85, FAST, seed=1).fit(X[:5000], t[:5000], y[:5000])
    b = learners.TLearner(0.85, FAST, seed=1).fit(X[:5000], t[:5000], y[:5000])
    assert np.allclose(a.predict(X[:200]), b.predict(X[:200]))


@pytest.mark.parametrize("cls", list(learners.LEARNERS.values()))
def test_an_empty_arm_raises_a_clear_error(cls, rct):
    X, _, y, _ = rct
    with pytest.raises(ValueError, match="control"):
        cls(0.85, FAST).fit(X[:2000], np.ones(2000, dtype=np.int8), y[:2000])


@pytest.mark.parametrize("cls", list(learners.LEARNERS.values()))
def test_an_arm_with_a_single_outcome_class_raises_a_clear_error(cls, rct):
    X, t, y, _ = rct
    y = y[:2000].copy()
    y[t[:2000] == 1] = 0  # the treated arm never converts
    with pytest.raises(ValueError, match="single outcome class"):
        cls(0.85, FAST).fit(X[:2000], t[:2000], y)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_propensity_must_be_strictly_between_zero_and_one(bad):
    with pytest.raises(ValueError, match="propensity"):
        learners.TLearner(bad)


class FixedPropensity:
    """A stand-in propensity model: e(x) is a known function of the first feature."""

    def predict(self, X):
        return 0.5 + 0.4 * np.tanh(2 * np.asarray(X)[:, 0])


def test_a_learner_accepts_a_propensity_model_in_place_of_a_constant():
    X = np.zeros((3, 12))
    X[:, 0] = [-10.0, 0.0, 10.0]
    model_based = learners.TLearner(FixedPropensity())._e(X)
    assert model_based == pytest.approx([0.1, 0.5, 0.9], abs=1e-6)
    assert learners.TLearner(0.85)._e(X) == pytest.approx([0.85] * 3)


def test_the_transformed_outcome_needs_the_true_propensity_when_treatment_is_confounded():
    _, t, y, e = confounded_rct()
    right = learners.transformed_outcome(y, t, e).mean()
    wrong = learners.transformed_outcome(y, t, np.full(len(y), t.mean())).mean()
    assert right == pytest.approx(0.02, abs=0.008)  # the true effect
    assert wrong > 0.08  # a constant e absorbs the confounding into the "effect"


def test_a_learner_given_the_propensity_model_is_not_fooled_by_confounding():
    x, t, y, _ = confounded_rct(n=200_000)
    X = np.column_stack([x, np.random.default_rng(0).normal(size=(len(x), 11))])
    X = X.astype(np.float32)
    fast = {"n_estimators": 60, "min_child_samples": 200, "num_leaves": 15}
    with_model = learners.TransformedOutcome(FixedPropensity(), fast).fit(X, t, y)
    constant = learners.TransformedOutcome(float(t.mean()), fast).fit(X, t, y)
    assert abs(with_model.predict(X).mean() - 0.02) < 0.03
    assert constant.predict(X).mean() > 0.07
