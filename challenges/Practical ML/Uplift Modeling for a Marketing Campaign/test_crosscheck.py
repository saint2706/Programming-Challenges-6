"""Independent oracles: scikit-uplift for the curves, causalml for the learners."""

import learners
import metrics
import numpy as np
import pytest

FAST = {"n_estimators": 60, "min_child_samples": 50, "num_leaves": 15}
N = 20_000


@pytest.fixture(scope="module")
def sample(rct):
    X, t, y, _ = rct
    score = np.random.default_rng(0).normal(size=N) + 0.5 * X[:N, 0]
    return X[:N], t[:N], y[:N], score


def test_qini_curve_matches_scikit_uplift(sample):
    sk = pytest.importorskip("sklift.metrics")
    _, t, y, score = sample
    x_sk, q_sk = sk.qini_curve(y, score, t)
    x, q, _ = metrics.curve(metrics.rank(score, t, y))
    n = len(y)
    assert np.allclose(x_sk / n, x)
    ok = np.isfinite(q_sk)  # scikit-uplift divides by an empty control group at the top
    assert ok.mean() > 0.99
    assert np.allclose(q_sk[ok] / n, q[ok], atol=1e-9)


def test_uplift_curve_matches_scikit_uplift(sample):
    sk = pytest.importorskip("sklift.metrics")
    _, t, y, score = sample
    _, u_sk = sk.uplift_curve(y, score, t)
    _, _, u = metrics.curve(metrics.rank(score, t, y))
    ok = np.isfinite(u_sk)
    assert ok.mean() > 0.99
    assert np.allclose(u_sk[ok] / len(y), u[ok], atol=1e-9)


def test_qini_coefficient_matches_the_area_of_scikit_uplifts_curve(sample):
    sk = pytest.importorskip("sklift.metrics")
    _, t, y, score = sample
    x_sk, q_sk = sk.qini_curve(y, score, t)
    ok = np.isfinite(q_sk)
    xs, qs = x_sk[ok] / len(y), q_sk[ok] / len(y)
    ref = np.trapezoid(qs - xs * qs[-1], xs)
    mine = metrics.qini_coefficient(metrics.rank(score, t, y))
    assert mine == pytest.approx(ref, rel=1e-3)


def test_tied_scores_match_scikit_uplift(sample):
    sk = pytest.importorskip("sklift.metrics")
    _, t, y, _ = sample
    score = np.random.default_rng(1).integers(0, 20, size=len(y)).astype(float)
    x_sk, q_sk = sk.qini_curve(y, score, t)
    x, q, _ = metrics.curve(metrics.rank(score, t, y))
    assert np.allclose(x_sk / len(y), x)
    ok = np.isfinite(q_sk)
    assert np.allclose(q_sk[ok] / len(y), q[ok], atol=1e-9)


def test_t_learner_matches_causalml(rct):
    meta = pytest.importorskip("causalml.inference.meta")
    X, t, y, _ = rct
    X, t, y = X[:N], t[:N], y[:N]
    mine = learners.TLearner(0.85, FAST, seed=0).fit(X, t, y).predict(X)
    ref = meta.BaseTClassifier(learner=learners._clf(FAST, 0))
    theirs = ref.fit_predict(X, t, y, verbose=False).ravel()
    assert np.corrcoef(mine, theirs)[0, 1] > 0.999


def test_x_learner_agrees_with_causalml(rct):
    meta = pytest.importorskip("causalml.inference.meta")
    X, t, y, _ = rct
    X, t, y = X[:N], t[:N], y[:N]
    mine = learners.XLearner(0.85, FAST, seed=0).fit(X, t, y).predict(X)
    ref = meta.BaseXClassifier(
        outcome_learner=learners._clf(FAST, 0), effect_learner=learners._reg(FAST, 0)
    )
    theirs = ref.fit_predict(X, t, y, p=np.full(len(y), 0.85), verbose=False).ravel()
    # observed 1.0000; a swapped propensity (e <-> 1-e) drops it to 0.955, so the floor must be tight
    assert np.corrcoef(mine, theirs)[0, 1] > 0.999


def test_s_learner_agrees_with_causalml(rct):
    meta = pytest.importorskip("causalml.inference.meta")
    X, t, y, _ = rct
    X, t, y = X[:N], t[:N], y[:N]
    mine = learners.SLearner(0.85, FAST, seed=0).fit(X, t, y).predict(X)
    ref = meta.BaseSClassifier(learner=learners._clf(FAST, 0))
    theirs = ref.fit_predict(X, t, y, verbose=False).ravel()
    # column order differs, so the two are not bit-identical (observed 0.995)
    assert np.corrcoef(mine, theirs)[0, 1] > 0.98


def test_dr_learner_agrees_loosely_with_causalml(rct):
    meta = pytest.importorskip("causalml.inference.meta")
    X, t, y, _ = rct
    X, t, y = X[:N], t[:N], y[:N]
    mine = learners.DRLearner(0.85, FAST, seed=0).fit(X, t, y).predict(X)
    ref = meta.BaseDRLearner(learner=learners._reg(FAST, 0), control_name=0)
    theirs = ref.fit_predict(X, t, y, p=np.full(len(y), 0.85), verbose=False).ravel()
    # causalml's DR uses regressors for the outcome nuisances and its own folds:
    # the rankings agree, the values do not (observed 0.93)
    assert np.corrcoef(mine, theirs)[0, 1] > 0.85
