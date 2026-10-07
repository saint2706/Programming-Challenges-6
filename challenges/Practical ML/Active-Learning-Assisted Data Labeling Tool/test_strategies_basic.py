import numpy as np
import pytest
import strategies
from helpers import assert_valid, make_state
from strategies import (
    Random,
    Selection,
    State,
    Uncertainty,
    entropy,
    least_confidence,
    margins,
    top_positions,
)

P3 = np.array([[0.5, 0.3, 0.2], [0.9, 0.05, 0.05], [1.0, 0.0, 0.0]])


def test_uncertainty_formulas_match_the_textbook():
    assert np.allclose(least_confidence(P3), [0.5, 0.1, 0.0])
    assert np.allclose(margins(P3), [0.2, 0.85, 1.0])
    first = -(0.5 * np.log(0.5) + 0.3 * np.log(0.3) + 0.2 * np.log(0.2))
    assert np.allclose(entropy(P3), [first, entropy(P3)[1], 0.0])
    assert np.isfinite(entropy(P3)).all()  # 0 * log 0 must not produce NaN


def test_state_unlabeled_is_the_sorted_complement():
    state = State(
        X=np.zeros((6, 2)),
        labeled=np.array([4, 1]),
        y=np.array([0, 1]),
        P=np.full((6, 2), 0.5),
        n_classes=2,
        C=1.0,
    )
    assert state.unlabeled().tolist() == [0, 2, 3, 5]


def test_top_positions_breaks_ties_at_random_not_by_pool_order():
    flat = np.zeros(50)
    a = top_positions(flat, 5, np.random.default_rng(0))
    b = top_positions(flat, 5, np.random.default_rng(1))
    assert a.tolist() == top_positions(flat, 5, np.random.default_rng(0)).tolist()
    assert set(a) != set(b)
    assert set(a) != {0, 1, 2, 3, 4}
    assert top_positions(
        np.array([1.0, 3.0, 2.0]), 2, np.random.default_rng(0)
    ).tolist() == [1, 2]


@pytest.mark.parametrize("name", ["random", "least-confidence", "margin", "entropy"])
def test_basic_strategies_meet_the_contract(name):
    state, _ = make_state()
    strat = strategies.make(name)
    assert strat.name == name
    for b in (1, 10, 500):  # 500 exceeds the 270 unlabeled items
        assert_valid(strat.select(state, b, np.random.default_rng(0)), state, b)


@pytest.mark.parametrize("name", ["random", "least-confidence", "margin", "entropy"])
def test_basic_strategies_are_reproducible_per_rng_and_return_empty_when_nothing_is_left(
    name,
):
    state, _ = make_state()
    strat = strategies.make(name)
    a = strat.select(state, 10, np.random.default_rng(4)).idx
    assert a.tolist() == strat.select(state, 10, np.random.default_rng(4)).idx.tolist()
    full = State(
        X=state.X,
        labeled=np.arange(len(state.X)),
        y=np.zeros(len(state.X), dtype=np.int64),
        P=state.P,
        n_classes=state.n_classes,
        C=state.C,
    )
    sel = strat.select(full, 5, np.random.default_rng(0))
    assert len(sel.idx) == 0 and len(sel.score) == 0


@pytest.mark.parametrize(
    "name,utility",
    [
        ("least-confidence", least_confidence),
        ("margin", lambda P: -margins(P)),
        ("entropy", entropy),
    ],
)
def test_uncertainty_picks_exactly_the_top_b_utilities_among_unlabeled(name, utility):
    state, _ = make_state()
    unl = state.unlabeled()
    u = utility(state.P[unl])
    sel = Uncertainty(name).select(state, 12, np.random.default_rng(0))
    assert set(sel.idx) == set(unl[np.argsort(-u)[:12]])
    assert np.allclose(np.sort(sel.score)[::-1], np.sort(u)[::-1][:12])


def test_random_ignores_the_model_and_selection_empty_is_empty():
    state, _ = make_state()
    sel = Random().select(state, 20, np.random.default_rng(0))
    assert np.all(sel.score == 0)
    assert len(Selection.empty().idx) == 0


def test_make_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown strategy"):
        strategies.make("nope")


def test_uncertainty_picks_items_the_model_gets_wrong_more_often_than_the_pool():
    """The grounding claim the benchmark reports: suggested items are the ones it fails on."""
    picked, pool = [], []
    for seed in range(5):
        state, gold = make_state(spread=0.9, seed=seed, n_labeled=30)
        unl = state.unlabeled()
        pred = state.P.argmax(axis=1)
        sel = strategies.make("margin").select(state, 20, np.random.default_rng(seed))
        picked.append((pred[sel.idx] != gold[sel.idx]).mean())
        pool.append((pred[unl] != gold[unl]).mean())
    assert np.mean(picked) > np.mean(pool) + 0.1


@pytest.mark.parametrize(
    "name,method",
    [
        ("least-confidence", "least_confident"),
        ("margin", "margin_sampling"),
        ("entropy", "entropy"),
    ],
)
def test_matches_scikit_activeml_utilities_up_to_a_constant(name, method):
    pytest.importorskip("skactiveml")
    from skactiveml.classifier import SklearnClassifier
    from skactiveml.pool import UncertaintySampling
    from sklearn.linear_model import LogisticRegression

    state, _ = make_state(n=120, k=4, n_labeled=24, seed=2)
    gold_partial = np.full(len(state.X), np.nan)
    gold_partial[state.labeled] = state.y
    clf = SklearnClassifier(
        LogisticRegression(C=state.C, max_iter=300),
        classes=list(range(4)),
        random_state=0,
    ).fit(state.X, gold_partial)
    assert np.allclose(clf.predict_proba(state.X), state.P, atol=1e-6)
    unl = state.unlabeled()
    idx, util = UncertaintySampling(method=method, random_state=0).query(
        state.X,
        gold_partial,
        clf=clf,
        fit_clf=False,
        batch_size=1,
        return_utilities=True,
    )
    theirs = util[0][unl]
    ours = Uncertainty(name)._utility(state.P[unl])
    assert np.allclose(ours - ours.mean(), theirs - theirs.mean(), atol=1e-8)
    assert int(idx[0]) == int(unl[np.argmax(ours)])
