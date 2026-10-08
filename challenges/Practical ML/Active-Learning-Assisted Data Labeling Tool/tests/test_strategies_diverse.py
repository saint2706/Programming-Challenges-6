import numpy as np
import pytest
from active_labeling import strategies
from active_labeling.model import Head
from active_labeling.strategies import (
    Badge,
    ClusterMargin,
    KCenter,
    QueryByCommittee,
    Random,
    State,
    cluster_pool,
    gradient_parts,
    gradient_sqdist,
    vote_entropy,
)
from helpers import assert_valid, make_pool, make_state
from scipy.spatial.distance import cdist

EMPTY = np.array([], dtype=np.int64)


def rng(seed=0):
    return np.random.default_rng(seed)


def cold_state(X, k):
    return State(X, EMPTY, EMPTY, np.full((len(X), k), 1.0 / k), k, 10.0)


@pytest.mark.parametrize("name", strategies.NAMES)
def test_every_strategy_meets_the_contract(name):
    state, _ = make_state(n=300, k=6)
    strat = strategies.make(name, n_clusters=20)
    assert strat.name == name
    for b in (1, 10, 500):  # 500 exceeds the 270 unlabeled items
        assert_valid(strat.select(state, b, rng(0)), state, b)


@pytest.mark.parametrize("name", strategies.NAMES)
def test_every_strategy_survives_cold_start_one_class_last_items_and_empty_pool(name):
    X, y = make_pool(n=60, k=4)
    strat = strategies.make(name, n_clusters=8)
    cold = cold_state(X, 4)
    assert_valid(strat.select(cold, 7, rng(0)), cold, 7)

    one = np.array([0, 1, 2])  # items 0..2 are all class 0
    single = State(X, one, y[one], Head(4).fit(X[one], y[one]).proba(X), 4, 10.0)
    assert_valid(strat.select(single, 7, rng(0)), single, 7)

    nearly = np.arange(57)
    P = Head(4).fit(X[nearly], y[nearly]).proba(X)
    last = State(X, nearly, y[nearly], P, 4, 10.0)
    assert_valid(strat.select(last, 10, rng(0)), last, 10)  # only 3 left

    done = State(X, np.arange(60), y, P, 4, 10.0)
    sel = strat.select(done, 5, rng(0))
    assert len(sel.idx) == 0 and len(sel.score) == 0


def brute_kcenter(X, centers, cand, b):
    centers, chosen = list(centers), []
    for _ in range(min(b, len(cand))):
        best, best_d = None, -1.0
        for i in cand:
            if i in chosen:
                continue
            d = min(((X[i] - X[c]) ** 2).sum() for c in centers)
            if d > best_d:
                best, best_d = i, d
        chosen.append(best)
        centers.append(best)
    return np.array(chosen)


def test_kcenter_matches_a_brute_force_farthest_first():
    state, _ = make_state(n=120, k=5, n_labeled=15)
    sel = KCenter().select(state, 12, rng(0))
    ref = brute_kcenter(state.X, state.labeled, state.unlabeled(), 12)
    assert sel.idx.tolist() == ref.tolist()


def test_kcenter_with_nothing_labeled_starts_at_random_then_goes_farthest():
    X, _ = make_pool(80, 4)
    sel = KCenter().select(cold_state(X, 4), 10, rng(3))
    first = int(sel.idx[0])
    ref = brute_kcenter(X, [first], np.setdiff1d(np.arange(80), [first]), 9)
    assert sel.idx[1:].tolist() == ref.tolist()


def test_kcenter_matches_scikit_activeml_coreset():
    pytest.importorskip("skactiveml")
    from skactiveml.pool import CoreSet

    state, _ = make_state(n=120, k=5, n_labeled=15)
    partial = np.full(len(state.X), np.nan)
    partial[state.labeled] = state.y
    theirs = CoreSet(random_state=0).query(state.X, partial, batch_size=12)
    assert KCenter().select(state, 12, rng(0)).idx.tolist() == theirs.tolist()


def test_kcenter_covers_more_intents_than_random_on_an_imbalanced_pool():
    k_cov, r_cov = [], []
    for seed in range(10):
        X, y = make_pool(600, 10, seed=seed)
        g = rng(seed)
        first = np.array([int(g.integers(600))])
        state = State(X, first, y[first], np.full((600, 10), 0.1), 10, 10.0)
        k = KCenter().select(state, 30, g).idx
        r = Random().select(state, 30, g).idx
        k_cov.append(len(set(y[k]) | {y[first[0]]}))
        r_cov.append(len(set(y[r]) | {y[first[0]]}))
    assert np.mean(k_cov) >= np.mean(r_cov) + 0.5


def explicit_gradients(P, X):
    A, H = gradient_parts(P, X)
    return A, H, np.stack([np.kron(A[i], H[i]) for i in range(len(A))])


def test_kronecker_distances_equal_explicit_gradient_embedding_distances():
    state, _ = make_state(n=60, k=3, n_labeled=12)
    A, H, G = explicit_gradients(state.P, state.X)
    D = cdist(G, G, "sqeuclidean")
    for c in (0, 7, 33):
        assert np.allclose(gradient_sqdist(A, H, c), D[:, c])


def test_gradient_embedding_is_p_minus_onehot_of_the_prediction():
    state, _ = make_state(n=60, k=3, n_labeled=12)
    A, H = gradient_parts(state.P, state.X)
    yhat = state.P.argmax(axis=1)
    expected = state.P.copy()
    expected[np.arange(60), yhat] -= 1.0
    assert np.allclose(A, expected) and np.array_equal(H, state.X)


def test_badge_equals_kmeanspp_on_the_explicit_gradient_embeddings():
    state, _ = make_state(n=80, k=4, n_labeled=16)
    cand = state.unlabeled()
    _, _, G = explicit_gradients(state.P, state.X)
    G = G[cand]

    def reference(b, g):
        chosen = [int(np.argmax((G**2).sum(axis=1)))]
        mind = cdist(G, G[[chosen[0]]], "sqeuclidean")[:, 0]
        while len(chosen) < b:
            w = mind.copy()
            w[chosen] = 0.0
            j = int(g.choice(len(G), p=w / w.sum()))
            chosen.append(j)
            mind = np.minimum(mind, cdist(G, G[[j]], "sqeuclidean")[:, 0])
        return cand[np.array(chosen)]

    sel = Badge().select(state, 10, rng(7))
    assert sel.idx.tolist() == reference(10, rng(7)).tolist()


def test_badge_with_identical_items_falls_back_to_uniform_instead_of_dividing_by_zero():
    X = np.tile(np.ones(3) / np.sqrt(3), (10, 1))
    sel = Badge().select(cold_state(X, 3), 5, rng(0))
    assert len(set(sel.idx.tolist())) == 5


def test_vote_entropy_matches_hand_computation():
    votes = np.array([[5.0, 0, 0], [3, 2, 0], [1, 1, 3]])
    assert np.allclose(vote_entropy(votes), [0.0, 0.6730117, 0.9502705], atol=1e-6)


def test_qbc_scores_are_the_committees_vote_entropy_and_top_b_win():
    state, _ = make_state(spread=0.9)
    qbc = QueryByCommittee(members=5)
    votes = qbc.votes(state, rng(5))
    assert np.allclose(votes.sum(axis=1), 5)
    sel = qbc.select(state, 15, rng(5))
    top = np.sort(vote_entropy(votes))[::-1][:15]
    assert np.allclose(np.sort(sel.score)[::-1], top)


def test_cluster_pool_separates_well_separated_blobs_and_bounds_the_cluster_count():
    g = rng(0)
    labels = np.repeat([0, 1, 2], 20)
    X = np.eye(3)[labels] * 10 + 0.1 * g.normal(size=(60, 3))
    got = cluster_pool(X, 3)
    assert len(set(got.tolist())) == 3
    for blob in range(3):
        assert len(set(got[labels == blob].tolist())) == 1
    many = cluster_pool(X, 500)
    assert many.min() == 0 and many.max() < 60


def margin_state(n=12):
    m = 0.01 * (np.arange(n) + 1)  # item i has margin 0.01 * (i + 1)
    P = np.stack([0.5 + m / 2, 0.5 - m / 2, np.zeros(n)], axis=1)
    return State(rng(0).normal(size=(n, 2)), EMPTY, EMPTY, P, 3, 1.0)


def test_cluster_margin_round_robins_over_clusters_smallest_first():
    clusters = np.array([0] * 6 + [1] * 3 + [2] * 3)
    sel = ClusterMargin(factor=3, clusters=clusters).select(margin_state(), 4, rng(0))
    assert sel.idx.tolist() == [6, 9, 0, 7]


def test_cluster_margin_only_considers_the_factor_times_b_smallest_margins():
    clusters = np.array([0] * 6 + [1] * 3 + [2] * 3)
    sel = ClusterMargin(factor=1, clusters=clusters).select(margin_state(), 4, rng(0))
    assert set(sel.idx.tolist()) == {0, 1, 2, 3}


def test_cluster_margin_spans_more_clusters_than_a_plain_top_margin_batch():
    cm, plain = [], []
    for seed in range(5):
        state, _ = make_state(spread=0.9, seed=seed)
        clusters = cluster_pool(state.X, 20)
        a = ClusterMargin(clusters=clusters).select(state, 20, rng(seed)).idx
        b = strategies.make("margin").select(state, 20, rng(seed)).idx
        cm.append(len(set(clusters[a].tolist())))
        plain.append(len(set(clusters[b].tolist())))
    assert np.mean(cm) > np.mean(plain)


def test_make_builds_every_name_and_cluster_margin_keeps_given_clusters():
    clusters = np.arange(300) % 7
    assert strategies.make("cluster-margin", clusters=clusters).clusters is clusters
    assert [strategies.make(n).name for n in strategies.NAMES] == list(strategies.NAMES)
