import numpy as np
from explain import explain, render
from helpers import make_state
from strategies import Selection, State

EMPTY = np.array([], dtype=np.int64)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_neighbors_equal_a_brute_force_search_over_the_labeled_items():
    state, _ = make_state(n=200, k=5, n_labeled=25)
    unl = state.unlabeled()
    sel = Selection(unl[:6], np.zeros(6))
    for r in explain(state, sel):
        sims = sorted(
            (
                (cos(state.X[r.idx], state.X[j]), int(j), int(c))
                for j, c in zip(state.labeled, state.y, strict=True)
            ),
            reverse=True,
        )[:3]
        assert [n[0] for n in r.neighbors] == [s[1] for s in sims]
        assert [n[1] for n in r.neighbors] == [s[2] for s in sims]
        assert np.allclose([n[2] for n in r.neighbors], [s[0] for s in sims])
        assert np.isclose(r.nearest_sim, sims[0][0])


def test_top_classes_and_margin_come_from_the_probabilities():
    state, _ = make_state()
    sel = Selection(state.unlabeled()[:4], np.zeros(4))
    for r in explain(state, sel):
        p = state.P[r.idx]
        order = np.argsort(-p)[:3]
        assert [c for c, _ in r.top] == order.tolist()
        assert np.allclose([q for _, q in r.top], p[order])
        assert np.isclose(r.margin, p[order[0]] - p[order[1]])


def test_an_item_far_from_everything_labeled_is_flagged_novel_and_a_twin_is_not():
    state, _ = make_state(n=200, k=5, n_labeled=25)
    X = state.X.copy()
    far = int(state.unlabeled()[0])
    X[far] = -X[state.labeled].mean(axis=0)  # opposite side of the labeled mass
    twin = int(state.unlabeled()[1])
    X[twin] = X[state.labeled[0]]
    state = State(X, state.labeled, state.y, state.P, state.n_classes, state.C)
    got = {
        r.idx: r for r in explain(state, Selection(np.array([far, twin]), np.zeros(2)))
    }
    assert got[far].novel and not got[twin].novel
    assert np.isclose(got[twin].nearest_sim, 1.0)
    assert got[far].nearest_sim < got[twin].nearest_sim


def test_nothing_labeled_yet_has_no_neighbors_and_is_not_novel():
    state, _ = make_state()
    cold = State(state.X, EMPTY, EMPTY, np.full(state.P.shape, 1 / 6), 6, 10.0)
    r = explain(cold, Selection(np.array([3]), np.zeros(1)))[0]
    assert r.neighbors == [] and r.nearest_sim is None and not r.novel


def test_cluster_info_reports_pool_size_and_labels_already_in_the_cluster():
    state, _ = make_state(n=100, k=4, n_labeled=12)
    clusters = np.arange(100) % 5
    i = int(state.unlabeled()[0])
    r = explain(state, Selection(np.array([i]), np.zeros(1)), clusters=clusters)[0]
    assert r.cluster == {
        "id": int(clusters[i]),
        "size": 20,
        "labeled": int((clusters[state.labeled] == clusters[i]).sum()),
    }
    assert explain(state, Selection(np.array([i]), np.zeros(1)))[0].cluster is None


def test_render_names_classes_snippets_and_the_novelty_note():
    state, _ = make_state(n=100, k=4, n_labeled=12)
    classes = ["alpha", "beta", "gamma", "delta"]
    texts = [f"message number {i}" for i in range(100)]
    reasons = explain(
        state,
        Selection(state.unlabeled()[:2], np.zeros(2)),
        clusters=np.arange(100) % 5,
    )
    out = render(reasons[0], classes, texts)
    assert reasons[0].top[0][0] in range(4) and classes[reasons[0].top[0][0]] in out
    assert f"margin {reasons[0].margin:.2f}" in out
    assert "message number" in out and "cluster" in out
    assert "message number" not in render(reasons[0], classes)
    reasons[0].novel = True
    assert "novel" in render(reasons[0], classes, texts)
    cold = explain(
        State(state.X, EMPTY, EMPTY, np.full(state.P.shape, 0.25), 4, 10.0),
        Selection(np.array([0]), np.zeros(1)),
    )
    assert "nothing labeled yet" in render(cold[0], classes)
