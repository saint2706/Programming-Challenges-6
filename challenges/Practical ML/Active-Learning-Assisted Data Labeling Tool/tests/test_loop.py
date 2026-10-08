from dataclasses import replace

import numpy as np
from active_labeling import strategies
from active_labeling.loop import METRICS, initial_labels, run_loop
from active_labeling.model import Head
from helpers import make_problem


def run(name="margin", prob=None, seed=0, **kw):
    kw = {"budget": 30, "b": 10, "init": 12} | kw
    return run_loop(
        prob or make_problem(), strategies.make(name, n_clusters=10), seed, **kw
    )


def test_curve_has_one_point_per_round_and_ends_at_init_plus_budget():
    c = run()
    assert c["n"] == [12, 22, 32, 42]
    for key in METRICS:
        assert len(c[key]) == 4
    assert np.isnan(c["change"][0]) and not np.isnan(c["change"][1:]).any()
    assert np.isnan(c["sel_err"][-1]) and np.isnan(c["sel_time"][-1])
    assert [len(p) for p in c["picked"]] == [10, 10, 10]


def test_a_batch_that_does_not_divide_the_budget_is_clipped():
    c = run(b=7, budget=20)
    assert c["n"] == [12, 19, 26, 32]
    assert [len(p) for p in c["picked"]] == [7, 7, 6]


def test_picks_are_unique_and_never_relabel_the_initial_set():
    prob = make_problem()
    c = run("badge", prob)
    flat = [i for batch in c["picked"] for i in batch]
    init = initial_labels(len(prob.X), 12, 0).tolist()
    assert len(set(flat + init)) == len(flat) + len(init)


def test_the_budget_is_clipped_to_what_the_pool_can_supply():
    prob = make_problem(n=60)
    c = run("random", prob, budget=500)
    assert c["n"][-1] == 60
    picked = sorted(i for batch in c["picked"] for i in batch)
    assert picked == sorted(set(range(60)) - set(initial_labels(60, 12, 0).tolist()))


def test_strategies_share_a_seeds_initial_labels_so_runs_are_paired():
    prob = make_problem()
    a, b = run("random", prob), run("margin", prob)
    assert a["n"] == b["n"] and a["acc"][0] == b["acc"][0] and a["f1"][0] == b["f1"][0]
    assert a["coverage"][0] == b["coverage"][0]
    assert a["picked"] != b["picked"]


def test_runs_are_reproducible_per_seed_and_differ_across_seeds():
    prob = make_problem()
    a, b = run("qbc", prob, seed=2), run("qbc", prob, seed=2)
    for key in METRICS:
        if key != "sel_time":
            assert np.allclose(a[key], b[key], equal_nan=True)
    assert a["picked"] == b["picked"]
    assert run("qbc", prob, seed=3)["picked"] != a["picked"]


def test_evaluation_labels_never_influence_what_is_picked():
    prob = make_problem()
    shuffled = replace(
        prob, y_eval=np.roll(prob.y_eval, 7), y_val=np.roll(prob.y_val, 3)
    )
    a, b = run("margin", prob), run("margin", shuffled)
    assert a["picked"] == b["picked"]
    assert a["acc"] != b["acc"]


def test_first_round_metrics_follow_from_the_initial_labels():
    prob = make_problem()
    c = run("margin", prob)
    init = initial_labels(len(prob.X), 12, 0)
    counts = np.bincount(prob.y[init], minlength=prob.n_classes)
    assert c["coverage"][0] == (counts > 0).sum()
    assert np.isclose(c["skew"][0], counts.max() / 12)
    head = Head(prob.n_classes, prob.C).fit(prob.X[init], prob.y[init])
    pred = head.predict(prob.X)
    first = c["picked"][0]
    assert np.isclose(c["sel_err"][0], (pred[first] != prob.y[first]).mean())
    unl = np.setdiff1d(np.arange(len(prob.X)), init)
    assert np.isclose(c["pool_err"][0], (pred[unl] != prob.y[unl]).mean())
    assert np.isclose(c["acc"][0], (head.predict(prob.X_eval) == prob.y_eval).mean())


def test_labeling_more_items_improves_held_out_accuracy_on_average():
    gains = []
    for seed in range(5):
        c = run("random", make_problem(spread=0.9), seed=seed, budget=60, b=12)
        gains.append(c["acc"][-1] - c["acc"][0])
    assert np.mean(gains) > 0
