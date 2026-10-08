import csv

import numpy as np
import pytest
from active_labeling import strategies
from active_labeling.model import Head
from active_labeling.project import Project
from helpers import make_problem

K = 4
CLASSES = [f"k{i}" for i in range(K)]


def build(path, rule=None, gold=True):
    prob = make_problem(n=200, k=K)
    p = Project.create(
        path,
        [f"item {i}" for i in range(200)],
        prob.X,
        CLASSES,
        gold=[CLASSES[c] for c in prob.y] if gold else None,
        C=prob.C,
        rule=rule,
        X_eval=prob.X_eval,
        y_eval=prob.y_eval,
    )
    return p, prob


@pytest.fixture
def proj(tmp_path):
    return build(tmp_path / "proj")


def labels_for(prob, ids):
    return {i: CLASSES[prob.y[i]] for i in ids}


def test_create_refuses_an_existing_folder_and_opening_a_missing_one_says_how_to_init(
    proj, tmp_path
):
    p, prob = proj
    with pytest.raises(FileExistsError):
        Project.create(p.dir, ["a", "b"], prob.X[:2], CLASSES)
    with pytest.raises(FileNotFoundError, match="init"):
        Project(tmp_path / "nope")


def test_cold_start_suggestions_are_valid_for_every_strategy(proj):
    p, _ = proj
    for name in strategies.NAMES:
        sel, reasons, _ = p.suggest(name, 8)
        assert len(sel.idx) == 8 and len(set(sel.idx.tolist())) == 8
        assert len(reasons) == 8 and all(r.neighbors == [] for r in reasons)


def test_suggest_rejects_a_batch_below_one_and_unknown_strategies(proj):
    p, _ = proj
    with pytest.raises(ValueError, match="at least 1"):
        p.suggest("margin", 0)
    with pytest.raises(ValueError, match="unknown strategy"):
        p.suggest("nope", 5)


def test_cluster_margin_caches_its_clusters_in_the_project_folder(proj):
    p, _ = proj
    p.suggest("cluster-margin", 5)
    assert (p.dir / "clusters-150.npy").exists()
    assert len(p.suggest("cluster-margin", 5)[0].idx) == 5


def test_suggestions_skip_labeled_items_and_neighbors_come_from_the_labeled_set(proj):
    p, prob = proj
    first = list(range(30))
    p.submit(labels_for(prob, first))
    sel, reasons, _ = p.suggest("margin", 6)
    assert not set(sel.idx.tolist()) & set(first)
    for r in reasons:
        assert r.neighbors
        assert all(j in first and c == prob.y[j] for j, c, _ in r.neighbors)


def test_a_round_is_recorded_once_per_change_with_the_prediction_change_fraction(proj):
    p, prob = proj
    assert p.refresh() is None and p.store.rounds() == []
    p.submit(labels_for(prob, range(10)))
    rounds = p.store.rounds()
    assert len(rounds) == 1 and rounds[0]["change"] is None
    assert rounds[0]["n_labeled"] == 10 and rounds[0]["accuracy"] is not None
    assert p.refresh() is None and len(p.store.rounds()) == 1
    p.submit(labels_for(prob, range(10, 25)))
    second = p.store.rounds()[1]
    before = Head(K, p.C).fit(p.X[:10], prob.y[:10]).predict(p.X)
    after = Head(K, p.C).fit(p.X[:25], prob.y[:25]).predict(p.X)
    assert np.isclose(second["change"], (before != after).mean())


def test_relabeling_records_a_new_round_and_keeps_the_history(proj):
    p, prob = proj
    p.submit(labels_for(prob, range(12)))
    p.submit({0: CLASSES[(prob.y[0] + 1) % K]})
    rounds = p.store.rounds()
    assert len(rounds) == 2 and rounds[1]["n_labeled"] == 12
    assert len(p.store.history(0)) == 2
    assert p.labeled()[1][0] == (prob.y[0] + 1) % K


def test_an_invalid_batch_writes_nothing_and_records_no_round(proj):
    p, _ = proj
    with pytest.raises(ValueError, match="not one of"):
        p.submit({0: "k0", 1: "zzz"})
    with pytest.raises(ValueError, match="no item"):
        p.submit({0: "k0", 500: "k1"})
    assert p.store.current_labels() == {} and p.store.rounds() == []
    assert p.submit({}) is None


def test_status_reports_coverage_and_fires_the_stopping_signal(tmp_path):
    p, prob = build(tmp_path / "p", rule={"threshold": 1.01, "k": 1, "spacing": 1})
    p.submit(labels_for(prob, range(12)))
    s = p.status()
    assert (s["items"], s["labeled"], s["classes"], s["classes_with_label"]) == (
        200,
        12,
        4,
        4,
    )
    assert not s["stop"]["fired"] and len(s["rounds"]) == 1
    p.submit(labels_for(prob, range(12, 20)))
    s = p.status()
    assert s["stop"] == {"fired": True, "at_labels": 20} and sum(s["counts"]) == 20


def test_simulate_uses_the_gold_labels_and_refuses_without_them(proj, tmp_path):
    p, prob = proj
    p.simulate([3, 4, 5])
    assert p.store.current_labels() == labels_for(prob, [3, 4, 5])
    bare, _ = build(tmp_path / "bare", gold=False)
    assert not bare.has_gold and p.has_gold
    with pytest.raises(ValueError, match="no gold"):
        bare.simulate([0])


def test_export_lists_every_item_with_its_current_label_and_gold(proj, tmp_path):
    p, prob = proj
    p.submit(labels_for(prob, [2]))
    out = tmp_path / "out.csv"
    p.export(out)
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert len(rows) == 200 and rows[2]["label"] == CLASSES[prob.y[2]]
    assert rows[3]["label"] == "" and rows[3]["gold"] == CLASSES[prob.y[3]]


def test_single_label_rounds_do_not_fake_a_stopping_signal(tmp_path):
    """The rule was tuned on 50-label rounds: tiny rounds change few predictions by construction."""
    p, prob = build(tmp_path / "p", rule={"threshold": 0.01, "k": 2, "spacing": 20})
    p.submit(labels_for(prob, range(20)))
    p.submit(labels_for(prob, range(20, 40)))
    p.submit(labels_for(prob, range(40, 60)))
    for i in range(60, 65):  # the one-label-at-a-time CLI workflow
        p.submit(labels_for(prob, [i]))
    s = p.status()
    assert len(s["rounds"]) == 8 and s["rule"]["spacing"] == 20
    picked = [r["n_labeled"] for r in s["signal_rounds"]]
    assert picked == [20, 40, 60]  # only rounds at least `spacing` labels apart count
    single = s["rounds"][-3:]
    assert all(
        r["change"] is not None and r["change"] < 0.01 for r in single
    )  # tiny, as the reviewer saw
    assert s["stop"]["at_labels"] != 65


def test_the_spacing_defaults_to_the_benchmark_batch_size(tmp_path):
    p, _ = build(tmp_path / "q")
    assert p.rule["spacing"] == 50
