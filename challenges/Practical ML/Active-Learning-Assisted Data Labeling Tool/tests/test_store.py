import numpy as np
import pytest
from active_labeling.store import Store

CLASSES = ["a", "b", "c"]


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "p.db")
    s.set_classes(CLASSES)
    s.add_items(["t0", "t1", "t2", "t3"], gold=["a", "b", None, "c"])
    return s


def test_items_round_trip_with_gold(store):
    assert store.texts() == ["t0", "t1", "t2", "t3"]
    assert store.gold() == ["a", "b", None, "c"]
    assert store.n_items() == 4 and store.classes() == CLASSES


def test_items_can_be_added_once_and_gold_must_be_a_known_class(tmp_path):
    s = Store(tmp_path / "p.db")
    s.set_classes(CLASSES)
    with pytest.raises(ValueError, match="gold label 'z'"):
        s.add_items(["x", "y"], gold=["a", "z"])
    assert s.n_items() == 0
    s.add_items(["x", "y"])
    with pytest.raises(ValueError, match="already has items"):
        s.add_items(["z"])


def test_last_label_wins_and_the_history_is_kept(store):
    store.label(0, "a")
    store.label(0, "b")
    store.label(1, "c")
    assert store.current_labels() == {0: "b", 1: "c"}
    assert [lab for lab, _ in store.history(0)] == ["a", "b"]
    assert store.history(2) == []


def test_labels_outside_the_class_list_or_for_missing_items_are_rejected(store):
    with pytest.raises(ValueError, match="not one of"):
        store.label(0, "zzz")
    with pytest.raises(ValueError, match="no item 99"):
        store.label(99, "a")
    with pytest.raises(ValueError, match="no item -1"):
        store.label(-1, "a")
    assert store.current_labels() == {}


def test_rounds_round_trip_and_survive_reopening(tmp_path):
    path = tmp_path / "p.db"
    s = Store(path)
    s.set_classes(CLASSES)
    s.add_round(5, 2, None, None, np.array([0, 1, 2, 1]), "sig1")
    s.add_round(8, 3, 0.25, 0.8, np.array([0, 1, 2, 2]), "sig2")
    again = Store(path).rounds()
    assert [r["n_labeled"] for r in again] == [5, 8]
    assert again[0]["change"] is None and again[0]["accuracy"] is None
    assert again[1]["accuracy"] == 0.8 and again[1]["sig"] == "sig2"
    assert again[1]["pred"].dtype == np.int16 and again[1]["pred"].tolist() == [
        0,
        1,
        2,
        2,
    ]


def test_meta_has_defaults(tmp_path):
    s = Store(tmp_path / "p.db")
    assert s.get("x") is None and s.get("x", "d") == "d"
    s.set("x", "1")
    s.set("x", "2")
    assert s.get("x") == "2"
