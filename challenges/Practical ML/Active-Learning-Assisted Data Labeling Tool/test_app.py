from pathlib import Path

import pytest
import strategies
from helpers import make_problem
from project import Project
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parent / "app.py")
CLASSES = [f"k{i}" for i in range(4)]


def make_project(path, gold=True):
    prob = make_problem(n=120, k=4)
    return Project.create(
        path,
        [f"item {i}" for i in range(120)],
        prob.X,
        CLASSES,
        gold=[CLASSES[c] for c in prob.y] if gold else None,
        C=prob.C,
        X_eval=prob.X_eval,
        y_eval=prob.y_eval,
    ), prob


def open_app(monkeypatch, path=""):
    monkeypatch.setenv("AL_PROJECT", str(path))
    return AppTest.from_file(APP, default_timeout=60).run()


def picks(at):
    return [s for s in at.selectbox if s.key and s.key.startswith("pick-")]


def ids(at):
    return [int(s.key.split("-")[1]) for s in picks(at)]


@pytest.fixture
def proj(tmp_path):
    return make_project(tmp_path / "p")


def test_without_a_project_the_app_explains_how_to_make_one(monkeypatch):
    at = open_app(monkeypatch)
    assert not at.exception and "init" in at.info[0].value


def test_a_missing_project_is_an_error_message_not_a_traceback(monkeypatch, tmp_path):
    at = open_app(monkeypatch, tmp_path / "nope")
    assert not at.exception and "init" in at.error[0].value


def test_suggesting_shows_a_picker_and_an_accept_button_per_item(monkeypatch, proj):
    p, _ = proj
    at = open_app(monkeypatch, p.dir)
    assert at.metric[0].value.startswith("0 /")
    at.button(key="suggest").click().run()
    assert not at.exception and len(picks(at)) == 10
    assert len([b for b in at.button if b.key and b.key.startswith("guess-")]) == 10


def test_accepting_the_models_guess_fills_the_picker(monkeypatch, proj):
    p, _ = proj
    at = open_app(monkeypatch, p.dir)
    at.button(key="suggest").click().run()
    first = ids(at)[0]
    at.button(key=f"guess-{first}").click().run()
    chosen = at.selectbox(key=f"pick-{first}").value
    assert (
        chosen in CLASSES
        and at.button(key=f"guess-{first}").label == f"Accept: {chosen}"
    )


def test_submitting_chosen_labels_saves_them_and_clears_the_batch(monkeypatch, proj):
    p, _ = proj
    at = open_app(monkeypatch, p.dir)
    at.button(key="suggest").click().run()
    a, b, *_ = ids(at)
    at.selectbox(key=f"pick-{a}").select("k1")
    at.selectbox(key=f"pick-{b}").select("k2")
    at.button(key="submit").click().run()
    assert not at.exception and not picks(at) and "Saved 2" in at.success[0].value
    assert Project(p.dir).store.current_labels() == {a: "k1", b: "k2"}
    assert at.metric[0].value.startswith("2 /")


def test_submitting_nothing_warns_and_writes_nothing(monkeypatch, proj):
    p, _ = proj
    at = open_app(monkeypatch, p.dir)
    at.button(key="suggest").click().run()
    at.button(key="submit").click().run()
    assert at.warning and Project(p.dir).store.current_labels() == {}


def test_simulating_the_annotator_labels_the_batch_with_gold(monkeypatch, proj):
    p, prob = proj
    at = open_app(monkeypatch, p.dir)
    at.button(key="suggest").click().run()
    batch = ids(at)
    at.button(key="simulate").click().run()
    assert not at.exception
    assert Project(p.dir).store.current_labels() == {
        i: CLASSES[prob.y[i]] for i in batch
    }


def test_there_is_no_simulate_button_without_gold_labels(monkeypatch, tmp_path):
    p, _ = make_project(tmp_path / "bare", gold=False)
    at = open_app(monkeypatch, p.dir)
    at.button(key="suggest").click().run()
    assert "simulate" not in [b.key for b in at.button] and "submit" in [
        b.key for b in at.button
    ]


def test_every_strategy_can_suggest_from_the_sidebar(monkeypatch, proj):
    p, _ = proj
    at = open_app(monkeypatch, p.dir)
    for name in strategies.NAMES:
        at.sidebar.selectbox(key="strategy").select(name).run()
        at.button(key="suggest").click().run()
        assert not at.exception and len(picks(at)) == 10, name


def test_charts_and_the_stopping_signal_render_after_several_rounds(monkeypatch, proj):
    p, _ = proj
    at = open_app(monkeypatch, p.dir)
    for _ in range(3):
        at.button(key="suggest").click().run()
        at.button(key="simulate").click().run()
    assert not at.exception
    assert at.metric[0].value.startswith("30 /")
    assert at.metric[2].value in ("not yet", "reached")
    assert len(Project(p.dir).store.rounds()) == 3
