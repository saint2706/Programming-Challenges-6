"""Headless Streamlit tests (streamlit.testing.v1.AppTest) against fake-encoder artifacts."""

from pathlib import Path

import data
import pipeline
import pytest
from streamlit.testing.v1 import AppTest
from test_pipeline import FakeImage, FakeText, make_catalog

APP = str(Path(__file__).parent / "app.py")


@pytest.fixture
def artifacts(tmp_path: Path, monkeypatch) -> Path:
    make_catalog(tmp_path, n_groups=16)
    monkeypatch.setattr(
        data, "fetch_catalog", lambda data_dir=tmp_path: data_dir / "train.csv"
    )
    monkeypatch.setattr(data, "fetch_images", lambda names, data_dir=tmp_path: [])
    df = pipeline.build_slice(n_groups=16, seed=0, data_dir=tmp_path)
    emb = pipeline.embed_slice(df, tmp_path, FakeText(), FakeImage())
    report, _, _ = pipeline.run_evaluation(df, emb, tmp_path, k=5)
    pipeline.save_report(
        tmp_path / pipeline.REPORT_FILE,
        report,
        settings={"k": 5, "val_fraction": 0.4, "seed": 0},
    )
    monkeypatch.setenv("DUPLICATE_DETECTOR_DATA_DIR", str(tmp_path))
    return tmp_path


def run_app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=60).run()


def metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def test_renders_without_exception_and_shows_metrics(artifacts: Path):
    at = run_app()
    assert not at.exception
    labels = {m.label for m in at.metric}
    assert {"Precision", "Recall", "F1", "Flagged pairs", "Candidate recall"} <= labels


def test_sliders_default_to_the_tuned_operating_point(artifacts: Path):
    at = run_app()
    fused = pipeline.load_report(artifacts / pipeline.REPORT_FILE)["modes"]["fused"]
    assert at.sidebar.slider(key="weight").value == pytest.approx(
        fused["weight"], abs=0.01
    )
    assert at.sidebar.slider(key="threshold").value == pytest.approx(
        fused["threshold"], abs=0.01
    )


def test_raising_threshold_flags_fewer_pairs(artifacts: Path):
    at = run_app()
    slider = at.sidebar.slider(key="threshold")
    at.sidebar.slider(key="threshold").set_value(
        slider.min
    ).run()  # lowest allowed: flags every candidate
    loose = int(metric(at, "Flagged pairs").replace(",", ""))
    at.sidebar.slider(key="threshold").set_value(slider.max).run()
    strict = int(metric(at, "Flagged pairs").replace(",", ""))
    assert not at.exception
    assert (
        loose > 20
    )  # more than just the exact-duplicate pairs the tuned default flags
    assert strict < loose


def test_ground_truth_hidden_until_toggled(artifacts: Path):
    at = run_app()
    text_before = " ".join(str(e.value) for e in at.markdown) + " ".join(
        str(e.value) for e in at.caption
    )
    assert "true duplicate" not in text_before.lower()
    at.sidebar.checkbox(key="show_truth").check().run()
    text_after = " ".join(str(e.value) for e in at.markdown) + " ".join(
        str(e.value) for e in at.caption
    )
    assert "true duplicate" in text_after.lower()


def test_switching_split_works(artifacts: Path):
    at = run_app()
    at.sidebar.selectbox(key="split").select("val").run()
    assert not at.exception
    assert at.metric


def test_missing_artifacts_show_actionable_error(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DUPLICATE_DETECTOR_DATA_DIR", str(tmp_path))
    at = run_app()
    assert not at.exception
    assert any("evaluate" in e.value for e in at.error)
