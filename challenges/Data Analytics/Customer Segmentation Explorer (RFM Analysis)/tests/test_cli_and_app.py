"""The command line and the Streamlit app, run against a temporary copy of the committed sample."""

import shutil
from pathlib import Path

import polars as pl
import pytest
from rfm_explorer import data
from rfm_explorer.cli import app
from rfm_explorer.paths import ENV_VAR
from streamlit.testing.v1 import AppTest
from typer.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "rfm_explorer" / "app.py"


@pytest.fixture()
def home(tmp_path, monkeypatch):
    """A project root holding only the committed sample, so results never depend on data/."""
    (tmp_path / "sample_data").mkdir()
    shutil.copy(ROOT / data.SAMPLE_FILE, tmp_path / data.SAMPLE_FILE)
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    return tmp_path


def test_score_writes_every_customer_and_prints_segments(home):
    out = home / "out" / "scored.csv"
    result = CliRunner().invoke(app, ["score", "--out", str(out)])
    assert result.exit_code == 0, result.output
    scored = pl.read_csv(out)
    assert scored.height > 100
    assert {
        "customer_id",
        "recency_days",
        "frequency",
        "monetary",
        "r",
        "f",
        "m",
        "segment",
    } <= set(scored.columns)
    assert "quintile segments" in result.output


@pytest.mark.parametrize("method", ["fixed", "kmeans"])
def test_score_other_methods(home, method):
    result = CliRunner().invoke(
        app, ["score", "--method", method, "--out", str(home / "o.csv")]
    )
    assert result.exit_code == 0, result.output


def test_score_rejects_an_unknown_method_and_a_bad_date(home):
    runner = CliRunner()
    assert runner.invoke(app, ["score", "--method", "magic"]).exit_code != 0
    assert runner.invoke(app, ["score", "--snapshot", "tomorrow"]).exit_code != 0


def test_validate_prints_both_tables(home):
    result = CliRunner().invoke(app, ["validate", "--resamples", "50"])
    assert result.exit_code == 0, result.output
    assert "next-period" in result.output
    assert "random = 20%" in result.output


def test_ledger_lists_every_rule(home):
    result = CliRunner().invoke(app, ["ledger"])
    assert result.exit_code == 0
    for rule in ("non_product", "bad_price", "bad_quantity", "no_customer"):
        assert rule in result.output


def run_app():
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def scoring_radio(at):
    return next(r for r in at.radio if r.label == "Scoring")


def test_app_renders_the_default_view(home):
    at = run_app()
    assert at.title[0].value == "Customer segments (RFM)"
    assert {m.label for m in at.metric} >= {"Customers", "Money in window"}
    assert len(at.tabs) == 5


@pytest.mark.parametrize("method", ["fixed", "kmeans"])
def test_app_rescores_when_the_method_changes(home, method):
    at = run_app()
    scoring_radio(at).set_value(method).run()
    assert not at.exception, [e.value for e in at.exception]
    assert scoring_radio(at).value == method


def test_app_rejects_bad_thresholds_without_crashing(home):
    at = run_app()
    scoring_radio(at).set_value("fixed").run()
    at.text_input[1].set_value("5, 3, 2, 1").run()  # not ascending
    assert any("four ascending numbers" in e.value for e in at.sidebar.error)
    assert not at.exception


def test_app_filters_by_country(home):
    at = run_app()
    before = int(at.metric[0].value.replace(",", ""))
    at.multiselect[0].set_value(["United Kingdom"]).run()
    assert not at.exception
    assert int(at.metric[0].value.replace(",", "")) <= before
