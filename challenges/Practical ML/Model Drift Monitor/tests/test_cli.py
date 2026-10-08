import json

import pytest
from drift_monitor import cli
from typer.testing import CliRunner

runner = CliRunner()


@pytest.fixture(autouse=True)
def fake_context(monkeypatch, tiny_context):
    monkeypatch.setattr(cli, "get_context", lambda: tiny_context)


def test_monitor_lists_every_window_with_alerts_and_the_delayed_accuracy():
    result = runner.invoke(cli.app, ["monitor"])
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0].split()[:2] == ["window", "alerts"]
    assert len([ln for ln in lines if ln.strip()[:1].isdigit()]) == 36
    assert (
        "nswprice" in result.output
    )  # the shifted feature shows up as an alerting signal
    assert "labels arrive 4 windows late" in result.output


def test_monitor_window_range_filters_the_rows():
    result = runner.invoke(
        cli.app, ["monitor", "--from-window", "10", "--to-window", "14"]
    )
    assert result.exit_code == 0, result.output
    rows = [ln for ln in result.output.splitlines() if ln.strip()[:1].isdigit()]
    assert [int(r.split()[0]) for r in rows] == [10, 11, 12, 13, 14]


@pytest.mark.parametrize(
    "args", [["--from-window", "20", "--to-window", "10"], ["--to-window", "99"]]
)
def test_a_bad_window_range_exits_two_and_names_the_valid_range(args):
    result = runner.invoke(cli.app, ["monitor", *args])
    assert result.exit_code == 2
    assert "0..35" in result.output


def test_report_without_a_report_file_says_what_to_run(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "RESULTS", tmp_path / "missing.json")
    result = runner.invoke(cli.app, ["report"])
    assert result.exit_code == 2 and "train" in result.output


def test_report_prints_the_headline_numbers_from_a_report_file(
    monkeypatch, tmp_path, tiny
):
    _, report, _ = tiny
    p = tmp_path / "report.json"
    p.write_text(json.dumps(report))
    monkeypatch.setattr(cli, "RESULTS", p)
    result = runner.invoke(cli.app, ["report"])
    assert result.exit_code == 0, result.output
    assert "natural run" in result.output and "adwin_error" in result.output
    assert "injected drift" in result.output and "concept" in result.output
    assert "unseen reference windows" in result.output
