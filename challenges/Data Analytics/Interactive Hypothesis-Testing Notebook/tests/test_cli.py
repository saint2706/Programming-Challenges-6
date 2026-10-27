from __future__ import annotations

from hypothesis_wizard.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_datasets_lists_every_bundled_table_with_its_licence():
    result = runner.invoke(app, ["datasets"])
    assert result.exit_code == 0
    for key in (
        "penguins",
        "iris",
        "wine_red",
        "sleep",
        "toothgrowth",
        "ucb_admissions",
    ):
        assert key in result.output
    assert "CC0 1.0" in result.output


def test_ask_paired_sleep_prints_plan_primary_and_verdict():
    result = runner.invoke(
        app,
        [
            "ask",
            "sleep",
            "--outcome",
            "extra",
            "--group",
            "group",
            "--pair",
            "ID",
            "--paired",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Primary: paired_t: t = -4.062" in result.output
    assert "Verdict:" in result.output


def test_ask_prints_the_session_threshold_with_alternatives():
    result = runner.invoke(
        app,
        [
            "ask",
            "sleep",
            "--outcome",
            "extra",
            "--group",
            "group",
            "--pair",
            "ID",
            "--paired",
            "--alternative",
            "wilcoxon_signed_rank",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "2 tests tried" in result.output
    assert "threshold now 0.025" in result.output


def test_ask_unknown_column_exits_with_code_two():
    result = runner.invoke(
        app, ["ask", "iris", "--outcome", "no_such_column", "--group", "species"]
    )
    assert result.exit_code == 2
    assert "error:" in result.output


def test_ask_unknown_alternative_exits_with_code_two():
    result = runner.invoke(
        app,
        [
            "ask",
            "iris",
            "--outcome",
            "petal_length",
            "--group",
            "species",
            "--alternative",
            "chi2",
        ],
    )
    assert result.exit_code == 2
