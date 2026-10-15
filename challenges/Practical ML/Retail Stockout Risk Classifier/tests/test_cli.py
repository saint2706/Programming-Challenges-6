from __future__ import annotations

import json

import pytest
from stockout_risk import cli, data, pipeline
from typer.testing import CliRunner

runner = CliRunner()


@pytest.fixture
def fitted(home, feat, monkeypatch):
    """A saved model, results folder and cached features, all from the synthetic table."""
    monkeypatch.setattr(pipeline, "load_features", lambda refresh=False: feat)
    result = runner.invoke(cli.app, ["evaluate", "--no-ablation", "--no-other-labels"])
    assert result.exit_code == 0, result.output
    return home, result


def test_evaluate_writes_results_model_and_predictions(fitted):
    home, result = fitted
    assert "test PR-AUC" in result.output
    for path in (
        "results/metrics.json",
        "results/report.md",
        "results/pr_curve.png",
        "data/model/model.txt",
        "data/model/model.json",
        "data/predictions.parquet",
    ):
        assert (home / path).exists(), path
    assert (
        json.loads((home / "results" / "metrics.json").read_text())["label_hours"] == 4
    )


def test_score_prints_a_risk_and_reasons(fitted, feat):
    row = feat.filter(feat["dt"] == feat["dt"].max()).row(0, named=True)
    args = [
        "score",
        "--store-id",
        str(row["store_id"]),
        "--product-id",
        str(row["product_id"]),
    ]
    out = runner.invoke(cli.app, args)
    assert out.exit_code == 0, out.output
    assert (
        "chance of 4+" in out.output
        and "out-of-stock hours on 2024-07-02" in out.output
    )
    assert out.output.count("\n") >= 4  # the headline and at least three reasons


def test_score_defaults_to_a_day_whose_outcome_is_known(fitted, feat):
    row = feat.row(0, named=True)
    args = [
        "score",
        "--store-id",
        str(row["store_id"]),
        "--product-id",
        str(row["product_id"]),
    ]
    out = runner.invoke(cli.app, [*args, "--json"])
    body = json.loads(out.output)
    assert body["date"] == "2024-07-01" and body["warning"] is None
    assert isinstance(body["actual_next_oos_hours"], int)
    assert "what happened" in runner.invoke(cli.app, args).output


def test_scoring_the_last_data_day_warns_that_tomorrow_is_unknown(fitted, feat):
    row = feat.row(0, named=True)
    args = [
        "score",
        "--store-id",
        str(row["store_id"]),
        "--product-id",
        str(row["product_id"]),
    ]
    out = runner.invoke(cli.app, [*args, "--date", "2024-07-02", "--json"])
    body = json.loads(out.output)
    assert body["warning"] and body["actual_next_oos_hours"] is None
    text = runner.invoke(cli.app, [*args, "--date", "2024-07-02"])
    assert "warning: tomorrow's promotion" in text.output
    work = runner.invoke(cli.app, ["worklist", "--date", "2024-07-02", "--top", "2"])
    assert "warning: tomorrow's promotion" in work.output
    assert "warning" not in runner.invoke(cli.app, ["worklist", "--top", "2"]).output


def test_score_json_has_a_probability_and_five_reasons(fitted, feat):
    row = feat.filter(feat["dt"] == feat["dt"].max()).row(0, named=True)
    out = runner.invoke(
        cli.app,
        [
            "score",
            "--store-id",
            str(row["store_id"]),
            "--product-id",
            str(row["product_id"]),
            "--date",
            "2024-06-30",
            "--json",
        ],
    )
    body = json.loads(out.output)
    assert (
        0 <= body["risk"] <= 1
        and len(body["reasons"]) == 5
        and body["date"] == "2024-06-30"
    )


def test_score_unknown_store_or_bad_date_fails_cleanly(fitted):
    out = runner.invoke(cli.app, ["score", "--store-id", "999999", "--product-id", "1"])
    assert out.exit_code == 1 and "no row for store 999999" in out.output
    out = runner.invoke(
        cli.app,
        ["score", "--store-id", "1", "--product-id", "1", "--date", "yesterday"],
    )
    assert out.exit_code == 2 and "YYYY-MM-DD" in out.output


def test_worklist_is_sorted_by_risk_and_limited(fitted):
    out = runner.invoke(cli.app, ["worklist", "--top", "5", "--json"])
    rows = json.loads(out.output)
    assert len(rows) == 5 and [r["risk"] for r in rows] == sorted(
        (r["risk"] for r in rows), reverse=True
    )
    table = runner.invoke(cli.app, ["worklist", "--top", "3", "--date", "2024-06-30"])
    assert table.exit_code == 0 and table.output.count("%") >= 3


def test_worklist_rejects_a_day_outside_the_data_and_absurd_sizes(fitted):
    assert runner.invoke(cli.app, ["worklist", "--date", "2030-01-01"]).exit_code == 1
    assert runner.invoke(cli.app, ["worklist", "--top", "0"]).exit_code == 2


def test_score_and_worklist_without_a_model_say_what_to_run(home):
    for args in (["worklist"], ["score", "--store-id", "1", "--product-id", "1"]):
        out = runner.invoke(cli.app, args)
        assert out.exit_code == 1 and "stockout evaluate" in out.output


def test_data_command_downloads_and_prints_the_summary(home, raw, monkeypatch):
    monkeypatch.setattr(data, "download", lambda force=False: {})
    monkeypatch.setattr(data, "load_raw", lambda: raw)
    out = runner.invoke(cli.app, ["data"])
    assert out.exit_code == 0 and "window_count_matches_flags: 1.0" in out.output
