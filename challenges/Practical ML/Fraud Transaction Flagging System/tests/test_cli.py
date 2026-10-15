from __future__ import annotations

import json
import re
from types import SimpleNamespace

import polars as pl
import pytest
from fraud_flagging import cli
from fraud_flagging.data import RAW_COLUMNS
from typer.testing import CliRunner

runner = CliRunner()


def run(args: list[str]) -> SimpleNamespace:
    """Invoke the CLI and normalise what it printed.

    On GitHub Actions typer forces Rich styling, which wraps error messages in a box and splits a
    word like ``--force`` across colour codes; on a plain terminal it prints the message as is. The
    same assertions have to hold in both, so colour codes and box characters are dropped and runs
    of whitespace collapse to one space.
    """
    result = runner.invoke(cli.app, args)
    text = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    text = re.sub(r"[│╭╮╰╯─]", " ", text)
    return SimpleNamespace(exit_code=result.exit_code, output=" ".join(text.split()))


@pytest.fixture(scope="module")
def fitted(tmp_path_factory, published):
    """One evaluation shared by the tests: FRAUD_HOME holds the data, model and results."""
    home = tmp_path_factory.mktemp("fraud_home")
    (home / "data").mkdir()
    published.write_parquet(home / "data" / "creditcard.parquet")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("FRAUD_HOME", str(home))
        result = run(["evaluate", "--no-ablation", "--no-rolling"])
        assert result.exit_code == 0, result.output
        yield home, result


@pytest.fixture
def transactions(published):
    """A few hundred transactions as a user would hand them over: raw columns only."""
    return published.select(RAW_COLUMNS).head(400)


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
    metrics = json.loads((home / "results" / "metrics.json").read_text())
    assert "main_strategy" in metrics


def test_results_are_written_with_unix_line_endings_and_a_final_newline(fitted):
    home, _ = fitted
    for name in ("metrics.json", "report.md"):
        raw = (home / "results" / name).read_bytes()
        assert b"\r" not in raw and raw.endswith(b"\n"), name


def test_data_command_prints_the_summary(installed):
    out = run(["data"])
    assert out.exit_code == 0 and "duplicate_rows: 25" in out.output


def test_score_one_transaction_of_the_data(fitted):
    out = run(["score", "--row", "100"])
    assert out.exit_code == 0, out.output
    assert "chance of fraud" in out.output and "largest contributions" in out.output
    body = json.loads(run(["score", "--row", "100", "--json"]).output)
    assert 0 <= body["risk"] <= 1
    assert set(body["flagged_by"]) == {"precision", "recall", "cost"}
    assert len(body["reasons"]) == 5


def test_score_unknown_row_and_bad_target_fail_cleanly(fitted):
    assert run(["score", "--row", "99999999"]).exit_code == 1
    bad = run(["score", "--row", "100", "--precision", "0.42"])
    assert bad.exit_code == 2 and "one of" in bad.output


def test_commands_without_a_model_say_what_to_run(installed, transactions):
    path = installed / "t.csv"
    transactions.write_csv(path)
    for args in (["score", "--row", "1"], ["flag", str(path)]):
        out = run(args)
        assert out.exit_code == 1 and "fraud evaluate" in out.output


@pytest.mark.parametrize("suffix", ["csv", "parquet"])
def test_flag_scores_a_file_and_writes_the_decisions(
    fitted, transactions, tmp_path, suffix
):
    src, dst = tmp_path / f"in.{suffix}", tmp_path / f"out.{suffix}"
    (transactions.write_csv if suffix == "csv" else transactions.write_parquet)(src)
    out = run(
        ["flag", str(src), "--policy", "cost", "--review-cost", "5", "--out", str(dst)]
    )
    assert out.exit_code == 0, out.output
    assert "400 transactions" in out.output and "wrote 400 rows to" in out.output
    written = pl.read_csv(dst) if suffix == "csv" else pl.read_parquet(dst)
    assert {"p", "flag"} <= set(written.columns) and "tod_sin" not in written.columns
    assert written["p"].is_between(0, 1).all()


def test_flag_only_flagged_writes_only_those_rows(fitted, transactions, tmp_path):
    src, dst = tmp_path / "in.parquet", tmp_path / "out.parquet"
    transactions.write_parquet(src)
    args = [
        "flag",
        str(src),
        "--policy",
        "capacity",
        "--per-hour",
        "1",
        "--out",
        str(dst),
    ]
    out = run([*args, "--only-flagged"])
    assert out.exit_code == 0, out.output
    assert pl.read_parquet(dst)["flag"].all()


def test_flag_refuses_to_overwrite_without_force_and_checks_paths(
    fitted, transactions, tmp_path
):
    src, dst = tmp_path / "in.csv", tmp_path / "out.csv"
    transactions.write_csv(src)
    dst.write_text("keep me")
    refused = run(["flag", str(src), "--out", str(dst)])
    assert refused.exit_code == 2 and "--force" in refused.output
    assert dst.read_text() == "keep me"
    assert run(["flag", str(src), "--out", str(dst), "--force"]).exit_code == 0
    assert run(["flag", str(src), "--out", str(tmp_path / "x.txt")]).exit_code == 2
    assert (
        run(["flag", str(src), "--out", str(tmp_path / "nodir" / "x.csv")]).exit_code
        == 2
    )


def test_flag_rejects_bad_inputs(fitted, transactions, tmp_path):
    (tmp_path / "t.txt").write_text("x")
    assert "only" in run(["flag", str(tmp_path / "t.txt")]).output
    assert "not a file" in run(["flag", str(tmp_path / "missing.csv")]).output
    transactions.drop("V5").write_csv(tmp_path / "short.csv")
    assert "missing columns" in run(["flag", str(tmp_path / "short.csv")]).output
    (tmp_path / "broken.parquet").write_bytes(b"not parquet")
    assert "could not read" in run(["flag", str(tmp_path / "broken.parquet")]).output
    transactions.with_columns(pl.lit(-1.0).alias("Amount")).write_csv(
        tmp_path / "neg.csv"
    )
    assert "negative" in run(["flag", str(tmp_path / "neg.csv")]).output


def test_flag_refuses_an_oversized_input(fitted, transactions, tmp_path, monkeypatch):
    transactions.write_csv(tmp_path / "big.csv")
    monkeypatch.setattr(cli, "MAX_INPUT_BYTES", 10)
    assert "larger than" in run(["flag", str(tmp_path / "big.csv")]).output
