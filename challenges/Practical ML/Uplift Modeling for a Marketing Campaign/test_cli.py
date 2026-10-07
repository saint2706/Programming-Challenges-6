import cli
import polars as pl
import pytest
from helpers import make_frame
from typer.testing import CliRunner

runner = CliRunner()


@pytest.fixture(autouse=True)
def fake_artifacts(monkeypatch, tiny):
    art = tiny[0]
    monkeypatch.setattr(cli, "get_artifacts", lambda with_models=False: art)


def test_check_prints_the_balance_table_and_passes_on_randomized_data(monkeypatch):
    monkeypatch.setattr(cli.data, "load", lambda *a, **k: make_frame(20000))
    result = runner.invoke(cli.app, ["check"])
    assert result.exit_code == 0, result.output
    assert "f0" in result.output and "smd" in result.output.lower()


def test_check_fails_with_exit_code_2_on_confounded_data(monkeypatch):
    monkeypatch.setattr(
        cli.data, "load", lambda *a, **k: make_frame(20000, confounded=True)
    )
    result = runner.invoke(cli.app, ["check"])
    assert result.exit_code == 2
    assert "imbalanced" in result.output


def test_check_without_data_says_to_fetch(monkeypatch):
    def missing(*a, **k):
        raise FileNotFoundError("run `python cli.py fetch` first")

    monkeypatch.setattr(cli.data, "load", missing)
    result = runner.invoke(cli.app, ["check"])
    assert result.exit_code == 2 and "fetch" in result.output


def test_target_reports_incremental_outcomes_against_random_targeting():
    result = runner.invoke(cli.app, ["target", "--budget", "0.2", "--n-boot", "40"])
    assert result.exit_code == 0, result.output
    assert "incremental" in result.output.lower()
    assert "random" in result.output.lower()
    assert "treat everyone" in result.output.lower()


def test_target_with_value_and_cost_reports_the_best_share_and_break_even():
    args = ["target", "--budget", "0.3", "--value", "10", "--cost", "0.05"]
    result = runner.invoke(cli.app, [*args, "--n-boot", "40"])
    assert result.exit_code == 0, result.output
    assert "break-even" in result.output.lower()
    assert "best share" in result.output.lower()


@pytest.mark.parametrize("budget", ["0", "-0.1", "1.5"])
def test_target_rejects_a_budget_outside_zero_one(budget):
    result = runner.invoke(cli.app, ["target", "--budget", budget])
    assert result.exit_code == 2
    assert "budget" in result.output


def test_target_rejects_an_unknown_learner():
    result = runner.invoke(cli.app, ["target", "--budget", "0.2", "--learner", "nope"])
    assert result.exit_code == 2 and "learner" in result.output


def test_score_adds_uplift_and_rank_columns(tmp_path):
    src = tmp_path / "in.parquet"
    make_frame(300).write_parquet(src)
    out = tmp_path / "out.parquet"
    result = runner.invoke(cli.app, ["score", str(src), "--output", str(out)])
    assert result.exit_code == 0, result.output
    scored = pl.read_parquet(out)
    assert scored.height == 300 and {"uplift", "uplift_rank"} <= set(scored.columns)
    assert scored["uplift_rank"].min() == 1 and scored["uplift_rank"].max() == 300


def test_score_with_missing_feature_columns_exits_2(tmp_path):
    src = tmp_path / "in.parquet"
    make_frame(50).drop("f3").write_parquet(src)
    out = tmp_path / "o.parquet"
    result = runner.invoke(cli.app, ["score", str(src), "--output", str(out)])
    assert result.exit_code == 2 and "f3" in result.output


def test_score_ignores_columns_it_must_not_use(tmp_path):
    src = tmp_path / "in.parquet"
    make_frame(100).write_parquet(src)  # includes treatment / outcomes / exposure
    out = tmp_path / "o.parquet"
    assert (
        runner.invoke(cli.app, ["score", str(src), "--output", str(out)]).exit_code == 0
    )
    assert {"treatment", "exposure"} <= set(
        pl.read_parquet(out).columns
    )  # kept, unused


def test_commands_before_a_benchmark_exit_2(monkeypatch):
    def missing(with_models=False):
        raise FileNotFoundError(
            "results/report.json not found; run `python cli.py benchmark` first"
        )

    monkeypatch.setattr(cli, "get_artifacts", missing)
    for args in (
        ["target", "--budget", "0.2"],
        ["score", "x.parquet", "--output", "y.parquet"],
    ):
        result = runner.invoke(cli.app, args)
        assert result.exit_code == 2 and "benchmark" in result.output


def test_benchmark_rejects_an_unknown_stage():
    result = runner.invoke(cli.app, ["benchmark", "--stage", "nope"])
    assert result.exit_code == 2 and "stage" in result.output


def test_format_report_lists_every_learner_with_intervals(tiny):
    text = cli.format_report(tiny[1])
    for name in ("T", "S", "X", "TO", "DR", "response", "random"):
        assert name in text
    assert "visit" in text and "Qini" in text and "[" in text
