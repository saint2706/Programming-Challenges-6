"""The analysis summary, the HTML report (including hostile labels) and the command line."""

import json
import re

import numpy as np
import polars as pl
import pytest
from helpers import r_cases
from survival_toolkit import analysis
from survival_toolkit.cli import app
from survival_toolkit.data import load_builtin
from survival_toolkit.report import build_html
from typer.testing import CliRunner

runner = CliRunner(env={"COLUMNS": "240", "NO_COLOR": "1"})
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def run(*args: str):
    result = runner.invoke(app, list(args))
    result.clean = " ".join(ANSI.sub("", result.output).split())
    return result


@pytest.fixture(scope="module")
def lung_analysis():
    spec, prepared = load_builtin("lung")
    return analysis.analyze(
        prepared, title=spec.title, by="sex", horizons=[180, 365], tau=500,
        covariates=["age", "sex", "ph_ecog"], strata=None,
    )  # fmt: skip


def test_analysis_numbers_agree_with_the_estimators(lung_analysis):
    a = lung_analysis
    assert (a.n, a.events) == (228, 165)
    assert [g.label for g in a.groups] == ["1", "2"]
    assert a.tests["logrank"].statistic == pytest.approx(10.32674, rel=1e-5)
    male = a.groups[0]
    ref = r_cases()["km"]["lung_male"]["rmst"]["mean"]
    assert male.rmst is not None and male.rmst[0] == pytest.approx(ref)
    assert (
        male.retention[0]["lower"] < male.retention[0]["s"] < male.retention[0]["upper"]
    )
    assert a.cox is not None and a.cox.dropped == 1
    assert a.follow_up_median is not None and a.follow_up_median > 300


def test_beyond_follow_up_horizons_are_not_estimated():
    _, prepared = load_builtin("lung")
    a = analysis.analyze(prepared, title="x", horizons=[100, 5000], tau=None)
    assert a.groups[0].retention[1]["s"] is None
    assert a.groups[0].rmst is None


def test_json_summary_is_strictly_valid(lung_analysis):
    doc = analysis.to_dict(lung_analysis)
    text = json.dumps(doc, allow_nan=False)  # raises on NaN or inf
    back = json.loads(text)
    assert back["cox"]["proportional_hazards"]["global"]["df"] == 3
    assert back["groups"][0]["rmst"]["mean"] > 0


def test_report_is_one_self_contained_file(lung_analysis):
    page = build_html(lung_analysis)
    assert page.startswith("<!doctype html>")
    markup = re.sub(
        r"<script>.*?</script>", "", page, flags=re.DOTALL
    )  # plotly.js has URLs in strings
    assert not re.search(r"""(src|href)=["']https?://""", markup)  # no network fetches
    assert page.count("Plotly.newPlot") >= 4
    for heading in (
        "Survival curves",
        "Do the curves differ?",
        "Hazard ratios",
        "Number at risk",
    ):
        assert heading in page


def test_hostile_labels_cannot_inject_markup(tmp_path):
    evil = "</script><img src=x onerror=alert(1)>"
    rng = np.random.default_rng(0)
    n = 120
    frame = pl.DataFrame(
        {
            "t": rng.exponential(10, n) + 0.1,
            "e": rng.integers(0, 2, n),
            f"arm{evil}": rng.choice([evil, "ok"], n),
            "x": rng.normal(size=n),
        }
    )
    path = tmp_path / "evil.csv"
    frame.write_csv(path)
    result = run(
        "report", str(path), "--time", "t", "--event", "e", "--by", f"arm{evil}", "-c", "x",
        "--out", str(tmp_path / "out"),
    )  # fmt: skip
    assert result.exit_code == 0, result.clean
    page = (tmp_path / "out" / "survival_report.html").read_text(encoding="utf-8")
    assert "<img src=x" not in page
    assert page.count("</script>") == page.count("<script")  # none added by the label
    assert "&lt;/script&gt;" in page


def test_datasets_and_km_commands():
    out = run("datasets")
    assert out.exit_code == 0 and "telco" in out.clean and "gbsg" in out.clean
    out = run("km", "lung", "--by", "sex", "-H", "365")
    assert out.exit_code == 0
    assert "165 events" in out.clean and "S(365)" in out.clean


def test_km_on_a_file_with_late_entry(tmp_path):
    frame = pl.DataFrame(
        {
            "start": [0.0, 1.0, 2.0, 0.5],
            "stop": [5.0, 6.0, 4.0, 7.0],
            "died": [1, 0, 1, 1],
        }
    )
    path = tmp_path / "d.csv"
    frame.write_csv(path)
    out = run(
        "km",
        str(path),
        "--time",
        "stop",
        "--event",
        "died",
        "--entry",
        "start",
        "-H",
        "3",
    )
    assert out.exit_code == 0, out.clean
    assert "3 events" in out.clean


def test_compare_prints_all_weights_and_pairs():
    out = run("compare", "gbsg", "--by", "grade")
    assert out.exit_code == 0
    for word in (
        "logrank",
        "wilcoxon",
        "tarone-ware",
        "peto",
        "fh:0,1",
        "1 vs 2",
        "RMST",
    ):
        assert word in out.clean


def test_cox_command_with_strata_and_penalty():
    out = run("cox", "lung", "-c", "age", "-c", "sex", "--strata", "ph_ecog")
    assert out.exit_code == 0, out.clean
    assert "stratified by ph_ecog" in out.clean and "GLOBAL" in out.clean
    out = run("cox", "telco", "--penalizer", "1")
    assert out.exit_code == 0 and "Contract[Two year]" in out.clean


def test_report_command_writes_files(tmp_path):
    out = run("report", "gbsg", "--out", str(tmp_path))
    assert out.exit_code == 0, out.clean
    assert (tmp_path / "survival_report.html").stat().st_size > 1_000_000
    doc = json.loads((tmp_path / "analysis.json").read_text(encoding="utf-8"))
    assert doc["n"] == 686 and doc["cox"]["terms"]


def test_errors_exit_cleanly():
    for args, text in (
        (["km", "nope"], "neither a bundled dataset"),
        (["cox", "lung", "-c", "zzz"], "no column"),
        (["km", "lung", "--ci", "wald"], "ci must"),
        (["cox", "telco", "--ties", "exact"], "ties must"),
    ):
        result = run(*args)
        assert result.exit_code == 2, (args, result.clean)
        assert text in result.clean


def test_compare_needs_groups_for_a_file(tmp_path):
    path = tmp_path / "d.csv"
    pl.DataFrame({"t": [1.0, 2.0, 3.0, 4.0], "e": [1, 0, 1, 1]}).write_csv(path)
    result = run("compare", str(path), "--time", "t", "--event", "e")
    assert result.exit_code == 2 and "--by" in result.clean
