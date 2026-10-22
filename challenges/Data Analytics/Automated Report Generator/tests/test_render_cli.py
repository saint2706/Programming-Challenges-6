"""The three output files, hostile text, and the command line."""

import json
import re
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from helpers import planted
from report_generator.cli import app
from report_generator.engine import build_report
from report_generator.render import to_dict, to_html, to_json, to_markdown
from typer.testing import CliRunner

runner = CliRunner(env={"COLUMNS": "200", "NO_COLOR": "1"})
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def run(*args: str):
    result = runner.invoke(app, list(args))
    result.clean = " ".join(ANSI.sub("", result.output).split())
    return result


@pytest.fixture(scope="module")
def report():
    return build_report(planted(0), title="Planted <demo>", row_label="days")


def test_markdown_has_the_sections_and_every_sentence(report):
    md = to_markdown(report)
    assert md.startswith("# Planted <demo>\n")
    for heading in (
        "## At a glance",
        "## Trends",
        "## Seasonality",
        "## How this was checked",
    ):
        assert heading in md
    for f in report.findings:
        assert f.text in md
    assert "recomputed from the raw table" in md and "checks across" in md
    assert "\n\n\n" not in md


def test_json_is_strict_and_traceable(report):
    doc = json.loads(
        to_json(report)
    )  # allow_nan=False inside: raises on NaN or infinity
    assert doc["verification"]["failed"] == 0 and doc["verification"]["checks"] > 30
    trend = next(f for f in doc["findings"] if f["id"] == "trend:visits")
    assert trend["facts"]["change"]["text"] in trend["text"]
    assert trend["facts"]["change"]["value"] == pytest.approx(
        report.findings[0].facts["change"].value, abs=1
    )
    assert (
        doc["columns"]["plan"]["role"] == "dimension" and doc["config"]["fdr"] == 0.01
    )
    assert to_dict(report)["candidates"] == report.candidates


def test_html_is_one_self_contained_file(report):
    page = to_html(report)
    markup = re.sub(
        r"<script\b[^>]*>.*?</script\s*>", "", page, flags=re.DOTALL | re.IGNORECASE
    )
    assert not re.search(r"""(src|href)=["']https?://""", markup)
    assert page.count("Plotly.newPlot") >= len(report.findings) - 1
    assert "Evidence" in page and "matches" in page and "How this was checked" in page
    assert "&lt;demo&gt;" in page and "<demo>" not in markup


def test_hostile_names_cannot_inject_markup(tmp_path):
    evil = "</script><img src=x onerror=alert(1)>"
    rng = np.random.default_rng(0)
    n = 900
    frame = pl.DataFrame(
        {
            "when": [date(2020, 1, 1) + timedelta(days=i) for i in range(n)],
            "n": rng.normal(50, 5, n) + np.arange(n) * 0.05,
            "n2": rng.normal(10, 1, n),
            evil: rng.choice([evil, "ok", "fine"], n, p=[0.2, 0.4, 0.4]),
        }
    )
    frame = frame.with_columns(
        pl.when(pl.col(evil) == evil)
        .then(pl.col("n") + 30)
        .otherwise(pl.col("n"))
        .alias("n")
    )
    path = tmp_path / "evil.csv"
    frame.write_csv(path)
    result = run("generate", str(path), "--title", evil, "--out", str(tmp_path / "out"))
    assert result.exit_code == 0, result.clean
    page = (tmp_path / "out" / "report.html").read_text(encoding="utf-8")
    assert "<img src=x" not in page
    assert page.count("</script>") == page.count("<script")
    assert "&lt;/script&gt;" in page
    assert (
        "&lt;img src=x" in page
    )  # the hostile name does appear in a finding, as inert text


def test_datasets_and_findings_commands():
    out = run("datasets")
    assert out.exit_code == 0 and all(
        n in out.clean for n in ("bike", "retail", "ames")
    )
    out = run("findings", "bike")
    assert (
        out.exit_code == 0 and "weekly total" in out.clean and "candidates" in out.clean
    )
    wide = run("findings", "bike", "--all")
    assert wide.clean.count("yes") >= 5


def test_verify_command_shows_claimed_and_recomputed():
    out = run("verify", "ames")
    assert out.exit_code == 0, out.clean
    assert (
        "DuckDB" in out.clean
        and "overview" in out.clean
        and "ok" in out.clean
        and "DIFFERS" not in out.clean
    )


def test_generate_writes_three_files_and_respects_options(tmp_path):
    out = run(
        "generate", "retail", "--out", str(tmp_path), "-n", "4", "--title", "Shop"
    )
    assert out.exit_code == 0, out.clean
    assert {p.name for p in tmp_path.iterdir()} == {
        "report.md",
        "report.html",
        "report.json",
    }
    doc = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert doc["title"] == "Shop" and len(doc["findings"]) <= 4
    skipped = run("generate", "retail", "--out", str(tmp_path / "b"), "--no-verify")
    assert (
        json.loads((tmp_path / "b" / "report.json").read_text(encoding="utf-8"))[
            "verification"
        ]["ran"]
        is False
    )
    assert skipped.exit_code == 0


def test_a_file_with_overrides(tmp_path):
    frame = (
        planted(1)
        .rename({"visits": "traffic", "date": "day_of"})
        .select("day_of", "traffic", "temp", "plan")
    )
    path = tmp_path / "site.parquet"
    frame.write_parquet(path)
    out = run(
        "generate",
        str(path),
        "--time",
        "day_of",
        "--sum",
        "traffic",
        "--rows-label",
        "days",
        "--out",
        str(tmp_path / "o"),
    )
    assert out.exit_code == 0, out.clean
    assert "weekly total traffic" in out.clean


def test_errors_exit_cleanly(tmp_path):
    for args, text in (
        (["generate", "nope"], "neither a bundled dataset"),
        (["generate", "bike", "--time", "weather"], "does not hold dates"),
        (["generate", "bike", "--time", "zzz"], "no column"),
    ):
        result = run(*args)
        assert result.exit_code == 2 and text in result.clean, (args, result.clean)
    bad = tmp_path / "x.xlsx"
    bad.write_text("x")
    assert run("generate", str(bad)).exit_code == 2
