"""CLI exit codes, JSON/HTML output, and HTML escaping of hostile CSV content."""

from __future__ import annotations

import json
import re

import dq_monitor as dq
import numpy as np
import polars as pl
import pytest
from corruptions import CORRUPTIONS
from dq_report import render_report
from taxi import load_taxi, split
from typer.testing import CliRunner

runner = CliRunner()
BY_NAME = {c.name: c for c in CORRUPTIONS}


def _write(df, path):
    df.write_csv(path, datetime_format="%Y-%m-%d %H:%M:%S")


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("dq")
    train, held = split(load_taxi())
    (root / "train").mkdir()
    for name, df in train:
        _write(df, root / "train" / name)
    clean = held[0][1]
    _write(clean, root / "clean.csv")
    rng = np.random.default_rng(3)
    for name in ("rename_column", "fare_x1.5", "payment_mix", "volume_tenth"):
        _write(BY_NAME[name].apply(clean, rng), root / f"{name}.csv")
    (root / "empty.csv").write_text("")
    result = runner.invoke(
        dq.app, ["profile", str(root / "train"), "-o", str(root / "baseline.json")]
    )
    assert result.exit_code == 0, result.output
    return root


def check(ws, name, *extra):
    return runner.invoke(
        dq.app, ["check", str(ws / name), "-b", str(ws / "baseline.json"), *extra]
    )


def test_profile_reports_and_writes_baseline(workspace):
    data = json.loads((workspace / "baseline.json").read_text())
    assert len(data["batches"]) == 21 and data["version"] == dq.BASELINE_VERSION


def test_profile_accepts_globs_and_forced_keys(workspace, tmp_path):
    out = tmp_path / "b.json"
    r = runner.invoke(
        dq.app,
        [
            "profile",
            str(workspace / "train" / "taxi_2019-03-0*.csv"),
            "-o",
            str(out),
            "--key",
            "trip_id",
            "--key",
            "nope",
        ],
    )
    assert r.exit_code == 0 and "nope" in r.output
    assert len(json.loads(out.read_text())["batches"]) == 9


def test_clean_batch_exits_zero(workspace):
    r = check(workspace, "clean.csv")
    assert r.exit_code == 0 and "status OK" in r.output


@pytest.mark.parametrize(
    "name", ["rename_column", "fare_x1.5", "payment_mix", "volume_tenth"]
)
def test_corrupted_batch_exits_one(workspace, name):
    r = check(workspace, f"{name}.csv")
    assert r.exit_code == 1, r.output


def test_fail_on_threshold(workspace):
    # column_order-style info alerts never fail the default run, and --fail-on critical ignores warnings
    assert (
        check(workspace, "fare_x1.5.csv", "--fail-on", "critical").exit_code == 1
    )  # PSI is critical
    r = runner.invoke(
        dq.app,
        [
            "check",
            str(workspace / "clean.csv"),
            "-b",
            str(workspace / "baseline.json"),
            "--fail-on",
            "info",
        ],
    )
    assert r.exit_code == 0
    assert check(workspace, "clean.csv", "--fail-on", "bogus").exit_code != 0


def test_json_and_html_outputs(workspace, tmp_path):
    j, h = tmp_path / "r.json", tmp_path / "r.html"
    r = check(workspace, "fare_x1.5.csv", "--json", str(j), "--html", str(h))
    assert r.exit_code == 1
    data = json.loads(j.read_text())
    assert data["status"] == "critical" and any(
        a["check"] == "drift.numeric" and a["column"] == "fare" for a in data["alerts"]
    )
    page = h.read_text(encoding="utf-8")
    assert (
        page.startswith("<!doctype html>")
        and "drift.numeric" in page
        and "plotly" in page.lower()
    )
    # Case-insensitive and attribute-tolerant: `<SCRIPT>`, `<script type=...>` and
    # `</script >` are all script blocks to a browser, so all must be stripped.
    assert "https://" not in re.sub(
        r"<script\b.*?</script\s*>", "", page, flags=re.DOTALL | re.IGNORECASE
    ).replace("http://www.w3.org", "")


def test_report_for_clean_batch_has_no_charts(workspace, tmp_path):
    h = tmp_path / "ok.html"
    assert check(workspace, "clean.csv", "--html", str(h)).exit_code == 0
    page = h.read_text(encoding="utf-8")
    assert "No alerts" in page and "plotly-graph-div" not in page


def test_volume_alert_gets_a_chart(workspace, tmp_path):
    h = tmp_path / "v.html"
    check(workspace, "volume_tenth.csv", "--html", str(h))
    assert "Rows per batch" in h.read_text(encoding="utf-8")


def test_bad_inputs_exit_two(workspace, tmp_path):
    base = str(workspace / "baseline.json")
    assert (
        runner.invoke(
            dq.app, ["check", str(workspace / "empty.csv"), "-b", base]
        ).exit_code
        == 2
    )
    assert (
        runner.invoke(
            dq.app, ["check", str(tmp_path / "missing.csv"), "-b", base]
        ).exit_code
        == 2
    )
    assert (
        check(workspace, "clean.csv", "--config", str(tmp_path / "nope.toml")).exit_code
        == 2
    )
    (tmp_path / "junk.json").write_text("[]")
    assert (
        runner.invoke(
            dq.app,
            ["check", str(workspace / "clean.csv"), "-b", str(tmp_path / "junk.json")],
        ).exit_code
        == 2
    )
    assert (
        runner.invoke(
            dq.app,
            ["profile", str(tmp_path / "zzz*.csv"), "-o", str(tmp_path / "x.json")],
        ).exit_code
        == 2
    )


def test_bad_as_of_is_a_usage_error(workspace):
    assert check(workspace, "clean.csv", "--as-of", "yesterday-ish").exit_code != 0


def test_config_file_changes_the_outcome(workspace, tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text(
        '[severity_overrides]\n"schema.possible_rename" = "off"\n"schema.missing_column" = "off"\n"schema.new_column" = "off"\n'
    )
    r = check(workspace, "rename_column.csv", "--config", str(cfg))
    assert r.exit_code == 0


def test_stale_batch_via_as_of(workspace):
    r = check(workspace, "clean.csv", "--as-of", "2019-04-20T00:00:00")
    assert r.exit_code == 1 and "freshness.stale" in r.output


# ---------------------------------------------------------- HTML escaping
HOSTILE_COL = "</script><script>alert(1)</script>"
HOSTILE_VAL = "<img src=x onerror=alert(1)>"


def test_hostile_csv_content_cannot_inject_html():

    def frame(seed, weights, scale=1.0):
        r = np.random.default_rng(seed)
        return pl.DataFrame(
            {
                HOSTILE_COL: r.normal(100, 10, 200) * scale,
                "cat": r.choice(["ok", HOSTILE_VAL], 200, p=weights),
            }
        )

    base = dq.build_baseline([(f"<b{i}>.csv", frame(i, [0.7, 0.3])) for i in range(6)])
    res = dq.check_batch(
        base, frame(9, [0.05, 0.95], scale=2.0), name="<h1>evil</h1>.csv"
    )
    assert res.by_check("drift.numeric") and res.by_check("drift.categorical")
    page = render_report(res, base)
    assert (
        HOSTILE_COL not in page
        and HOSTILE_VAL not in page
        and "<h1>evil</h1>" not in page
    )
    assert "&lt;/script&gt;" in page  # escaped in the alerts table and the schema table
    assert r"\u003cimg" in page  # Plotly's own JSON escaping for the chart labels
    assert (
        "<img" not in page
        and "<script>alert" not in page
        and "</script><script>alert" not in page
    )
