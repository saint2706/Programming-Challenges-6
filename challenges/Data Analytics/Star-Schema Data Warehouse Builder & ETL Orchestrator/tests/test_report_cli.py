"""The HTML report escapes everything it shows, and the CLI works end to end on the sample feed.

All files are written under pytest's tmp_path.
"""

from __future__ import annotations

from datetime import datetime

from helpers import SAMPLE
from star_warehouse import report
from star_warehouse.cli import app
from star_warehouse.dag import Task
from star_warehouse.warehouse import connect
from typer.testing import CliRunner

PAYLOAD = '<img src=x onerror="alert(1)"></script><script>alert(2)</script>'


def test_report_escapes_task_names_and_error_text(tmp_path):
    con = connect(tmp_path / "w.duckdb")
    con.execute(
        "INSERT INTO ops.run VALUES ('abc12345', DATE '2010-03-01', DATE '2010-04-01', ?, 'failed', ?, ?, NULL)",
        [PAYLOAD, datetime(2010, 1, 1, 2), datetime(2010, 1, 1, 3)],
    )
    con.execute(
        "INSERT INTO ops.task_run VALUES ('abc12345', ?, 1, 'failed', ?, ?, ?)",
        [PAYLOAD, datetime(2010, 1, 1, 2), datetime(2010, 1, 1, 3), PAYLOAD],
    )
    tasks = [Task(PAYLOAD, lambda: None), Task("ok", lambda: None, deps=(PAYLOAD,))]
    html = report.render(con, tasks, title=PAYLOAD)
    con.close()

    assert "<script" not in html.lower()
    assert "<img" not in html.lower()
    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;" in html
    assert '<meta http-equiv="Content-Security-Policy"' in html
    assert "default-src &#x27;none&#x27;" in html  # the policy, attribute-escaped
    # No external references: nothing is fetched from elsewhere.
    assert 'http-equiv="refresh"' not in html.lower()
    assert 'src="http' not in html.lower() and 'href="http' not in html.lower()


def test_cli_run_status_and_report_on_the_sample(tmp_path):
    db = tmp_path / "cli.duckdb"
    runner = CliRunner()

    result = runner.invoke(
        app, ["run", "2010-03", "--db", str(db), "--feed", str(SAMPLE)]
    )
    assert result.exit_code == 0, result.output
    assert "succeeded" in result.output

    status = runner.invoke(app, ["status", "--db", str(db)])
    assert status.exit_code == 0
    assert "2010-03-01 to 2010-04-01" in status.output

    out = tmp_path / "report.html"
    rep = runner.invoke(app, ["report", "--db", str(db), "--out", str(out)])
    assert rep.exit_code == 0, rep.output
    text = out.read_text(encoding="utf-8")
    assert "Attempts in the latest run" in text
    assert "<script" not in text.lower()


def test_cli_rejects_bad_input_without_creating_a_warehouse(tmp_path):
    db = tmp_path / "nope.duckdb"
    runner = CliRunner()
    bad = runner.invoke(app, ["run", "2010-13", "--db", str(db), "--feed", str(SAMPLE)])
    assert bad.exit_code == 2
    missing = runner.invoke(
        app,
        ["run", "2010-03", "--db", str(db), "--feed", str(tmp_path / "absent.parquet")],
    )
    assert missing.exit_code == 2
    assert not db.exists()


def test_cli_backfill_and_tick_report_bad_arguments_as_usage_errors(tmp_path):
    db = tmp_path / "nope.duckdb"
    runner = CliRunner()
    common = ["--db", str(db), "--feed", str(SAMPLE)]
    backfill = runner.invoke(app, ["backfill", "2010-13", "2011-01", *common])
    tick = runner.invoke(app, ["tick", "0 3 * * *", "--now", "garbage", *common])
    for result in (backfill, tick):
        assert result.exit_code == 2, result.output
        assert isinstance(result.exception, SystemExit)
    assert not db.exists()
