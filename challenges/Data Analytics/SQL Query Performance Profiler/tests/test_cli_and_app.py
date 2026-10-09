"""The command line and the Streamlit app, run against a temporary copy of the committed sample."""

from pathlib import Path

import pytest
from sql_profiler.cli import app
from streamlit.testing.v1 import AppTest
from typer.testing import CliRunner

APP = Path(__file__).resolve().parents[1] / "src" / "sql_profiler" / "app.py"
SLOW = "SELECT count(*) FROM invoice_lines WHERE substr(invoice, 1, 3) = '536'"
FAST = "SELECT count(*) FROM invoice_lines WHERE invoice >= '536' AND invoice < '537'"


def run(*args):
    return CliRunner().invoke(app, list(args))


def test_explain_prints_the_plan_and_the_findings(home):
    result = run("explain", SLOW)
    assert result.exit_code == 0, result.output
    assert "SEQ_SCAN invoice_lines" in result.output
    assert "rows returned" in result.output


def test_explain_writes_an_html_report(home, tmp_path):
    out = tmp_path / "nested" / "report.html"
    result = run("explain", FAST, "--html", str(out))
    assert result.exit_code == 0, result.output
    html = out.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>") and "<svg" in html


def test_explain_reads_a_sql_file(home, tmp_path):
    path = tmp_path / "q.sql"
    path.write_text("SELECT count(*) FROM products", encoding="utf-8")
    result = run("explain", str(path))
    assert result.exit_code == 0 and "products" in result.output


@pytest.mark.parametrize(
    "sql", ["DROP TABLE invoices", "SELECT 1; SELECT 2", "SELEC oops"]
)
def test_unsafe_or_broken_sql_is_refused_with_exit_code_2(home, sql):
    result = run("explain", sql)
    assert result.exit_code == 2
    assert "error:" in result.output


def test_a_query_that_runs_too_long_is_stopped(home):
    result = run(
        "explain",
        "SELECT sum(a.price * b.price) FROM invoice_lines a, invoice_lines b",
        "--timeout",
        "0.3",
    )
    assert result.exit_code == 2 and "stopped after" in result.output


def test_compare_checks_the_rows_and_reports_a_verdict(home):
    result = run("compare", SLOW, FAST, "--repeats", "3")
    assert result.exit_code == 0, result.output
    assert "identical rows" in result.output
    assert "rows read from tables" in result.output
    assert any(
        v in result.output
        for v in ("B is faster", "A is faster", "no clear difference")
    )


def test_compare_says_when_the_answers_differ(home):
    result = run("compare", "SELECT 1 AS x", "SELECT 2 AS x", "--repeats", "3")
    assert result.exit_code == 0
    assert "only in A" in result.output


def test_lab_markdown_has_one_row_per_experiment(home):
    result = run("lab", "--repeats", "3", "--markdown")
    assert result.exit_code == 0, result.output
    rows = [
        l for l in result.output.splitlines() if l.startswith("| ") and "---" not in l
    ]
    assert len(rows) == 1 + 8  # header + experiments
    assert "sort-vs-top-n | by design" in result.output


def test_index_lab_prints_a_table(home):
    result = run("index-lab", "--repeats", "3")
    assert result.exit_code == 0, result.output
    assert "CREATE INDEX" in result.output


def test_schema_lists_every_table(home):
    result = run("schema")
    assert result.exit_code == 0
    for table in ("invoice_lines", "invoices", "products", "customers"):
        assert table in result.output


def start():
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_app_loads_with_four_tabs(home):
    at = start()
    assert len(at.tabs) == 4
    assert at.title[0].value == "SQL query performance profiler"


def test_profiling_the_starter_query_shows_metrics_and_findings(home):
    at = start()
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert {m.label for m in at.metric} >= {
        "Wall clock",
        "CPU time",
        "Rows returned",
        "Operators",
    }


def test_the_app_refuses_a_destructive_statement_without_running_it(home):
    at = start()
    at.text_area[0].set_value("DROP TABLE invoices").run()
    at.button[0].click().run()
    assert any("only SELECT" in e.value for e in at.error)
    at.text_area[0].set_value("SELECT count(*) FROM invoices").run()
    at.button[0].click().run()
    assert not at.error and not at.exception


def test_a_sql_error_is_shown_not_raised(home):
    at = start()
    at.text_area[0].set_value("SELECT nope FROM invoices").run()
    at.button[0].click().run()
    assert any("Binder" in e.value or "nope" in e.value for e in at.error)
    assert not at.exception


def test_comparing_an_experiment_reports_the_rows_and_a_verdict(home):
    at = start()
    at.selectbox[0].set_value("function-on-key").run()
    compare_button = next(b for b in at.button if b.label == "Compare")
    compare_button.click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("identical rows" in s.value for s in at.success)
    assert {m.label for m in at.metric} >= {"A median", "B median", "Verdict"}
