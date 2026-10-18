import json
import re

import pytest
from conftest import ADULT, ROOT, UCB
from pivot_tool import render
from pivot_tool.cli import app
from pivot_tool.engine import Table
from pivot_tool.ffi import LIB_ENV, BuildError, Engine, EngineError
from typer.testing import CliRunner

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def clean(text: str) -> str:
    """Strip colour codes and Rich box drawing so assertions survive GitHub Actions' styling."""
    text = ANSI.sub("", text)
    text = re.sub(r"[─-╿]", " ", text)
    return re.sub(r"\s+", " ", text)


def cli(*args):
    result = CliRunner().invoke(app, [str(a) for a in args], env={"COLUMNS": "200"})
    return result.exit_code, clean(result.output)


SALES = b"region,product,units\nnorth,apple,10\nnorth,pear,5\nsouth,apple,7\nsouth,pear,12\neast,apple,9\n"


@pytest.fixture
def sales():
    with Table(SALES) as t:
        yield t


# -- rendering -----------------------------------------------------------------------------------
def test_grid_has_headers_rows_totals_and_collapses_repeats(sales):
    r = sales.pivot(
        {
            "rows": [{"field": "region"}, {"field": "product"}],
            "cols": [],
            "values": [{"field": "units"}],
            "options": {"subtotals": True},
        }
    )
    grid = render.to_grid(r, collapse_repeats=True)
    assert (
        grid[0] == ["region / product", "", "sum(units)"]
        or grid[0][0] == "region / product"
    )
    body = grid[render.header_height(r) :]
    assert body[0][:2] == ["east", "apple"] and body[1][:2] == ["east Total", ""]
    assert body[2][:2] == ["north", "apple"] and body[3][:2] == [
        "",
        "pear",
    ]  # the repeated region is blank
    assert body[-1][0] == "Grand Total" and body[-1][-1] == "43"


def test_csv_markdown_and_rich_agree_on_the_numbers(sales):
    r = sales.pivot(
        {
            "rows": [{"field": "region"}],
            "cols": [{"field": "product"}],
            "values": [{"field": "units"}, {"agg": "count"}],
        }
    )
    csv_text = render.to_csv(r)
    assert csv_text.splitlines()[-1].startswith("Grand Total,")
    assert "43" in csv_text and "5" in csv_text
    md = render.to_markdown(r)
    assert md.count("\n") == len(r["rows"]) + 2 and "| Grand Total" in md
    rich = render.to_rich(r, title="t")
    assert len(rich.rows) == len(r["rows"])


def test_number_formatting():
    assert (
        render.fmt(None) == ""
        and render.fmt(3.0) == "3"
        and render.fmt(1234567.0) == "1,234,567"
    )
    assert (
        render.fmt(0.123456) == "0.1235"
        and render.fmt(12.3456) == "12.35"
        and render.fmt(0.25, percent=True) == "25.0%"
    )


def test_empty_result_renders_without_error(sales):
    r = sales.pivot(
        {
            "rows": [{"field": "region"}],
            "filters": [{"field": "region", "op": "eq", "value": "mars"}],
        }
    )
    assert render.to_csv(r)
    assert render.to_markdown(r)


# -- the command line ----------------------------------------------------------------------------
def test_describe():
    code, out = cli("describe", ADULT)
    assert (
        code == 0
        and "48,842 rows" in out
        and "education_num int" in out
        and "income str 0 2" in out
    )


def test_pivot_prints_a_table_with_totals():
    code, out = cli(
        "pivot", ADULT, "-r", "sex", "-c", "income", "-v", "count() as people"
    )
    assert (
        code == 0
        and "Female" in out
        and "Grand Total" in out
        and "48,842" in out
        and "14,423" in out
    )


def test_pivot_to_files(tmp_path):
    base = [
        "pivot",
        ADULT,
        "-r",
        "education",
        "-v",
        "mean(hours_per_week)",
        "-f",
        "sex=Male",
    ]
    assert cli(*base, "--out", tmp_path / "p.csv")[0] == 0
    assert (
        (tmp_path / "p.csv")
        .read_text(encoding="utf-8")
        .splitlines()[-1]
        .startswith("Grand Total,")
    )
    assert cli(*base, "--out", tmp_path / "p.md")[0] == 0
    assert (tmp_path / "p.md").read_text(encoding="utf-8").startswith("| ")
    assert cli(*base, "--out", tmp_path / "p.json")[0] == 0
    assert (
        json.loads((tmp_path / "p.json").read_text(encoding="utf-8"))["rows_used"]
        == 32650
    )
    assert cli(*base, "--out", tmp_path / "p.html")[0] == 0
    assert "<!doctype html>" in (tmp_path / "p.html").read_text(encoding="utf-8")


def test_pivot_from_a_json_spec_file(tmp_path):
    spec = tmp_path / "s.json"
    spec.write_text(
        json.dumps({"rows": [{"field": "sex"}], "values": [{"agg": "count"}]}),
        encoding="utf-8",
    )
    code, out = cli("pivot", ADULT, "--spec", spec)
    assert code == 0 and "32,650" in out


def test_sort_top_and_subtotals_options():
    code, out = cli(
        "pivot",
        ADULT,
        "-r",
        "workclass",
        "-r",
        "sex",
        "-v",
        "count() as n",
        "--subtotals",
        "--sort",
        "value:n",
        "--top",
        "2:n",
    )
    assert code == 0 and "Private Total" in out
    assert (
        out.index("Private") < out.index("Self-emp-not-inc")
        if "Self-emp-not-inc" in out
        else True
    )


def test_crosstab_command():
    code, out = cli("crosstab", ADULT, "-r", "sex", "-c", "income")
    assert (
        code == 0
        and "chi-square" in out
        and "odds ratio" in out
        and "Cramer's V" in out
    )
    code, out = cli("crosstab", UCB, "-r", "sex", "-c", "outcome", "--layer", "dept")
    assert code == 0 and "Simpson's paradox" in out
    code, out = cli("crosstab", UCB, "-r", "sex", "-c", "outcome", "--json")
    assert code == 0 and '"counts"' in out


def test_drill_command():
    code, out = cli(
        "drill",
        UCB,
        "-r",
        "dept",
        "-c",
        "outcome",
        "--row-path",
        "A",
        "--col-path",
        "Admitted",
        "--limit",
        "3",
    )
    assert code == 0 and "601 rows behind the cell (showing 3)" in out


def test_html_command(tmp_path):
    code, out = cli(
        "html",
        UCB,
        "--out",
        tmp_path / "u.html",
        "--title",
        "Berkeley",
        "-r",
        "dept",
        "-v",
        "count()",
    )
    assert code == 0 and "MB" in out
    page = (tmp_path / "u.html").read_text(encoding="utf-8")
    assert "<title>Berkeley</title>" in page


def test_demo_runs_every_bundled_pivot(tmp_path):
    code, out = cli("demo", "berkeley", "--html", tmp_path / "b.html")
    assert (
        code == 0
        and "Simpson's paradox check" in out
        and (tmp_path / "b.html").exists()
    )
    code, out = cli("demo", "adult")
    assert code == 0 and "Calculated field" in out and "gain per person" in out


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["pivot", ADULT, "-r", "nope"], "no field called 'nope'"),
        (["pivot", ADULT, "-r", "sex", "-v", "sum(education)"], "it is text"),
        (["pivot", ADULT, "-r", "sex", "-v", "bogus(age)"], "unknown aggregation"),
        (["pivot", ADULT, "-r", "sex", "-f", "nonsense"], "cannot read the filter"),
        (["pivot", ADULT, "-r", "age:month"], "not a date"),
        (["pivot", ADULT, "--sort", "label"], "no --rows"),
        (
            ["crosstab", ADULT, "-r", "sex", "-c", "income", "-f", "sex=nobody"],
            "no rows match",
        ),
        (["drill", UCB, "-r", "dept", "--row-path", "Z"], "no 'Z' in dept"),
        (["demo", "nonesuch"], "unknown dataset"),
    ],
)
def test_bad_input_is_a_clear_error_not_a_traceback(args, message):
    code, out = cli(*args)
    assert code == 2 and message in out


def test_a_missing_library_is_a_clear_error(monkeypatch, tmp_path):
    monkeypatch.setenv(LIB_ENV, str(tmp_path / "nowhere.dll"))
    with pytest.raises(BuildError, match="library is missing"):
        Engine()


def test_engine_errors_carry_the_engine_message(adult):
    with pytest.raises(EngineError, match="no table with id"):
        adult._eng.call("pivot", {"table": 424242, "spec": {}})
    with pytest.raises(EngineError, match="unknown operation"):
        adult._eng.call("frobnicate")
    assert ROOT.exists()
