"""Tests for the spending categorizer.

Run with:  uv run --with pytest --with polars --with plotly pytest -q
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from categorizer import (
    UNCATEGORIZED,
    categorize,
    categorize_transactions,
    compile_rules,
    load_rules,
    load_transactions,
    render_report,
    resolve_column,
)

SAMPLE_DIR = Path(__file__).parent / "sample_data"
RULES_PATH = Path(__file__).parent / "rules.json"


def test_resolve_column_auto_detects_common_alias() -> None:
    df = pl.DataFrame({"Date": ["2023-01-01"], "Description": ["x"], "Amount": ["1"]})
    assert resolve_column(df, "date", None) == "Date"
    assert resolve_column(df, "description", None) == "Description"
    assert resolve_column(df, "amount", None) == "Amount"


def test_resolve_column_raises_helpful_error() -> None:
    df = pl.DataFrame({"foo": [1]})
    with pytest.raises(ValueError, match="couldn't auto-detect"):
        resolve_column(df, "date", None)


def test_resolve_column_honors_override() -> None:
    df = pl.DataFrame({"weird_col": [1]})
    assert resolve_column(df, "amount", "weird_col") == "weird_col"


def test_load_transactions_format_a(tmp_path: Path) -> None:
    df, mapping = load_transactions(
        SAMPLE_DIR / "checking_export_a.csv", (None, None, None)
    )
    assert mapping.date == "Date"
    assert mapping.amount == "Amount"
    assert len(df) == 19
    assert df["date"].dtype == pl.Date
    payroll_row = df.filter(pl.col("description").str.contains("PAYROLL"))
    assert (payroll_row["amount"] > 0).all()


def test_load_transactions_format_b_parses_parens_and_currency(tmp_path: Path) -> None:
    df, mapping = load_transactions(
        SAMPLE_DIR / "checking_export_b.csv", (None, None, None)
    )
    assert mapping.date == "Posted Date"
    assert mapping.description == "Memo"
    assert mapping.amount == "Debit"
    assert len(df) == 11

    mortgage = df.filter(pl.col("description").str.contains("MORTGAGE"))
    assert mortgage["amount"].to_list() == [-1800.00]

    payroll = df.filter(pl.col("description").str.contains("PAYROLL"))
    assert payroll["amount"].to_list() == [2600.00]


def test_categorize_matches_keyword_case_insensitively() -> None:
    rules = compile_rules({"Dining": ["starbucks"]})
    assert categorize("STARBUCKS STORE #04521", rules) == "Dining"
    assert categorize("Unrelated purchase", rules) == UNCATEGORIZED


def test_categorize_transactions_end_to_end() -> None:
    df, _ = load_transactions(SAMPLE_DIR / "checking_export_a.csv", (None, None, None))
    rules = load_rules(RULES_PATH)
    categorized = categorize_transactions(df, rules)

    by_desc = {
        row["description"]: row["category"] for row in categorized.iter_rows(named=True)
    }
    assert by_desc["STARBUCKS STORE #04521"] == "Dining"
    assert by_desc["RENT PAYMENT JAN"] == "Rent/Mortgage"
    assert by_desc["PAYROLL DIRECT DEP ACME CORP"] == "Income"
    assert by_desc["MYSTERY VENDOR XYZ123"] == UNCATEGORIZED


def test_render_report_is_self_contained_html() -> None:
    df, _ = load_transactions(SAMPLE_DIR / "checking_export_a.csv", (None, None, None))
    rules = load_rules(RULES_PATH)
    categorized = categorize_transactions(df, rules)
    report = render_report(categorized)
    assert report.startswith("<!doctype html>")
    assert "<script" in report
    assert "Monthly Spending Report" in report
    assert "Uncategorized" in report


def test_main_writes_report(tmp_path: Path) -> None:
    from categorizer import main

    output = tmp_path / "out.html"
    exit_code = main([str(SAMPLE_DIR / "checking_export_b.csv"), "-o", str(output)])
    assert exit_code == 0
    assert output.exists()
    assert "Monthly Spending Report" in output.read_text(encoding="utf-8")


def test_main_missing_file_returns_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from categorizer import main

    exit_code = main([str(tmp_path / "nope.csv")])
    assert exit_code == 1
    assert "does not exist" in capsys.readouterr().err


def test_main_reports_unmappable_columns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from categorizer import main

    bad = tmp_path / "bad.csv"
    bad.write_text("foo,bar\n1,2\n")
    exit_code = main([str(bad)])
    assert exit_code == 1
    assert "couldn't auto-detect" in capsys.readouterr().err
