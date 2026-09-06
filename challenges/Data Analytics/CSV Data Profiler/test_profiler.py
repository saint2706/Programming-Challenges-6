"""Tests for the CSV Data Profiler.

Run with:  uv run --with pytest --with polars --with plotly pytest -q
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from profiler import (
    classify_semantic_type,
    iqr_outlier_count,
    profile_dataset,
    render_report,
    sniff_dialect,
)

SAMPLE_DIR = Path(__file__).parent / "sample_data"


def test_sniff_dialect_detects_comma(tmp_path: Path) -> None:
    path = tmp_path / "comma.csv"
    path.write_text("a,b,c\n1,2,3\n")
    encoding, delimiter = sniff_dialect(path)
    assert delimiter == ","
    assert encoding in {"utf-8", "latin-1"}


def test_sniff_dialect_detects_semicolon(tmp_path: Path) -> None:
    path = tmp_path / "semi.csv"
    path.write_text("a;b;c\n1;2;3\n4;5;6\n")
    _, delimiter = sniff_dialect(path)
    assert delimiter == ";"


def test_classify_semantic_type_numeric() -> None:
    assert classify_semantic_type(pl.Series("x", [1, 2, 3, None, 5])) == "numeric"


def test_classify_semantic_type_boolean_from_strings() -> None:
    assert (
        classify_semantic_type(pl.Series("x", ["true", "false", "true"])) == "boolean"
    )


def test_classify_semantic_type_categorical_vs_text() -> None:
    categorical = pl.Series("dept", ["Eng", "Sales", "Eng", "Sales", "Eng"] * 4)
    text = pl.Series("bio", [f"free form note number {i}" for i in range(20)])
    assert classify_semantic_type(categorical) == "categorical"
    assert classify_semantic_type(text) == "text"


def test_classify_semantic_type_datetime() -> None:
    dates = pl.Series("d", ["2023-01-01", "2023-02-01", "2023-03-01"]).str.to_date()
    assert classify_semantic_type(dates) == "datetime"


def test_iqr_outlier_count_flags_extremes() -> None:
    values = pl.Series("x", [10, 11, 12, 13, 14, 12, 11, 10, 9999])
    assert iqr_outlier_count(values) == 1


def test_iqr_outlier_count_zero_iqr_is_safe() -> None:
    values = pl.Series("x", [5, 5, 5, 5])
    assert iqr_outlier_count(values) == 0


def test_iqr_outlier_count_too_few_values() -> None:
    assert iqr_outlier_count(pl.Series("x", [1, 2])) == 0


def test_profile_dataset_clean_employees() -> None:
    profile = profile_dataset(SAMPLE_DIR / "clean_employees.csv", sep=None, top_n=10)
    assert profile.n_rows == 15
    assert profile.n_cols == 6
    assert profile.duplicate_rows == 0

    by_name = {c.name: c for c in profile.columns}
    assert by_name["salary"].semantic_type == "numeric"
    assert by_name["department"].semantic_type == "categorical"
    assert by_name["hire_date"].semantic_type == "datetime"
    assert by_name["active"].semantic_type == "boolean"
    assert by_name["name"].null_count == 0


def test_profile_dataset_messy_transactions_flags_issues() -> None:
    profile = profile_dataset(SAMPLE_DIR / "messy_transactions.csv", sep=None, top_n=10)
    by_name = {c.name: c for c in profile.columns}

    assert by_name["customer"].null_count == 1
    assert by_name["amount"].null_count == 1
    assert by_name["amount"].stats["outliers"] >= 2  # the two "data entry error" rows


def test_profile_dataset_detects_duplicate_rows(tmp_path: Path) -> None:
    path = tmp_path / "dupes.csv"
    path.write_text("a,b\n1,x\n2,y\n1,x\n3,z\n1,x\n")
    profile = profile_dataset(path, sep=None, top_n=10)
    assert profile.duplicate_rows == 3  # all three "1,x" rows count as duplicated


def test_render_report_is_self_contained_html() -> None:
    profile = profile_dataset(SAMPLE_DIR / "clean_employees.csv", sep=None, top_n=10)
    report = render_report(profile)
    assert report.startswith("<!doctype html>")
    assert "<script" in report  # plotly bundle embedded inline
    assert "CSV Data Profile" in report
    assert "salary" in report


def test_main_writes_report(tmp_path: Path) -> None:
    from profiler import main

    output = tmp_path / "out.html"
    exit_code = main([str(SAMPLE_DIR / "clean_employees.csv"), "-o", str(output)])
    assert exit_code == 0
    assert output.exists()
    assert "CSV Data Profile" in output.read_text(encoding="utf-8")


def test_main_missing_file_returns_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from profiler import main

    exit_code = main([str(tmp_path / "does_not_exist.csv")])
    assert exit_code == 1
    assert "does not exist" in capsys.readouterr().err
