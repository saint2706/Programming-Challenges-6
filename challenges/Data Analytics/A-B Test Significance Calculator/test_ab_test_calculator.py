"""Tests for the A/B test significance calculator.

Run with:  uv run --with pytest --with polars --with plotly --with scipy --with numpy pytest -q
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import stats as scipy_stats

from ab_test_calculator import (
    analyze_csv,
    build_arg_parser,
    is_binary_outcome,
    main,
    plain_english,
    resolve_column,
    two_proportion_test,
    welch_t_test_from_raw,
    welch_t_test_from_stats,
)
import polars as pl

SAMPLE_DIR = Path(__file__).parent / "sample_data"


def test_resolve_column_auto_detects() -> None:
    assert resolve_column(["Group", "Converted"], ["group"], None, "group") == "Group"


def test_resolve_column_raises_helpful_error() -> None:
    with pytest.raises(ValueError, match="couldn't auto-detect"):
        resolve_column(["foo", "bar"], ["group"], None, "group")


def test_is_binary_outcome_detects_zero_one() -> None:
    assert is_binary_outcome(pl.Series([0, 1, 1, 0]))


def test_is_binary_outcome_rejects_continuous() -> None:
    assert not is_binary_outcome(pl.Series([1.2, 3.4, 5.6, 2.1]))


def test_two_proportion_test_matches_scipy_chi2() -> None:
    result = two_proportion_test("control", "variant", 120, 1000, 145, 1000, alpha=0.05)
    table = np.array([[120, 880], [145, 855]])
    chi2, p, _, _ = scipy_stats.chi2_contingency(table, correction=False)
    assert result.chi2 == pytest.approx(chi2)
    assert result.p_value == pytest.approx(p)
    assert result.control_rate == pytest.approx(0.12)
    assert result.variant_rate == pytest.approx(0.145)
    assert result.diff == pytest.approx(0.025)


def test_two_proportion_test_significant_case() -> None:
    # Huge, clearly-different samples -> should be significant.
    result = two_proportion_test("control", "variant", 100, 1000, 400, 1000, alpha=0.05)
    assert result.significant


def test_two_proportion_test_not_significant_case() -> None:
    # Tiny samples, small difference -> should not be significant.
    result = two_proportion_test("control", "variant", 5, 50, 6, 50, alpha=0.05)
    assert not result.significant


def test_welch_t_test_from_stats_matches_scipy() -> None:
    result = welch_t_test_from_stats(
        "control", "variant",
        control_mean=2.4, control_std=0.5, control_n=300,
        variant_mean=2.1, variant_std=0.45, variant_n=300,
        alpha=0.05,
    )
    t_stat, p = scipy_stats.ttest_ind_from_stats(
        mean1=2.1, std1=0.45, nobs1=300, mean2=2.4, std2=0.5, nobs2=300, equal_var=False
    )
    assert result.t_stat == pytest.approx(t_stat)
    assert result.p_value == pytest.approx(p)
    assert result.significant


def test_welch_t_test_from_raw_matches_from_stats() -> None:
    rng = np.random.default_rng(0)
    control = rng.normal(2.4, 0.5, 300)
    variant = rng.normal(2.1, 0.45, 300)
    from_raw = welch_t_test_from_raw("control", "variant", control, variant, alpha=0.05)
    from_stats = welch_t_test_from_stats(
        "control", "variant",
        control_mean=control.mean(), control_std=control.std(ddof=1), control_n=len(control),
        variant_mean=variant.mean(), variant_std=variant.std(ddof=1), variant_n=len(variant),
        alpha=0.05,
    )
    assert from_raw.t_stat == pytest.approx(from_stats.t_stat)
    assert from_raw.p_value == pytest.approx(from_stats.p_value)


def test_analyze_csv_conversion_data_auto_detects_proportions() -> None:
    result = analyze_csv(SAMPLE_DIR / "checkout_conversion.csv", None, None, alpha=0.05)
    assert result.kind == "proportions"
    assert result.control_total == 1000
    assert result.variant_total == 1000


def test_analyze_csv_load_time_data_detects_means() -> None:
    # "load_time_seconds" is a domain-specific metric name, not one of the generic
    # outcome aliases -- this is exactly what --outcome-col is for.
    result = analyze_csv(SAMPLE_DIR / "page_load_time.csv", None, "load_time_seconds", alpha=0.05)
    assert result.kind == "means"
    assert result.control_n == 300
    assert result.variant_n == 300
    assert result.variant_mean < result.control_mean


def test_analyze_csv_rejects_more_than_two_groups(tmp_path: Path) -> None:
    path = tmp_path / "three_groups.csv"
    path.write_text("group,converted\nA,1\nB,0\nC,1\n")
    with pytest.raises(ValueError, match="expected exactly 2 groups"):
        analyze_csv(path, None, None, alpha=0.05)


def test_plain_english_mentions_significance_language() -> None:
    sig = two_proportion_test("control", "variant", 100, 1000, 400, 1000, alpha=0.05)
    not_sig = two_proportion_test("control", "variant", 5, 50, 6, 50, alpha=0.05)
    assert "statistically significant" in plain_english(sig)
    assert "not statistically significant" in plain_english(not_sig)


def test_main_csv_mode(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main([str(SAMPLE_DIR / "checkout_conversion.csv")])
    assert exit_code == 0
    assert "variant" in capsys.readouterr().out.lower()


def test_main_writes_html_report(tmp_path: Path) -> None:
    output = tmp_path / "report.html"
    exit_code = main([str(SAMPLE_DIR / "checkout_conversion.csv"), "-o", str(output)])
    assert exit_code == 0
    assert output.exists()
    assert "A/B Test" in output.read_text(encoding="utf-8")


def test_main_summary_flags_proportions(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "--control-conversions", "120", "--control-total", "1000",
            "--variant-conversions", "145", "--variant-total", "1000",
        ]
    )
    assert exit_code == 0
    assert "12.00%" in capsys.readouterr().out


def test_main_summary_flags_means(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "--control-mean", "2.4", "--control-std", "0.5", "--control-n", "300",
            "--variant-mean", "2.1", "--variant-std", "0.45", "--variant-n", "300",
        ]
    )
    assert exit_code == 0
    assert "Cohen's d" in capsys.readouterr().out


def test_main_rejects_csv_and_flags_together(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            str(SAMPLE_DIR / "checkout_conversion.csv"),
            "--control-conversions", "1", "--control-total", "2",
            "--variant-conversions", "1", "--variant-total", "2",
        ]
    )
    assert exit_code == 1
    assert "either a CSV path or summary-stat flags" in capsys.readouterr().err


def test_main_rejects_incomplete_flag_set(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--control-conversions", "1", "--control-total", "2"])
    assert exit_code == 1
    assert "required together" in capsys.readouterr().err


def test_main_missing_file_returns_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main([str(tmp_path / "nope.csv")])
    assert exit_code == 1
    assert "does not exist" in capsys.readouterr().err


def test_arg_parser_builds() -> None:
    assert build_arg_parser() is not None
