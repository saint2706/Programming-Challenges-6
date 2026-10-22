"""Planted patterns are found, noise is not, sentences say what the numbers say."""

from dataclasses import replace

import numpy as np
import polars as pl
import pytest
from helpers import events, noise_table, planted
from report_generator.config import Config
from report_generator.engine import build_report


@pytest.fixture(scope="module")
def report():
    return build_report(planted(0), title="planted", row_label="days")


def by_id(report, fid):
    found = [f for f in report.candidate_findings if f.id == fid]
    assert found, f"{fid} not among {[f.id for f in report.candidate_findings]}"
    return found[0]


def test_trend_is_found_with_the_planted_size(report):
    f = by_id(report, "trend:visits")
    assert f.template == "trend.sa" and f.facts["verb"].text == "climbed"
    assert f.facts["change"].value == pytest.approx(
        0.57, abs=0.08
    )  # planted: +60% over the window
    assert f.stats["direction"] == "up" and f.stats["p_adjusted"] < Config().fdr
    assert "between 2019-01-01 and 2021-12-21" in f.text


def test_seasonality_peaks_where_it_was_planted(report):
    f = by_id(report, "season:visits")
    assert f.facts["peak"].text == "June" and f.facts["trough"].text == "December"
    assert f.facts["swing"].value == pytest.approx(0.65, abs=0.15)
    assert f.stats["cycles"] == 3 and f.stats["p"] <= 0.01


def test_the_level_shift_is_dated_and_sized(report):
    f = by_id(report, "shift:score:86")
    assert f.template == "shift.up" and f.facts["date"].text.startswith("2020-08")
    assert f.facts["before"].value == pytest.approx(100, abs=3)
    assert f.facts["after"].value == pytest.approx(140, abs=4)


def test_the_collapsed_week_is_the_outlier(report):
    f = by_id(report, "anomaly:visits")
    assert (
        f.facts["date"].text.startswith("2021-03") and f.facts["side"].text == "below"
    )
    assert f.facts["deviation"].value > 0.3


def test_group_difference_and_relationship(report):
    g = by_id(report, "group:plan:spend")
    assert (g.facts["hi"].text, g.facts["lo"].text) == ("pro", "basic")
    assert g.facts["gap"].value == pytest.approx(2.6, abs=0.15)
    c = by_id(report, "correlation:x:y")
    assert c.template == "correlation.pos" and c.facts["rho"].value > 0.9


def test_nothing_is_said_about_the_noise_column(report):
    assert not any("noise" in f.columns for f in report.candidate_findings)


def test_every_planted_statement_passes_audit_and_verification(report):
    assert report.dropped == []
    assert report.verified and all(c.ok for cs in report.checks.values() for c in cs)
    assert sum(len(v) for v in report.checks.values()) > 30


def test_pure_noise_produces_almost_nothing():
    total = trends = 0
    for seed in range(25):
        r = build_report(noise_table(seed), verify=False)
        total += len(r.candidate_findings)
        trends += sum(
            f.type in ("trend", "seasonality", "shift", "group", "correlation")
            for f in r.candidate_findings
        )
    assert trends == 0 and total <= 3, (trends, total)


def test_concentration_in_an_event_table():
    r = build_report(events(0), row_label="orders")
    f = next(
        f for f in r.candidate_findings if f.id.startswith("concentration:customer_id")
    )
    assert f.template == "concentration.entity" and f.facts["share"].value >= 0.6
    assert "customers" in f.text and r.dropped == []


def test_config_moves_the_gates():
    strict = replace(Config(), min_trend_change=2.0)
    assert not any(
        f.id == "trend:visits"
        for f in build_report(
            planted(0), config=strict, verify=False
        ).candidate_findings
    )
    capped = build_report(
        planted(0), config=replace(Config(), max_findings=2), verify=False
    )
    assert len(capped.findings) == 2


def test_selection_caps_sections_and_orders_them(report):
    sections = [f.section for f in report.findings]
    assert max(sections.count(s) for s in set(sections)) <= Config().max_per_section
    from report_generator.engine import SECTION_ORDER

    assert sections == sorted(sections, key=SECTION_ORDER.index)


def test_data_notes_report_the_quality_problems():
    frame = planted(0).with_columns(
        pl.when(pl.int_range(pl.len()) % 4 == 0)
        .then(None)
        .otherwise(pl.col("temp"))
        .alias("temp")
    )
    frame = pl.concat([frame, frame.head(30)])
    r = build_report(frame, row_label="days")
    kinds = {f.id for f in r.notes}
    assert "quality:missing" in kinds and "quality:duplicates" in kinds
    assert (
        next(f for f in r.notes if f.id == "quality:duplicates").facts["n"].value == 30
    )


def test_missing_days_are_reported():
    frame = planted(0)
    frame = frame.filter(
        ~((pl.int_range(pl.len()) >= 100) & (pl.int_range(pl.len()) < 260))
    )
    notes = {f.id: f for f in build_report(frame, verify=False).notes}
    assert notes["quality:gaps"].facts["n"].value == 160


def test_series_centred_on_zero_get_no_percent_claims():
    rng = np.random.default_rng(0)
    t = np.arange(730)
    frame = noise_table(0).with_columns(
        pl.Series("swing", -10 + 0.03 * t + rng.normal(0, 1, 730))
    )
    r = build_report(
        frame, verify=False
    )  # a strong trend, but "up 200%" from a start near zero is nonsense
    assert not any(
        "swing" in f.columns and f.type in ("trend", "shift", "anomaly", "latest")
        for f in r.candidate_findings
    )
