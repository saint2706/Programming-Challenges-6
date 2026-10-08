import json

import numpy as np
import pytest
from helpers import make_features, tiny_config
from review_stars import pipeline, report

PNG = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def full(tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    features = make_features(seed=0)
    features.meta["prepared"] = {
        "windows": {"end": "2023-03", "test": ["2022-04", "2023-03"], "cal": ["2021-10", "2022-03"],
                    "val": ["2021-04", "2021-09"], "train_before": "2021-04", "cal_months": 6,
                    "val_months": 6, "notes": ["cal window widened from 6 to 9 months (target 5 rows)"]},
        "duplicates_removed": {"Appliances": 12, "Software": 40},
        "auto_title_share_by_split": {"train": 0.12, "val": 0.0, "cal": 0.0, "test": 0.0,
                                       "ood_test": 0.001, "ood_pool": 0.0},
        "splits": {s: {"target": 100, "available": 120, "selected": 100} for s in
                   ("train", "val", "cal", "test", "ood_test", "ood_pool")},
    }  # fmt: skip
    rep = pipeline.run_all(features, out, tiny_config())
    return rep, out


def test_formatting_helpers_show_missing_values_as_a_dash_not_none():
    assert (
        report.fmt(0.12345) == "0.123"
        and report.fmt(None) == "–"
        and report.fmt(float("nan")) == "–"
    )
    ci = {"est": 0.1234, "lo": 0.1, "hi": 0.15}
    assert report.fmt_ci(ci) == "0.123 [0.100, 0.150]"
    assert report.fmt_ci({"est": 0.2, "lo": None, "hi": None}) == "0.200"
    assert report.fmt_ci(None) == "–"
    assert report.fmt_pct(0.1234) == "12.3%"
    # the recalibration study reports a spread over random draws (mean, lo, hi), not est
    assert report.fmt_ci({"mean": 0.2, "lo": 0.1, "hi": 0.3}) == "0.200 [0.100, 0.300]"


def test_the_markdown_tables_are_built_from_the_results_not_typed_in(full):
    rep, _ = full
    md = report.tables_markdown(rep)
    for heading in (
        "## Data and windows", "## In-time test", "## Out-of-domain test (Software)",
        "## Calibrators", "## Paired differences", "## Conformal prediction",
        "## Recalibration budget", "## Selective prediction", "## Input-text ablation", "## Slices",
    ):  # fmt: skip
        assert heading in md, heading
    row = rep["stages"]["evaluate"]["splits"]["test"]["rows"][
        "ens-classification|temperature"
    ]
    assert (
        report.fmt_ci(row["ci"]["ece"]) in md
    )  # the exact interval, formatted from the report
    assert f"{rep['data']['sizes']['train']:,}" in md
    assert (
        "cal window widened from 6 to 9 months" in md
    )  # the notes travel with the numbers
    assert "12.0%" in md  # the era effect: the auto-title share per split


def test_every_model_and_calibrator_gets_a_row(full):
    rep, _ = full
    md = report.tables_markdown(rep)
    for m in rep["models"]:
        assert f"| {m} |" in md
    section = md.split("## Calibrators")[1].split("##")[0]
    assert "vector" in section and "isotonic" in section


def test_a_partial_report_renders_what_exists_and_says_what_is_missing(tmp_path):
    features = make_features(seed=1)
    rep = pipeline.run_all(
        features, tmp_path, tiny_config(), stages=("train", "calibrate")
    )
    md = report.tables_markdown(rep)
    assert "## Data and windows" in md
    assert "not computed yet" in md and "evaluate" in md


def test_skipped_ablation_modes_are_reported_as_skipped(full):
    md = report.tables_markdown(full[0])
    section = md.split("## Input-text ablation")[1].split("\n## ")[0]
    assert "title_text" in section and "skipped" in section


def test_render_writes_the_tables_and_every_figure(full, tmp_path):
    rep, _ = full
    written = report.render(rep, tmp_path)
    names = {p.name for p in written}
    assert names == {
        "tables.md", "reliability.png", "coverage.png", "recalibration.png",
        "risk_coverage.png", "framings.png",
    }  # fmt: skip
    for p in written:
        assert p.stat().st_size > 500
        if p.suffix == ".png":
            assert p.read_bytes()[:8] == PNG
    assert "## In-time test" in (tmp_path / "tables.md").read_text(encoding="utf-8")


def test_render_skips_figures_whose_stage_is_missing_without_failing(tmp_path):
    rep = pipeline.run_all(
        make_features(seed=2),
        tmp_path / "r",
        tiny_config(),
        stages=("train", "calibrate"),
    )
    written = report.render(rep, tmp_path / "out")
    assert {p.name for p in written} == {"tables.md"}


def test_render_from_a_report_file_on_disk(full, tmp_path):
    _, out = full
    written = report.render_file(out / "report.json", tmp_path / "again")
    assert (tmp_path / "again" / "tables.md").exists() and len(written) == 6


def test_render_refuses_a_missing_report_with_the_command_to_make_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="review-stars benchmark"):
        report.render_file(tmp_path / "nope.json", tmp_path)


def test_numbers_in_the_markdown_survive_a_round_trip_through_the_json_file(full):
    rep, out = full
    again = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report.tables_markdown(again) == report.tables_markdown(rep)
    assert np.isfinite(
        again["stages"]["evaluate"]["splits"]["test"]["rows"]["linear-ordinal|none"][
            "point"
        ]["nll"]
    )


def test_the_recalibration_table_has_a_number_in_every_budget_row(full):
    rep, _ = full
    md = report.tables_markdown(rep)
    section = md.split("## Recalibration budget")[1].split("\n## ")[0]
    budget_rows = [
        line for line in section.splitlines() if line.startswith(("| 30 |", "| 80 |"))
    ]
    assert len(budget_rows) == 2
    for line in budget_rows:
        cells = [c.strip() for c in line.strip("|").split("|")]
        assert all(c != "–" for c in cells), line  # every ECE and T cell is filled
