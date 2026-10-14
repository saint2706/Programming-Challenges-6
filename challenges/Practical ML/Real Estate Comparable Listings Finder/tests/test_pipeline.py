"""The real experiment, run once (``result`` fixture). Structure and invariants, plus loose sanity."""

from __future__ import annotations

import json

import numpy as np
from comp_finder import pipeline, report

METHODS = ("knn", "ridge", "hedonic", "leaf", "avm")


def test_windows_and_targets(result):
    d = result.metrics["data"]
    assert (d["fit"], d["validation"], d["test"]) == (14_633, 2_228, 4_752)
    assert d["test_first_sales"] == len(result.targets) == 4_654
    eng = result.engine
    assert (result.targets >= d["fit"] + d["validation"]).all()
    assert (eng.sales["n_prior"].to_numpy()[result.targets] == 0).all()
    assert eng.n_fit == d["fit"] + d["validation"]


def test_every_target_got_k_comps_from_earlier_sales(result):
    eng, rows = result.engine, result.targets
    comps = eng.rank("leaf", rows[::25], 5)
    assert (comps >= 0).all()
    for q, row in zip(rows[::25], comps, strict=True):
        assert (eng.sold_days[row] < eng.sold_days[q]).all()


def test_headline_table_is_complete_and_intervals_contain_the_estimates(result):
    main = result.metrics["main"]
    assert tuple(main) == METHODS
    for m, v in main.items():
        assert v["n"] == 4_654
        assert 0.04 < v["mdape"] < 0.25, m
        assert 0 < v["within10"] < v["within20"] <= 1
        for stat in ("mdape", "within10", "within20"):
            lo, hi = v["ci"][stat]
            assert lo <= v[stat] <= hi


def test_similarity_definitions_rank_as_measured(result):
    main, paired = result.metrics["main"], result.metrics["paired"]
    assert main["leaf"]["mdape"] < main["hedonic"]["mdape"] < main["knn"]["mdape"]
    _, _, hi = paired["leaf-knn"]["mdape"]
    assert hi < 0  # the whole paired interval says leaf proximity beats scaled kNN


def test_validation_choice_is_the_grid_minimum(result):
    grid = result.metrics["validation_grid"]
    assert len(grid) == 4
    best = min(grid, key=lambda g: g["mean_mdape"])
    s = result.settings
    assert (s.extrapolate, s.combine) == (best["extrapolate"], best["combine"])
    assert s.k == 5


def test_adjustments_help_and_segments_partition_the_targets(result):
    for m, v in result.metrics["adjust_ablation"].items():
        assert v["none"] > v["full"], m
    for segs in result.metrics["segments"].values():
        assert sum(s["n"] for s in segs.values()) == 4_654


def test_metrics_are_json_serialisable_and_the_report_is_written(
    result, tmp_path, monkeypatch
):
    json.dumps(result.metrics, default=str)
    path = report.write_report(result, tmp_path / "results")
    text = path.read_text(encoding="utf-8")
    assert "## Test window" in text and "GBM AVM (no comps)" in text
    for name in ("metrics.json", "mdape.png", "k_sweep.png"):
        assert (tmp_path / "results" / name).stat().st_size > 1000
    monkeypatch.setenv("COMP_FINDER_HOME", str(tmp_path))
    saved = pipeline.saved_settings()
    assert (saved.extrapolate, saved.combine) == (
        result.settings.extrapolate,
        result.settings.combine,
    )


def test_timing_and_pool_stats_are_recorded(result):
    ms = result.metrics["ms_per_subject"]
    assert all(0 < v < 200 for v in ms.values())
    assert ms["leaf"] > ms["knn"]
    pool = result.metrics["pool_size_at_test"]
    assert pool["first"] > 10_000 and pool["last"] < 21_613
    assert np.isfinite(result.metrics["zipcode"]["with_zip_categorical"])
