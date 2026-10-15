from __future__ import annotations

import json

import polars as pl
from stockout_risk import features, pipeline, report
from stockout_risk.features import GROUPS


def test_result_has_every_section_and_is_json_serialisable(out):
    r = out["result"]
    for key in (
        "baselines",
        "logistic",
        "lightgbm",
        "lightgbm_raw",
        "worklist",
        "cost",
        "newly_at_risk",
        "per_category",
        "drift",
        "censoring",
        "importance",
        "ablation",
        "label_definitions",
        "reliability",
    ):
        assert key in r, key
    json.dumps(r)


def test_the_model_beats_every_baseline_on_the_synthetic_table(out):
    r = out["result"]
    best_baseline = max(b["pr_auc"] for b in r["baselines"].values())
    assert r["lightgbm"]["pr_auc"] > best_baseline
    assert r["lightgbm"]["pr_auc"] > r["base_rate"]["test"] + 0.1


def test_calibrated_scores_stay_sane_on_a_small_validation_window(out):
    """Isotonic regression can overfit a few thousand validation rows (here it does, a little), so
    the test only holds it to "still a usable probability"; the real run has 700,000 rows."""
    r = out["result"]
    assert (
        r["lightgbm"]["ece"] < 0.1
        and r["lightgbm"]["log_loss"] < r["lightgbm_raw"]["log_loss"] + 0.05
    )
    assert r["lightgbm"]["pr_auc"] > r["lightgbm_raw"]["pr_auc"] - 0.03


def test_one_ablation_row_per_group_and_one_label_row_per_definition(out):
    assert [a["dropped"] for a in out["result"]["ablation"]] == list(GROUPS)
    assert [d["hours"] for d in out["result"]["label_definitions"]] == list(
        pipeline.LABEL_HOURS
    )


def test_the_main_label_row_reuses_the_main_model(out):
    main = next(d for d in out["result"]["label_definitions"] if d["hours"] == 4)
    assert main["lightgbm"]["pr_auc"] == out["result"]["lightgbm_raw"]["pr_auc"]


def test_importance_is_sorted_and_covers_all_features(out):
    imp = out["result"]["importance"]
    assert len(imp) == len(features.FEATURES)
    shap = [i["mean_abs_shap"] for i in imp]
    assert (
        shap == sorted(shap, reverse=True)
        and abs(sum(i["gain_share"] for i in imp) - 1) < 1e-6
    )


def test_the_worklist_beats_chance_at_every_size(out):
    for row in out["result"]["worklist"]["lightgbm"]:
        assert row["lift"] > 1.0


def test_censoring_table_shows_sales_understating_demand_on_stockout_days(out):
    rows = out["result"]["censoring"]
    assert rows[0]["oos_hours"] == "0"
    # with half the day out of stock, the stocked hours' rate implies far more than was sold
    deep = next(c for c in rows if c["oos_hours"] == "8-11")
    assert deep["mean_demand_16h"] > deep["mean_sale"]


def test_predictions_table_holds_validation_and_test(out):
    preds = out["predictions"]
    assert set(preds["part"].unique()) == {"validation", "test"}
    assert preds["p"].is_between(0, 1).all()


def test_no_ablation_or_label_study_when_switched_off(feat):
    r = pipeline.run(feat, ablate=False, other_labels=False, rounds=20)["result"]
    assert "ablation" not in r and "label_definitions" not in r


def test_baselines_are_scores_over_the_frame(feat):
    b = pipeline.baselines(feat.head(100), 4)
    assert set(b) == {
        "yesterday",
        "same_weekday_last_week",
        "heavy_rate_7d",
        "mean_oos_hours_14d",
    }
    assert all(len(v) == 100 for v in b.values())


def test_report_markdown_names_the_sections(out):
    md = report.markdown(out["result"])
    for heading in (
        "Models on the test window",
        "Worklist",
        "Cost of a threshold",
        "Newly at risk",
        "Censored sales",
    ):
        assert heading in md
    assert isinstance(md, str) and "nan" not in md.lower()


def test_report_and_charts_are_written(out, tmp_path):
    report.write(out, tmp_path)
    for name in (
        "report.md",
        "reliability.png",
        "pr_curve.png",
        "cost.png",
        "importance.png",
    ):
        assert (tmp_path / name).stat().st_size > 500


def test_load_features_caches_next_to_the_data(home, monkeypatch, feat):
    calls = []
    monkeypatch.setattr(pipeline.data, "load", lambda: calls.append(1) or None)
    monkeypatch.setattr(pipeline.features, "build", lambda _: feat.head(50))
    first = pipeline.load_features()
    second = pipeline.load_features()
    assert (
        calls == [1]
        and first.equals(second)
        and (home / "data" / "features.parquet").exists()
    )
    pipeline.load_features(refresh=True)
    assert calls == [1, 1]
    assert isinstance(first, pl.DataFrame)


def test_pr_points_are_spaced_in_recall_and_span_the_curve():
    import numpy as np

    rng = np.random.default_rng(0)
    y = (rng.random(100_000) < 0.3).astype(int)
    s = rng.normal(np.where(y == 1, 1.0, 0.0), 1.0)
    recall, precision = report.pr_points(y, s, n=200)
    assert len(recall) <= 201 and recall.max() == 1.0 and recall.min() == 0.0
    assert np.all(
        np.diff(np.sort(recall)) < 0.02
    )  # no gaps: the top of the curve is there
    assert precision[recall.argmin()] == 1.0  # sklearn's closing point
