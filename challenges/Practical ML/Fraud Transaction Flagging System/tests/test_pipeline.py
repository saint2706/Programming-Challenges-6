from __future__ import annotations

import json

from fraud_flagging import models, pipeline, report
from fraud_flagging.features import FEATURES, GROUPS


def test_result_has_every_section_and_is_json_serialisable(out):
    r = out["result"]
    for key in (
        "candidates",
        "calibration",
        "fixed_precision",
        "fixed_recall",
        "cost",
        "cost_sweep",
        "capacity",
        "threshold_stability",
        "precision_at_k",
        "errors_at_precision_0.9",
        "importance",
        "drift",
        "ablation",
        "rolling_origin",
    ):
        assert key in r, key
    json.dumps(r)


def test_the_main_model_is_chosen_on_validation_not_test(out):
    r = out["result"]
    val = {
        s: r["candidates"][f"lightgbm_{s}"]["validation"]["pr_auc"]
        for s in models.STRATEGIES
    }
    assert r["main_strategy"] == max(val, key=val.get)


def test_the_supervised_models_beat_an_unsupervised_one_and_chance_by_a_wide_margin(
    out,
):
    r = out["result"]["candidates"]
    base = out["result"]["fraud_rate"]["test"]
    assert r[f"lightgbm_{out['result']['main_strategy']}"]["test"]["pr_auc"] > 10 * base
    assert r["logistic"]["test"]["pr_auc"] > r["isolation_forest"]["test"]["pr_auc"]
    for c in r.values():
        lo, hi = c["test_pr_auc_ci"]
        assert lo <= c["test"]["pr_auc"] <= hi + 0.05


def test_platt_scaling_does_not_hurt_the_probabilities(out):
    cal = out["result"]["calibration"]
    assert cal["platt_scaled"]["log_loss"] <= cal["raw"]["log_loss"] + 0.01
    assert set(cal["by_strategy"]) == set(models.STRATEGIES)


def test_higher_precision_targets_give_higher_thresholds(out):
    thr = [p["threshold"] for p in out["result"]["fixed_precision"].values() if p]
    assert thr == sorted(thr)
    rec = [p["threshold"] for p in out["result"]["fixed_recall"].values() if p]
    assert rec == sorted(rec, reverse=True)  # more recall needs a lower threshold


def test_fixed_recall_points_reach_their_recall_on_validation(out):
    for target, p in out["result"]["fixed_recall"].items():
        if p:
            assert p["validation"]["recall"] >= float(target) - 1e-9


def test_the_oracle_is_the_cheapest_policy_and_a_cheap_review_makes_flagging_pay(out):
    rows = out["result"]["cost"]
    for row in rows:
        for policy in ("static", "amount_aware"):
            assert row["trivial_oracle"] <= row[policy]["total"] + 1e-6
    # With cheap reviews and a strong signal, amount-aware flagging beats doing nothing. At a high
    # review cost one false alarm on a large amount can outweigh the few frauds caught, so no such
    # claim is made there.
    for row in rows[:2]:
        assert row["amount_aware"]["total"] < row["trivial_flag_nothing"]


def test_cost_intervals_bracket_the_cost(out):
    for row in out["result"]["cost"]:
        lo, hi = row["amount_aware"]["cost_ci"]
        assert lo <= row["amount_aware"]["total"] <= hi


def test_capacity_never_flags_more_than_the_hourly_budget(out):
    for row in out["result"]["capacity"]:
        assert row["by_risk"]["flagged"] <= row["per_hour"] * row["hours"]
        assert row["by_expected_loss"]["flagged"] <= row["per_hour"] * row["hours"]


def test_rolling_origin_trains_only_on_the_past(out):
    folds = out["result"]["rolling_origin"]
    assert [f["from_hour"] for f in folds] == [24.0, 30.0, 36.0, 42.0]
    assert all(f["frauds"] > 0 and f["lightgbm"]["pr_auc"] > 0 for f in folds)


def test_importance_covers_all_features_and_one_ablation_per_group(out):
    imp = out["result"]["importance"]
    assert {i["feature"] for i in imp} == set(FEATURES)
    assert abs(sum(i["gain_share"] for i in imp) - 1) < 1e-6
    assert [a["dropped"] for a in out["result"]["ablation"]] == list(GROUPS)
    assert (
        next(a for a in out["result"]["ablation"] if a["dropped"] == "pca")[
            "delta_pr_auc"
        ]
        < -0.3
    )


def test_meta_and_predictions_hold_what_the_cli_and_app_need(out):
    meta = out["meta"]
    assert meta["features"] == FEATURES and set(meta["thresholds"]) == {
        "precision",
        "recall",
        "cost_static",
    }
    preds = out["predictions"]
    assert (
        set(preds["part"].unique()) == {"validation", "test"}
        and preds["p"].is_between(0, 1).all()
    )


def test_switches_turn_the_slow_parts_off(frame):
    r = pipeline.run(frame, n_boot=5, rounds=30, ablate=False, rolling=False)["result"]
    assert "ablation" not in r and "rolling_origin" not in r


def test_report_markdown_and_charts_are_written(out, tmp_path):
    md = report.markdown(out["result"])
    for heading in (
        "Fixed precision",
        "Fixed recall",
        "Cost",
        "Review capacity",
        "Rolling origin",
    ):
        assert heading in md
    assert "nan" not in md.lower().replace("finance", "")
    report.write(out, tmp_path)
    for name in (
        "report.md",
        "pr_curve.png",
        "cost.png",
        "stability.png",
        "reliability.png",
        "importance.png",
    ):
        assert (tmp_path / name).stat().st_size > 500
