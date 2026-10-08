from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from helpers import synthetic
from inbox_sorter.evaluate import (
    ablation,
    bootstrap_ci,
    daily_inbox_metrics,
    ece,
    pr_auc,
    reliability,
    roc_auc,
    summarize,
)
from inbox_sorter.features import FEATURE_GROUPS
from inbox_sorter.models import time_split

T0 = datetime.fromisoformat("2001-06-04T15:00")  # a Monday, 09:00 Houston


def test_pr_auc_matches_a_hand_computed_average_precision():
    # ranking: pos, neg, pos, neg -> precision at the positives 1/1 and 2/3
    assert pr_auc([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]) == pytest.approx((1 + 2 / 3) / 2)


def test_roc_auc_and_degenerate_labels():
    assert roc_auc([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]) == pytest.approx(0.75)
    assert np.isnan(pr_auc([0, 0, 0], [0.1, 0.2, 0.3]))
    assert np.isnan(roc_auc([1, 1], [0.1, 0.2]))


def test_ece_matches_a_hand_value_and_includes_probability_one():
    # bins 0.1 and 0.9: gap 0.1 each, half the mass in each -> ECE 0.1
    assert ece([0, 0, 1, 1], [0.1, 0.1, 0.9, 0.9], bins=10) == pytest.approx(0.1)
    assert ece([1, 1], [1.0, 1.0], bins=10) == pytest.approx(0.0)
    assert ece([0, 1], [0.0, 1.0], bins=10) == pytest.approx(0.0)


def test_reliability_lists_only_populated_bins_in_order():
    rows = reliability([0, 0, 1, 1], [0.1, 0.1, 0.9, 0.9], bins=10)
    assert [r["n"] for r in rows] == [2, 2]
    assert rows[0]["mean_pred"] == pytest.approx(0.1) and rows[0]["frac_pos"] == 0.0
    assert rows[1]["mean_pred"] == pytest.approx(0.9) and rows[1]["frac_pos"] == 1.0


def test_summarize_reports_base_rate_and_counts():
    s = summarize(np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.7, 0.1]))
    assert s["n"] == 4 and s["positives"] == 2 and s["pos_rate"] == 0.5
    assert s["pr_auc"] == pytest.approx((1 + 2 / 3) / 2)


def _day_frame(days):
    """days: list of acted-lists, one list per (mailbox 'm') day."""
    rows = []
    for d, acted in enumerate(days):
        for i, a in enumerate(acted):
            rows.append(
                {
                    "mailbox": "m",
                    "date": T0 + timedelta(days=d, minutes=i),
                    "acted": a,
                }
            )
    return pl.DataFrame(rows, schema_overrides={"date": pl.Datetime("us")})


def test_perfect_scores_give_ndcg_one_and_hand_precision_and_recall():
    df = _day_frame([[False, True, False, False, True, False]])  # 2 of 6 acted
    scores = df["acted"].to_numpy().astype(float)
    m = daily_inbox_metrics(df, scores)
    assert m.height == 1
    row = m.row(0, named=True)
    assert row["ndcg_at_5"] == pytest.approx(1.0)
    assert row["precision_at_3"] == pytest.approx(2 / 3)
    assert row["recall_top20"] == pytest.approx(1.0)  # top ceil(0.2*6)=2 holds both


def test_worst_scores_put_acted_mail_last():
    df = _day_frame([[True, False, False, False, False, False]])
    m = daily_inbox_metrics(df, -df["acted"].to_numpy().astype(float))
    row = m.row(0, named=True)
    assert row["precision_at_3"] == 0.0 and row["ndcg_at_5"] == 0.0
    assert row["recall_top20"] == 0.0


def test_days_without_a_mix_or_with_fewer_than_five_messages_are_excluded():
    df = _day_frame(
        [
            [True] * 6,  # all acted
            [False] * 6,  # none acted
            [True, False, False, False],  # only 4 messages
            [True, False, False, False, False],  # eligible
        ]
    )
    m = daily_inbox_metrics(df, np.arange(df.height, dtype=float))
    assert m.height == 1
    assert not m.drop_nulls().is_empty()
    assert np.isfinite(m.select(pl.exclude("mailbox", "day")).to_numpy()).all()


def test_ties_are_broken_by_a_seeded_shuffle_not_by_arrival_order():
    # 5 messages, acted one is first in arrival order; all scores tie
    df = _day_frame([[True, False, False, False, False]] * 60)
    flat = np.zeros(df.height)
    a = daily_inbox_metrics(df, flat, seed=1)["precision_at_3"].mean()
    b = daily_inbox_metrics(df, flat, seed=1)["precision_at_3"].mean()
    assert a == b
    # expectation for one positive among 5 with top-3 is 1/5, never the optimistic 1/3
    assert a == pytest.approx(1 / 5, abs=0.06)


def test_random_scores_recover_about_twenty_percent_in_the_top_fifth():
    rng = np.random.default_rng(0)
    days = []
    for _ in range(400):
        acted = np.zeros(10, bool)
        acted[rng.choice(10, 2, replace=False)] = True
        days.append(acted.tolist())
    df = _day_frame(days)
    m = daily_inbox_metrics(df, rng.random(df.height))
    assert m["recall_top20"].mean() == pytest.approx(0.2, abs=0.03)


def test_bootstrap_ci_is_deterministic_and_brackets_the_mean():
    x = np.random.default_rng(0).random(200)
    a = bootstrap_ci(x, seed=3)
    assert a == bootstrap_ci(x, seed=3)
    mean, lo, hi = a
    assert lo <= mean <= hi and hi - lo < 0.2
    assert all(np.isnan(v) for v in bootstrap_ci([], seed=0))


def test_ablation_drops_the_group_that_carries_the_signal_the_most():
    df = synthetic(900)
    train, val, test = time_split(df)
    drops = ablation(train, val, test, FEATURE_GROUPS, seed=0)
    assert set(drops) == set(FEATURE_GROUPS)
    assert max(drops, key=drops.get) == "sender_history"
    assert drops["sender_history"] > 0.05
