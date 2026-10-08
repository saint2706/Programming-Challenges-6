import numpy as np
import polars as pl
import pytest
from helpers import make_classification_preds
from review_stars import calibrate, metrics, shift


def pool_and_test(sharpen=2.2, n_pool=6000, n_test=15_000):
    pool, y_pool = make_classification_preds(n=n_pool, seed=1, sharpen=sharpen)
    test, y_test = make_classification_preds(n=n_test, seed=2, sharpen=sharpen)
    return pool, y_pool, test, y_test


# ---------------------------------------------------------------- recalibration budget


def test_more_target_labels_give_a_better_and_steadier_temperature():
    pool, y_pool, test, y_test = pool_and_test()
    out = shift.recalibration_curve(
        pool, y_pool, test, y_test, ns=(50, 500, 5000), draws=12, seed=0
    )
    rows = {r["n"]: r for r in out["rows"]}
    assert set(rows) == {50, 500, 5000}
    ece = [rows[n]["ece"]["mean"] for n in (50, 500, 5000)]
    assert ece[0] > ece[1] > ece[2]
    assert (
        rows[50]["ece"]["hi"] - rows[50]["ece"]["lo"]
        > rows[5000]["ece"]["hi"] - rows[5000]["ece"]["lo"]
    )
    assert rows[5000]["T"]["mean"] == pytest.approx(2.2, rel=0.08)
    assert out["uncalibrated"]["ece"] > 0.12 and rows[5000]["ece"]["mean"] < 0.02


def test_the_curve_is_reproducible_for_a_seed_and_the_draws_really_differ():
    pool, y_pool, test, y_test = pool_and_test(n_pool=3000, n_test=4000)
    a = shift.recalibration_curve(
        pool, y_pool, test, y_test, ns=(100,), draws=8, seed=3
    )
    b = shift.recalibration_curve(
        pool, y_pool, test, y_test, ns=(100,), draws=8, seed=3
    )
    c = shift.recalibration_curve(
        pool, y_pool, test, y_test, ns=(100,), draws=8, seed=4
    )
    assert a == b and a != c
    assert a["rows"][0]["T"]["hi"] > a["rows"][0]["T"]["lo"]


def test_budgets_larger_than_the_pool_are_skipped_and_said_so():
    pool, y_pool, test, y_test = pool_and_test(n_pool=300, n_test=1000)
    out = shift.recalibration_curve(
        pool, y_pool, test, y_test, ns=(50, 300, 5000), draws=3, seed=0
    )
    assert [r["n"] for r in out["rows"]] == [50, 300]
    assert out["skipped"] == [5000]


def test_a_tiny_pool_containing_one_star_does_not_crash():
    pool, y_pool, test, y_test = pool_and_test(n_pool=600, n_test=500)
    only_fives = np.full(len(y_pool), 5, dtype=np.int64)
    out = shift.recalibration_curve(
        pool, only_fives, test, y_test, ns=(50,), draws=3, seed=0
    )
    lo, hi = calibrate.T_BOUNDS
    assert lo <= out["rows"][0]["T"]["mean"] <= hi and np.isfinite(
        out["rows"][0]["nll"]["mean"]
    )


def test_reference_rows_report_the_uncalibrated_and_full_pool_fits():
    pool, y_pool, test, y_test = pool_and_test()
    out = shift.recalibration_curve(
        pool, y_pool, test, y_test, ns=(100,), draws=3, seed=0
    )
    assert set(out) >= {"rows", "skipped", "uncalibrated", "full_pool"}
    assert out["full_pool"]["ece"] < out["uncalibrated"]["ece"]
    assert out["full_pool"]["T"] == pytest.approx(2.2, rel=0.08)


# ---------------------------------------------------------------- slices


def frame(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "parent_asin": [f"B{i}" for i in rng.integers(0, 300, n)],
            "verified": rng.uniform(size=n) < 0.8,
        }
    ), rng.integers(2, 600, n).astype(np.int32)


def test_slice_masks_split_by_product_novelty_verification_and_length():
    df, n_tok = frame()
    seen = {f"B{i}" for i in range(150)}
    masks = shift.slice_masks(df, seen, n_tok)
    assert set(masks) == {
        "all", "unseen_product", "seen_product", "verified", "unverified",
        "tokens<=64", "tokens 65-128", "tokens>128",
    }  # fmt: skip
    n = len(df)
    assert masks["all"].all() and masks["all"].shape == (n,)
    assert np.array_equal(masks["unseen_product"], ~masks["seen_product"])
    assert np.array_equal(masks["verified"], ~masks["unverified"])
    assert (masks["tokens<=64"] | masks["tokens 65-128"] | masks["tokens>128"]).all()
    assert not (masks["tokens<=64"] & masks["tokens 65-128"]).any()
    expected_unseen = ~df["parent_asin"].is_in(list(seen)).to_numpy()
    assert np.array_equal(masks["unseen_product"], expected_unseen)


def test_slice_table_scores_each_slice_and_skips_the_ones_that_are_too_small():
    P, y = make_classification_preds(n=3000, seed=1, sharpen=1.5)
    P = P.proba()
    masks = {
        "all": np.ones(3000, bool),
        "first_half": np.arange(3000) < 1500,
        "tiny": np.arange(3000) < 10,
    }
    table = shift.slice_table(P, y, masks, bins=10, min_n=50)
    assert table["all"]["n"] == 3000 and table["all"]["acc"] == pytest.approx(
        metrics.accuracy(P, y)
    )
    assert table["first_half"]["ece"] == pytest.approx(
        metrics.top_label_ece(P[:1500], y[:1500], 10)
    )
    assert table["tiny"] == {"n": 10, "skipped": "fewer than 50 reviews"}
    assert set(table["all"]) >= {"n", "acc", "nll", "ece", "mean_conf"}
