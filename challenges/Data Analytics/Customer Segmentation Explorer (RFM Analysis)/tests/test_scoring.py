import numpy as np
import polars as pl
import pytest
from rfm_explorer import scoring
from rfm_explorer.scoring import (
    SEGMENT_GRID,
    SEGMENT_ORDER,
    fixed_scores,
    fm_score,
    kmeans_segments,
    quantile_scores,
    score_customers,
    silhouette_by_k,
    summarize,
)


def table(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "customer_id": np.arange(n),
            "country": ["UK"] * n,
            "recency_days": rng.integers(1, 365, n),
            "frequency": rng.geometric(0.3, n),
            "monetary": rng.lognormal(6, 1, n).round(2),
        }
    )


def test_quantile_scores_split_distinct_values_evenly():
    assert quantile_scores(np.arange(10)).tolist() == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]


def test_lower_is_better_reverses_the_scale():
    scores = quantile_scores(np.arange(10), higher_is_better=False)
    assert scores.tolist() == [5, 5, 4, 4, 3, 3, 2, 2, 1, 1]


def test_tied_values_always_share_a_score():
    values = [1] * 40 + list(range(2, 62))
    assert len(set(quantile_scores(values)[:40])) == 1
    # what the common qcut(rank(method="first")) recipe does to the same ties
    first_rank = np.argsort(np.argsort(values, kind="stable"), kind="stable")
    assert len(set(np.floor(first_rank[:40] / 20))) > 1


def test_a_heavy_tie_block_can_skip_a_score_value():
    # 35% of customers at 1 all land in group 1 and the next block in group 3, so score 2 is
    # never used (the real data has 35.5% at one purchase). That is what honouring ties costs.
    assert set(quantile_scores([1] * 35 + [2] * 20 + list(range(3, 48)))) == {
        1,
        3,
        4,
        5,
    }


def test_quantile_scores_of_nothing_and_of_one_value():
    assert quantile_scores([]).size == 0
    assert len(set(quantile_scores([5, 5, 5, 5]))) == 1  # one tie block holds everyone


def test_fixed_scores_edges_belong_to_the_lower_score():
    assert fixed_scores([1, 2, 3, 4, 9, 10], [1, 2, 4, 9]).tolist() == [
        1,
        2,
        3,
        3,
        4,
        5,
    ]


def test_fixed_scores_reversed_for_recency():
    # edges are the upper bounds of scores 5, 4, 3, 2
    values = [5, 30, 31, 90, 270, 271]
    assert fixed_scores(
        values, [30, 90, 180, 270], higher_is_better=False
    ).tolist() == [5, 5, 4, 4, 2, 1]


def test_fixed_scores_reject_unsorted_edges():
    with pytest.raises(ValueError, match="ascending"):
        fixed_scores([1], [5, 1])


def test_segment_grid_covers_every_cell_with_known_names():
    assert set(SEGMENT_GRID) == {(r, c) for r in range(1, 6) for c in range(1, 6)}
    assert set(SEGMENT_GRID.values()) == set(SEGMENT_ORDER)


def test_best_and_worst_corners():
    assert SEGMENT_GRID[(5, 5)] == "Champions"
    assert SEGMENT_GRID[(1, 1)] == "Lost"
    assert SEGMENT_GRID[(1, 5)] == "Can't Lose Them"


def test_fm_score_rounds_half_up():
    assert fm_score(np.array([2, 1, 5, 4]), np.array([3, 1, 5, 5])).tolist() == [
        3,
        1,
        5,
        5,
    ]


@pytest.mark.parametrize("method", scoring.METHODS)
def test_scored_table_shape(method):
    t = table()
    s = score_customers(t, method)
    assert s.height == t.height
    assert {"r", "f", "m", "segment", "priority"} <= set(s.columns)
    assert s["r"].min() >= 1 and s["r"].max() <= 5
    assert s["segment"].null_count() == 0


def test_identical_customers_get_identical_segments():
    base = table(50)
    t = pl.concat([base, base.with_columns(customer_id=pl.col("customer_id") + 1000)])
    s = score_customers(t, "quintile").sort("customer_id")
    assert s["segment"][:50].to_list() == s["segment"][50:].to_list()


def test_recent_frequent_big_spenders_are_champions():
    t = pl.DataFrame(
        {
            "customer_id": list(range(100)),
            "country": ["UK"] * 100,
            "recency_days": list(range(1, 101)),
            "frequency": list(range(100, 0, -1)),
            "monetary": [float(x) for x in range(10000, 0, -100)],
        }
    )
    s = score_customers(t, "quintile").sort("customer_id")
    assert s["segment"][0] == "Champions"
    assert s["segment"][-1] == "Lost"
    assert s["priority"][0] > s["priority"][-1]


def test_custom_edges_change_the_scores():
    t = table()
    default = score_customers(t, "fixed")
    strict = score_customers(t, "fixed", edges={"monetary": [1e6, 2e6, 3e6, 4e6]})
    assert strict["m"].max() == 1 < default["m"].max()


def test_kmeans_is_deterministic_and_ranks_clusters_best_first():
    t = table(300)
    names1, p1 = kmeans_segments(t, 4, seed=1)
    names2, p2 = kmeans_segments(t, 4, seed=1)
    assert names1 == names2 and p1.tolist() == p2.tolist()
    assert sorted(set(p1)) == [1.0, 2.0, 3.0, 4.0]
    assert next(n for n, p in zip(names1, p1) if p == 4.0).startswith("K1 ")
    s = (
        score_customers(t, "kmeans", k=4, seed=1)
        .group_by("priority")
        .agg(pl.col("monetary").mean())
        .sort("priority")
    )
    assert s["monetary"][-1] > s["monetary"][0]


def test_kmeans_survives_negative_net_spend():
    t = table(60).with_columns(
        monetary=pl.when(pl.col("customer_id") < 5)
        .then(-10.0)
        .otherwise(pl.col("monetary"))
    )
    assert score_customers(t, "kmeans", k=3).height == 60


def test_bad_arguments():
    with pytest.raises(ValueError, match="method"):
        score_customers(table(), "magic")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="k must"):
        kmeans_segments(table(10), 11)
    with pytest.raises(ValueError, match="k must"):
        kmeans_segments(table(10), 1)


def test_empty_input_gives_empty_scored_table():
    s = score_customers(table().head(0), "quintile")
    assert s.is_empty()
    assert {"r", "f", "m", "segment", "priority"} <= set(s.columns)


def test_silhouette_by_k_returns_one_score_per_k():
    out = silhouette_by_k(table(150), ks=range(2, 5))
    assert sorted(out) == [2, 3, 4]
    assert all(-1 <= v <= 1 for v in out.values())


def test_summary_shares_add_up_and_best_segment_is_first():
    s = summarize(score_customers(table(300), "quintile"))
    assert s["customer_share"].sum() == pytest.approx(1.0)
    assert s["money_share"].sum() == pytest.approx(1.0)
    assert s["priority"].to_list() == sorted(s["priority"].to_list(), reverse=True)
