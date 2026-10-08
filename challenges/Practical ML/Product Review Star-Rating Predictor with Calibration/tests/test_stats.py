import numpy as np
import pytest
from helpers import make_probs
from review_stars import metrics, stats


def clustered(n_clusters=300, size=5, rho=0.0, seed=0):
    """Binary outcomes with an intra-cluster correlation: ``(correct, groups)``."""
    rng = np.random.default_rng(seed)
    shared = rng.normal(size=n_clusters)[:, None] * np.sqrt(rho)
    noise = rng.normal(size=(n_clusters, size)) * np.sqrt(1 - rho)
    correct = (shared + noise > 0.0).ravel()
    groups = np.repeat([f"p{i}" for i in range(n_clusters)], size)
    return correct, groups


def test_bootstrap_resamples_whole_clusters_with_replacement():
    groups = np.repeat(["a", "b", "c", "d"], [3, 1, 4, 2])
    seen = []
    for idx in stats.boot_indices(groups, n_boot=50, seed=0):
        counts = np.bincount(idx, minlength=len(groups))
        for g in "abcd":
            members = counts[groups == g]
            assert (
                len(set(members)) == 1
            )  # every row of a cluster is drawn the same number of times
        seen.append(tuple(counts[[0, 3, 4, 8]]))
    assert len(set(seen)) > 10  # and the draws really vary


def test_each_resample_has_as_many_clusters_as_the_original():
    groups = np.repeat(np.arange(20), 3)
    for idx in stats.boot_indices(groups, n_boot=20, seed=1):
        # rows drawn = sum over drawn clusters of their size: all sizes are 3
        assert len(idx) == 60


def test_same_seed_gives_the_same_resamples_which_is_what_makes_comparisons_paired():
    groups = np.repeat(np.arange(30), 4)
    a = list(stats.boot_indices(groups, n_boot=10, seed=7))
    b = list(stats.boot_indices(groups, n_boot=10, seed=7))
    c = list(stats.boot_indices(groups, n_boot=10, seed=8))
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert not all(np.array_equal(x, y) for x, y in zip(a, c, strict=True))


def test_cluster_bootstrap_is_wider_than_a_row_bootstrap_when_reviews_of_a_product_correlate():
    correct, groups = clustered(rho=0.6, seed=2)
    rows = np.arange(len(correct))  # every review its own cluster = the naive bootstrap

    def width(g):
        reps = [correct[i].mean() for i in stats.boot_indices(g, n_boot=300, seed=0)]
        lo, hi = np.percentile(reps, [2.5, 97.5])
        return hi - lo

    assert width(groups) > 1.4 * width(rows)


def test_row_and_cluster_bootstrap_agree_when_clusters_are_independent():
    correct, groups = clustered(rho=0.0, seed=3)
    rows = np.arange(len(correct))

    def width(g):
        reps = [correct[i].mean() for i in stats.boot_indices(g, n_boot=300, seed=0)]
        lo, hi = np.percentile(reps, [2.5, 97.5])
        return hi - lo

    assert width(groups) == pytest.approx(width(rows), rel=0.25)


def test_boot_metrics_returns_one_array_per_model_and_metric():
    P1, y = make_probs(n=1500, sharpen=1.0, seed=0)
    P2, _ = make_probs(n=1500, sharpen=2.0, seed=0)
    groups = np.repeat(np.arange(300), 5)
    reps = stats.boot_metrics(
        {"calibrated": P1, "sharp": P2}, y, groups, n_boot=12, seed=0, bins=10
    )
    assert set(reps) == {"calibrated", "sharp"}
    assert set(reps["sharp"]) == set(metrics.BUNDLE_KEYS)
    assert reps["sharp"]["ece"].shape == (12,)
    assert reps["sharp"]["ece"].mean() > reps["calibrated"]["ece"].mean()


def test_boot_metrics_is_the_same_serially_and_in_parallel():
    P, y = make_probs(n=1200, seed=1)
    groups = np.repeat(np.arange(240), 5)
    a = stats.boot_metrics({"m": P}, y, groups, n_boot=8, seed=3, bins=10, n_jobs=1)
    b = stats.boot_metrics({"m": P}, y, groups, n_boot=8, seed=3, bins=10, n_jobs=2)
    for k in a["m"]:
        assert np.allclose(a["m"][k], b["m"][k], equal_nan=True)


def test_paired_difference_is_tighter_than_the_difference_of_two_intervals():
    P1, y = make_probs(n=4000, sharpen=1.0, seed=2)
    P2, _ = make_probs(n=4000, sharpen=1.05, seed=2)  # nearly the same model
    groups = np.repeat(np.arange(800), 5)
    reps = stats.boot_metrics(
        {"a": P1, "b": P2}, y, groups, n_boot=200, seed=0, bins=10
    )
    a, b = reps["a"]["nll"], reps["b"]["nll"]
    paired = stats.ci(a - b)
    separate = (
        stats.ci(a)["hi"] - stats.ci(a)["lo"] + stats.ci(b)["hi"] - stats.ci(b)["lo"]
    )
    assert (paired["hi"] - paired["lo"]) < 0.3 * separate


def test_ci_reports_the_point_estimate_the_percentile_interval_and_how_many_reps_were_nan():
    reps = np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.nan])
    out = stats.ci(reps, point=3.0)
    assert out["est"] == 3.0 and out["n_nan"] == 1 and out["n"] == 6
    assert out["lo"] == pytest.approx(np.nanpercentile(reps, 2.5))
    assert out["hi"] == pytest.approx(np.nanpercentile(reps, 97.5))
    empty = stats.ci(np.array([np.nan, np.nan]))
    assert np.isnan(empty["lo"]) and empty["n_nan"] == 2


def test_ci_without_a_point_uses_the_replicate_mean():
    assert stats.ci(np.array([1.0, 3.0]))["est"] == 2.0


def test_a_bootstrap_of_an_empty_split_is_an_error():
    with pytest.raises(ValueError, match="empty"):
        list(stats.boot_indices(np.array([]), n_boot=3, seed=0))


def test_a_single_cluster_still_resamples_without_crashing():
    idx = next(iter(stats.boot_indices(np.array(["only"] * 6), n_boot=1, seed=0)))
    assert len(idx) == 6


def test_boot_metrics_can_also_return_per_bin_counts_for_reliability_bands():
    P, y = make_probs(n=1500, seed=0)
    groups = np.repeat(np.arange(300), 5)
    reps = stats.boot_metrics(
        {"m": P}, y, groups, n_boot=9, seed=0, bins=10, curve_bins=8
    )
    n, k = reps["m"]["reliability_n"], reps["m"]["reliability_k"]
    assert n.shape == k.shape == (9, 8) and (k <= n).all()
    sizes = [len(i) for i in stats.boot_indices(groups, 9, 0)]
    assert (
        n.sum(axis=1).astype(int).tolist() == sizes
    )  # every resampled review is in exactly one bin
    plain = stats.boot_metrics({"m": P}, y, groups, n_boot=9, seed=0, bins=10)
    assert "reliability_n" not in plain["m"]
    parallel = stats.boot_metrics(
        {"m": P}, y, groups, n_boot=9, seed=0, bins=10, n_jobs=2, curve_bins=8
    )
    assert np.array_equal(parallel["m"]["reliability_n"], n)
