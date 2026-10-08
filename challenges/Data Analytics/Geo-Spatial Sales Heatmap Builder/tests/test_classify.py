"""Classification schemes, checked against brute force and against independent libraries."""

from __future__ import annotations

import itertools

import jenkspy
import numpy as np
import pytest
from sales_heatmap.classify import (
    assign,
    class_counts,
    compute_edges,
    equal_interval_edges,
    gvf,
    jenks_edges,
    quantile_edges,
    std_dev_edges,
)


def within_ss(values: np.ndarray, edges: np.ndarray) -> float:
    cls = assign(values, edges)
    return sum(
        float(((values[cls == c] - values[cls == c].mean()) ** 2).sum())
        for c in np.unique(cls)
    )


def brute_force_best_ss(values: np.ndarray, k: int) -> float:
    """Optimum over *every* way to cut the sorted values into k contiguous groups."""
    v = np.sort(values)
    best = np.inf
    for cuts in itertools.combinations(range(1, len(v)), k - 1):
        groups = np.split(v, cuts)
        best = min(best, sum(float(((g - g.mean()) ** 2).sum()) for g in groups))
    return best


class TestJenks:
    @pytest.mark.parametrize("seed", range(8))
    @pytest.mark.parametrize("k", [2, 3, 4])
    def test_matches_brute_force_optimum(self, seed, k):
        rng = np.random.default_rng(seed)
        values = rng.lognormal(0, 1.2, size=12)
        edges = jenks_edges(values, k)
        assert within_ss(values, edges) == pytest.approx(
            brute_force_best_ss(values, k), rel=1e-9
        )

    @pytest.mark.parametrize("k", [3, 5, 7])
    def test_matches_jenkspy_on_continuous_data(self, k):
        rng = np.random.default_rng(42)
        values = rng.lognormal(3, 1.0, size=700)
        ours = jenks_edges(values, k)
        theirs = np.array(jenkspy.jenks_breaks(values, n_classes=k))
        np.testing.assert_allclose(ours, theirs, rtol=1e-9)

    def test_never_worse_than_jenkspy_when_values_tie(self):
        rng = np.random.default_rng(3)
        values = rng.integers(0, 40, size=500).astype(float)  # heavy ties
        ours = jenks_edges(values, 5)
        theirs = np.array(jenkspy.jenks_breaks(values, n_classes=5))
        assert within_ss(values, ours) <= within_ss(values, theirs) + 1e-6

    def test_edges_span_the_data_and_increase(self):
        values = np.random.default_rng(1).exponential(5, 300)
        edges = jenks_edges(values, 6)
        assert edges[0] == values.min() and edges[-1] == values.max()
        assert np.all(np.diff(edges) > 0)

    def test_fewer_distinct_values_than_classes(self):
        edges = jenks_edges([1, 1, 2, 2, 3], 5)
        assert len(edges) - 1 == 3  # one class per distinct value, no empty classes
        assert list(assign([1, 2, 3], edges)) == [0, 1, 2]

    def test_all_identical_values(self):
        edges = jenks_edges([7.0] * 20, 4)
        assert list(edges) == [7.0, 7.0]
        assert set(assign([7.0] * 3, edges)) == {0}

    def test_numerically_stable_for_large_offsets(self):
        # sales in dollars: a 1e12 offset would swamp a naive sum-of-squares formulation
        base = 1e12
        values = base + np.array(
            [0, 1, 2, 3, 100, 101, 102, 103, 1000, 1001, 1002, 1003], dtype=float
        )
        edges = jenks_edges(values, 3)
        assert list(assign(values, edges)) == [0] * 4 + [1] * 4 + [2] * 4

    def test_sampled_path_for_many_distinct_values(self, monkeypatch):
        monkeypatch.setattr("sales_heatmap.classify.JENKS_MAX_DISTINCT", 50)
        values = np.random.default_rng(5).normal(size=400)
        edges = jenks_edges(values, 4)
        assert len(edges) == 5 and np.all(np.diff(edges) > 0)
        assert gvf(values, assign(values, edges)) > 0.85

    def test_beats_other_schemes_on_variance_explained(self):
        values = np.random.default_rng(9).lognormal(2, 1.5, size=800)
        scores = {
            s: gvf(values, assign(values, compute_edges(values, s, 5)))
            for s in ("quantile", "equal_interval", "jenks")
        }
        assert scores["jenks"] >= max(scores["quantile"], scores["equal_interval"])


class TestOtherSchemes:
    def test_quantile_matches_numpy(self):
        values = np.arange(1, 101, dtype=float)
        np.testing.assert_allclose(
            quantile_edges(values, 4), np.quantile(values, [0, 0.25, 0.5, 0.75, 1])
        )

    def test_quantile_classes_are_balanced(self):
        values = np.random.default_rng(0).lognormal(size=1000)
        counts = class_counts(assign(values, quantile_edges(values, 5)), 5)
        assert max(counts) - min(counts) <= 2

    def test_quantile_collapses_tied_edges(self):
        values = [0] * 80 + list(range(1, 21))
        edges = quantile_edges(values, 5)
        assert len(edges) - 1 < 5 and np.all(np.diff(edges) > 0)

    def test_equal_interval_is_linear(self):
        np.testing.assert_allclose(
            equal_interval_edges([0, 10, 3, 7], 5), [0, 2, 4, 6, 8, 10]
        )

    def test_std_dev_edges_are_mean_plus_multiples_of_sd(self):
        values = np.random.default_rng(2).normal(100, 15, 5000)
        edges = std_dev_edges(values, 5)
        mean, sd = values.mean(), values.std()
        expected = [mean + sd * m for m in (-1.5, -0.5, 0.5, 1.5)]
        np.testing.assert_allclose(edges[1:-1], expected)
        assert edges[0] == values.min() and edges[-1] == values.max()

    def test_std_dev_drops_edges_outside_a_skewed_range(self):
        values = np.random.default_rng(2).exponential(1.0, 2000)
        assert len(std_dev_edges(values, 7)) - 1 < 7

    @pytest.mark.parametrize(
        "scheme", ["quantile", "equal_interval", "jenks", "std_dev"]
    )
    def test_constant_input_is_one_degenerate_class(self, scheme):
        edges = compute_edges([3.0, 3.0, 3.0], scheme, 4)
        assert len(edges) == 2

    def test_unknown_scheme_and_bad_k(self):
        with pytest.raises(ValueError, match="unknown scheme"):
            compute_edges([1, 2, 3], "magic", 3)
        with pytest.raises(ValueError, match="at least 2"):
            compute_edges([1, 2, 3], "jenks", 1)

    def test_no_finite_values(self):
        with pytest.raises(ValueError, match="no finite"):
            compute_edges([np.nan, np.inf], "quantile", 3)


class TestAssignAndGvf:
    def test_right_closed_with_minimum_in_first_class(self):
        edges = np.array([0.0, 10.0, 20.0, 30.0])
        assert list(assign([0, 5, 10, 10.0001, 20, 30], edges)) == [0, 0, 0, 1, 1, 2]

    def test_nan_maps_to_minus_one_and_out_of_range_is_clamped(self):
        edges = np.array([0.0, 1.0, 2.0])
        assert list(assign([np.nan, -5, 99], edges)) == [-1, 0, 1]

    def test_gvf_is_one_when_classes_are_the_groups(self):
        values = np.array([1, 1, 1, 9, 9, 9], dtype=float)
        assert gvf(values, np.array([0, 0, 0, 1, 1, 1])) == pytest.approx(1.0)

    def test_gvf_is_zero_for_one_big_class(self):
        values = np.arange(10, dtype=float)
        assert gvf(values, np.zeros(10, dtype=int)) == pytest.approx(0.0)

    def test_gvf_ignores_unclassified_regions(self):
        values = np.array([1, 1, 9, 9, np.nan], dtype=float)
        assert gvf(values, np.array([0, 0, 1, 1, -1])) == pytest.approx(1.0)
