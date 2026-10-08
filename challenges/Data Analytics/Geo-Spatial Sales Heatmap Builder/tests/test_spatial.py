"""Contiguity, Moran's I and local Moran's I, checked against dense-matrix maths and known geography."""

from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import grid_features
from sales_heatmap import spatial
from sales_heatmap.geo_heatmap import DATA
from sales_heatmap.spatial import (
    benjamini_hochberg,
    build_weights,
    contiguity,
    local_moran,
    morans_i,
)
from scipy import stats


def dense_weights(neighbors: list[set[int]]) -> np.ndarray:
    n = len(neighbors)
    w = np.zeros((n, n))
    for i, nbrs in enumerate(neighbors):
        for j in nbrs:
            w[i, j] = 1.0 / len(nbrs)
    return w


def dense_moran(x: np.ndarray, w: np.ndarray) -> float:
    z = x - x.mean()
    return float(len(x) / w.sum() * (z @ w @ z) / (z @ z))


@pytest.fixture(scope="module")
def states():
    feats = json.loads(
        (DATA / "us_states_2017_20m.geojson").read_text(encoding="utf-8")
    )["features"]
    return feats, [f["properties"]["abbr"] for f in feats]


class TestContiguity:
    def test_grid_queen_vs_rook(self):
        feats = grid_features(3, 3)  # centre cell is index 4
        queen, rook = contiguity(feats, "queen"), contiguity(feats, "rook")
        assert queen[4] == {0, 1, 2, 3, 5, 6, 7, 8}
        assert rook[4] == {1, 3, 5, 7}
        assert queen[0] == {1, 3, 4} and rook[0] == {1, 3}

    def test_symmetry(self):
        nbrs = contiguity(grid_features(4, 3))
        assert all(i in nbrs[j] for i in range(len(nbrs)) for j in nbrs[i])

    def test_unknown_kind(self):
        with pytest.raises(ValueError):
            contiguity(grid_features(2, 2), "bishop")

    def test_real_state_adjacency(self, states):
        feats, abbr = states
        nb = contiguity(feats, "queen")
        of = {a: {abbr[j] for j in nb[i]} for i, a in enumerate(abbr)}
        assert of["ME"] == {"NH"}
        assert of["MO"] == {
            "AR",
            "IA",
            "IL",
            "KS",
            "KY",
            "NE",
            "OK",
            "TN",
        }  # the famous 8
        assert of["TN"] == {"AL", "AR", "GA", "KY", "MO", "MS", "NC", "VA"}
        assert of["DC"] == {"MD", "VA"}
        assert of["HI"] == set() and of["AK"] == set()

    def test_four_corners_is_queen_but_not_rook(self, states):
        feats, abbr = states
        i = abbr.index("NM")
        queen = {abbr[j] for j in contiguity(feats, "queen")[i]}
        rook = {abbr[j] for j in contiguity(feats, "rook")[i]}
        assert "UT" in queen and "UT" not in rook
        assert {"AZ", "CO", "OK", "TX"} <= rook

    def test_real_counties_look_like_a_county_map(self):
        feats = json.loads(
            (DATA / "us_counties_2017_20m.geojson").read_text(encoding="utf-8")
        )["features"]
        nb = contiguity(feats)
        degrees = np.array([len(x) for x in nb])
        assert (
            5.5 < degrees.mean() < 6.5
        )  # a planar polygon map averages about 6 (Euler)
        assert (degrees == 0).sum() <= 12  # islands only


class TestWeights:
    def test_missing_regions_are_removed_from_the_neighbour_sets(self):
        nbrs = contiguity(grid_features(3, 1))  # 0 - 1 - 2 in a row
        w = build_weights(nbrs, np.array([True, False, True]))
        assert list(w.index) == []  # 0 and 2 only touch via the missing cell 1
        assert w.islands == 2

    def test_island_is_dropped_and_counted(self):
        feats = grid_features(2, 1) + [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[9, 9], [10, 9], [10, 10], [9, 10], [9, 9]]],
                },
            }
        ]
        w = build_weights(contiguity(feats), np.array([True, True, True]))
        assert list(w.index) == [0, 1] and w.islands == 1
        assert list(w.degree) == [1, 1]


class TestGlobalMoran:
    def test_matches_dense_formula(self):
        rng = np.random.default_rng(0)
        feats = grid_features(6, 5)
        nbrs = contiguity(feats)
        x = rng.normal(size=len(feats)) + np.linspace(0, 2, len(feats))
        w = build_weights(nbrs, np.ones(len(feats), bool))
        got = morans_i(x, w, permutations=19, seed=1)
        assert got.i == pytest.approx(dense_moran(x, dense_weights(nbrs)), rel=1e-12)
        assert got.expected == pytest.approx(-1 / (len(x) - 1))

    def test_checkerboard_is_exactly_minus_one(self):
        feats = grid_features(4, 4)
        x = np.array([(r + c) % 2 for r in range(4) for c in range(4)], dtype=float)
        w = build_weights(contiguity(feats, "rook"), np.ones(16, bool))
        assert morans_i(x, w, permutations=9, seed=0).i == pytest.approx(-1.0)

    def test_clustered_pattern_is_significant_and_positive(self):
        feats = grid_features(8, 8)
        x = np.array([1.0 if c < 4 else 0.0 for _r in range(8) for c in range(8)])
        w = build_weights(contiguity(feats, "rook"), np.ones(64, bool))
        res = morans_i(x, w, permutations=499, seed=0)
        assert res.i > 0.7 and res.p_value <= 0.01 and res.z_score > 3

    def test_random_pattern_is_not_significant(self):
        rng = np.random.default_rng(11)
        feats = grid_features(10, 10)
        w = build_weights(contiguity(feats), np.ones(100, bool))
        res = morans_i(rng.normal(size=100), w, permutations=499, seed=0)
        assert abs(res.i) < 0.2 and res.p_value > 0.05

    def test_reproducible_with_seed(self):
        feats = grid_features(5, 5)
        w = build_weights(contiguity(feats), np.ones(25, bool))
        x = np.random.default_rng(0).normal(size=25)
        assert morans_i(x, w, 99, seed=4).p_value == morans_i(x, w, 99, seed=4).p_value

    def test_constant_values_and_tiny_inputs_raise(self):
        feats = grid_features(3, 3)
        w = build_weights(contiguity(feats), np.ones(9, bool))
        with pytest.raises(ValueError, match="identical"):
            morans_i(np.ones(9), w)
        tiny = build_weights(contiguity(grid_features(2, 1)), np.ones(2, bool))
        with pytest.raises(ValueError, match="at least 3"):
            morans_i(np.array([1.0, 2.0]), tiny)

    def test_ignores_regions_without_data(self):
        feats = grid_features(4, 4)
        x = np.arange(16, dtype=float)
        valid = np.ones(16, bool)
        valid[5] = False
        w = build_weights(contiguity(feats), valid)
        polluted = x.copy()
        polluted[5] = 1e9  # must not matter: region 5 is not analysed
        assert morans_i(x, w, 9, 0).i == pytest.approx(morans_i(polluted, w, 9, 0).i)


class TestLocalMoran:
    def setup_method(self):
        rng = np.random.default_rng(7)
        self.feats = grid_features(9, 9)
        self.nbrs = contiguity(self.feats)
        # one hot corner, one cold corner, noise elsewhere
        x = rng.normal(0, 0.3, 81)
        for r in range(3):
            for c in range(3):
                x[r * 9 + c] += 4
                x[(8 - r) * 9 + (8 - c)] -= 4
        self.x = x
        self.w = build_weights(self.nbrs, np.ones(81, bool))

    def test_local_values_sum_to_n_times_global(self):
        res = local_moran(self.x, self.w, permutations=9, seed=0)
        glob = morans_i(self.x, self.w, permutations=9, seed=0).i
        assert res.local_i.sum() == pytest.approx(len(self.x) * glob, rel=1e-10)

    def test_matches_dense_formula(self):
        res = local_moran(self.x, self.w, permutations=9, seed=0)
        z = self.x - self.x.mean()
        lag = dense_weights(self.nbrs) @ z
        m2 = (z * z).mean()
        np.testing.assert_allclose(res.local_i, z * lag / m2, rtol=1e-10)
        np.testing.assert_allclose(res.lag, lag, rtol=1e-10)

    def test_finds_the_planted_hot_and_cold_spots(self):
        res = local_moran(self.x, self.w, permutations=999, seed=3)
        hot = {r * 9 + c for r in range(3) for c in range(3)}
        cold = {(8 - r) * 9 + (8 - c) for r in range(3) for c in range(3)}
        assert {i for i in range(81) if res.cluster[i] == 1} <= hot and len(
            hot & set(np.flatnonzero(res.cluster == 1))
        ) >= 3
        assert {i for i in range(81) if res.cluster[i] == 2} <= cold and len(
            cold & set(np.flatnonzero(res.cluster == 2))
        ) >= 3

    def test_quadrants_follow_sign_of_value_and_lag(self):
        res = local_moran(self.x, self.w, permutations=9, seed=0)
        z = self.x - self.x.mean()
        assert np.all((res.quadrant == 1) == ((z >= 0) & (res.lag >= 0)))
        assert np.all((res.quadrant == 3) == ((z < 0) & (res.lag < 0)))

    def test_fdr_never_flags_more_than_uncorrected(self):
        loose = local_moran(self.x, self.w, permutations=199, seed=1, fdr=False)
        strict = local_moran(self.x, self.w, permutations=199, seed=1, fdr=True)
        assert strict.n_significant <= loose.n_significant == strict.n_uncorrected

    def test_pure_noise_flags_almost_nothing_after_fdr(self):
        w = build_weights(contiguity(grid_features(15, 15)), np.ones(225, bool))
        res = local_moran(
            np.random.default_rng(5).normal(size=225), w, permutations=199, seed=2
        )
        assert res.n_significant <= 3

    def test_p_values_are_valid_pseudo_p_values(self):
        res = local_moran(self.x, self.w, permutations=99, seed=0)
        assert res.p_value.min() >= 1 / 100 and res.p_value.max() <= 1.0


class TestHelpers:
    @pytest.mark.parametrize("seed", range(5))
    def test_benjamini_hochberg_matches_scipy(self, seed):
        p = np.random.default_rng(seed).beta(0.4, 3, size=200)
        adjusted = stats.false_discovery_control(p, method="bh")
        np.testing.assert_array_equal(benjamini_hochberg(p, 0.05), adjusted <= 0.05)

    def test_benjamini_hochberg_all_null(self):
        assert not benjamini_hochberg(np.array([0.9, 0.8, 0.7]), 0.05).any()

    def test_sample_others_draws_distinct_non_self_indices(self):
        rng = np.random.default_rng(0)
        n, width = 40, 12
        degree = rng.integers(1, width + 1, size=n)
        for _ in range(50):
            picks = spatial._sample_others(rng, n, degree, width)
            for i in range(n):
                row = picks[i, : degree[i]]
                assert (
                    len(set(row.tolist())) == degree[i]
                    and i not in row
                    and row.min() >= 0
                    and row.max() < n
                )
