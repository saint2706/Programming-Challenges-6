from __future__ import annotations

import numpy as np
import pytest
from helpers import reference
from hypothesis_wizard import posthoc

REF = reference()
P_VECTOR = REF["multiplicity_input"]


def test_holm_matches_r_p_adjust():
    assert posthoc.holm(P_VECTOR) == pytest.approx(REF["holm"], rel=1e-12)


def test_benjamini_hochberg_matches_r_p_adjust():
    assert posthoc.benjamini_hochberg(P_VECTOR) == pytest.approx(
        REF["benjamini_hochberg"], rel=1e-12
    )


@pytest.mark.parametrize("fn", [posthoc.holm, posthoc.benjamini_hochberg])
def test_adjustments_never_shrink_p_values_and_stay_in_unit_interval(fn):
    rng = np.random.default_rng(11)
    for _ in range(50):
        p = rng.uniform(size=rng.integers(1, 12)).tolist()
        adjusted = fn(p)
        assert len(adjusted) == len(p)
        assert all(a >= raw - 1e-15 for a, raw in zip(adjusted, p, strict=True))
        assert all(0.0 <= a <= 1.0 for a in adjusted)


def test_holm_is_no_less_conservative_than_bh():
    rng = np.random.default_rng(2)
    p = rng.uniform(size=8).tolist()
    assert all(
        h >= b - 1e-12
        for h, b in zip(posthoc.holm(p), posthoc.benjamini_hochberg(p), strict=True)
    )


def test_dunn_holm_reports_adjusted_not_raw_p_values():
    rng = np.random.default_rng(4)
    groups = [rng.normal(0, 1, 15), rng.normal(0.2, 1, 15), rng.normal(1.5, 1, 15)]
    rows = posthoc.dunn_holm(groups, ["a", "b", "c"])
    assert len(rows) == 3
    for row in rows:
        assert row["p"] >= row["p_raw"] - 1e-12
        assert 0.0 <= row["p"] <= 1.0
    # The clearly separated pair must come out significant after correction.
    far = next(r for r in rows if {r["a"], r["b"]} == {"a", "c"})
    assert far["p"] < 0.05


def test_dunn_holm_identical_groups_are_not_significant():
    rng = np.random.default_rng(9)
    base = rng.normal(size=10)
    rows = posthoc.dunn_holm([base, base.copy(), base.copy()], ["x", "y", "z"])
    assert all(r["p"] > 0.5 for r in rows)


def test_games_howell_identical_groups_have_zero_difference():
    rng = np.random.default_rng(1)
    base = rng.normal(size=12)
    rows = posthoc.games_howell([base, base.copy()], ["x", "y"], alpha=0.05)
    assert len(rows) == 1
    assert rows[0]["diff"] == pytest.approx(0.0, abs=1e-12)
    assert rows[0]["ci_low"] < 0 < rows[0]["ci_high"]
