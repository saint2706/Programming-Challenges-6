from __future__ import annotations

import numpy as np
import pytest
from comp_finder import evaluate

ACTUAL = np.array([100.0, 100.0, 100.0, 100.0])
PRED = np.array([105.0, 95.0, 100.0, 150.0])


def test_summarize_known_values():
    m = evaluate.summarize(PRED, ACTUAL)
    assert m["n"] == 4
    assert m["mdape"] == pytest.approx(0.05)
    assert m["within10"] == 0.75 and m["within20"] == 0.75
    assert m["bias"] == pytest.approx(0.025)


def test_signed_error_is_relative_to_the_sale_price():
    assert evaluate.signed_error(np.array([90.0]), np.array([120.0]))[
        0
    ] == pytest.approx(-0.25)


def test_interval_brackets_the_point_estimate_and_is_reproducible():
    rng = np.random.default_rng(1)
    actual = rng.uniform(2e5, 9e5, 800)
    pred = actual * np.exp(rng.normal(0, 0.1, 800))
    idx = evaluate.boot_indices(800, 400)
    ci = evaluate.interval(pred, actual, idx)
    point = evaluate.summarize(pred, actual)
    for stat in evaluate.STATS:
        lo, hi = ci[stat]
        assert lo <= point[stat] <= hi
    assert ci == evaluate.interval(pred, actual, evaluate.boot_indices(800, 400))
    narrow = evaluate.interval(pred, actual, evaluate.boot_indices(800, 400))["mdape"]
    assert narrow[1] - narrow[0] < 0.03


def test_paired_difference_is_negative_when_a_is_better_and_zero_against_itself():
    rng = np.random.default_rng(2)
    actual = rng.uniform(2e5, 9e5, 600)
    good = actual * np.exp(rng.normal(0, 0.05, 600))
    bad = actual * np.exp(rng.normal(0, 0.15, 600))
    idx = evaluate.boot_indices(600, 400)
    d = evaluate.paired_difference(good, bad, actual, idx)
    assert d["mdape"][0] < 0 and d["mdape"][2] < 0  # the whole interval is below zero
    assert d["within10"][0] > 0 and d["within10"][1] > 0
    same = evaluate.paired_difference(good, good, actual, idx)
    assert all(v == (0.0, 0.0, 0.0) for v in same.values())


def test_unknown_statistic_is_rejected():
    with pytest.raises(ValueError):
        evaluate._stat(np.array([0.1]), "rmse")
