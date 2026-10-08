from __future__ import annotations

import numpy as np
import pytest
from ts_anomaly.evaluate import (
    event_counts,
    nab_score,
    point_counts,
    runs,
    windows_to_index,
    windows_to_truth,
)


def mask(n, *spans):
    m = np.zeros(n, dtype=bool)
    for lo, hi in spans:
        m[lo : hi + 1] = True
    return m


def test_runs():
    assert runs(np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1], dtype=bool)) == [
        (1, 2),
        (4, 4),
        (7, 9),
    ]
    assert runs(np.zeros(5, dtype=bool)) == []
    assert runs(np.ones(4, dtype=bool)) == [(0, 3)]


def test_point_counts_and_derived_scores():
    truth = mask(10, (2, 5))
    flags = mask(10, (4, 7))
    c = point_counts(flags, truth)
    assert (c.tp, c.fp, c.fn) == (2, 2, 2)
    assert c.precision == 0.5 and c.recall == 0.5 and c.f1 == 0.5
    empty = point_counts(np.zeros(10, dtype=bool), np.zeros(10, dtype=bool))
    assert (empty.precision, empty.recall, empty.f1) == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="same length"):
        point_counts(np.zeros(3, dtype=bool), np.zeros(4, dtype=bool))


def test_event_counts_credit_a_single_flag_inside_a_long_event():
    truth = mask(100, (20, 69))
    flags = mask(100, (20, 21))  # noticed only at the very start
    assert point_counts(flags, truth).recall == pytest.approx(2 / 50)
    c = event_counts(flags, truth)
    assert (c.tp, c.fp, c.fn) == (1, 0, 0)
    assert c.f1 == 1.0


def test_event_counts_false_alarms_and_misses():
    truth = mask(100, (10, 10), (50, 50), (90, 90))
    flags = mask(100, (10, 10), (30, 31), (70, 70))
    c = event_counts(flags, truth)
    assert (c.tp, c.fp, c.fn) == (1, 2, 2)


def test_event_tolerance_forgives_near_misses_only():
    truth = mask(100, (50, 50))
    assert event_counts(mask(100, (52, 52)), truth, tolerance=0).tp == 0
    assert event_counts(mask(100, (52, 52)), truth, tolerance=2).tp == 1
    assert event_counts(mask(100, (53, 53)), truth, tolerance=2).tp == 0
    late = event_counts(mask(100, (53, 53)), truth, tolerance=2)
    assert (late.fp, late.fn) == (1, 1)


def test_windows_to_index_and_truth():
    ts = np.array(
        [
            "2024-01-01T00:00",
            "2024-01-01T01:00",
            "2024-01-01T02:00",
            "2024-01-01T03:00",
        ],
        dtype="datetime64[us]",
    )
    idx = windows_to_index(
        ts,
        [
            ("2024-01-01 00:30:00.000000", "2024-01-01 02:00:00.000000"),
            ("2025-01-01 00:00:00", "2025-01-02 00:00:00"),
        ],
    )
    assert idx == [(1, 2)]  # the window with no rows is dropped, bounds are inclusive
    assert windows_to_truth(4, idx).tolist() == [False, True, True, False]


# NAB scaled sigmoid: 2/(1+exp(5y)) - 1 == -tanh(5y/2)
def sigma(y):
    return -np.tanh(2.5 * y)


def test_nab_hit_at_window_start_scores_near_perfect_and_at_end_scores_half():
    windows = [(20, 39)]  # width 20
    start = nab_score(mask(100, (20, 20)), windows)
    assert start.raw == pytest.approx(sigma(-19 / 20))
    assert start.normalized == pytest.approx(100 * (sigma(-19 / 20) + 1) / 2)
    assert start.normalized > 99
    end = nab_score(mask(100, (39, 39)), windows)
    assert end.raw == pytest.approx(0.0)
    assert end.normalized == pytest.approx(50.0)
    assert (start.windows_hit, start.windows_total, start.false_positive_events) == (
        1,
        1,
        0,
    )


def test_nab_missed_window_equals_the_null_detector():
    r = nab_score(np.zeros(100, dtype=bool), [(20, 39), (60, 79)])
    assert r.raw == -2.0
    assert r.normalized == 0.0
    assert r.windows_hit == 0


def test_nab_false_positive_penalty_depends_on_distance_from_the_previous_window():
    windows = [(20, 39)]
    hit = sigma(-19 / 20)
    # before any window: full A_fp = 0.11
    early = nab_score(mask(100, (20, 20), (5, 5)), windows)
    assert early.raw == pytest.approx(hit - 0.11)
    # right after the window: cheap; far after: nearly the full 0.11
    near = nab_score(mask(100, (20, 20), (40, 40)), windows)
    assert near.raw == pytest.approx(hit + 0.11 * sigma(1 / 20))
    far = nab_score(mask(100, (20, 20), (99, 99)), windows)
    assert far.raw == pytest.approx(hit + 0.11 * sigma(60 / 20))
    assert near.raw > far.raw
    assert far.false_positive_events == 1


def test_nab_collapses_runs_and_scores_only_the_first_detection_per_window():
    windows = [(20, 39)]
    once = nab_score(mask(100, (20, 20), (70, 70)), windows)
    run_of_fps = nab_score(mask(100, (20, 20), (70, 74)), windows)
    assert run_of_fps.raw == once.raw  # five flagged samples, one false alarm
    two_hits = nab_score(mask(100, (20, 20), (30, 30)), windows)
    assert two_hits.raw == nab_score(mask(100, (20, 20)), windows).raw


def test_nab_with_no_windows_reports_only_false_alarms():
    r = nab_score(mask(50, (10, 11), (30, 30)), [])
    assert r.false_positive_events == 2
    assert np.isnan(r.normalized)
    assert r.raw == pytest.approx(-0.22)


def test_nab_profile_weights_change_the_fp_cost():
    windows = [(20, 39)]
    flags = mask(100, (20, 20), (90, 90))
    standard = nab_score(flags, windows)
    harsh = nab_score(flags, windows, profile=(1.0, 0.5, 1.0))
    assert harsh.raw < standard.raw
