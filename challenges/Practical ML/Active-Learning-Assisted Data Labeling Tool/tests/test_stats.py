import numpy as np
from active_labeling import stats


def test_alc_is_the_mean_accuracy_over_the_labeling_range():
    assert np.isclose(stats.alc([10, 20, 30], [0.7, 0.7, 0.7]), 0.7)
    assert np.isclose(stats.alc([0, 100], [0.0, 1.0]), 0.5)
    assert np.isclose(stats.alc([0, 50, 100], [0.0, 1.0, 1.0]), 0.75)


def test_labels_to_target_interpolates_and_reports_never_as_nan():
    n, acc = [0, 10, 20], [0.5, 0.7, 0.9]
    assert np.isclose(stats.labels_to_target(n, acc, 0.8), 15.0)
    assert stats.labels_to_target(n, acc, 0.5) == 0.0
    assert stats.labels_to_target(n, acc, 0.9) == 20.0
    assert np.isnan(stats.labels_to_target(n, acc, 0.95))


def test_mean_ci_brackets_the_mean_is_seeded_and_handles_degenerate_input():
    ci = stats.mean_ci([1.0, 2.0, 3.0, 4.0, 5.0], n_boot=500, seed=1)
    assert ci["est"] == 3.0 and ci["lo"] < 3.0 < ci["hi"] and ci["n"] == 5
    assert ci == stats.mean_ci([1.0, 2.0, 3.0, 4.0, 5.0], n_boot=500, seed=1)
    same = stats.mean_ci([2.0, 2.0, 2.0])
    assert same["est"] == same["lo"] == same["hi"] == 2.0
    empty = stats.mean_ci([])
    assert empty["n"] == 0 and np.isnan(empty["est"])


def test_as_float_turns_none_into_nan():
    out = stats.as_float([None, 1, 2.5])
    assert np.isnan(out[0]) and out[1:].tolist() == [1.0, 2.5]


def test_stop_round_needs_k_consecutive_changes_below_the_threshold():
    change = [float("nan"), 0.10, 0.05, 0.02, 0.01, 0.01, 0.005]
    assert stats.stop_round(change, 0.02, 2) == 5  # 0.02 is not < 0.02
    assert stats.stop_round(change, 0.02, 1) == 4
    assert stats.stop_round(change, 0.001, 2) is None
    assert stats.stop_round([float("nan")] * 4, 0.5, 1) is None
    assert (
        stats.stop_round([float("nan"), 0.0], 0.5, 2) is None
    )  # window would include NaN


def curve(n_stop_at, acc_end=0.9):
    n = list(range(0, 90, 10))
    acc = [0.5, 0.6, 0.7, 0.8, 0.85, 0.89, 0.9, 0.9, acc_end]
    change = [float("nan"), 0.2, 0.1, 0.05, 0.03, 0.02, 0.005, 0.004, 0.003]
    return {"n": n, "acc": acc, "change": change}


def test_stop_outcome_reports_labels_and_the_accuracy_left_on_the_table():
    out = stats.stop_outcome(curve(0), 0.01, 2)
    assert out == {"fired": True, "n": 70.0, "gap": 0.0}
    early = stats.stop_outcome(
        curve(0), 0.05, 2
    )  # fires at round 5, where accuracy is 0.89
    assert early["fired"] and early["n"] == 50.0 and early["gap"] > 0
    never = stats.stop_outcome(curve(0), 0.0001, 2)
    assert never == {"fired": False, "n": 80.0, "gap": 0.0}


def test_pick_stop_rule_is_the_earliest_rule_within_tolerance():
    curves = [curve(0), curve(0, acc_end=0.91)]
    rule = stats.pick_stop_rule(curves, tol=0.01)
    assert rule["threshold"] in stats.THRESHOLDS and rule["k"] in stats.KS
    assert rule["acc_gap"] <= 0.01 and rule["tol"] == 0.01
    for thr in stats.THRESHOLDS:
        for k in stats.KS:
            outs = [stats.stop_outcome(c, thr, k) for c in curves]
            if np.mean([o["gap"] for o in outs]) <= 0.01:
                assert np.mean([o["n"] for o in outs]) >= rule["labels"] - 1e-9


def test_pick_stop_rule_falls_back_to_the_strictest_rule_when_nothing_is_within_tolerance():
    rule = stats.pick_stop_rule([curve(0)], thresholds=(0.5,), ks=(1,), tol=-1.0)
    assert (rule["threshold"], rule["k"]) == (0.5, 1)
