import numpy as np
from active_labeling import evaluate
from active_labeling.model import Head
from helpers import make_problem


def fake_curve(acc, n=None, change=None):
    n = n or list(range(0, 10 * len(acc), 10))
    m = len(acc)
    return {
        "n": n,
        "acc": acc,
        "f1": acc,
        "coverage": [3] * m,
        "skew": [0.5] * m,
        "change": change or [float("nan")] + [0.1] * (m - 1),
        "sel_err": [0.4] * (m - 1) + [float("nan")],
        "pool_err": [0.1] * m,
        "sel_time": [0.01] * (m - 1) + [float("nan")],
        "picked": [],
    }


RANDOM = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
FAST = [0.7, 0.8, 0.9, 1.0, 1.0, 1.0]


def curves(strategy=FAST, seeds=(0, 1, 2)):
    return {
        "random": {s: fake_curve(RANDOM) for s in seeds},
        "fast": {s: fake_curve(strategy) for s in seeds},
    }


def test_summarize_paired_alc_and_label_savings_match_hand_computation():
    out = evaluate.summarize(curves(), ceiling_acc=1.0, n_boot=200)
    assert out["random"]["alc_vs_random"] is None
    assert out["random"]["targets"]["0.90"]["saving_vs_random"] is None
    assert np.isclose(out["random"]["alc"]["est"], 0.75)
    assert np.isclose(out["fast"]["alc"]["est"], 0.91)
    assert np.isclose(out["fast"]["alc_vs_random"]["est"], 0.16)
    t = out["fast"]["targets"]["0.90"]  # random reaches 0.9 at 40 labels, fast at 20
    assert t["reached"] == 3 and np.isclose(t["labels"], 20.0)
    assert np.isclose(t["saving_vs_random"]["est"], 20.0)
    assert np.isclose(t["saving_vs_random"]["lo"], 20.0)
    assert np.isclose(t["saving_vs_random"]["hi"], 20.0)
    assert out["fast"]["acc_final"]["est"] == 1.0 and out["fast"]["coverage_final"] == 3
    assert np.isclose(out["fast"]["sel_err"], 0.4) and np.isclose(
        out["fast"]["pool_err"], 0.1
    )
    assert np.isclose(out["fast"]["select_seconds"], 0.01)


def test_savings_only_use_seeds_where_both_curves_reach_the_target():
    c = curves()
    c["fast"][2] = fake_curve([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])  # never reaches 0.9
    t = evaluate.summarize(c, 1.0, n_boot=100)["fast"]["targets"]["0.90"]
    assert t["reached"] == 2 and t["saving_vs_random"]["n"] == 2
    assert np.isclose(t["saving_vs_random"]["est"], 20.0)


def test_a_target_nobody_reaches_is_reported_as_none_not_a_crash():
    t = evaluate.summarize(curves(), ceiling_acc=2.0, n_boot=100)["fast"]["targets"][
        "0.95"
    ]
    assert t["reached"] == 0 and t["labels"] is None
    assert t["saving_vs_random"]["n"] == 0


def test_summaries_survive_a_json_round_trip_of_the_curves():
    c = curves()
    for by_seed in c.values():
        for cv in by_seed.values():
            for key in ("change", "sel_err", "sel_time"):
                cv[key] = [None if np.isnan(v) else v for v in cv[key]]
    out = evaluate.summarize(c, 1.0, n_boot=100)
    assert np.isclose(out["fast"]["sel_err"], 0.4)


def test_stop_summary_reports_labels_fired_count_and_gap():
    rule = {"threshold": 0.15, "k": 2}
    out = evaluate.summarize(curves(), 1.0, rule=rule, n_boot=100)["fast"]["stop"]
    assert out["fired"] == 3 and np.isclose(out["labels"], 20.0)
    assert np.isclose(out["acc_gap"]["est"], 0.1)


def test_mean_curve_is_the_pointwise_mean_with_a_95_percent_band():
    cv = evaluate.mean_curve([fake_curve([0.4, 0.6]), fake_curve([0.6, 0.8])])
    assert cv["n"] == [0, 10] and np.allclose(cv["acc"], [0.5, 0.7])
    assert np.all(np.array(cv["lo"]) < cv["acc"]) and np.all(
        np.array(cv["hi"]) > cv["acc"]
    )
    assert cv["coverage"] == [3.0, 3.0] and cv["skew"] == [
        0.5,
        0.5,
    ]  # intents with a label, top share


def test_ceiling_is_the_all_labels_head_accuracy_and_run_job_wraps_a_curve():
    prob = make_problem()
    head = Head(prob.n_classes, prob.C).fit(prob.X, prob.y)
    assert evaluate.ceiling(prob) == (head.predict(prob.X_eval) == prob.y_eval).mean()
    job = evaluate.run_job(prob, "margin", 3, budget=20, b=10, init=12)
    assert (job["strategy"], job["seed"], job["b"]) == ("margin", 3, 10)
    assert job["curve"]["n"] == [12, 22, 32]
