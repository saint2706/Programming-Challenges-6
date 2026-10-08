import numpy as np
import pytest
from helpers import make_probs
from review_stars import metrics, selective


def test_risk_coverage_by_hand():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    loss = np.array([0.0, 0.0, 1.0, 1.0])  # the two most confident are right
    rc = selective.risk_coverage(conf, loss)
    assert rc["coverage"].tolist() == [0.25, 0.5, 0.75, 1.0]
    assert rc["risk"].tolist() == pytest.approx([0.0, 0.0, 1 / 3, 0.5])
    assert rc["aurc"] == pytest.approx((0 + 0 + 1 / 3 + 0.5) / 4)


def test_a_perfect_ranking_beats_a_random_one_and_random_matches_the_overall_error():
    rng = np.random.default_rng(0)
    loss = (rng.uniform(size=20_000) < 0.3).astype(float)
    best = selective.risk_coverage(
        1.0 - loss + rng.uniform(size=loss.shape) * 1e-3, loss
    )["aurc"]
    random = selective.risk_coverage(rng.uniform(size=loss.shape), loss)["aurc"]
    assert best < 0.5 * random and random == pytest.approx(loss.mean(), abs=0.01)


def test_tied_confidences_do_not_let_row_order_change_the_answer():
    # a piecewise-constant (isotonic) confidence: only the group matters, never the order inside it
    rng = np.random.default_rng(1)
    conf = np.repeat([0.4, 0.7, 0.95], [500, 300, 200])
    loss = (rng.uniform(size=1000) < 1 - conf + 0.05).astype(float)
    a = selective.risk_coverage(conf, loss)
    perm = rng.permutation(1000)
    b = selective.risk_coverage(conf[perm], loss[perm])
    worst_first = np.argsort(-loss, kind="stable")
    c = selective.risk_coverage(conf[worst_first], loss[worst_first])
    assert a["aurc"] == pytest.approx(b["aurc"]) == pytest.approx(c["aurc"])
    assert len(a["coverage"]) == 3  # one point per distinct confidence


def test_at_threshold_reports_coverage_and_risk_and_nan_when_nothing_is_kept():
    conf = np.array([0.9, 0.5, 0.4])
    loss = np.array([0.0, 1.0, 1.0])
    kept = selective.at_threshold(conf, loss, 0.5)
    assert kept == {"tau": 0.5, "coverage": pytest.approx(2 / 3), "risk": 0.5}
    none = selective.at_threshold(conf, loss, 0.99)
    assert none["coverage"] == 0.0 and np.isnan(none["risk"])


def test_choose_threshold_is_the_loosest_one_that_meets_the_target_on_calibration_data():
    P, y = make_probs(n=20_000, seed=2)
    conf, correct = metrics.top_label(P, y)
    loss = (~correct).astype(float)
    tau = selective.choose_threshold(conf, loss, target_risk=0.1)
    # brute force over every distinct confidence: the best coverage among those meeting the target
    feasible = [
        (conf >= t).mean() for t in np.unique(conf) if loss[conf >= t].mean() <= 0.1
    ]
    assert (conf >= tau).mean() == pytest.approx(max(feasible))
    assert loss[conf >= tau].mean() <= 0.1


def test_a_threshold_that_cannot_be_met_abstains_on_everything():
    conf = np.array([0.9, 0.8])
    assert (
        selective.choose_threshold(conf, np.array([1.0, 1.0]), target_risk=0.1)
        == np.inf
    )


def test_a_calibrated_confidence_means_what_it_says_at_a_fixed_threshold_an_overconfident_one_does_not():
    # with no labeled data at all: "keep only reviews the model is 80% sure of" works only if the
    # confidence is calibrated
    kept = {}
    for name, sharpen in (("calibrated", 1.0), ("overconfident", 2.5)):
        P, y = make_probs(n=40_000, sharpen=sharpen, seed=4)
        conf, correct = metrics.top_label(P, y)
        at = selective.at_threshold(conf, (~correct).astype(float), 0.8)
        kept[name] = 1.0 - at["risk"]  # selective accuracy
    assert kept["calibrated"] > 0.8 - 0.01
    assert kept["overconfident"] < 0.8 - 0.05


def test_transfer_picks_the_threshold_on_one_split_and_reports_it_on_another():
    P_cal, y_cal = make_probs(n=30_000, seed=3)
    P_test, y_test = make_probs(n=30_000, seed=4)
    conf_c, ok_c = metrics.top_label(P_cal, y_cal)
    conf_t, ok_t = metrics.top_label(P_test, y_test)
    rows = selective.transfer(
        conf_c, (~ok_c).astype(float), conf_t, (~ok_t).astype(float), (0.1, 0.3)
    )
    assert [r["target_risk"] for r in rows] == [0.1, 0.3]
    for r in rows:
        assert (
            abs(r["test"]["risk"] - r["target_risk"]) < 0.02
        )  # in-distribution: it holds
        assert r["cal"]["risk"] <= r["target_risk"] + 1e-9
    assert rows[1]["test"]["coverage"] > rows[0]["test"]["coverage"]


def test_summary_has_error_and_star_error_curves_with_a_bounded_number_of_points():
    P, y = make_probs(n=5000, seed=5)
    out = selective.summarize(P, y, n_points=25)
    assert set(out) == {"error", "mae"}
    assert len(out["error"]["curve"]["coverage"]) <= 25
    assert out["error"]["aurc"] > 0 and out["mae"]["aurc"] > out["error"]["aurc"] * 0.5
    assert out["error"]["curve"]["coverage"][-1] == pytest.approx(1.0)


def test_empty_input_is_nan_not_a_crash():
    rc = selective.risk_coverage(np.array([]), np.array([]))
    assert np.isnan(rc["aurc"]) and len(rc["coverage"]) == 0
