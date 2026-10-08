import numpy as np
import polars as pl
import pytest
from helpers import confounded_rct, make_frame
from uplift import data, evaluate, learners, propensity

TINY = {"n_estimators": 40, "min_child_samples": 30, "num_leaves": 15}
GRID = [TINY, {**TINY, "num_leaves": 7}]


@pytest.fixture(scope="module")
def frames():
    return data.split(make_frame(30000, strength=2.0), seed=0)


def fit_propensity(train):
    X, t, _ = data.xy(train)
    return propensity.PropensityModel(seed=0).fit(X, t)


@pytest.fixture(scope="module")
def real(frames):
    return evaluate.evaluate_real(
        *frames,
        "visit",
        propensity_model=fit_propensity(frames[0]),
        grid=GRID,
        n_boot=40,
        seed=0,
    )


def test_tune_returns_a_grid_entry(frames):
    tr, va, _ = frames
    best = evaluate.tune(
        learners.TLearner, data.xy(tr), data.xy(va), GRID, data.propensity(tr), 0
    )
    assert best in GRID


def test_summary_covers_every_learner_and_baseline(real):
    summary, scores, models, _ = real
    names = set(summary["learners"])
    assert names == {"T", "S", "X", "TO", "DR", "response", "random"}
    assert set(scores) == names and set(models) == names
    assert summary["n_test"] == len(scores["T"])


def test_every_learner_entry_has_intervals(real):
    entry = real[0]["learners"]["T"]
    for key in ("qini", "auuc", "uplift@10", "incremental@20"):
        assert entry[key]["lo"] <= entry[key]["est"] <= entry[key]["hi"]
    assert len(entry["calibration"]) == 10
    assert entry["qini_vs_T"] is None
    assert real[0]["learners"]["S"]["qini_vs_T"] is not None


def test_a_real_effect_is_detected_and_random_is_chance(real):
    learners_out = real[0]["learners"]
    assert learners_out["T"]["qini"]["lo"] > 0  # the T-learner beats random targeting
    rnd = learners_out["random"]["qini"]
    assert rnd["lo"] < 0 < rnd["hi"]  # the chance baseline's CI covers zero


def persuadable_when_response_is_low(n=30000, seed=0):
    """Customers with high f0 convert often anyway and are slightly put off by contact;
    customers with low f0 convert rarely but are persuaded. Response and uplift disagree."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 12)).astype(np.float32)
    t = (rng.random(n) < 0.85).astype(np.int8)
    high = X[:, 0] > 0
    p = np.where(high, 0.5 - 0.05 * t, 0.05 + 0.2 * t)
    y = (rng.random(n) < p).astype(np.int8)
    return pl.DataFrame(
        {
            **{f"f{i}": X[:, i] for i in range(12)},
            "treatment": t,
            "visit": y,
            "conversion": y,
            "exposure": t,
        }
    )


def test_a_response_model_is_not_credited_as_an_uplift_model():
    tr, va, te = data.split(persuadable_when_response_is_low(), seed=0)
    summary, *_ = evaluate.evaluate_real(
        tr, va, te, "visit", propensity_model=fit_propensity(tr), grid=GRID, n_boot=40
    )
    out = summary["learners"]
    assert out["T"]["qini"]["lo"] > 0
    assert (
        out["response"]["qini"]["hi"] < 0
    )  # contacting the likely converters is worse than random
    assert out["response"]["qini_vs_T"]["hi"] < 0


def test_the_ate_interval_brackets_the_estimate(real):
    ate = real[0]["ate"]
    assert ate["lo"] <= ate["est"] <= ate["hi"]


def test_test_split_is_scored_not_trained_on(frames, real):
    _, _, te = frames
    assert real[0]["n_test"] == te.height


def test_mean_ci_handles_nan_single_and_empty():
    assert evaluate.mean_ci([1.0, 3.0])["mean"] == 2.0
    assert evaluate.mean_ci([2.0])["lo"] == 2.0
    assert evaluate.mean_ci([float("nan")]) is None
    assert evaluate.mean_ci([1.0, float("nan"), 3.0])["mean"] == 2.0


@pytest.fixture(scope="module")
def synth_out():
    df = make_frame(12000)
    X, t, _ = data.xy(df)
    # A stronger, commoner outcome than Criteo's 4.7% so 6,000 training rows carry enough signal.
    return evaluate.evaluate_synth(
        X,
        t,
        scenarios=("heterogeneous", "none"),
        n_seeds=2,
        params=TINY,
        seed=0,
        base_rate=0.2,
        strength=2.0,
    )


def test_synthetic_benchmark_recovers_a_known_effect(synth_out):
    het = synth_out["heterogeneous"]
    assert het["T"]["spearman"]["mean"] > 0.2
    assert het["oracle"]["qini"]["mean"] > 0
    assert het["T"]["qini"]["mean"] <= het["oracle"]["qini"]["mean"] * 1.2


def test_synthetic_no_effect_scenario_has_an_uninformative_oracle_and_no_spearman(
    synth_out,
):
    none = synth_out["none"]
    assert none["oracle"]["qini"]["mean"] == 0.0  # tau == 0 everywhere: all scores tie
    assert none["T"]["spearman"] is None  # tau is constant, rank correlation undefined
    assert none["T"]["rmse"]["mean"] > 0  # the learner still predicts a nonzero effect


def confounded_frame(n=120_000, seed=0):
    x, t, y, _ = confounded_rct(n=n, seed=seed)
    rng = np.random.default_rng(seed + 1)
    noise = rng.normal(size=(n, 11)).astype(np.float32)
    return pl.DataFrame(
        {
            "f0": x.astype(np.float32),
            **{f"f{i + 1}": noise[:, i] for i in range(11)},
            "treatment": t.astype(np.int8),
            "visit": y.astype(np.int8),
            "conversion": y.astype(np.int8),
            "exposure": t.astype(np.int8),
        }
    )


def test_evaluate_real_removes_the_confounding_from_the_headline_effect():
    tr, va, te = data.split(confounded_frame(), seed=0)
    summary, *_ = evaluate.evaluate_real(
        tr, va, te, "visit", propensity_model=fit_propensity(tr), grid=[TINY], n_boot=30
    )
    assert summary["ate"]["est"] == pytest.approx(0.02, abs=0.015)  # true effect
    assert summary["ate_unadjusted"]["est"] > 0.06  # the plain difference in means
    entry = summary["learners"]["T"]
    assert {"qini", "auuc", "incremental@20"} <= set(entry["unadjusted"])


def test_every_learner_entry_carries_the_unadjusted_sensitivity(real):
    for entry in real[0]["learners"].values():
        ci = entry["unadjusted"]["qini"]
        assert ci["lo"] <= ci["est"] <= ci["hi"]
    assert "ate_unadjusted" in real[0]
