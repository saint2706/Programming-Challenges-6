"""Cox regression against R's coxph/cox.zph/survfit, lifelines and scikit-survival."""

import numpy as np
import pandas as pd
import pytest
from helpers import arr, gbsg, lung, r_cases
from lifelines import CoxPHFitter
from survival_toolkit import cox
from survival_toolkit.design import encode
from survival_toolkit.types import Survival, SurvivalError


def _lung_inputs(entry: bool = False):
    df = lung()
    keep = df.select(["age", "sex", "ph_ecog"]).drop_nulls().height
    d = encode(df, ["age", "sex", "ph_ecog"])
    assert d.keep.sum() == keep
    time, event = df["time"].to_numpy()[d.keep], df["event"].to_numpy()[d.keep] == 1
    ent = None
    if entry:
        ent = arr(r_cases()["km"]["lung_late_entry"]["entry"])[d.keep]
    return Survival(time, event, ent), d.x, d.names


def _gbsg_inputs(columns, strata=None):
    g = gbsg()
    d = encode(g, columns)
    s = Survival(g["rfstime"].to_numpy(), g["event"].to_numpy() == 1)
    return s, d.x, d.names, None if strata is None else g[strata].to_numpy()


GBSG_ALL = ["age", "meno", "size", "grade", "nodes", "pgr", "er", "hormon"]


def _build(case: str):
    if case == "lung_efron":
        s, x, n = _lung_inputs()
        return cox.fit(s, x, n, ties="efron")
    if case == "lung_breslow":
        s, x, n = _lung_inputs()
        return cox.fit(s, x, n, ties="breslow")
    if case == "lung_late_entry":
        s, x, n = _lung_inputs(entry=True)
        return cox.fit(s, x, n)
    if case == "gbsg_efron":
        s, x, n, _ = _gbsg_inputs(GBSG_ALL)
        return cox.fit(s, x, n)
    s, x, n, strata = _gbsg_inputs(
        ["age", "size", "grade", "nodes", "pgr", "er"], "hormon"
    )
    return cox.fit(s, x, n, strata=strata)


COX_CASES = [
    "lung_efron",
    "lung_breslow",
    "lung_late_entry",
    "gbsg_efron",
    "gbsg_strata_hormon",
]


@pytest.mark.parametrize("case", COX_CASES)
def test_fit_matches_r_coxph(case):
    ref, model = r_cases()["cox"][case], _build(case)
    assert model.converged
    assert model.names == ref["names"]
    np.testing.assert_allclose(model.beta, arr(ref["coef"]), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(model.se, arr(ref["se"]), rtol=1e-6)
    assert model.loglik == pytest.approx(ref["loglik"], abs=1e-6)
    assert model.loglik_null == pytest.approx(ref["loglik_null"], abs=1e-6)
    assert model.score_statistic == pytest.approx(ref["score"], rel=1e-6)
    assert (model.n, model.n_events) == (ref["n"], ref["nevent"])
    np.testing.assert_allclose(model.cov, np.array(ref["vcov"]), rtol=1e-5, atol=1e-10)


@pytest.mark.parametrize("case", [c for c in COX_CASES if c != "lung_late_entry"])
def test_concordance_matches_r(case):
    ref, model = r_cases()["cox"][case], _build(case)
    assert model.concordance()["c"] == pytest.approx(ref["concordance"], abs=2e-4)


@pytest.mark.parametrize("case", COX_CASES)
def test_proportional_hazards_test_matches_r_cox_zph(case):
    model = _build(case)
    for ref in r_cases()["cox"][case]["zph"]:
        got = model.proportional_hazards_test(ref["transform"])
        chi2 = [t["chi2"] for t in got["terms"]] + [got["global"]["chi2"]]
        np.testing.assert_allclose(chi2, arr(ref["chisq"]), rtol=1e-5, atol=1e-9)
        assert [t["name"] for t in got["terms"]] + ["GLOBAL"] == ref["names"]
        assert got["global"]["df"] == len(model.names)


@pytest.mark.parametrize("case", ["lung_efron", "lung_breslow"])
def test_predicted_survival_matches_r_survfit(case):
    ref = r_cases()["cox"][case]["predict"]
    model = _build(case)
    x = np.array(
        [[r["age"], r["sex"], r["ph_ecog"]] for r in ref["newdata"]], dtype=float
    )
    pred = model.predict_survival(x, arr(ref["time"]))
    want = np.array(
        ref["surv"], dtype=float
    )  # times down the rows, one column per new subject
    np.testing.assert_allclose(pred.T, want, rtol=1e-6)
    assert np.all(np.diff(pred, axis=1) <= 1e-15)


def test_schoenfeld_residuals_sum_to_zero_at_the_optimum():
    for case in COX_CASES:
        sch = _build(case).schoenfeld()
        assert sch["residuals"].shape[0] == r_cases()["cox"][case]["nevent"]
        scale = np.abs(sch["residuals"]).max()
        np.testing.assert_allclose(sch["residuals"].sum(axis=0), 0, atol=1e-6 * scale)


def _random_problem(seed, n=150, p=3, ties=False, entry=False):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, p))
    beta = np.array([0.8, -0.5, 0.0])[:p]
    t = rng.exponential(1 / np.exp(x @ beta))
    c = rng.exponential(1.5, n)
    time, event = np.minimum(t, c), t <= c
    if ties:
        time = np.maximum(np.ceil(time * 4) / 4, 0.25)
    ent = time * rng.uniform(0, 0.6, n) if entry else None
    return Survival(time, event, ent), x


@pytest.mark.parametrize("seed", range(4))
def test_matches_lifelines_on_tied_data(seed):
    """lifelines always uses Efron's likelihood."""
    data, x = _random_problem(seed, ties=True)
    model = cox.fit(data, x, ties="efron")
    df = pd.DataFrame(x, columns=["a", "b", "c"]).assign(T=data.time, E=data.event)
    ref = CoxPHFitter().fit(df, "T", "E")
    # lifelines stops at a step size of 1e-7, R and this module at 1e-10, so allow for that
    np.testing.assert_allclose(model.beta, ref.params_.to_numpy(), rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(model.se, ref.standard_errors_.to_numpy(), rtol=1e-3)
    assert model.loglik == pytest.approx(ref.log_likelihood_, abs=1e-4)


@pytest.mark.parametrize("ties", ["efron", "breslow"])
def test_matches_scikit_survival(ties):
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.util import Surv

    data, x = _random_problem(11, n=200, ties=True)
    model = cox.fit(data, x, ties=ties)
    ref = CoxPHSurvivalAnalysis(ties=ties, alpha=0, tol=1e-12, n_iter=200)
    ref.fit(x, Surv.from_arrays(data.event, data.time))
    np.testing.assert_allclose(model.beta, ref.coef_, rtol=1e-5, atol=1e-7)
    risk = model.linear_predictor()
    assert model.concordance()["c"] == pytest.approx(
        ref.score(x, Surv.from_arrays(data.event, data.time)), abs=5e-3
    )
    assert np.corrcoef(risk, ref.predict(x))[0, 1] == pytest.approx(1.0)


def test_efron_and_breslow_differ_only_when_there_are_ties():
    data, x = _random_problem(2, ties=False)
    np.testing.assert_allclose(
        cox.fit(data, x, ties="efron").beta,
        cox.fit(data, x, ties="breslow").beta,
        rtol=1e-9,
    )
    tied, xt = _random_problem(2, ties=True)
    a, b = cox.fit(tied, xt, ties="efron"), cox.fit(tied, xt, ties="breslow")
    assert not np.allclose(a.beta, b.beta, rtol=1e-4)


def test_recovers_known_coefficients():
    data, x = _random_problem(5, n=4000, p=2)
    model = cox.fit(data, x)
    np.testing.assert_allclose(model.beta, [0.8, -0.5], atol=0.08)
    lo, hi = model.hazard_ratio_ci()
    assert np.all(lo < np.exp([0.8, -0.5])) and np.all(np.exp([0.8, -0.5]) < hi)


def test_penalty_shrinks_and_matches_the_penalised_gradient():
    data, x = _random_problem(3, n=120)
    free = cox.fit(data, x)
    ridge = cox.fit(data, x, penalizer=5.0)
    assert np.linalg.norm(ridge.beta * ridge.scale) < np.linalg.norm(
        free.beta * free.scale
    )
    huge = cox.fit(data, x, penalizer=1e7)
    assert np.abs(huge.beta * huge.scale).max() < 1e-4
    # the fitted point is a stationary point of the penalised likelihood (central differences)
    z = (x - ridge.mean) / ridge.scale
    bz = ridge.beta * ridge.scale
    blocks = ridge._blocks
    grad = cox._evaluate(blocks, z, bz, 5.0)[1]
    assert np.abs(grad).max() < 1e-5
    eps = 1e-6
    for j in range(3):
        step = np.zeros(3)
        step[j] = eps
        numeric = (
            cox._evaluate(blocks, z, bz + step, 5.0, hessian=False)[0]
            - cox._evaluate(blocks, z, bz - step, 5.0, hessian=False)[0]
        ) / (2 * eps)
        assert numeric == pytest.approx(grad[j], abs=1e-4)


def test_analytic_gradient_and_information_match_finite_differences():
    data, x = _random_problem(8, n=90, ties=True)
    free = cox.fit(data, x)
    z = (x - free.mean) / free.scale
    beta = np.array([0.3, -0.2, 0.1])
    _ll, grad, info = cox._evaluate(free._blocks, z, beta, 0.0)
    eps = 1e-6
    num_grad = np.zeros(3)
    num_info = np.zeros((3, 3))
    for j in range(3):
        e = np.zeros(3)
        e[j] = eps
        up = cox._evaluate(free._blocks, z, beta + e, 0.0)
        dn = cox._evaluate(free._blocks, z, beta - e, 0.0)
        num_grad[j] = (up[0] - dn[0]) / (2 * eps)
        num_info[:, j] = -(up[1] - dn[1]) / (2 * eps)
    np.testing.assert_allclose(grad, num_grad, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(info, num_info, rtol=1e-4, atol=1e-5)


def test_strata_share_coefficients_but_not_baselines():
    data, x = _random_problem(4, n=200)
    strata = np.where(x[:, 2] > 0, "hi", "lo")
    model = cox.fit(data, x[:, :2], strata=strata)
    base = model.baseline_hazard()
    assert set(base) == {0, 1}
    assert not np.array_equal(base[0][0], base[1][0])
    surv = model.predict_survival(np.zeros((1, 2)), np.array([0.5, 1.0]), stratum=1)
    assert surv.shape == (1, 2) and np.all(surv <= 1)
    with pytest.raises(SurvivalError, match="stratum"):
        model.predict_survival(np.zeros((1, 2)), np.array([1.0]), stratum=7)


def test_summary_and_global_tests_are_consistent():
    model = _build("lung_efron")
    rows = model.summary()
    assert [r["name"] for r in rows] == ["age", "sex", "ph_ecog"]
    assert rows[1]["hr"] == pytest.approx(np.exp(rows[1]["coef"]))
    assert rows[1]["hr_lower"] < rows[1]["hr"] < rows[1]["hr_upper"]
    lr_stat, df, p = model.likelihood_ratio()
    assert (df, round(lr_stat, 1)) == (3, 30.5) and p < 1e-5
    assert model.wald()[2] < 1e-4 and model.score()[2] < 1e-4
    assert model.aic == pytest.approx(-2 * model.loglik + 6)


def test_a_covariate_with_a_time_varying_effect_is_flagged():
    rng = np.random.default_rng(0)
    n = 1500
    x = rng.integers(0, 2, n).astype(float)
    # hazard ratio 3 early, 0.3 late: far from proportional
    early = rng.exponential(1 / np.where(x == 1, 3.0, 1.0))
    late = rng.exponential(1 / np.where(x == 1, 0.3, 1.0))
    t = np.where(early < 0.5, early, 0.5 + late)
    c = rng.exponential(3, n)
    data = Survival(np.minimum(t, c), t <= c)
    test = cox.fit(data, x[:, None], ["arm"]).proportional_hazards_test("km")
    assert test["terms"][0]["p"] < 1e-6
    prop, xp = _random_problem(1, n=1500, p=2)
    ok = cox.fit(prop, xp).proportional_hazards_test("km")
    assert ok["global"]["p"] > 0.01


def test_errors_are_clear():
    data, x = _random_problem(0, n=40)
    with pytest.raises(SurvivalError, match="constant"):
        cox.fit(data, np.column_stack([x[:, 0], np.ones(40)]), ["a", "const"])
    with pytest.raises(SurvivalError, match="one row"):
        cox.fit(data, x[:10])
    with pytest.raises(SurvivalError, match="missing"):
        bad = x.copy()
        bad[0, 0] = np.nan
        cox.fit(data, bad)
    with pytest.raises(SurvivalError, match="ties"):
        cox.fit(data, x, ties="exact")
    with pytest.raises(SurvivalError, match="collinear"):
        cox.fit(data, np.column_stack([x[:, 0], 2 * x[:, 0]]))
    with pytest.raises(SurvivalError, match="no events"):
        cox.fit(Survival(data.time, np.zeros(40, bool)), x)
    with pytest.raises(SurvivalError, match="transform"):
        cox.fit(data, x).proportional_hazards_test("sqrt")
    with pytest.raises(SurvivalError, match="penalizer"):
        cox.fit(data, x, penalizer=-1)
