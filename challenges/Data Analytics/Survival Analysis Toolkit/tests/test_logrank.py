"""Log-rank family against R's survdiff and lifelines, plus trend, pairwise and RMST tests."""

import numpy as np
import pytest
from helpers import arr, gbsg, lung, r_cases, rng_data
from lifelines.statistics import logrank_test
from scipy import stats
from survival_toolkit import logrank as lr
from survival_toolkit.types import Survival, SurvivalError


def _case(name: str):
    g, lu = gbsg(), lung()
    lung_surv = Survival(lu["time"].to_numpy(), lu["event"].to_numpy() == 1)
    gbsg_surv = Survival(g["rfstime"].to_numpy(), g["event"].to_numpy() == 1)
    table = {
        "lung_sex": (lung_surv, lu["sex"].to_numpy(), None, "logrank"),
        "lung_sex_rho1": (lung_surv, lu["sex"].to_numpy(), None, "fh:1,0"),
        "gbsg_grade": (gbsg_surv, g["grade"].to_numpy(), None, "logrank"),
        "gbsg_grade_rho1": (gbsg_surv, g["grade"].to_numpy(), None, "fh:1,0"),
        "gbsg_hormon_strata_meno": (gbsg_surv, g["hormon"].to_numpy(), g["meno"].to_numpy(), "logrank"),
        "gbsg_grade_strata_hormon": (gbsg_surv, g["grade"].to_numpy(), g["hormon"].to_numpy(), "logrank"),
    }  # fmt: skip
    return table[name]


@pytest.mark.parametrize("name", list(r_cases()["survdiff"]))
def test_matches_r_survdiff(name):
    ref = r_cases()["survdiff"][name]
    data, groups, strata, weight = _case(name)
    res = lr.logrank(data, groups, weight=weight, strata=strata)
    assert res.statistic == pytest.approx(ref["chisq"], rel=1e-9)
    assert res.df == ref["df"]
    if (
        weight == "logrank" and strata is None
    ):  # R reports weighted or per-stratum counts otherwise
        np.testing.assert_allclose(res.observed, arr(ref["obs"]), rtol=1e-12)
        np.testing.assert_allclose(res.expected, arr(ref["exp"]), rtol=1e-9)
        np.testing.assert_array_equal(res.n, arr(ref["n"]))
    assert res.p_value == pytest.approx(stats.chi2.sf(ref["chisq"], ref["df"]))


@pytest.mark.parametrize(
    ("ours", "theirs"),
    [
        ("logrank", None),
        ("wilcoxon", "wilcoxon"),
        ("tarone-ware", "tarone-ware"),
        ("peto", "peto"),
        ("fh:0.5,0.5", ("fleming-harrington", 0.5, 0.5)),
        ("fh:0,1", ("fleming-harrington", 0, 1)),
    ],
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_weights_match_lifelines_with_ties(ours, theirs, seed):
    data = rng_data(seed, n=80, ties=True)
    groups = np.where(np.random.default_rng(seed).random(80) < 0.5, "a", "b")
    a, b = data.subset(groups == "a"), data.subset(groups == "b")
    kwargs = {}
    if isinstance(theirs, tuple):
        kwargs = {"weightings": theirs[0], "p": theirs[1], "q": theirs[2]}
    elif theirs:
        kwargs = {"weightings": theirs}
    ref = logrank_test(a.time, b.time, a.event, b.event, **kwargs)
    res = lr.logrank(data, groups, weight=ours)
    assert res.statistic == pytest.approx(ref.test_statistic, rel=1e-8)
    assert res.p_value == pytest.approx(ref.p_value, rel=1e-6)


def test_two_group_z_is_signed_and_squares_to_the_statistic():
    data, groups, _, _ = _case("lung_sex")
    res = lr.logrank(data, groups)
    assert res.z**2 == pytest.approx(res.statistic)
    # sex 1 (men) has more deaths than expected: positive z for the first label
    assert res.groups == ["1", "2"] and res.z > 0
    assert res.observed[0] > res.expected[0]


def test_observed_minus_expected_sums_to_zero():
    data, groups, _, _ = _case("gbsg_grade")
    res = lr.logrank(data, groups)
    assert res.observed.sum() == pytest.approx(res.expected.sum())
    rows = res.summary_rows()
    assert rows[0]["group"] == "1" and rows[0]["n"] == res.n[0]


def test_late_entry_changes_the_answer_and_matches_a_hand_count():
    data = Survival(
        np.array([2.0, 3.0, 5.0, 6.0]),
        np.array([True, True, True, False]),
        np.array([0.0, 0.0, 4.0, 1.0]),
    )
    groups = np.array(["a", "a", "b", "b"])
    res = lr.logrank(data, groups)
    # t=2: a has 2 at risk (entry<2<=time), b has 1 (the entry-1 subject): 3 at risk, 1 event in a
    # t=3: a 1, b 1 (entry 1<3<=6; the entry-4 subject is not yet in): 2 at risk, 1 event in a
    # t=5: a 0, b 2 (both entered): 1 event in b
    assert res.observed.tolist() == [2.0, 1.0]
    exp_a = 1 * 2 / 3 + 1 * 1 / 2 + 0
    assert res.expected[0] == pytest.approx(exp_a)
    ignoring = lr.logrank(Survival(data.time, data.event), groups)
    assert ignoring.statistic != pytest.approx(res.statistic)


def test_trend_test_beats_omnibus_on_ordered_groups():
    g = gbsg()
    data = Survival(g["rfstime"].to_numpy(), g["event"].to_numpy() == 1)
    omni = lr.logrank(data, g["grade"].to_numpy())
    trend = lr.logrank(data, g["grade"].to_numpy(), trend=True)
    assert trend.df == 1 and omni.df == 2
    assert trend.p_value < omni.p_value
    assert trend.z > 0 or trend.z < 0  # a signed statistic
    custom = lr.logrank(
        data, g["grade"].to_numpy(), trend=True, scores=np.array([1.0, 2.0, 3.0])
    )
    assert custom.statistic == pytest.approx(trend.statistic)
    with pytest.raises(SurvivalError, match="scores"):
        lr.logrank(data, g["grade"].to_numpy(), trend=True, scores=np.array([1.0, 2.0]))


def test_pairwise_adjustment():
    g = gbsg()
    data = Survival(g["rfstime"].to_numpy(), g["event"].to_numpy() == 1)
    rows = lr.pairwise(data, g["grade"].to_numpy(), adjust="holm")
    assert [(r["a"], r["b"]) for r in rows] == [("1", "2"), ("1", "3"), ("2", "3")]
    for r in rows:
        assert r["p"] <= r["p_adjusted"] <= 1
    direct = lr.logrank(
        data.subset(g["grade"].to_numpy() != 3),
        g["grade"].to_numpy()[g["grade"].to_numpy() != 3],
    )
    assert rows[0]["statistic"] == pytest.approx(direct.statistic)


def test_adjust_p_matches_known_procedures():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    holm = lr.adjust_p(p, "holm")
    np.testing.assert_allclose(holm, [0.03, 0.06, 0.06, 0.02])
    bh = lr.adjust_p(p, "bh")
    np.testing.assert_allclose(bh, stats.false_discovery_control(p, method="bh"))
    np.testing.assert_allclose(lr.adjust_p(p, "none"), p)
    with pytest.raises(SurvivalError):
        lr.adjust_p(p, "bonferroni")


def test_rmst_difference():
    lu = lung()
    male = lu["sex"].to_numpy() == 1
    surv = Survival(lu["time"].to_numpy(), lu["event"].to_numpy() == 1)
    out = lr.rmst_difference(surv.subset(male), surv.subset(~male), 500)
    ref = r_cases()["km"]
    assert out["rmst_a"] == pytest.approx(ref["lung_male"]["rmst"]["mean"], rel=1e-10)
    assert out["rmst_b"] == pytest.approx(ref["lung_female"]["rmst"]["mean"], rel=1e-10)
    assert out["difference"] < 0 and out["lower"] < out["difference"] < out["upper"]
    assert out["p"] < 0.05
    se = np.hypot(ref["lung_male"]["rmst"]["se"], ref["lung_female"]["rmst"]["se"])
    assert out["se"] == pytest.approx(se, rel=1e-8)


def test_weight_parsing_and_input_errors():
    assert lr.parse_weight("FH:1,0") == ("fh", 1.0, 0.0)
    assert lr.parse_weight("logrank") == ("logrank", 0.0, 0.0)
    for bad in ("fh:1", "fh:a,b", "fh:-1,0", "mantel"):
        with pytest.raises(SurvivalError):
            lr.parse_weight(bad)
    data = rng_data(0)
    with pytest.raises(SurvivalError, match="at least two"):
        lr.logrank(data, np.array(["a"] * len(data)))
    with pytest.raises(SurvivalError, match="one label"):
        lr.logrank(data, np.array(["a", "b"]))
