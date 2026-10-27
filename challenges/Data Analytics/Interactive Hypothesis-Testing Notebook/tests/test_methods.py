"""Every statistic the wizard reports, checked against an independent implementation.

R values come from tests/reference/r_tests.json (scripts/make_reference.R, base R 4.6.1).
pingouin values are computed live (dev dependency).
"""

from __future__ import annotations

import numpy as np
import pingouin as pg
import polars as pl
import pytest
from helpers import reference, table, values
from hypothesis_wizard import effects, methods, posthoc

REF = reference()
REL = 1e-6


def approx(value: float, rel: float = REL, abs_: float = 1e-9):
    return pytest.approx(value, rel=rel, abs=abs_)


@pytest.fixture(scope="module")
def sleep() -> tuple[np.ndarray, np.ndarray]:
    df = table("sleep")
    return values(df, "extra", where=("group", 1)), values(
        df, "extra", where=("group", 2)
    )


SPECIES = ("Iris-setosa", "Iris-versicolor", "Iris-virginica")


@pytest.fixture(scope="module")
def iris_groups() -> list[np.ndarray]:
    df = table("iris")
    return [values(df, "petal_length", where=("species", s)) for s in SPECIES]


def _pandas(df: pl.DataFrame, columns: list[str]):
    """A pandas frame for pingouin without pyarrow (pandas comes with pingouin)."""
    import pandas as pd

    return pd.DataFrame({c: df[c].to_list() for c in columns})


# Two-sample tests on sleep (R: t.test, wilcox.test)


def test_welch_t_matches_r(sleep):
    x, y = sleep
    got = methods.welch_t(x, y)
    want = REF["welch_t_sleep"]
    assert got.statistic == approx(want["statistic"])
    assert got.df == approx(want["parameter"])
    assert got.p_value == approx(want["p_value"])
    assert got.ci[0] == approx(want["conf_int"][0])
    assert got.ci[1] == approx(want["conf_int"][1])


def test_student_t_matches_r(sleep):
    x, y = sleep
    got = methods.student_t(x, y)
    want = REF["student_t_sleep"]
    assert got.statistic == approx(want["statistic"])
    assert got.df == approx(want["parameter"])
    assert got.p_value == approx(want["p_value"])
    assert got.ci[0] == approx(want["conf_int"][0])


def test_paired_t_matches_r():
    df = table("sleep")
    wide = df.pivot(on="group", index="ID", values="extra").sort("ID")
    x, y = wide["1"].to_numpy(), wide["2"].to_numpy()
    got = methods.paired_t(x, y)
    want = REF["paired_t_sleep"]
    assert got.statistic == approx(want["statistic"])
    assert got.df == approx(want["parameter"])
    assert got.p_value == approx(want["p_value"])
    assert got.ci[0] == approx(want["conf_int"][0])


def test_mann_whitney_matches_r_normal_approximation(sleep):
    """Sleep has ties, so both sides use the normal approximation with continuity correction
    (R: wilcox.test(exact = FALSE)); scipy's asymptotic p-value agrees to 1e-8."""
    x, y = sleep
    got = methods.mann_whitney(x, y)
    want = REF["mann_whitney_sleep_normal"]
    assert got.statistic == approx(want["statistic"])
    assert got.p_value == approx(want["p_value"], rel=1e-7)


def test_mann_whitney_default_r_p_value_is_documented_not_matched(sleep):
    """R's default wilcox.test on tied data reports 0.0658 (not the normal approximation,
    0.0693). This records the gap instead of hiding it: it stays under 0.01."""
    x, y = sleep
    got = methods.mann_whitney(x, y)
    want = REF["mann_whitney_sleep"]["p_value"]
    assert abs(got.p_value - want) < 0.01
    assert got.p_value != pytest.approx(want, rel=1e-3)


def test_welch_and_student_agree_on_equal_variances_and_sizes():
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=25), rng.normal(0.4, 1.0, size=25)
    assert methods.welch_t(x, y).p_value == approx(
        methods.student_t(x, y).p_value, rel=0.05
    )


# Ordinary and Welch one-way ANOVA (R: oneway.test, kruskal.test)


def test_welch_anova_matches_r(iris_groups):
    got = methods.welch_anova(iris_groups)
    want = REF["welch_anova_iris_petal_length"]
    assert got.statistic == approx(want["statistic"])
    assert got.df == approx(want["parameter"][1])
    assert got.p_value == approx(want["p_value"], rel=1e-6, abs_=1e-300)


def test_welch_anova_matches_pingouin(iris_groups):
    df = table("iris")
    pg_df = _pandas(df, ["petal_length", "species"])
    want = pg.welch_anova(data=pg_df, dv="petal_length", between="species").iloc[0]
    got = methods.welch_anova(iris_groups)
    assert got.statistic == approx(float(want["F"]))
    assert got.p_value == approx(float(want["p_unc"]), rel=1e-6, abs_=1e-300)
    assert got.df == approx(float(want["ddof2"]))


def test_anova_matches_r(iris_groups):
    got = methods.anova(iris_groups)
    want = REF["anova_iris_petal_length"]
    assert got.statistic == approx(want["statistic"])
    assert got.p_value == approx(want["p_value"], rel=1e-6, abs_=1e-300)


def test_kruskal_wallis_matches_r(iris_groups):
    got = methods.kruskal_wallis(iris_groups)
    want = REF["kruskal_wallis_iris_petal_length"]
    assert got.statistic == approx(want["statistic"])
    assert got.p_value == approx(want["p_value"], rel=1e-6, abs_=1e-300)


def test_games_howell_matches_pingouin(iris_groups):
    df = table("iris")
    pg_df = _pandas(df, ["petal_length", "species"])
    want = pg.pairwise_gameshowell(data=pg_df, dv="petal_length", between="species")
    rows = posthoc.games_howell(iris_groups, list(SPECIES), alpha=0.05)
    for row in rows:
        match = want[
            ((want["A"] == row["a"]) & (want["B"] == row["b"]))
            | ((want["A"] == row["b"]) & (want["B"] == row["a"]))
        ].iloc[0]
        sign = 1.0 if match["A"] == row["a"] else -1.0
        assert row["diff"] == approx(sign * float(match["diff"]), rel=1e-6)
        assert row["df"] == approx(float(match["df"]), rel=1e-6)
        assert row["p"] == approx(float(match["pval"]), rel=1e-4, abs_=1e-12)


# Effect sizes (pingouin compute_effsize)


def test_hedges_g_matches_pingouin(sleep):
    x, y = sleep
    want = pg.compute_effsize(x, y, eftype="hedges")
    got, _ = effects.hedges_g(x, y)
    assert got == approx(float(want), rel=1e-9)


def test_hedges_g_is_smaller_in_magnitude_than_cohens_d():
    rng = np.random.default_rng(5)
    x, y = rng.normal(size=8), rng.normal(0.5, size=8)
    d = pg.compute_effsize(x, y, eftype="cohen")
    g, _ = effects.hedges_g(x, y)
    assert abs(g) < abs(d)


# Chi-square, Fisher, Mantel-Haenszel (R: chisq.test, fisher.test, mantelhaen.test)


def test_chi2_matches_r():
    got = methods.chi2(np.array([[12.0, 7.0], [5.0, 16.0]]))
    want = REF["chi2_2x2"]
    assert got.statistic == approx(want["statistic"])
    assert got.p_value == approx(want["p_value"], rel=1e-6, abs_=1e-300)


def test_fisher_matches_r():
    got = methods.fisher_exact(np.array([[8, 1], [2, 5]]))
    want = REF["fisher_2x2"]
    assert got.p_value == approx(want["p_value"], rel=1e-6, abs_=1e-300)
    # R solves these with uniroot at its default tolerance, so its numbers sit a little off the
    # exact roots that this code solves to 1e-14. The upper bound differs by about 1%: at R's
    # value P(X <= 8) is 0.02523, at ours it is 0.02500 (see the defining-equation test below).
    assert got.effect == approx(want["estimate"], rel=5e-4)
    assert got.ci[0] == approx(want["conf_int"][0], rel=5e-4)
    assert got.ci[1] == approx(want["conf_int"][1], rel=2e-2)


def test_fisher_interval_satisfies_its_defining_equations():
    from scipy.stats import nchypergeom_fisher

    low, high = methods.fisher_exact(np.array([[8, 1], [2, 5]])).ci
    # Totals for [[8, 1], [2, 5]]: 16 observations, 10 in column 1, 9 in row 1.
    assert nchypergeom_fisher.sf(7, 16, 10, 9, low) == approx(0.025, abs_=1e-9)
    assert nchypergeom_fisher.cdf(8, 16, 10, 9, high) == approx(0.025, abs_=1e-9)


def test_cmh_matches_r_on_ucb():
    df = table("ucb_admissions")
    strata = []
    for dept in sorted(df["Dept"].unique().to_list()):
        sub = df.filter(pl.col("Dept") == dept)
        t = np.zeros((2, 2))
        for admit, gender, freq in sub.select(["Admit", "Gender", "Freq"]).iter_rows():
            i = 0 if admit == "Admitted" else 1
            j = 0 if gender == "Male" else 1
            t[i, j] = freq
        strata.append(t)
    got = methods.cmh(strata)
    want = REF["cmh_ucb"]
    assert got.statistic == approx(want["statistic"], rel=1e-6)
    assert got.p_value == approx(want["p_value"], rel=1e-6, abs_=1e-300)


# Correlations (R: cor.test)


def _tooth():
    df = table("toothgrowth")
    return df["dose"].cast(pl.Float64).to_numpy(), df["len"].cast(pl.Float64).to_numpy()


def test_pearson_matches_r():
    x, y = _tooth()
    got = methods.pearson(x, y)
    want = REF["pearson_tooth"]
    assert got.statistic == approx(want["estimate"], rel=1e-9)
    assert got.p_value == approx(want["p_value"], rel=1e-6, abs_=1e-300)
    assert got.ci[0] == approx(want["conf_int"][0], rel=1e-6)
    assert got.ci[1] == approx(want["conf_int"][1], rel=1e-6)


def test_spearman_matches_r():
    x, y = _tooth()
    got = methods.spearman(x, y)
    want = REF["spearman_tooth"]
    assert got.statistic == approx(want["estimate"], rel=1e-9)
    assert got.p_value == approx(want["p_value"], rel=1e-6, abs_=1e-300)


def test_kendall_matches_r():
    x, y = _tooth()
    got = methods.kendall(x, y)
    want = REF["kendall_tooth"]
    assert got.statistic == approx(want["estimate"], rel=1e-9)
    assert got.p_value == approx(want["p_value"], rel=1e-4, abs_=1e-300)


def test_spearman_perfect_correlation_has_degenerate_ci():
    x = np.arange(10.0)
    res = methods.spearman(x, 2 * x + 1)
    assert res.ci[0] == pytest.approx(1.0) and res.ci[1] == pytest.approx(1.0)
