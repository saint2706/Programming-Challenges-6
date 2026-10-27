"""The tests the wizard can run. Each returns a ``TestResult``. The numbers come from scipy and
statsmodels, except Welch's ANOVA, which scipy does not ship and which is written out from
Welch (1951). That one is cross-checked against pingouin and R.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.optimize import brentq
from scipy.stats import nchypergeom_fisher
from scipy.stats.contingency import odds_ratio
from statsmodels.stats.contingency_tables import StratifiedTable

from hypothesis_wizard import effects


@dataclass(frozen=True)
class TestResult:
    test: str
    statistic: float
    statistic_name: str
    p_value: float
    df: float | None = None
    ci: tuple[float, float] | None = None
    ci_name: str = ""
    effect: float | None = None
    effect_name: str = ""
    notes: tuple[str, ...] = ()


def _mean_ci(res, level: float = 0.95) -> tuple[float, float]:
    ci = res.confidence_interval(level)
    return float(ci.low), float(ci.high)


def welch_t(x, y) -> TestResult:
    res = stats.ttest_ind(x, y, equal_var=False)
    g, _ = effects.hedges_g(np.asarray(x), np.asarray(y))
    return TestResult(
        "welch_t",
        float(res.statistic),
        "t",
        float(res.pvalue),
        df=float(res.df),
        ci=_mean_ci(res),
        ci_name="difference in means (x minus y)",
        effect=g,
        effect_name="Hedges g",
    )


def student_t(x, y) -> TestResult:
    res = stats.ttest_ind(x, y, equal_var=True)
    g, _ = effects.hedges_g(np.asarray(x), np.asarray(y))
    return TestResult(
        "student_t",
        float(res.statistic),
        "t",
        float(res.pvalue),
        df=float(res.df),
        ci=_mean_ci(res),
        ci_name="difference in means (x minus y)",
        effect=g,
        effect_name="Hedges g",
        notes=(
            "assumes equal variances; shown for comparison, not recommended by default",
        ),
    )


def mann_whitney(x, y) -> TestResult:
    res = stats.mannwhitneyu(x, y, alternative="two-sided")
    return TestResult(
        "mann_whitney",
        float(res.statistic),
        "U",
        float(res.pvalue),
        effect=effects.cliffs_delta(np.asarray(x), np.asarray(y)),
        effect_name="Cliff's delta",
        notes=(
            (
                "tests whether one distribution tends to exceed the other; a test of medians only "
                "when the shapes match"
            ),
        ),
    )


def brunner_munzel(x, y) -> TestResult:
    res = stats.brunnermunzel(x, y)
    return TestResult(
        "brunner_munzel",
        float(res.statistic),
        "W",
        float(res.pvalue),
        effect=effects.cliffs_delta(np.asarray(x), np.asarray(y)),
        effect_name="Cliff's delta",
        notes=(
            "compares P(X < Y) directly; no equal-variance or equal-shape assumption",
        ),
    )


def permutation_mean(x, y, *, n_resamples: int = 9_999, seed: int = 0) -> TestResult:
    x_arr, y_arr = np.asarray(x), np.asarray(y)

    def diff_of_means(a, b, axis=-1):
        return np.mean(a, axis=axis) - np.mean(b, axis=axis)

    res = stats.permutation_test(
        (x_arr, y_arr),
        diff_of_means,
        permutation_type="independent",
        vectorized=True,
        n_resamples=n_resamples,
        random_state=np.random.default_rng(seed),
    )
    g, _ = effects.hedges_g(x_arr, y_arr)
    return TestResult(
        "permutation_mean",
        float(res.statistic),
        "mean difference",
        float(res.pvalue),
        effect=g,
        effect_name="Hedges g",
        notes=(
            (
                f"{n_resamples} random relabellings; exact only when both groups share one "
                "distribution under the null"
            ),
        ),
    )


def paired_t(x, y) -> TestResult:
    res = stats.ttest_rel(x, y)
    value, _ = effects.dz(np.asarray(x), np.asarray(y))
    return TestResult(
        "paired_t",
        float(res.statistic),
        "t",
        float(res.pvalue),
        df=float(res.df),
        ci=_mean_ci(res),
        ci_name="mean paired difference (x minus y)",
        effect=value,
        effect_name="dz",
    )


def wilcoxon_signed_rank(x, y) -> TestResult:
    res = stats.wilcoxon(x, y)
    return TestResult(
        "wilcoxon_signed_rank",
        float(res.statistic),
        "W+",
        float(res.pvalue),
        effect=effects.rank_biserial_signed(np.asarray(x), np.asarray(y)),
        effect_name="rank-biserial r",
    )


def welch_anova(groups: Sequence[np.ndarray]) -> TestResult:
    """Welch (1951) one-way ANOVA: F with unequal-variance adjusted denominator df."""
    k = len(groups)
    n = np.array([len(g) for g in groups], dtype=float)
    means = np.array([np.mean(g) for g in groups])
    variances = np.array([np.var(g, ddof=1) for g in groups])
    w = n / variances
    w_total = w.sum()
    grand = np.sum(w * means) / w_total
    a = np.sum(w * (means - grand) ** 2) / (k - 1)
    tmp = np.sum((1 - w / w_total) ** 2 / (n - 1))
    b = 2 * (k - 2) / (k * k - 1) * tmp
    f = a / (1 + b)
    df2 = (k * k - 1) / (3 * tmp)
    p = float(stats.f.sf(f, k - 1, df2))
    everything = np.concatenate(groups)
    grand_mean = everything.mean()
    ss_between = float(np.sum(n * (means - grand_mean) ** 2))
    ss_total = float(np.sum((everything - grand_mean) ** 2))
    return TestResult(
        "welch_anova",
        float(f),
        "F",
        p,
        df=float(df2),
        effect=ss_between / ss_total,
        effect_name="eta squared (descriptive)",
        notes=(
            f"numerator df {k - 1}, denominator df {df2:.2f} (Welch-Satterthwaite)",
        ),
    )


def anova(groups: Sequence[np.ndarray]) -> TestResult:
    res = stats.f_oneway(*groups)
    everything = np.concatenate(groups)
    grand_mean = everything.mean()
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_total = float(np.sum((everything - grand_mean) ** 2))
    return TestResult(
        "anova",
        float(res.statistic),
        "F",
        float(res.pvalue),
        df=float(len(everything) - len(groups)),
        effect=float(ss_between / ss_total),
        effect_name="eta squared (descriptive)",
        notes=(
            "assumes equal variances; shown for comparison, not recommended by default",
        ),
    )


def kruskal_wallis(groups: Sequence[np.ndarray]) -> TestResult:
    res = stats.kruskal(*groups)
    n = sum(len(g) for g in groups)
    return TestResult(
        "kruskal_wallis",
        float(res.statistic),
        "H",
        float(res.pvalue),
        df=float(len(groups) - 1),
        effect=effects.epsilon_squared_kruskal(float(res.statistic), n),
        effect_name="epsilon squared",
    )


def chi2(table: np.ndarray) -> TestResult:
    res = stats.chi2_contingency(table, correction=False)
    rows, cols = table.shape
    n = int(table.sum())
    return TestResult(
        "chi2",
        float(res.statistic),
        "chi-square",
        float(res.pvalue),
        df=float(res.dof),
        effect=effects.cramers_v(float(res.statistic), n, rows, cols),
        effect_name="Cramer's V",
    )


def fisher_exact(table: np.ndarray) -> TestResult:
    _, p = stats.fisher_exact(table)
    estimate, ci = _conditional_odds(np.asarray(table))
    return TestResult(
        "fisher_exact",
        estimate,
        "odds ratio (conditional MLE)",
        float(p),
        ci=ci,
        ci_name="odds ratio, 95% conditional CI",
        effect=estimate,
        effect_name="odds ratio",
    )


def _conditional_odds(table: np.ndarray) -> tuple[float, tuple[float, float]]:
    """Conditional MLE and 95% CI of the odds ratio, solved as R's fisher.test does (uniroot on
    the noncentral hypergeometric), not with scipy's looser root-finder.

    Zero cells have no finite interior estimate; scipy's value is used there.
    """
    a, b, c, d = (int(v) for v in table.ravel())
    if min(a, b, c, d) == 0:
        cond = odds_ratio(table, kind="conditional")
        ci = cond.confidence_interval(0.95)
        return float(cond.statistic), (float(ci.low), float(ci.high))
    total, col1, row1 = a + b + c + d, a + c, a + b
    xs = np.arange(max(0, row1 - (total - col1)), min(col1, row1) + 1)

    def mean_at(log_or: float) -> float:
        pmf = nchypergeom_fisher.pmf(xs, total, col1, row1, math.exp(log_or))
        return float(np.sum(xs * pmf))

    def bound(log_or: float, prob: float, upper: bool) -> float:
        if upper:  # P(X <= a) falls as the odds ratio grows
            return nchypergeom_fisher.cdf(a, total, col1, row1, math.exp(log_or)) - prob
        return nchypergeom_fisher.sf(a - 1, total, col1, row1, math.exp(log_or)) - prob

    estimate = brentq(lambda t: mean_at(t) - a, -30.0, 30.0, xtol=1e-14, rtol=1e-14)
    lo = brentq(lambda t: bound(t, 0.025, False), -30.0, 30.0, xtol=1e-14, rtol=1e-14)
    hi = brentq(lambda t: bound(t, 0.025, True), -30.0, 30.0, xtol=1e-14, rtol=1e-14)
    return math.exp(estimate), (math.exp(lo), math.exp(hi))


def barnard_exact(table: np.ndarray) -> TestResult:
    res = stats.barnard_exact(table, alternative="two-sided")
    return TestResult(
        "barnard_exact",
        float(res.statistic),
        "Wald statistic",
        float(res.pvalue),
        notes=(
            "unconditional exact test; slower and slightly more powerful than Fisher",
        ),
    )


def cmh(tables: Sequence[np.ndarray]) -> TestResult:
    """Cochran-Mantel-Haenszel test of no association across 2x2 strata (no continuity fix)."""
    stacked = np.stack(tables, axis=2)
    st = StratifiedTable(stacked)
    res = st.test_null_odds(correction=False)
    lo, hi = st.oddsratio_pooled_confint(0.05)
    return TestResult(
        "cmh",
        float(res.statistic),
        "chi-square",
        float(res.pvalue),
        df=1.0,
        ci=(float(lo), float(hi)),
        ci_name="pooled odds ratio, 95% CI (Mantel-Haenszel)",
        effect=float(st.oddsratio_pooled),
        effect_name="pooled odds ratio",
    )


def pearson(x, y) -> TestResult:
    res = stats.pearsonr(x, y)
    ci = res.confidence_interval(0.95)
    return TestResult(
        "pearson",
        float(res.statistic),
        "r",
        float(res.pvalue),
        ci=(float(ci.low), float(ci.high)),
        ci_name="Fisher z interval",
        effect=float(res.statistic),
        effect_name="r",
    )


def spearman(x, y) -> TestResult:
    res = stats.spearmanr(x, y)
    rho = float(res.statistic)
    n = len(x)
    if abs(rho) >= 1:
        ci = (rho, rho)
    else:
        z = math.atanh(rho)
        half = 1.959964 / math.sqrt(n - 3)
        ci = (math.tanh(z - half), math.tanh(z + half))
    return TestResult(
        "spearman",
        rho,
        "rho",
        float(res.pvalue),
        ci=ci,
        ci_name="Fisher z approximation on ranks",
        effect=rho,
        effect_name="rho",
    )


def kendall(x, y) -> TestResult:
    res = stats.kendalltau(x, y)
    return TestResult(
        "kendall",
        float(res.statistic),
        "tau-b",
        float(res.pvalue),
        effect=float(res.statistic),
        effect_name="tau-b",
    )


TWO_GROUP = {
    "welch_t": welch_t,
    "student_t": student_t,
    "mann_whitney": mann_whitney,
    "brunner_munzel": brunner_munzel,
    "permutation_mean": permutation_mean,
    "paired_t": paired_t,
    "wilcoxon_signed_rank": wilcoxon_signed_rank,
}
K_GROUP = {"welch_anova": welch_anova, "anova": anova, "kruskal_wallis": kruskal_wallis}
TABLE = {"chi2": chi2, "fisher_exact": fisher_exact, "barnard_exact": barnard_exact}
CORRELATION = {"pearson": pearson, "spearman": spearman, "kendall": kendall}
ALL_TESTS: dict[str, Callable] = {
    **TWO_GROUP,
    **K_GROUP,
    **TABLE,
    **CORRELATION,
    "cmh": cmh,
}
