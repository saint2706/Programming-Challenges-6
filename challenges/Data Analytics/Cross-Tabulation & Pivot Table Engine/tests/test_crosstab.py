"""The cross-tab statistics against scipy, on real tables, and the Berkeley admissions paradox."""

import math

import numpy as np
import pytest
from scipy import stats

SPEC = lambda r, c, **kw: {"row": {"field": r}, "col": {"field": c}, **kw}


def close(a, b, rel=1e-9):
    assert math.isclose(a, b, rel_tol=rel, abs_tol=1e-12), (a, b)


@pytest.mark.parametrize(
    ("row", "col"),
    [
        ("sex", "income"),
        ("race", "income"),
        ("workclass", "income"),
        ("marital_status", "occupation"),
        ("education", "sex"),
    ],
)
def test_pearson_g_and_cramers_v_match_scipy(adult, row, col):
    ct = adult.crosstab(SPEC(row, col))
    table = np.array(ct["counts"]).astype(np.int64)
    table = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]
    s = ct["stats"]
    chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
    close(s["chi2"], chi2)
    close(s["p_value"], p, rel=1e-6) if p > 1e-300 else None
    assert s["dof"] == dof
    np.testing.assert_allclose(np.array(s["expected"]), expected, rtol=1e-10)
    g, *_ = stats.chi2_contingency(table, correction=False, lambda_="log-likelihood")
    close(s["g"], g)
    close(
        s["cramers_v"], stats.contingency.association(table, method="cramer"), rel=1e-9
    )
    assert 0 <= s["cramers_v_corrected"] <= s["cramers_v"] + 1e-12
    n = table.sum()
    close(s["contingency_coefficient"], math.sqrt(chi2 / (chi2 + n)))


def test_two_by_two_matches_scipy_including_yates_fisher_and_odds_ratio(adult):
    ct = adult.crosstab(SPEC("sex", "income"))
    table = np.array(ct["counts"]).astype(np.int64)
    s = ct["stats"]
    yates, p_yates, *_ = stats.chi2_contingency(table, correction=True)
    close(s["chi2_yates"], yates)
    close(s["p_yates"], p_yates, rel=1e-6)
    odds, fisher_p = stats.fisher_exact(table)
    close(s["odds_ratio"], odds)
    assert s["fisher_p"] <= 1 and (
        s["fisher_p"] == 0
        or fisher_p == 0
        or math.isclose(s["fisher_p"], fisher_p, rel_tol=1e-6)
    )
    lo, hi = s["odds_ratio_ci"]
    se = math.sqrt((1 / table).sum())  # Woolf's standard error of the log odds ratio
    close(lo, math.exp(math.log(odds) - 1.959963984540054 * se))
    close(hi, math.exp(math.log(odds) + 1.959963984540054 * se))
    close(
        s["phi"],
        (table[0, 0] * table[1, 1] - table[0, 1] * table[1, 0])
        / math.sqrt(np.prod(table.sum(axis=0)) * np.prod(table.sum(axis=1))),
    )


@pytest.mark.parametrize(
    "table",
    [[[3, 1], [1, 3]], [[10, 2], [3, 15]], [[0, 5], [6, 4]], [[12, 5], [29, 3]]],
)
def test_fisher_exact_matches_scipy_on_small_tables(table):
    # build a CSV with exactly these counts
    from pivot_tool.engine import Table

    rows = ["a,b"] + [
        f"{'xy'[i]},{'pq'[j]}"
        for i in range(2)
        for j in range(2)
        for _ in range(table[i][j])
    ]
    with Table(("\n".join(rows) + "\n").encode()) as t:
        s = t.crosstab(SPEC("a", "b"))["stats"]
    odds, p = stats.fisher_exact(table)
    assert math.isclose(s["fisher_p"], p, rel_tol=1e-7, abs_tol=1e-12)
    if odds not in (0, math.inf):
        close(s["odds_ratio"], odds)


def test_standardized_residuals_have_the_textbook_formula(adult):
    ct = adult.crosstab(SPEC("race", "income"))
    table = np.array(ct["counts"]).astype(np.int64)
    n = table.sum()
    rt, ct_ = table.sum(axis=1, keepdims=True), table.sum(axis=0, keepdims=True)
    expected = rt * ct_ / n
    want = (table - expected) / np.sqrt(expected * (1 - rt / n) * (1 - ct_ / n))
    np.testing.assert_allclose(np.array(ct["stats"]["residuals"]), want, rtol=1e-9)


def test_small_expected_counts_are_warned_about(adult):
    ct = adult.crosstab(SPEC("native_country", "occupation"))
    assert ct["stats"]["share_expected_below_5"] > 0.2
    assert any("expected count below 5" in w for w in ct["warnings"])
    assert not adult.crosstab(SPEC("sex", "income"))["warnings"]


def test_grouped_fields_and_filters_work_in_a_crosstab(adult, adult_frame):
    import polars as pl

    spec = {
        "row": {"field": "age", "group": {"kind": "edges", "edges": [30, 50]}},
        "col": {"field": "income"},
        "filters": [{"field": "sex", "op": "eq", "value": "Male"}],
    }
    ct = adult.crosstab(spec)
    assert ct["row_labels"] == ["< 30", "[30, 50)", ">= 50"]
    men = adult_frame.filter(pl.col("sex") == "Male")
    assert ct["n"] == men.height
    assert (
        ct["counts"][0][1]
        == men.filter((pl.col("age") < 30) & (pl.col("income") == ">50K")).height
    )


def test_empty_selection_and_degenerate_tables(adult):
    from pivot_tool.ffi import EngineError

    with pytest.raises(EngineError, match="no rows match"):
        adult.crosstab(
            SPEC(
                "sex",
                "income",
                filters=[{"field": "sex", "op": "eq", "value": "nobody"}],
            )
        )
    one_value = adult.crosstab(
        SPEC("sex", "income", filters=[{"field": "sex", "op": "eq", "value": "Male"}])
    )
    assert one_value["stats"] is None
    assert one_value["warnings"]


# -- Berkeley 1973 ---------------------------------------------------------------------------------
def test_berkeley_pooled_table_is_the_published_one(ucb):
    ct = ucb.crosstab(SPEC("sex", "outcome"))
    assert ct["row_labels"] == ["Female", "Male"] and ct["col_labels"] == [
        "Admitted",
        "Rejected",
    ]
    assert ct["counts"] == [[557, 1278], [1198, 1493]] and ct["n"] == 4526
    close(ct["counts"][0][0] / ct["row_totals"][0], 0.3036, rel=1e-3)
    close(ct["counts"][1][0] / ct["row_totals"][1], 0.4452, rel=1e-3)
    assert ct["stats"]["p_value"] < 1e-20
    assert ct["stats"]["odds_ratio"] < 1  # women's odds of admission are lower, pooled


def test_berkeley_departments_reverse_the_pooled_association(ucb):
    ct = ucb.crosstab(SPEC("sex", "outcome", layer={"field": "dept"}))
    sp = ct["simpson"]
    assert sp["reversal"] is True
    assert sp["pooled_odds_ratio"] < 1 < sp["mantel_haenszel_or"]
    assert sp["layers_counted"] == 6 and sp["layers_agreeing_with_pooled"] == 2
    assert "Simpson's paradox" in sp["verdict"]
    by_dept = {layer["label"]: layer for layer in ct["layers"]}
    # the published department A rates: 82.4% of women, 62.1% of men admitted
    close(by_dept["A"]["rates"][0], 89 / 108)
    close(by_dept["A"]["rates"][1], 512 / 825)
    assert sum(layer["n"] for layer in ct["layers"]) == 4526


def test_mantel_haenszel_matches_the_textbook_formula(ucb):
    ct = ucb.crosstab(SPEC("sex", "outcome", layer={"field": "dept"}))
    num = den = 0.0
    for layer in ct["layers"]:
        (a, b), (c, d) = layer["counts"]
        n = a + b + c + d
        num += a * d / n
        den += b * c / n
    close(ct["simpson"]["mantel_haenszel_or"], num / den)
    lo, hi = ct["simpson"]["mantel_haenszel_ci"]
    assert lo < num / den < hi
    # the interval for the layer-adjusted odds ratio includes 1: no evidence of bias against women
    assert lo < 1 < hi


def test_a_layer_without_a_reversal_is_reported_as_such(adult):
    ct = adult.crosstab(SPEC("sex", "income", layer={"field": "marital_status"}))
    assert (
        ct["simpson"]["reversal"] is False and "No reversal" in ct["simpson"]["verdict"]
    )


def test_simpson_needs_a_two_by_two_table(adult):
    ct = adult.crosstab(SPEC("race", "income", layer={"field": "sex"}))
    assert ct["simpson"] is None
    assert any("needs a 2x2" in w for w in ct["warnings"])
