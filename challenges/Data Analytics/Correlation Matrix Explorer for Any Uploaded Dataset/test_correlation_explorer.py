from pathlib import Path

import correlation_explorer as ce
import numpy as np
import polars as pl
import pytest
from scipy import stats

SAMPLE = Path(__file__).parent / "sample_data" / "health_survey.csv"


@pytest.fixture(scope="module")
def sample_df() -> pl.DataFrame:
    return ce.load_csv(SAMPLE)


@pytest.fixture(scope="module")
def sample_analysis(sample_df: pl.DataFrame) -> ce.Analysis:
    return ce.analyze(sample_df)


def _rand_df(n: int = 60, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    return pl.DataFrame(
        {"a": a, "b": a * 2 + rng.normal(size=n), "c": rng.normal(size=n)}
    )


# --- correctness against scipy -------------------------------------------------


@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_matches_scipy(method):
    df = _rand_df()
    res = ce.compute_matrix(df, method)
    fn = stats.pearsonr if method == "pearson" else stats.spearmanr
    for i, ci in enumerate(df.columns):
        for j, cj in enumerate(df.columns):
            if i == j:
                continue
            expected = fn(df[ci].to_numpy(), df[cj].to_numpy())
            assert res.r[i, j] == pytest.approx(expected[0])
            assert res.p[i, j] == pytest.approx(expected[1])
            assert res.n[i, j] == df.height


def test_matrix_symmetric_with_unit_diagonal():
    res = ce.compute_matrix(_rand_df(), "pearson")
    assert np.allclose(res.r, res.r.T, equal_nan=True)
    assert np.allclose(np.diag(res.r), 1.0)
    assert not res.significant.diagonal().any()


def test_invalid_method():
    with pytest.raises(ValueError):
        ce.compute_matrix(_rand_df(), "kendall")


# --- missing data / degenerate input ------------------------------------------


def test_pairwise_deletion_reports_n_per_pair():
    df = pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "y": [2.0, 1.0, None, 5.0, 4.0, 7.0],
            "z": [1.0, None, None, 3.0, 2.0, 6.0],
        }
    )
    res = ce.compute_matrix(df, "pearson")
    assert res.n[0, 1] == 5
    assert res.n[0, 2] == 4
    assert res.n[1, 2] == 4
    mask = np.array([True, True, False, True, True, True])
    expected = stats.pearsonr(df["x"].to_numpy()[mask], df["y"].to_numpy()[mask])
    assert res.r[0, 1] == pytest.approx(expected[0])


def test_pair_with_too_few_overlapping_rows_is_nan():
    df = pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, None, None],
            "y": [None, None, 3.0, 4.0, 5.0],
            "w": [1.0, 5.0, 2.0, 7.0, 3.0],
        }
    )
    res = ce.compute_matrix(df, "pearson")
    assert res.n[0, 1] == 1
    assert np.isnan(res.r[0, 1]) and np.isnan(res.p[0, 1])
    assert not res.significant[0, 1]


def test_constant_after_pairwise_deletion_is_nan():
    df = pl.DataFrame(
        {"x": [1.0, 1.0, 1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0, None, None]}
    )
    res = ce.compute_matrix(df, "spearman")
    assert res.n[0, 1] == 3
    assert np.isnan(res.r[0, 1])


def test_select_numeric_skips_and_explains():
    df = pl.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0],
            "num2": [4, 3, 1, 2],
            "const": [7, 7, 7, 7],
            "text": ["a", "b", "c", "d"],
            "flag": [True, False, True, False],
            "sparse": [1.0, None, None, None],
        }
    )
    numeric, skipped = ce.select_numeric(df)
    assert numeric.columns == ["num", "num2"]
    assert "constant" in skipped["const"]
    assert skipped["text"] == "not numeric"
    assert skipped["flag"] == "not numeric"
    assert "fewer than" in skipped["sparse"]


def test_nan_values_treated_as_missing():
    df = pl.DataFrame(
        {"a": [1.0, 2.0, float("nan"), 4.0, 5.0], "b": [2.0, 1.0, 5.0, 3.0, 4.0]}
    )
    numeric, _ = ce.select_numeric(df)
    assert numeric["a"].null_count() == 1
    assert ce.compute_matrix(numeric, "pearson").n[0, 1] == 4


def test_too_few_numeric_columns():
    with pytest.raises(ce.CorrelationError, match="at least 2"):
        ce.analyze(pl.DataFrame({"a": [1.0, 2.0, 3.0], "t": ["x", "y", "z"]}))


def test_too_few_rows():
    # Columns with < MIN_PAIR_N non-null values are dropped, so this surfaces as "no usable columns".
    with pytest.raises(ce.CorrelationError, match="at least 2 usable"):
        ce.analyze(pl.DataFrame({"a": [1.0, 2.0], "b": [2.0, 1.0]}))


# --- Benjamini-Hochberg --------------------------------------------------------


def test_bh_known_example():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    # thresholds 0.05*k/10: 0.005, 0.010, 0.015, ... -> only first two pass
    assert ce.benjamini_hochberg(p, 0.05).tolist() == [True, True] + [False] * 8


def test_bh_step_up_rejects_earlier_larger_p():
    # p[1]=0.04 fails its own threshold (0.025) but p[2]=0.03 <= 0.05*3/3.. step-up keeps all
    p = np.array([0.01, 0.04, 0.03])
    assert ce.benjamini_hochberg(p, 0.05).all()


def test_bh_ignores_nan_and_handles_empty():
    p = np.array([0.001, np.nan, 0.9])
    assert ce.benjamini_hochberg(p, 0.05).tolist() == [True, False, False]
    assert ce.benjamini_hochberg(np.array([np.nan]), 0.05).tolist() == [False]
    assert ce.benjamini_hochberg(np.array([]), 0.05).size == 0


def test_correction_is_stricter_than_raw_alpha():
    rng = np.random.default_rng(3)
    df = pl.DataFrame({f"v{i}": rng.normal(size=40) for i in range(12)})
    res = ce.compute_matrix(df, "pearson")
    iu = np.triu_indices(12, 1)
    assert res.significant[iu].sum() <= (res.p[iu] < 0.05).sum()


# --- sample dataset ------------------------------------------------------------


def test_sample_columns_and_skips(sample_analysis):
    assert "site_code" in sample_analysis.skipped
    assert "region" in sample_analysis.skipped
    assert "age" in sample_analysis.columns
    assert "site_code" not in sample_analysis.columns


def test_sample_known_relationships(sample_analysis):
    cols = sample_analysis.columns
    pear = sample_analysis.results["pearson"]
    ix = cols.index
    assert pear.r[ix("age"), ix("income")] > 0.5
    assert pear.r[ix("weekly_exercise_hrs"), ix("resting_hr")] < -0.5
    assert pear.significant[ix("age"), ix("income")]
    assert not pear.significant[ix("noise"), ix("age")]


def test_sample_spearman_beats_pearson_on_nonlinear_monotonic(sample_analysis):
    cols = sample_analysis.columns
    i, j = cols.index("dose_mg"), cols.index("response")
    pearson = sample_analysis.results["pearson"].r[i, j]
    spearman = sample_analysis.results["spearman"].r[i, j]
    assert spearman > 0.95
    assert spearman - pearson > 0.05


def test_sample_missing_values_lower_pair_n(sample_analysis):
    cols = sample_analysis.columns
    n = sample_analysis.results["pearson"].n
    assert (
        n[cols.index("income"), cols.index("weekly_exercise_hrs")]
        < sample_analysis.n_rows
    )


# --- CSV loading ---------------------------------------------------------------


def test_load_csv_semicolon_and_na(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("a;b\n1;2\n2;NA\n3;7\n4;1\n", encoding="utf-8")
    df = ce.load_csv(p)
    assert df.columns == ["a", "b"]
    assert df["b"].null_count() == 1


def test_load_csv_latin1(tmp_path):
    p = tmp_path / "d.csv"
    p.write_bytes("caf\xe9,b\n1,2\n2,3\n3,1\n".encode("latin-1"))
    assert ce.load_csv(p).columns == ["café", "b"]


# --- HTML output / XSS ---------------------------------------------------------


def test_report_is_self_contained_with_toggle(sample_analysis):
    out = ce.render_report(sample_analysis)
    assert out.startswith("<!DOCTYPE html>")
    assert "Plotly" in out
    assert "Pearson" in out and "Spearman" in out
    assert "<script src=" not in out  # bundle is inline, no external fetch
    assert "<h2>Significant pairs</h2>" in out


def test_column_names_are_escaped():
    evil = "<script>alert(1)</script>"
    rng = np.random.default_rng(1)
    a = rng.normal(size=30)
    df = pl.DataFrame(
        {evil: a, "b<img src=x onerror=alert(2)>": a + rng.normal(size=30) * 0.1}
    )
    out = ce.render_report(ce.analyze(df), title="<b>t</b>")
    assert "<script>alert(1)</script>" not in out
    assert "<img src=x onerror=alert(2)>" not in out
    assert "<title>&lt;b&gt;t&lt;/b&gt;</title>" in out
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in out


def test_skipped_column_names_are_escaped():
    a = np.arange(10.0)
    df = pl.DataFrame({"a": a, "b": a[::-1], "<i>k</i>": [1.0] * 10})
    out = ce.render_report(ce.analyze(df))
    assert "<i>k</i>" not in out
    assert "&lt;i&gt;k&lt;/i&gt;" in out


# --- CLI -----------------------------------------------------------------------


def test_cli_writes_report(tmp_path, capsys):
    out = tmp_path / "r.html"
    assert ce.main([str(SAMPLE), "-o", str(out)]) == 0
    assert out.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")
    assert "wrote" in capsys.readouterr().out


def test_cli_missing_file(tmp_path, capsys):
    assert ce.main([str(tmp_path / "nope.csv")]) == 2
    assert "not found" in capsys.readouterr().err


def test_cli_unusable_dataset(tmp_path, capsys):
    p = tmp_path / "t.csv"
    p.write_text("a,b\nx,y\nz,w\nq,r\n", encoding="utf-8")
    assert ce.main([str(p), "-o", str(tmp_path / "o.html")]) == 1
    assert "error" in capsys.readouterr().err
