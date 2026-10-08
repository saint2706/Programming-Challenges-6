"""Aggregation, missing-vs-zero handling, per-capita maths, the bundled data, and the CLI."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sales_heatmap import geo_heatmap as gh
from sales_heatmap.geo_heatmap import (
    BLANK,
    NO_POPULATION,
    NO_RECORD,
    OK,
    aggregate_to_regions,
    build_model,
    parse_numbers,
)
from scipy import stats


@pytest.fixture
def grid(grid_geojson):
    path, _ = grid_geojson
    return gh.load_custom_boundaries(
        path, "id", "name"
    )  # cells C00 C10 C20 C01 C11 C21


def tiny_model(grid, **kwargs):
    data = pl.DataFrame(
        {
            "cell": ["C00", "C10", "C20", "C01", "C01", "C11", "ZZ"],
            "sales": ["100", "200", "", "40", "60", "30", "5"],
        }
    )
    pop = pl.DataFrame(
        {
            "cell": ["C00", "C10", "C20", "C01", "C21"],
            "pop": ["10", "20", "5", "0", "6"],
        }
    )
    values, join = aggregate_to_regions(data, "cell", "sales", "sum", grid.index)
    pop_values, pop_join = aggregate_to_regions(pop, "cell", "pop", "sum", grid.index)
    return build_model(
        grid,
        values,
        join,
        title="t",
        value_label="Sales",
        prefix="$",
        population=(pop_values, pop_join),
        permutations=19,
        lisa_permutations=19,
        **kwargs,
    )


class TestParseNumbers:
    def test_formats_blanks_and_junk(self):
        values, blank, junk = parse_numbers(
            pl.Series(["$1,234.5", " 7 ", "12%", "", None, "abc", "nan", "inf", "-3"])
        )
        assert values.to_list() == [
            1234.5,
            7.0,
            12.0,
            None,
            None,
            None,
            None,
            None,
            -3.0,
        ]
        assert blank == 2 and junk == 3  # abc, nan, inf are present-but-not-numeric

    def test_numeric_column_passes_through(self):
        values, blank, junk = parse_numbers(pl.Series([1.5, 2.0, None]))
        assert values.to_list() == [1.5, 2.0, None] and (blank, junk) == (1, 0)


class TestAggregation:
    def frame(self):
        return pl.DataFrame(
            {
                "k": ["C00", "C00", "C10", "C20", "C20", "nope"],
                "v": ["1", "2", "5", "", "", "9"],
            }
        )

    def test_sum_merges_rows_and_counts_duplicates(self, grid):
        out, rep = aggregate_to_regions(self.frame(), "k", "v", "sum", grid.index)
        vals = dict(zip(out["pos"].to_list(), out["value"].to_list(), strict=True))
        assert vals[0] == 3.0 and vals[1] == 5.0
        assert (
            rep.matched_rows == 5
            and rep.matched_regions == 3
            and rep.duplicate_rows_merged == 2
        )
        assert rep.blank_values == 2 and rep.unparseable_values == 0

    def test_all_blank_region_is_null_not_zero(self, grid):
        """A sum over nothing is unknown, not 0: this is the whole 'missing is not zero' point."""
        out, _ = aggregate_to_regions(self.frame(), "k", "v", "sum", grid.index)
        vals = dict(zip(out["pos"].to_list(), out["value"].to_list(), strict=True))
        assert vals[2] is None

    @pytest.mark.parametrize(
        "agg,expected",
        [("mean", 1.5), ("median", 1.5), ("min", 1.0), ("max", 2.0), ("count", 2.0)],
    )
    def test_other_aggregations(self, grid, agg, expected):
        out, _ = aggregate_to_regions(self.frame(), "k", "v", agg, grid.index)
        assert (
            dict(zip(out["pos"].to_list(), out["value"].to_list(), strict=True))[0]
            == expected
        )

    def test_count_of_an_all_blank_region_is_zero_rows_counted(self, grid):
        out, _ = aggregate_to_regions(self.frame(), "k", "v", "count", grid.index)
        assert (
            dict(zip(out["pos"].to_list(), out["value"].to_list(), strict=True))[2]
            == 0.0
        )

    def test_unmatched_and_missing_regions_are_reported(self, grid):
        _, rep = aggregate_to_regions(self.frame(), "k", "v", "sum", grid.index)
        assert (
            rep.unmatched == [("nope", "no region with this key", 1)]
            and rep.unmatched_rows == 1
        )
        assert (
            rep.regions_without_record == ["Cell 0,1", "Cell 1,1", "Cell 2,1"]
            and rep.total_regions == 6
        )

    def test_bad_aggregation(self, grid):
        with pytest.raises(ValueError, match="agg must be"):
            aggregate_to_regions(self.frame(), "k", "v", "mode", grid.index)

    def test_empty_frame(self, grid):
        out, rep = aggregate_to_regions(
            pl.DataFrame({"k": [], "v": []}, schema={"k": pl.Utf8, "v": pl.Utf8}),
            "k",
            "v",
            "sum",
            grid.index,
        )
        assert (
            out.height == 0
            and rep.matched_regions == 0
            and len(rep.regions_without_record) == 6
        )


class TestModel:
    def test_statuses_distinguish_blank_absent_and_zero_population(self, grid):
        m = tiny_model(grid)
        raw, pc = m.measures["raw"], m.measures["percap"]
        assert raw.status.tolist() == [OK, OK, BLANK, OK, OK, NO_RECORD]
        np.testing.assert_array_equal(
            np.isnan(raw.values), [False, False, True, False, False, True]
        )
        assert raw.values[3] == 100.0  # two rows summed
        assert pc.status.tolist() == [
            OK,
            OK,
            BLANK,
            NO_POPULATION,
            NO_POPULATION,
            NO_RECORD,
        ]
        # C01 has population 0 (division by zero must not produce inf), C11 has no population row at all
        assert pc.values[:2].tolist() == [10.0, 10.0] and np.isnan(pc.values[2:]).all()

    def test_zero_is_a_value_not_missing(self, grid):
        data = pl.DataFrame({"c": ["C00", "C10", "C20"], "v": ["0", "5", "10"]})
        values, join = aggregate_to_regions(data, "c", "v", "sum", grid.index)
        m = build_model(
            grid,
            values,
            join,
            title="t",
            value_label="V",
            permutations=9,
            lisa_permutations=9,
        )
        assert m.measures["raw"].status[0] == OK and m.measures["raw"].values[0] == 0.0
        assert m.views[0].classes[0] == 0  # zero lands in the lowest class...
        assert (
            m.views[0].classes[3] == -NO_RECORD
        )  # ...while an absent region gets a status code

    def test_views_cover_every_measure_and_scheme(self, grid):
        m = tiny_model(grid)
        assert {(v.measure, v.scheme) for v in m.views} == {
            (a, b) for a in ("raw", "percap") for b in gh.SCHEMES
        }
        for v in m.views:
            assert v.classes.shape == (6,)
            assert sum(v.counts) == int((v.classes >= 0).sum())

    def test_spatial_stats_degrade_gracefully_on_tiny_maps(self, grid):
        m = tiny_model(grid)
        assert m.moran["percap"] is None  # only 2 regions have a per-resident value
        assert any("skipped" in n for n in m.notes)

    def test_per_capita_needs_a_sum(self, grid):
        data = pl.DataFrame({"c": ["C00", "C10"], "v": ["1", "2"]})
        values, join = aggregate_to_regions(data, "c", "v", "mean", grid.index)
        pop = aggregate_to_regions(
            pl.DataFrame({"c": ["C00"], "p": ["3"]}), "c", "p", "max", grid.index
        )
        m = build_model(
            grid,
            values,
            join,
            title="t",
            value_label="V",
            population=pop,
            aggregation="mean",
            permutations=9,
            lisa_permutations=9,
        )
        assert "percap" not in m.measures and any(
            "only meaningful for a sum" in n for n in m.notes
        )

    def test_all_regions_missing_makes_no_views(self, grid):
        data = pl.DataFrame({"c": ["C00"], "v": [""]})
        values, join = aggregate_to_regions(data, "c", "v", "sum", grid.index)
        m = build_model(
            grid,
            values,
            join,
            title="t",
            value_label="V",
            permutations=9,
            lisa_permutations=9,
        )
        assert m.views == [] and (m.measures["raw"].status == OK).sum() == 0


class TestStatsHelpers:
    def test_spearman_matches_scipy_including_ties(self):
        rng = np.random.default_rng(0)
        a = rng.integers(0, 15, 200).astype(float)
        b = a + rng.normal(0, 4, 200)
        assert gh.spearman(a, b) == pytest.approx(
            stats.spearmanr(a, b).statistic, rel=1e-12
        )

    def test_spearman_skips_non_finite_pairs_and_degenerate_inputs(self):
        assert gh.spearman([1, 2, 3, np.nan], [2, 4, 6, 1]) == pytest.approx(1.0)
        assert np.isnan(gh.spearman([1, 1, 1], [1, 2, 3])) and np.isnan(
            gh.spearman([1, 2], [1, 2])
        )

    def test_top_overlap(self):
        a = np.arange(100.0)
        assert gh.top_overlap(a, a) == 1.0
        assert gh.top_overlap(a, -a) == 0.0
        b = a.copy()
        b[95:] = -1  # five of the ten highest by `a` fall out of the top ten by `b`
        assert gh.top_overlap(a, b, 0.1) == pytest.approx(0.5)


class TestBundledData:
    def test_withheld_cells_are_null_not_zero(self):
        """The Census stores a withheld cell as 0 with flag 'D'; loading it as 0 would fake a real zero."""
        raw = pl.read_csv(
            gh.DATA / "retail_sales_2017.csv",
            schema_overrides={"fips": pl.Utf8, "naics": pl.Utf8},
        )
        withheld = raw.filter(pl.col("sales_flag") == "D")
        assert withheld.height > 1000
        assert withheld["sales_k"].null_count() == withheld.height
        aleutians = raw.filter(
            (pl.col("fips") == "02013") & (pl.col("naics") == "44-45")
        )
        assert aleutians["sales_k"][0] is None and aleutians["sales_flag"][0] == "D"

    def test_state_sales_sum_to_the_published_national_total(self):
        sales, _ = gh.load_bundled_tables("state", "44-45")
        assert (
            sales["sales"].sum() == 4_949_601_481 * 1000
        )  # United States row of the same file

    def test_every_bundled_geography_is_present(self):
        for level, n in (("state", 52), ("county", 3220)):
            b = gh.load_bundled_boundaries(level)
            assert len(b.features) == n == len(b.index)

    def test_population_is_plausible(self):
        _, pop = gh.load_bundled_tables("state", "44-45")
        assert (
            320e6 < pop["population"].sum() < 330e6
        )  # 2017 U.S. population, excluding Puerto Rico

    def test_unknown_naics(self):
        with pytest.raises(ValueError, match="unknown NAICS"):
            gh.load_bundled_tables("state", "999")

    def test_state_model_end_to_end(self):
        b = gh.load_bundled_boundaries("state")
        sales, pop = gh.load_bundled_tables("state", "44-45")
        values, join = aggregate_to_regions(sales, "fips", "sales", "sum", b.index)
        pop_values, pop_join = aggregate_to_regions(
            pop, "fips", "population", "sum", b.index
        )
        m = build_model(
            b,
            values,
            join,
            title="t",
            value_label="Sales",
            population=(pop_values, pop_join),
            permutations=99,
            lisa_permutations=99,
        )
        assert join.regions_without_record == [
            "Puerto Rico"
        ]  # the Economic Census file has no PR row
        assert m.measures["raw"].status.tolist().count(NO_RECORD) == 1
        assert len(m.views) == 8 and m.lisa is not None
        assert (
            m.stats["spearman_raw_population"] > 0.9
        )  # totals are mostly a population map
        assert m.moran["raw"] is not None and m.moran["percap"] is not None


class TestCli:
    def run(self, argv, capsys):
        code = gh.main(argv)
        return code, capsys.readouterr()

    def state_csv(self, tmp_path):
        path = tmp_path / "sales.csv"
        path.write_text(
            "state,revenue,pop\n"
            'AL,"$1,200",4900000\n'
            "Alabama,300,4900000\n"
            "texas,5000,28000000\n"
            "TX,abc,28000000\n"
            "Narnia,7,1\n"
            "1,50,4900000\n",
            encoding="utf-8",
        )
        return path

    def test_build_from_csv_reports_unmatched_and_writes_html(self, tmp_path, capsys):
        out = tmp_path / "map.html"
        code, cap = self.run(
            [
                "build",
                str(self.state_csv(tmp_path)),
                "--region-col",
                "state",
                "--value-col",
                "revenue",
                "--population-col",
                "pop",
                "--level",
                "state",
                "--prefix",
                "$",
                "--permutations",
                "9",
                "--lisa-permutations",
                "9",
                "-o",
                str(out),
            ],
            capsys,
        )
        text = out.read_text(encoding="utf-8")
        assert code == 0 and "1 unmatched key(s) covering 1 row(s)" in cap.out
        assert "Narnia" in text and "no region with this key" in text
        assert "0 blank and 1 non-numeric" in text  # the 'abc' cell

    def test_auto_level_detection(self, tmp_path, capsys):
        counties = tmp_path / "c.csv"
        counties.write_text("fips,v\n01001,5\n1003,6\n48201,7\n", encoding="utf-8")
        code, cap = self.run(
            [
                "build",
                str(counties),
                "--region-col",
                "fips",
                "--value-col",
                "v",
                "--permutations",
                "9",
                "--lisa-permutations",
                "9",
                "-o",
                str(tmp_path / "o.html"),
            ],
            capsys,
        )
        assert code == 0 and "3/3220 regions have data" in cap.out

    def test_custom_geojson(self, tmp_path, grid_geojson, capsys):
        path, _ = grid_geojson
        csv = tmp_path / "d.csv"
        csv.write_text("cell,v\nC00,1\nC10,2\nC20,3\nC01,4\n", encoding="utf-8")
        code, cap = self.run(
            [
                "build",
                str(csv),
                "--region-col",
                "cell",
                "--value-col",
                "v",
                "--geojson",
                str(path),
                "--geo-key",
                "id",
                "--permutations",
                "9",
                "--lisa-permutations",
                "9",
                "-o",
                str(tmp_path / "o.html"),
            ],
            capsys,
        )
        assert code == 0 and "4/6 regions have data" in cap.out

    def test_missing_column_is_a_clean_error(self, tmp_path, capsys):
        code, cap = self.run(
            [
                "build",
                str(self.state_csv(tmp_path)),
                "--region-col",
                "nope",
                "--value-col",
                "revenue",
                "-o",
                str(tmp_path / "o.html"),
            ],
            capsys,
        )
        assert (
            code == 2
            and "column(s) not found: nope" in cap.err
            and "Available" in cap.err
        )

    def test_keys_that_match_nothing_are_a_clean_error(self, tmp_path, capsys):
        bad = tmp_path / "bad.csv"
        bad.write_text("k,v\nfoo,1\nbar,2\n", encoding="utf-8")
        code, cap = self.run(
            [
                "build",
                str(bad),
                "--region-col",
                "k",
                "--value-col",
                "v",
                "-o",
                str(tmp_path / "o.html"),
            ],
            capsys,
        )
        assert code == 2 and "no key matched" in cap.err

    def test_missing_file(self, tmp_path, capsys):
        code, cap = self.run(
            [
                "build",
                str(tmp_path / "absent.csv"),
                "--region-col",
                "k",
                "--value-col",
                "v",
            ],
            capsys,
        )
        assert code == 2 and cap.err.startswith("error:")

    @pytest.mark.parametrize("classes", ["1", "10"])
    def test_class_count_is_validated(self, tmp_path, capsys, classes):
        code, cap = self.run(
            [
                "demo",
                "--level",
                "state",
                "--classes",
                classes,
                "-o",
                str(tmp_path / "o.html"),
            ],
            capsys,
        )
        assert code == 2 and "--classes" in cap.err

    def test_demo_writes_a_self_contained_file(self, tmp_path, capsys):
        out = tmp_path / "demo.html"
        code, cap = self.run(
            [
                "demo",
                "--level",
                "state",
                "--naics",
                "445",
                "--permutations",
                "9",
                "--lisa-permutations",
                "9",
                "-o",
                str(out),
            ],
            capsys,
        )
        assert code == 0 and out.stat().st_size > 100_000 and "51/52 regions" in cap.out


class TestSampleData:
    def test_messy_orders_sample_demonstrates_the_join_repairs(self, tmp_path):
        args = gh.build_parser().parse_args(
            [
                "build",
                str(gh.HERE / "sample_data" / "messy_orders.csv"),
                "--region-col",
                "state",
                "--value-col",
                "order_value",
                "--population-col",
                "customer_pop",
                "--permutations",
                "9",
                "--lisa-permutations",
                "9",
            ]
        )
        m = gh.cmd_build(args)
        j = m.join
        assert {k for k, _r, _n in j.unmatched} == {
            "Californa",
            "Guam",
            "",
        }  # typo, territory, blank key
        assert j.matched_regions == 14 and j.duplicate_rows_merged == 7
        assert (j.blank_values, j.unparseable_values) == (1, 1)
        names = m.boundaries.index.names
        raw = m.measures["raw"]
        value = {names[i]: (raw.values[i], raw.status[i]) for i in range(len(names))}
        assert value["California"][0] == 24900.0  # "6" (FIPS) + "California"
        assert value["Texas"][0] == 7650.0  # "Texas" + "texas " + "TX", "1,800" parsed
        assert value["Vermont"] == (0.0, OK)  # a real zero...
        assert (
            value["Ohio"][1] == BLANK and value["Oregon"][1] == BLANK
        )  # ...is not a blank
        assert value["District of Columbia"][0] == 500.0
