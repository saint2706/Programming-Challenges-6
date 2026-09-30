"""Key normalisation and join reporting: nothing may fail to match silently."""

from __future__ import annotations

import pytest
from conftest import grid_features
from geo_heatmap import load_bundled_boundaries
from regions import RegionIndex, norm_text, normalize_fips, resolve_keys


@pytest.fixture(scope="module")
def states():
    return load_bundled_boundaries("state").index


@pytest.fixture(scope="module")
def counties():
    return load_bundled_boundaries("county").index


class TestNormalizeFips:
    @pytest.mark.parametrize(
        "raw,width,expected",
        [
            ("01001", 5, "01001"),
            (1001, 5, "01001"),  # Excel dropped the leading zero
            ("1001", 5, "01001"),
            (1001.0, 5, "01001"),
            ("01001.0", 5, "01001"),
            ("'01001", 5, "01001"),  # spreadsheet text marker
            (" 06 ", 2, "06"),
            (6, 2, "06"),
            ("6", 2, "06"),
        ],
    )
    def test_repairs(self, raw, width, expected):
        assert normalize_fips(raw, width) == expected

    @pytest.mark.parametrize(
        "raw", ["", None, "abc", "1e3", 1001.5, float("nan"), "123456", "-1", "01 001"]
    )
    def test_rejects(self, raw):
        assert normalize_fips(raw, 5) is None

    def test_never_truncates(self):
        assert normalize_fips("010012", 5) is None and normalize_fips(123, 2) is None


class TestNormText:
    def test_case_accents_punctuation_saint(self):
        assert norm_text("  Doña  Ana ") == "dona ana"
        assert norm_text("St. Louis") == norm_text("Saint Louis") == "st louis"
        assert norm_text("Washington, D.C.") == "washington d c"


class TestStates:
    @pytest.mark.parametrize(
        "raw",
        ["AL", "al", " Alabama ", "ALABAMA", "01", 1, "1", 1.0, "1.0"],
    )
    def test_alabama_in_all_its_spellings(self, states, raw):
        assert states.names[states.resolve(raw).index] == "Alabama"

    @pytest.mark.parametrize(
        "raw",
        [
            "DC",
            "D.C.",
            "Washington DC",
            "washington d.c.",
            "District of Columbia",
            "11",
        ],
    )
    def test_district_of_columbia(self, states, raw):
        assert states.names[states.resolve(raw).index] == "District of Columbia"

    def test_similar_abbreviations_stay_distinct(self, states):
        got = {
            k: states.names[states.resolve(k).index]
            for k in ("IN", "OR", "ME", "MA", "MD", "MO", "MS")
        }
        assert (
            got["IN"] == "Indiana"
            and got["OR"] == "Oregon"
            and got["MS"] == "Mississippi"
            and got["MO"] == "Missouri"
        )

    @pytest.mark.parametrize(
        "raw,reason",
        [
            ("Atlantis", "no region"),
            ("", "blank"),
            (None, "blank"),
            (float("nan"), "blank"),
            ("99", "no region"),
        ],
    )
    def test_failures_carry_a_reason(self, states, raw, reason):
        res = states.resolve(raw)
        assert res.index is None and reason in res.reason

    def test_all_states_resolve_by_fips_abbr_and_name(self):
        b = load_bundled_boundaries("state")
        for i, feat in enumerate(b.features):
            p = feat["properties"]
            assert [
                b.index.resolve(k).index for k in (p["fips"], p["abbr"], p["name"])
            ] == [i, i, i]


class TestCounties:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("01001", "Autauga County, Alabama"),
            (1001, "Autauga County, Alabama"),
            ("Autauga County, Alabama", "Autauga County, Alabama"),
            ("autauga, AL", "Autauga County, Alabama"),
            ("Autauga, al", "Autauga County, Alabama"),
            ("Saint Louis County, Missouri", "St. Louis County, Missouri"),
            ("St. Louis city, MO", "St. Louis city, Missouri"),
            ("Doña Ana County, NM", "Doña Ana County, New Mexico"),
            ("Dona Ana, New Mexico", "Doña Ana County, New Mexico"),
            ("LaSalle Parish, Louisiana", "LaSalle Parish, Louisiana"),
            ("Carson City, Nevada", "Carson City, Nevada"),
            ("Richmond city, Virginia", "Richmond city, Virginia"),
            ("Richmond County, Virginia", "Richmond County, Virginia"),
        ],
    )
    def test_spellings(self, counties, raw, expected):
        res = counties.resolve(raw)
        assert res.index is not None, res.reason
        assert counties.names[res.index] == expected

    def test_bare_name_shared_by_a_city_and_a_county_is_ambiguous_not_guessed(
        self, counties
    ):
        res = counties.resolve("Richmond, Virginia")
        assert res.index is None and res.reason.startswith("ambiguous")

    def test_county_name_without_state(self, counties):
        res = counties.resolve("Washington County")
        assert res.index is None and "without a state" in res.reason

    def test_unknown_state_and_unknown_county(self, counties):
        assert "unrecognised state" in counties.resolve("Autauga County, Narnia").reason
        assert "no region" in counties.resolve("Nowhere County, Alabama").reason

    def test_state_code_is_not_a_county_code(self, counties):
        assert (
            counties.resolve("01").index is None
        )  # padded to 00001, which does not exist

    def test_every_county_name_round_trips_to_itself(self, counties):
        wrong = [
            n for i, n in enumerate(counties.names) if counties.resolve(n).index != i
        ]
        assert wrong == []

    def test_names_are_unique(self, counties):
        assert len(set(counties.names)) == len(counties.names)


class TestCustomIndex:
    def test_matches_by_key_and_by_name_case_insensitively(self):
        idx = RegionIndex.for_custom(grid_features(2, 1), "id", "name")
        assert idx.resolve("c00").index == 0 and idx.resolve("CELL 1,0").index == 1
        assert idx.resolve("zzz").index is None


class TestResolveKeys:
    def test_reports_every_failure_with_row_counts(self, states):
        keys = ["AL", "Alabama", "Narnia", "Narnia", "", "1", "Mordor"]
        positions, failed = resolve_keys(keys, states)
        assert (
            positions[:2] == [positions[0], positions[0]] and positions[0] is not None
        )
        assert positions[5] == positions[0]
        assert failed[0] == ("Narnia", "no region with this key", 2)
        assert {k for k, _r, _n in failed} == {"Narnia", "", "Mordor"}
        assert sum(n for _k, _r, n in failed) == 4
