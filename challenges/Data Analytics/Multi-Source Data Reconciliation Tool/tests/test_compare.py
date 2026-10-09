import math

import pytest
from data_reconciler import compare as c
from data_reconciler.config import Field


def cat(result):
    return result.category


# --- text -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("London Heathrow", "London Heathrow", c.MATCH),
        ("London Heathrow", "LONDON  HEATHROW", c.FORMAT_ONLY),
        ("Zürich Airport", "Zurich airport", c.FORMAT_ONLY),
        ("St. Louis-Lambert", "St Louis Lambert", c.FORMAT_ONLY),
        (
            "Rabah Bitat Airport",
            "Annaba Rabah Bitat Airport",
            c.NEAR,
        ),  # one contains the other
        (
            "Al Ghaidah International Airport",
            "Al Ghaydah International Airport",
            c.NEAR,
        ),  # typo
        ("Buariki Airport", "Aranuka Airport", c.MISMATCH),
        (None, "x", c.LEFT_MISSING),
        ("x", "  ", c.RIGHT_MISSING),
        (None, None, c.BOTH_MISSING),
    ],
)
def test_text_categories(left, right, expected):
    assert cat(c.compare_text(left, right, near=0.85)) == expected


def test_text_near_threshold_is_configurable_and_reports_the_similarity():
    r = c.compare_text("Kabri Dehar Airport", "Kebri Dahar Airport", near=0.85)
    assert r.category == c.NEAR and 0.85 <= r.metric < 1
    assert (
        c.compare_text("Kabri Dehar Airport", "Kebri Dahar Airport", near=0.99).category
        == c.MISMATCH
    )


def test_normalize_text_strips_accents_case_punctuation_and_space():
    assert (
        c.normalize_text("  Ünited-STATES,  of   America ")
        == "united states of america"
    )


# --- codes and numbers ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("LHR", "LHR", c.MATCH),
        (" lhr", "LHR", c.FORMAT_ONLY),
        ("LHR", "LGW", c.MISMATCH),
        (None, "LHR", c.LEFT_MISSING),
    ],
)
def test_code_categories(left, right, expected):
    assert cat(c.compare_code(left, right)) == expected


def test_number_tolerances_absolute_and_relative():
    assert cat(c.compare_number("83", "83.0", 0, 0)) == c.MATCH
    assert cat(c.compare_number("83", "86", 5, 0)) == c.WITHIN
    assert cat(c.compare_number("83", "100", 5, 0)) == c.MISMATCH
    assert (
        cat(c.compare_number("1000", "1040", 0, 0.05)) == c.WITHIN
    )  # 4% of the larger
    r = c.compare_number("1000", "1100", 0, 0.05)
    assert r.category == c.MISMATCH and r.metric == 100


@pytest.mark.parametrize("bad", ["abc", "NaN", "inf", "1,5"])
def test_unreadable_numbers_are_invalid_not_silently_zero(bad):
    r = c.compare_number(bad, "5", 0, 0)
    assert r.category == c.INVALID and "left" in r.note
    assert c.compare_number("5", bad, 0, 0).category == c.INVALID


def test_number_missing_sides():
    assert cat(c.compare_number(None, "5", 0, 0)) == c.LEFT_MISSING
    assert cat(c.compare_number("5", "", 0, 0)) == c.RIGHT_MISSING
    assert cat(c.compare_number("", None, 0, 0)) == c.BOTH_MISSING


# --- geography --------------------------------------------------------------------------------


def test_haversine_matches_known_distances():
    assert c.haversine_km(0, 0, 0, 1) == pytest.approx(
        111.195, abs=0.01
    )  # one degree of longitude at the equator
    assert c.haversine_km(51.4706, -0.4619, 49.0097, 2.5479) == pytest.approx(
        348, abs=5
    )  # Heathrow to CDG
    assert c.haversine_km(10, 20, 10, 20) == 0
    assert c.haversine_km(0, 0, 0, 180) == pytest.approx(
        math.pi * c.EARTH_RADIUS_KM, rel=1e-9
    )  # antipodes: no domain error


def test_geo_categories():
    assert cat(c.compare_geo(("51.47", "-0.46"), ("51.47", "-0.46"), 0.5)) == c.MATCH
    near = c.compare_geo(("51.4706", "-0.461941"), ("51.470748", "-0.459909"), 0.5)
    assert near.category == c.WITHIN and 0 < near.metric < 0.5
    far = c.compare_geo(("29.72", "-95.59"), ("-0.37", "117.25"), 0.5)
    assert far.category == c.MISMATCH and far.metric > 10000


def test_geo_missing_and_invalid():
    assert cat(c.compare_geo((None, None), ("1", "2"), 1)) == c.LEFT_MISSING
    assert (
        cat(c.compare_geo(("1", None), ("1", "2"), 1)) == c.LEFT_MISSING
    )  # half a coordinate is no coordinate
    assert cat(c.compare_geo(("1", "2"), ("", ""), 1)) == c.RIGHT_MISSING
    assert cat(c.compare_geo((None, None), ("", ""), 1)) == c.BOTH_MISSING
    out = c.compare_geo(("91", "0"), ("1", "2"), 1)
    assert out.category == c.INVALID and "outside the valid range" in out.note
    assert cat(c.compare_geo(("1", "2"), ("x", "2"), 1)) == c.INVALID


# --- crosswalk --------------------------------------------------------------------------------


def spec(**kw):
    return Field("country", ("a",), ("b",), "crosswalk", **kw)


def test_the_crosswalk_is_learned_by_majority_and_flags_the_odd_one_out():
    lefts = ["United States"] * 9 + ["United States"]
    rights = ["US"] * 9 + ["ID"]
    results, table = c.compare_crosswalk(
        lefts, rights, spec(min_support=3, min_share=0.6)
    )
    assert [r.category for r in results] == [c.MATCH] * 9 + [c.MISMATCH]
    assert "'united states' maps to 'us' in 90% of 10 pairs" == results[-1].note
    entry = table[0]
    assert (entry.left_value, entry.right_value, entry.pairs, entry.trusted) == (
        "united states",
        "us",
        10,
        True,
    )


def test_a_value_with_too_little_evidence_is_unverifiable_not_wrong():
    results, table = c.compare_crosswalk(
        ["Narnia", "Narnia"], ["NA", "ZZ"], spec(min_support=5)
    )
    assert [r.category for r in results] == [c.UNVERIFIABLE, c.UNVERIFIABLE]
    assert not table[0].trusted


def test_a_split_vote_is_not_trusted():
    lefts, rights = ["X"] * 6, ["a", "a", "a", "b", "b", "b"]
    results, table = c.compare_crosswalk(
        lefts, rights, spec(min_support=3, min_share=0.6)
    )
    assert {r.category for r in results} == {c.UNVERIFIABLE} and not table[0].trusted


def test_crosswalk_ignores_case_and_space_and_handles_missing_values():
    results, _ = c.compare_crosswalk(
        ["Peru", " peru", "PERU", None, "Peru"],
        ["pe", "PE ", "pe", "pe", None],
        spec(min_support=3),
    )
    assert [r.category for r in results] == [
        c.MATCH,
        c.MATCH,
        c.MATCH,
        c.LEFT_MISSING,
        c.RIGHT_MISSING,
    ]


def test_ties_resolve_deterministically():
    a = c.learn_crosswalk([("x", "b"), ("x", "a")], 1, 0.5)
    b = c.learn_crosswalk([("x", "a"), ("x", "b")], 1, 0.5)
    assert a["x"].right_value == b["x"].right_value == "a"


# --- dispatch ---------------------------------------------------------------------------------


def test_compare_field_dispatches_by_comparator_and_returns_no_crosswalk_for_others():
    results, table = c.compare_field(
        Field("n", ("a",), ("b",), "number", abs_tolerance=2), ["1", "9"], ["2", "1"]
    )
    assert [r.category for r in results] == [c.WITHIN, c.MISMATCH] and table == []


def test_the_differences_set_is_what_a_person_must_look_at():
    assert c.DIFFERENCES == {
        c.MISMATCH,
        c.NEAR,
        c.INVALID,
        c.LEFT_MISSING,
        c.RIGHT_MISSING,
    }
    assert set(c.CATEGORIES) >= c.DIFFERENCES | {
        c.MATCH,
        c.FORMAT_ONLY,
        c.WITHIN,
        c.BOTH_MISSING,
        c.UNVERIFIABLE,
    }
