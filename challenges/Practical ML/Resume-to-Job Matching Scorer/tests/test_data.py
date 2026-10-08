import polars as pl
import pytest
from resume_matcher.data import (
    check_split_disjoint,
    clean_text,
    load_postings,
    load_resumes,
    split_ids,
    strip_headline,
)


def test_clean_text_collapses_spaces_keeps_single_line_breaks_and_unescapes():
    assert clean_text("  A&amp;B \n\n C\t") == "A&B\nC"


def test_strip_headline_drops_first_line_only():
    assert strip_headline("HR ADMINISTRATOR\n\nSummary\nBody") == "Summary\nBody"


def test_strip_headline_skips_leading_blank_lines():
    assert strip_headline("   \n\nHEADLINE\nBody") == "Body"


def test_strip_headline_single_line_returns_empty_not_crash():
    assert strip_headline("ONLY TITLE") == ""


def test_split_is_stratified_and_disjoint():
    ids = [f"r{i}" for i in range(100)]
    strata = ["a"] * 50 + ["b"] * 50
    val, test = split_ids(ids, strata, 0.5, seed=0)
    assert set(val).isdisjoint(test)
    assert len(val) == 50 and len(test) == 50
    assert sum(int(i[1:]) < 50 for i in val) == 25


def test_split_is_deterministic_for_a_seed():
    ids = [f"r{i}" for i in range(40)]
    strata = ["a", "b"] * 20
    assert split_ids(ids, strata, 0.5, seed=3) == split_ids(ids, strata, 0.5, seed=3)


def test_check_split_disjoint_raises_on_overlap():
    with pytest.raises(ValueError, match="overlap"):
        check_split_disjoint(["a", "b"], ["b"])


def test_load_postings_drops_null_and_tiny_descriptions(tmp_path):
    path = tmp_path / "postings.csv"
    pl.DataFrame(
        {
            "job_id": ["1", "2", "3", "4"],
            "title": ["Nurse", "Chef", "Clerk", "Welder"],
            "description": [
                "Care for patients in a hospital ward " * 3,
                None,
                "ab",
                "Weld steel " * 10,
            ],
        }
    ).write_csv(path)
    out = load_postings(path)
    assert out["job_id"].to_list() == ["1", "4"]
    assert out.columns == ["job_id", "title", "text"]


def test_load_resumes_renames_and_cleans(tmp_path):
    path = tmp_path / "Resume.csv"
    pl.DataFrame(
        {
            "ID": [1, 2],
            "Resume_str": [
                "  CHEF \n\n Cooks &amp; bakes" + " pastry" * 30,
                "TEACHER\nTeaches" + " pupils" * 30,
            ],
            "Resume_html": ["<b>x</b>", "<b>y</b>"],
            "Category": ["CHEF", "TEACHER"],
        }
    ).write_csv(path)
    out = load_resumes(path)
    assert out.columns == ["id", "category", "text"]
    assert out["id"].to_list() == ["1", "2"]
    assert out["text"][0].startswith("CHEF\nCooks & bakes pastry")


def test_strip_headline_also_drops_the_title_repeated_at_the_start_of_the_body():
    text = "HR ADMINISTRATOR/MARKETING ASSOCIATE\nHR ADMINISTRATOR Summary Dedicated manager"
    assert strip_headline(text) == "Summary Dedicated manager"


def test_strip_headline_keeps_body_that_does_not_repeat_the_title():
    assert (
        strip_headline("CHEF\nSummary of a chef career") == "Summary of a chef career"
    )


def test_load_resumes_drops_resumes_too_short_to_score(tmp_path):
    path = tmp_path / "Resume.csv"
    pl.DataFrame(
        {
            "ID": [1, 2],
            "Resume_str": [
                "",
                "Seasoned accountant with ten years of reconciliation experience " * 3,
            ],
            "Resume_html": ["", "<b>y</b>"],
            "Category": ["CHEF", "ACCOUNTANT"],
        }
    ).write_csv(path)
    assert load_resumes(path)["id"].to_list() == ["2"]
