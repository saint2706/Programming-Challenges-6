from pathlib import Path

import polars as pl
import pytest
from labels import (
    CATEGORY_PATTERNS,
    RESUME_CATEGORIES,
    audit_sample,
    balanced_pool,
    label_postings,
    title_category,
)


@pytest.mark.parametrize(
    ("title", "category"),
    [
        ("Registered Nurse - RN - LTAC", "HEALTHCARE"),
        ("Staff Accountant", "ACCOUNTANT"),
        ("Senior Software Engineer", "INFORMATION-TECHNOLOGY"),
        ("Mechanical Engineer", "ENGINEERING"),
        ("Executive Chef", "CHEF"),
        ("HR Generalist", "HR"),
        ("Middle School Teacher", "TEACHER"),
        ("Customer Service Representative", "BPO"),
        ("Outside Sales Representative", "SALES"),
        ("Corporate Attorney", "ADVOCATE"),
        ("Graphic Designer", "DESIGNER"),
        ("Personal Trainer", "FITNESS"),
    ],
)
def test_title_maps_to_expected_category(title, category):
    assert title_category(title) == category


@pytest.mark.parametrize(
    "title", ["Finance Manager and HR Manager", "Sales Accountant"]
)
def test_title_matching_two_categories_is_dropped_not_first_wins(title):
    assert title_category(title) is None


@pytest.mark.parametrize("title", ["Xylophone Wrangler", "", "   "])
def test_unmapped_or_blank_title_is_none(title):
    assert title_category(title) is None


def test_every_resume_category_has_exactly_one_pattern():
    names = [c for c, _ in CATEGORY_PATTERNS]
    assert len(names) == len(set(names)) == 24
    assert set(names) == RESUME_CATEGORIES


def test_patterns_are_case_insensitive():
    assert title_category("REGISTERED NURSE") == title_category("registered nurse")


def _postings():
    return pl.DataFrame(
        {
            "job_id": [str(i) for i in range(9)],
            "title": ["Registered Nurse"] * 4
            + ["Executive Chef"] * 3
            + ["Xylophone Wrangler", "Sales Accountant"],
            "text": ["body"] * 9,
        }
    )


def test_label_postings_drops_unmapped_and_ambiguous():
    out = label_postings(_postings())
    assert out.height == 7
    assert set(out["category"]) == {"HEALTHCARE", "CHEF"}


def test_balanced_pool_caps_each_category_and_is_seeded():
    pool = balanced_pool(label_postings(_postings()), per_category=3, seed=0)
    counts = dict(pool.group_by("category").len().iter_rows())
    assert counts == {"HEALTHCARE": 3, "CHEF": 3}
    again = balanced_pool(label_postings(_postings()), per_category=3, seed=0)
    assert pool["job_id"].to_list() == again["job_id"].to_list()


def test_audit_sample_has_blank_ok_column_and_fixed_size():
    s = audit_sample(label_postings(_postings()), n=5, seed=1)
    assert s.height == 5
    assert s.columns == ["job_id", "title", "category", "ok"]
    assert s["ok"].to_list() == [""] * 5


@pytest.mark.parametrize(
    ("title", "category"),
    [
        ("Instructional Designer", None),
        ("Structural Revit Designer", None),
        ("IATF Lead Auditor", None),
        ("Data Solutions Engineer III (Managed Care - Finance Focus)", None),
        ("Internal Auditor", "ACCOUNTANT"),
        ("Finance Manager", "FINANCE"),
        ("Director of Finance", "FINANCE"),
        ("UX Designer", "DESIGNER"),
    ],
)
def test_audit_regressions_do_not_overreach(title, category):
    assert title_category(title) == category


def test_committed_audit_file_still_matches_the_label_table():
    audit = pl.read_csv(
        Path(__file__).parent / "audit" / "labels_audit.csv",
        schema_overrides={"job_id": pl.Utf8},
    )
    assert audit.height == 100
    assert set(audit["ok"].to_list()) <= {0, 1}
    for title, category in zip(audit["title"], audit["category"], strict=True):
        assert title_category(title) == category
