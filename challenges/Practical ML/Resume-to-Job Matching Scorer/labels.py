"""Proxy relevance labels for job postings.

The postings dataset has no resume-style category, so a posting's category is
inferred from its **title** with one regex per resume category. A title that
matches no category, or more than one, is dropped rather than guessed.

This table is an *evaluation device*: no scorer ever sees it. It is audited by
hand on a fixed sample (see :func:`audit_sample`), and the README reports how
often it is right.
"""

from __future__ import annotations

import re

import polars as pl


def _rx(*alts: str) -> re.Pattern[str]:
    return re.compile(r"\b(?:" + "|".join(alts) + r")\b", re.IGNORECASE)


# One entry per resume category. Words, not substrings, so "rn" does not fire
# inside "intern" and "pr" does not fire inside "project".
CATEGORY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "HEALTHCARE",
        _rx(
            r"nurses?",
            r"rn",
            r"lpn",
            r"cna",
            r"physicians?",
            r"nurse practitioner",
            r"medical assistant",
            r"pharmacists?",
            r"pharmacy technician",
            r"dental (?:hygienist|assistant)",
            r"dentist",
            r"phlebotomist",
            r"(?:physical|occupational|respiratory|speech) therapist",
            r"radiolog\w*",
            r"sonographer",
            r"surgical technologist",
            r"caregiver",
            r"patient care",
            r"clinical (?:coordinator|manager|specialist)",
        ),
    ),
    (
        "ACCOUNTANT",
        _rx(
            r"accountants?",
            r"accounting",
            r"bookkeeper",
            r"controller",
            # bare "auditor" is mostly quality/compliance (IATF, ISO), not accounting
            r"(?:financial|internal|external|tax) auditor",
            r"accounts? (?:payable|receivable)",
            r"tax (?:preparer|accountant|manager)",
        ),
    ),
    (
        "FINANCE",
        _rx(
            r"financial (?:analyst|planner|advisor|manager)",
            # "finance" alone also appears as a focus area of unrelated roles
            r"finance (?:manager|director|analyst|associate|specialist)",
            r"(?:director|manager|vp|head|vice president) of finance",
            r"treasury",
            r"fp&a",
            r"investment (?:analyst|associate)",
            r"credit analyst",
            r"portfolio manager",
        ),
    ),
    (
        "BANKING",
        _rx(
            r"bank(?:er|ing)?",
            r"teller",
            r"loan (?:officer|processor)",
            r"mortgage",
            r"branch manager",
            r"underwriter",
        ),
    ),
    (
        "INFORMATION-TECHNOLOGY",
        _rx(
            r"software",
            r"developer",
            r"devops",
            r"it (?:support|specialist|manager|analyst)",
            r"help ?desk",
            r"network (?:administrator|engineer|technician)",
            r"(?:system|systems|database) administrator",
            r"data (?:analyst|scientist|engineer)",
            r"cyber ?security",
            r"programmer",
            r"web developer",
            r"qa (?:analyst|tester)",
            r"cloud engineer",
        ),
    ),
    (
        "ENGINEERING",
        _rx(
            r"(?:mechanical|electrical|civil|chemical|process|manufacturing|industrial|structural|"
            r"quality|reliability|petroleum|environmental|materials|test|field|project) engineer",
            r"engineering (?:manager|technician)",
        ),
    ),
    (
        "SALES",
        _rx(
            r"sales",
            r"salesperson",
            r"account executive",
            r"account manager",
            r"store manager",
            r"retail (?:associate|sales)",
            r"cashier",
        ),
    ),
    (
        "BUSINESS-DEVELOPMENT",
        _rx(r"business development", r"business developer", r"partnerships?"),
    ),
    ("CONSULTANT", _rx(r"consultants?", r"consulting")),
    (
        "HR",
        _rx(
            r"hr",
            r"human resources",
            r"recruiter",
            r"talent acquisition",
            r"benefits (?:specialist|administrator)",
        ),
    ),
    (
        "TEACHER",
        _rx(
            r"teachers?",
            r"instructors?",
            r"tutors?",
            r"professors?",
            r"educators?",
            r"teaching assistant",
            r"lecturer",
        ),
    ),
    (
        "CHEF",
        _rx(
            r"chef",
            r"cook",
            r"culinary",
            r"sous",
            r"line cook",
            r"kitchen (?:manager|staff)",
            r"baker",
            r"pastry",
        ),
    ),
    (
        "FITNESS",
        _rx(
            r"fitness",
            r"personal trainer",
            r"yoga",
            r"pilates",
            r"strength and conditioning",
            r"gym",
        ),
    ),
    (
        "AVIATION",
        _rx(
            r"pilot",
            r"flight (?:attendant|instructor|operations|dispatcher)",
            r"aircraft",
            r"aviation",
            r"avionics",
            r"air traffic",
            r"airline",
        ),
    ),
    (
        "CONSTRUCTION",
        _rx(
            r"construction",
            r"carpenter",
            r"electrician",
            r"plumber",
            r"superintendent",
            r"foreman",
            r"roofer",
            r"mason",
            r"hvac",
            r"estimator",
        ),
    ),
    (
        "PUBLIC-RELATIONS",
        _rx(
            r"public relations",
            r"media relations",
            r"publicist",
            r"communications? (?:manager|specialist|director|coordinator)",
            r"corporate communications",
        ),
    ),
    (
        "DESIGNER",
        _rx(
            # instructional/structural designers are education and drafting roles
            r"(?<!instructional )(?<!structural )(?<!structural revit )designer",
            r"ux",
            r"ui",
            r"graphic",
            r"art director",
            r"creative director",
            r"illustrator",
        ),
    ),
    (
        "DIGITAL-MEDIA",
        _rx(
            r"digital marketing",
            r"social media",
            r"seo",
            r"content (?:writer|creator|manager|strategist|producer)",
            r"video (?:editor|producer)",
            r"videographer",
            r"copywriter",
            r"journalist",
            r"multimedia",
        ),
    ),
    (
        "ARTS",
        _rx(
            r"artist",
            r"musician",
            r"actor",
            r"actress",
            r"curator",
            r"gallery",
            r"dancer",
            r"photographer",
            r"theat(?:er|re)",
        ),
    ),
    (
        "APPAREL",
        _rx(
            r"apparel",
            r"fashion",
            r"garment",
            r"merchandis(?:er|ing)",
            r"stylist",
            r"textile",
            r"seamstress",
            r"tailor",
        ),
    ),
    (
        "AGRICULTURE",
        _rx(
            r"agricultur\w*",
            r"farm\w*",
            r"crop",
            r"agronom\w*",
            r"livestock",
            r"horticultur\w*",
            r"grower",
            r"ranch\w*",
            r"forestry",
            r"greenhouse",
        ),
    ),
    (
        "AUTOMOBILE",
        _rx(
            r"automotive",
            r"auto (?:mechanic|technician|body|sales)",
            r"collision",
            r"service advisor",
            r"vehicle",
            r"dealership",
            r"diesel",
            r"mechanic",
        ),
    ),
    (
        "BPO",
        _rx(
            r"call cent(?:er|re)",
            r"customer service",
            r"customer support",
            r"telemarketer",
            r"data entry",
            r"back office",
            r"process associate",
            r"technical support representative",
        ),
    ),
    (
        "ADVOCATE",
        _rx(
            r"attorney",
            r"lawyer",
            r"paralegal",
            r"legal (?:assistant|secretary|counsel)",
            r"counsel",
            r"advocate",
            r"law clerk",
        ),
    ),
]

RESUME_CATEGORIES = {name for name, _ in CATEGORY_PATTERNS}


def title_category(title: str) -> str | None:
    """The one category a title matches, or ``None`` if it matches none or several."""
    matched = [name for name, rx in CATEGORY_PATTERNS if rx.search(title)]
    return matched[0] if len(matched) == 1 else None


def label_postings(df: pl.DataFrame) -> pl.DataFrame:
    """Add ``category`` and drop postings whose title is unmapped or ambiguous."""
    cats = [title_category(t) for t in df["title"].to_list()]
    return df.with_columns(pl.Series("category", cats, dtype=pl.Utf8)).drop_nulls(
        "category"
    )


def balanced_pool(df: pl.DataFrame, per_category: int, seed: int) -> pl.DataFrame:
    """At most ``per_category`` postings per category, sampled with a fixed seed.

    A few frequent categories (sales, healthcare) would otherwise dominate the
    pool and make every scorer look better at them.
    """
    parts = [
        group.sample(n=min(per_category, group.height), seed=seed)
        for _, group in df.sort("job_id").group_by("category", maintain_order=True)
    ]
    return pl.concat(parts).sort("job_id")


def audit_sample(df: pl.DataFrame, n: int, seed: int) -> pl.DataFrame:
    """A fixed random sample to judge by hand; ``ok`` is filled in by the reviewer."""
    s = df.sort("job_id").sample(n=min(n, df.height), seed=seed)
    return s.select("job_id", "title", "category").with_columns(pl.lit("").alias("ok"))
