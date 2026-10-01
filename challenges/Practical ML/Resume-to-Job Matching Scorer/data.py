"""Loading, cleaning and splitting the resume and job-posting data.

Resumes: Kaggle "Resume Dataset" (livecareer, 24 categories). Postings: Kaggle
"LinkedIn Job Postings". Both are downloaded once into ``data/`` (gitignored).
"""

from __future__ import annotations

import html
import random
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import polars as pl

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"

RESUME_DATASET = "snehaanbhawal/resume-dataset"
RESUME_FILE = "Resume/Resume.csv"
POSTINGS_DATASET = "arshkon/linkedin-job-postings"
POSTINGS_FILE = "postings.csv"

# Postings whose description is shorter than this are noise (the data has a
# 2-character one); a real description is hundreds of characters.
MIN_POSTING_CHARS = 50
# Resumes under this are scanned stubs ("", 21 chars exist) with nothing to rank on.
MIN_RESUME_CHARS = 100

_HSPACE = re.compile(r"[^\S\n]+")
_BLANKS = re.compile(r"\n\s*\n+")
_TITLE_SEPARATORS = re.compile(r"\s*[/|,]\s*|\s+-\s+")


def clean_text(s: str) -> str:
    """Unescape HTML entities, collapse spaces/tabs, keep single line breaks.

    Line structure is kept because the first line of a resume is its headline
    (see :func:`strip_headline`).
    """
    s = html.unescape(s)
    s = _HSPACE.sub(" ", s)
    s = "\n".join(line.strip() for line in s.split("\n"))
    s = _BLANKS.sub("\n", s)
    return s.strip()


def strip_headline(s: str) -> str:
    """Drop the first non-blank line.

    Each resume starts with the job title its category was derived from, so a
    scorer that sees it is partly matching the label. Stripped text is the
    skills-only view.
    """
    lines = s.strip().split("\n", 1)
    if len(lines) < 2:
        return ""
    headline, body = lines[0], lines[1].strip()
    # The body usually opens by repeating the title ("HR ADMINISTRATOR Summary
    # ..."), often just one half of "A/B"; drop that too or the label leaks back in.
    for seg in sorted(_TITLE_SEPARATORS.split(headline), key=len, reverse=True):
        if seg and body.lower().startswith(seg.lower()):
            return body[len(seg) :].strip()
    return body


def load_resumes(path: Path) -> pl.DataFrame:
    df = pl.read_csv(path)
    out = pl.DataFrame(
        {
            "id": [str(i) for i in df["ID"].to_list()],
            "category": df["Category"].to_list(),
            "text": [clean_text(t or "") for t in df["Resume_str"].to_list()],
        }
    )
    return out.filter(pl.col("text").str.len_chars() >= MIN_RESUME_CHARS)


def load_postings(path: Path) -> pl.DataFrame:
    """Postings with a usable description; ``text`` is title + description."""
    df = pl.read_csv(
        path,
        columns=["job_id", "title", "description"],
        schema_overrides={"job_id": pl.Utf8},
        infer_schema_length=0,
    ).drop_nulls(["title", "description"])
    desc = [clean_text(d) for d in df["description"].to_list()]
    keep = [len(d) >= MIN_POSTING_CHARS for d in desc]
    return pl.DataFrame(
        {
            "job_id": [
                j for j, k in zip(df["job_id"].to_list(), keep, strict=True) if k
            ],
            "title": [t for t, k in zip(df["title"].to_list(), keep, strict=True) if k],
            "text": [d for d, k in zip(desc, keep, strict=True) if k],
        }
    )


def split_ids(
    ids: list[str], strata: list[str], val_frac: float, seed: int
) -> tuple[list[str], list[str]]:
    """Stratified id split. Whole ids go to one side, so nothing is on both."""
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for i, s in zip(ids, strata, strict=True):
        by_stratum[s].append(i)
    rng = random.Random(seed)
    val: list[str] = []
    test: list[str] = []
    for stratum in sorted(by_stratum):
        members = sorted(by_stratum[stratum])
        rng.shuffle(members)
        cut = round(len(members) * val_frac)
        val += members[:cut]
        test += members[cut:]
    return val, test


def check_split_disjoint(a: list[str], b: list[str]) -> None:
    both = set(a) & set(b)
    if both:
        raise ValueError(f"splits overlap on {len(both)} ids, e.g. {sorted(both)[:3]}")


def _unzip_if_needed(path: Path) -> None:
    if path.exists() and zipfile.is_zipfile(path):
        tmp = path.with_suffix(path.suffix + ".zip")
        path.rename(tmp)
        with zipfile.ZipFile(tmp) as z:
            z.extract(z.namelist()[0], path.parent)
        tmp.unlink()


def fetch(data_dir: Path = DATA_DIR) -> None:
    """Download both CSVs with the Kaggle API (token in ``~/.kaggle/kaggle.json``)."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    data_dir.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    for dataset, remote, local in (
        (RESUME_DATASET, RESUME_FILE, "Resume.csv"),
        (POSTINGS_DATASET, POSTINGS_FILE, "postings.csv"),
    ):
        target = data_dir / local
        if target.exists():
            continue
        api.dataset_download_file(
            dataset, remote, path=str(data_dir), force=False, quiet=False
        )
        _unzip_if_needed(target)
