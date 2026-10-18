"""Rebuild the two committed sample files from their public sources.

    uv run python -m pivot_tool.fetch_data

* ``sample_data/adult.csv.gz``: the UCI Adult census extract (Becker and Kohavi, 1996; CC BY 4.0),
  train and test files stacked, with a header, a ``split`` column, and the test file's trailing
  full stops removed from ``income``. 48,842 rows, 15 fields plus ``split``.
* ``sample_data/ucb_admissions.csv``: the 1973 UC Berkeley graduate admissions by department and
  sex (Bickel, Hammel and O'Connell, Science 1975), one row per applicant. The counts are the
  24 cells of R's ``UCBAdmissions`` table (4,526 applicants over the six largest departments).
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import urllib.request
import zipfile

from pivot_tool.paths import project_root

ADULT_URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"
ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

# (department, sex) -> (admitted, rejected): Bickel, Hammel and O'Connell (1975), Table 1.
UCB = {
    ("A", "Male"): (512, 313),
    ("A", "Female"): (89, 19),
    ("B", "Male"): (353, 207),
    ("B", "Female"): (17, 8),
    ("C", "Male"): (120, 205),
    ("C", "Female"): (202, 391),
    ("D", "Male"): (138, 279),
    ("D", "Female"): (131, 244),
    ("E", "Male"): (53, 138),
    ("E", "Female"): (94, 299),
    ("F", "Male"): (22, 351),
    ("F", "Female"): (24, 317),
}


def ucb_rows() -> list[tuple[str, str, str]]:
    rows = []
    for (dept, sex), (admitted, rejected) in UCB.items():
        rows += [(dept, sex, "Admitted")] * admitted
        rows += [(dept, sex, "Rejected")] * rejected
    return rows


def adult_csv(archive: bytes) -> str:
    out = [",".join([*ADULT_COLUMNS, "split"])]
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        for name, split in (("adult.data", "train"), ("adult.test", "test")):
            for line in zf.read(name).decode("utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("|"):
                    continue  # the test file begins with a comment line
                cells = [c.strip() for c in line.split(",")]
                cells[-1] = cells[-1].rstrip(".")
                if len(cells) != len(ADULT_COLUMNS):
                    raise ValueError(f"{name}: unexpected row {line!r}")
                out.append(",".join([*cells, split]))
    return "\n".join(out) + "\n"


def main() -> None:
    root = project_root() / "sample_data"
    root.mkdir(exist_ok=True)
    print(f"downloading {ADULT_URL}")
    with urllib.request.urlopen(ADULT_URL, timeout=120) as resp:
        archive = resp.read()
    csv = adult_csv(archive)
    with gzip.GzipFile(
        root / "adult.csv.gz", "wb", mtime=0
    ) as fh:  # mtime 0: reproducible bytes
        fh.write(csv.encode())
    rows = ucb_rows()
    ucb = "dept,sex,outcome\n" + "".join(f"{d},{s},{o}\n" for d, s, o in rows)
    (root / "ucb_admissions.csv").write_text(ucb, encoding="utf-8", newline="\n")
    info = {
        "adult.csv.gz": {
            "url": ADULT_URL,
            "download_sha256": hashlib.sha256(archive).hexdigest(),
            "rows": csv.count("\n") - 1,
        },
        "ucb_admissions.csv": {
            "source": "Bickel, Hammel and O'Connell, Science 187 (1975); R datasets::UCBAdmissions",
            "rows": len(rows),
        },
    }
    (root / "SOURCES.json").write_text(
        json.dumps(info, indent=4) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"adult: {info['adult.csv.gz']['rows']:,} rows, ucb: {len(rows):,} rows")


if __name__ == "__main__":
    main()
