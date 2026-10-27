"""Regenerate the committed test fixture from the real March 2019 part files.

The fixture is 20,000 real trips split over two part files, plus the zone lookup. It lets the
tests run every engine on real data without the full download. Run from the challenge folder:
``uv run python scripts/make_fixtures.py`` (needs ``data/`` from ``dist-pipeline prepare``).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data"
DST = ROOT / "tests" / "fixtures" / "data"
ROWS_PER_PART = 10_000


def main() -> None:
    part = pl.read_parquet(SRC / "normalized" / "1m" / "part-0000.parquet")
    out = DST / "normalized" / "mini"
    out.mkdir(parents=True, exist_ok=True)
    for i in range(2):
        chunk = part.slice(i * ROWS_PER_PART, ROWS_PER_PART)
        chunk.write_parquet(out / f"part-{i:04d}.parquet", compression="zstd")
    (DST / "raw").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        SRC / "raw" / "taxi_zone_lookup.csv", DST / "raw" / "taxi_zone_lookup.csv"
    )
    print(f"wrote {2 * ROWS_PER_PART} rows to {out}")


if __name__ == "__main__":
    main()
