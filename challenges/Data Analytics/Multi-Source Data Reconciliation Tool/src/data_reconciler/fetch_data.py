r"""Download the two public airport files and cut the small committed sample.

Run with:  uv run python -m data_reconciler.fetch_data

* OpenFlights ``airports.dat`` (Open Database License), headerless CSV with ``\N`` for missing.
* OurAirports ``airports.csv`` (public domain), kept current by volunteers.

Writes ``data/`` (not committed) and ``sample_data/`` (committed): 500 random OpenFlights rows, the
OurAirports rows that carry any of their codes, plus 150 random OurAirports airports that have an
IATA code but are not in that slice, so every category of difference and every kind of
unmatched row appears in the sample. Both keep the original file format, so one config reads either.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import polars as pl

from data_reconciler.paths import project_root

SOURCES = {
    "openflights_airports.dat": "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat",
    "ourairports_airports.csv": "https://davidmegginson.github.io/ourairports-data/airports.csv",
}
OPENFLIGHTS_COLUMNS = [
    "id",
    "name",
    "city",
    "country",
    "iata",
    "icao",
    "lat",
    "lon",
    "alt",
    "tz",
    "dst",
    "tzdb",
    "type",
    "source",
]
SEED = 20261009
NULL = chr(92) + "N"  # OpenFlights writes a missing value as backslash-N


def download(name: str, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / name
    if not target.exists():
        print(f"downloading {SOURCES[name]}")
        req = urllib.request.Request(
            SOURCES[name], headers={"User-Agent": "data-reconciler-fetch/1.0"}
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            target.write_bytes(resp.read())
    return target


def main() -> None:
    root = project_root()
    paths = {name: download(name, root / "data") for name in SOURCES}
    of = pl.read_csv(
        paths["openflights_airports.dat"],
        has_header=False,
        new_columns=OPENFLIGHTS_COLUMNS,
        null_values=[NULL],
        infer_schema_length=0,
    )
    oa = pl.read_csv(paths["ourairports_airports.csv"], infer_schema_length=0)

    of_sample = of.sample(500, seed=SEED).sort(pl.col("id").cast(pl.Int64))
    iata, icao = (
        of_sample["iata"].drop_nulls().implode(),
        of_sample["icao"].drop_nulls().implode(),
    )
    linked = oa.filter(
        pl.col("iata_code").is_in(iata)
        | pl.col("icao_code").is_in(icao)
        | pl.col("ident").is_in(icao)
    )
    extra = (
        oa.filter(pl.col("iata_code").is_not_null())
        .join(linked.select("id"), on="id", how="anti")
        .sample(150, seed=SEED)
    )
    oa_sample = (
        pl.concat([linked, extra]).unique("id").sort(pl.col("id").cast(pl.Int64))
    )

    out = root / "sample_data"
    out.mkdir(exist_ok=True)
    # re-quote exactly as the source does so one config reads both
    of_sample.write_csv(
        out / "openflights_airports.dat",
        include_header=False,
        null_value=NULL,
        quote_style="non_numeric",
    )
    oa_sample.write_csv(out / "ourairports_airports.csv")
    (out / "SOURCES.json").write_text(
        json.dumps(
            {
                "sources": {
                    n: {
                        "url": u,
                        "sha256": hashlib.sha256(paths[n].read_bytes()).hexdigest(),
                    }
                    for n, u in SOURCES.items()
                },
                "rows": {"openflights": of.height, "ourairports": oa.height},
                "sample_rows": {
                    "openflights": of_sample.height,
                    "ourairports": oa_sample.height,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"full: {of.height:,} / {oa.height:,} rows; sample: {of_sample.height:,} / {oa_sample.height:,}"
    )


if __name__ == "__main__":
    main()
