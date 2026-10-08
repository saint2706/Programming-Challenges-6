"""Rebuild the vendored data files from their original public sources.

Run with:  uv run --group fetch python -m sales_heatmap.fetch_data

Everything under ``data/`` and ``vendor/`` is committed, so you only need this
to reproduce or refresh them.  Downloads are cached under ``.cache/`` and each
one's SHA-256 is recorded in ``data/SOURCES.json`` so a silent upstream change
is visible as a diff.

Sources (all U.S. federal public domain, except Leaflet which is BSD-2):

* 2017 Economic Census, Retail Trade (sector 44-45): sales by state/county.
* Census Population Estimates, vintage 2019: July-1-2017 county/state population.
* Census cartographic boundary files (GENZ2017, 1:20,000,000): state/county shapes.
* Leaflet 1.9.4: inlined into the report so it works offline.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

from sales_heatmap.paths import project_root

HERE = project_root()
CACHE = HERE / ".cache"
DATA = HERE / "data"
VENDOR = HERE / "vendor"

SOURCES = {
    "economic_census": "https://www2.census.gov/programs-surveys/economic-census/data/2017/sector44/EC1744BASIC.zip",
    "population": "https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/counties/totals/co-est2019-alldata.csv",
    "counties_shp": "https://www2.census.gov/geo/tiger/GENZ2017/shp/cb_2017_us_county_20m.zip",
    "states_shp": "https://www2.census.gov/geo/tiger/GENZ2017/shp/cb_2017_us_state_20m.zip",
    "leaflet_js": "https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js",
    "leaflet_css": "https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css",
}

# Census "legal/statistical area description" codes that occur in the county file.
LSAD = {
    "03": "city and borough",
    "04": "borough",
    "05": "census area",
    "06": "county",
    "12": "municipality",
    "13": "municipio",
    "15": "parish",
    "25": "city",
}
COORD_DECIMALS = 3  # ~110 m at the equator; well below the 1:20M generalisation error


def download(name: str) -> Path:
    CACHE.mkdir(exist_ok=True)
    target = CACHE / Path(SOURCES[name]).name
    if not target.exists():
        print(f"downloading {SOURCES[name]}")
        req = urllib.request.Request(
            SOURCES[name], headers={"User-Agent": "geo-heatmap-fetch/1.0"}
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            target.write_bytes(resp.read())
    return target


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _int_or_none(text: str, flag: str) -> int | None:
    """A withheld ("D") cell is stored by the Census as 0. It is *not* zero: it is unknown."""
    if flag.strip() or not text.strip():
        return None
    try:
        return int(text)
    except ValueError:
        return None


def build_sales() -> int:
    path = download("economic_census")
    rows: list[dict] = []
    titles: dict[str, str] = {}
    with zipfile.ZipFile(path) as zf, zf.open("EC1744BASIC.dat") as raw:
        text = io.TextIOWrapper(raw, encoding="latin-1", newline="")
        reader = csv.DictReader(text, delimiter="|")
        for r in reader:
            naics = r["NAICS2017"]
            if not (naics == "44-45" or (len(naics) == 3 and naics.isdigit())):
                continue
            titles[naics] = r["NAICS2017_TTL"]
            geo_id = r["GEO_ID"]
            if r["GEOTYPE"] == "02" and geo_id.startswith("0400000US"):
                level, code = "state", geo_id[-2:]
            elif r["GEOTYPE"] == "03" and geo_id.startswith("0500000US"):
                level, code = "county", geo_id[-5:]
            else:
                continue
            rows.append(
                {
                    "level": level,
                    "fips": code,
                    "naics": naics,
                    "sales_k": _int_or_none(r["RCPTOT"], r["RCPTOT_F"]),
                    "sales_flag": r["RCPTOT_F"].strip(),
                    "establishments": _int_or_none(r["ESTAB"], r["ESTAB_F"]),
                }
            )
    rows.sort(key=lambda x: (x["level"], x["naics"], x["fips"]))
    DATA.mkdir(exist_ok=True)
    with (DATA / "retail_sales_2017.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if v is None else v) for k, v in row.items()})
    with (DATA / "naics_titles.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["naics", "title"])
        w.writerows(sorted(titles.items()))
    return len(rows)


def build_population() -> int:
    path = download("population")
    out: list[tuple[str, str, str]] = []
    with path.open(encoding="latin-1", newline="") as fh:
        for r in csv.DictReader(fh):
            state, county = r["STATE"].zfill(2), r["COUNTY"].zfill(3)
            if county == "000":
                out.append(("state", state, r["POPESTIMATE2017"]))
            else:
                out.append(("county", state + county, r["POPESTIMATE2017"]))
    out.sort()
    with (DATA / "population_2017.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["level", "fips", "population"])
        w.writerows(out)
    return len(out)


def _ring(coords) -> list[list[float]]:
    return [[round(x, COORD_DECIMALS), round(y, COORD_DECIMALS)] for x, y in coords]


def _shape_to_geometry(shape) -> dict:
    """Convert a pyshp polygon to GeoJSON, sorting rings into outer rings + their holes."""
    parts = list(shape.parts) + [len(shape.points)]
    rings = [shape.points[parts[i] : parts[i + 1]] for i in range(len(parts) - 1)]

    def signed_area(ring) -> float:
        return (
            sum(
                x1 * y2 - x2 * y1
                for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1], strict=True)
            )
            / 2
        )

    polygons: list[list[list[list[float]]]] = []
    for ring in rings:
        # Shapefile convention: outer rings are clockwise (negative area), holes counter-clockwise.
        if signed_area(ring) < 0 or not polygons:
            polygons.append([_ring(ring)])
        else:
            polygons[-1].append(_ring(ring))
    return {"type": "MultiPolygon", "coordinates": polygons}


def build_geojson(source: str, out_name: str, level: str) -> int:
    import shapefile  # pyshp; only needed to (re)build the vendored GeoJSON

    path = download(source)
    with zipfile.ZipFile(path) as zf:
        stem = next(n[:-4] for n in zf.namelist() if n.endswith(".shp"))
        reader = shapefile.Reader(
            shp=io.BytesIO(zf.read(stem + ".shp")),
            shx=io.BytesIO(zf.read(stem + ".shx")),
            dbf=io.BytesIO(zf.read(stem + ".dbf")),
            encoding="utf-8",  # the Census .dbf files really are UTF-8 ("Doña Ana")
        )
    features = []
    for sr in reader.iterShapeRecords():
        rec = sr.record.as_dict()
        if level == "state":
            fips, name = rec["GEOID"], rec["NAME"]
            props = {"fips": fips, "name": name, "abbr": rec["STUSPS"]}
        else:
            fips, name = rec["GEOID"], rec["NAME"]
            props = {
                "fips": fips,
                "name": name,
                "state_fips": rec["STATEFP"],
                "lsad": LSAD.get(rec["LSAD"], rec["LSAD"]),
            }
        features.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": _shape_to_geometry(sr.shape),
            }
        )
    features.sort(key=lambda f: f["properties"]["fips"])
    (DATA / out_name).write_text(
        json.dumps(
            {"type": "FeatureCollection", "features": features}, separators=(",", ":")
        ),
        encoding="utf-8",
    )
    return len(features)


def build_vendor() -> None:
    VENDOR.mkdir(exist_ok=True)
    for key, name in (("leaflet_js", "leaflet.js"), ("leaflet_css", "leaflet.css")):
        (VENDOR / name).write_bytes(download(key).read_bytes())


def main() -> int:
    DATA.mkdir(exist_ok=True)
    print("sales rows     :", build_sales())
    print("population rows:", build_population())
    print(
        "state shapes   :",
        build_geojson("states_shp", "us_states_2017_20m.geojson", "state"),
    )
    print(
        "county shapes  :",
        build_geojson("counties_shp", "us_counties_2017_20m.geojson", "county"),
    )
    build_vendor()
    manifest = {
        name: {"url": url, "sha256": sha256(CACHE / Path(url).name)}
        for name, url in SOURCES.items()
    }
    (DATA / "SOURCES.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("wrote data/SOURCES.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
