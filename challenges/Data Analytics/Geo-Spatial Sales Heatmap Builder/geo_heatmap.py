"""Geo-Spatial Sales Heatmap Builder: regional aggregate data -> one self-contained choropleth HTML map.

Run with:
    uv run python geo_heatmap.py demo --level county -o retail.html
    uv run python geo_heatmap.py build my_sales.csv --region-col state --value-col revenue -o map.html
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
from classify import SCHEMES, assign, class_counts, compute_edges, gvf
from projection import project_collection
from regions import JoinReport, RegionIndex, resolve_keys
from spatial import (
    GlobalMoran,
    LocalMoran,
    Weights,
    build_weights,
    contiguity,
    local_moran,
    morans_i,
)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# Why a region has no colour on a given measure.  0 means it has a value.
OK, BLANK, NO_RECORD, NO_POPULATION = 0, 1, 2, 3
STATUS_LABELS = {
    BLANK: "Value blank or withheld in the data",
    NO_RECORD: "No row for this region in the data",
    NO_POPULATION: "No population figure (per-resident value undefined)",
}
AGGREGATIONS = ("sum", "mean", "median", "min", "max", "count")


# --------------------------------------------------------------------------- #
# Boundaries
# --------------------------------------------------------------------------- #


@dataclass
class Boundaries:
    level: str
    features: list[dict]  # lon/lat GeoJSON features, in a fixed order used everywhere
    index: RegionIndex
    projection: str = "albers-usa"  # or "plate-carree" for custom GeoJSON


def load_geojson(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_bundled_boundaries(level: str) -> Boundaries:
    states = load_geojson(DATA / "us_states_2017_20m.geojson")["features"]
    if level == "state":
        return Boundaries("state", states, RegionIndex.for_states(states))
    if level == "county":
        counties = load_geojson(DATA / "us_counties_2017_20m.geojson")["features"]
        return Boundaries(
            "county", counties, RegionIndex.for_counties(counties, states)
        )
    raise ValueError(f"level must be 'state' or 'county', got {level!r}")


def load_custom_boundaries(
    path: Path, key_prop: str, name_prop: str | None
) -> Boundaries:
    fc = load_geojson(path)
    feats = fc["features"]
    missing = [
        i for i, f in enumerate(feats) if key_prop not in f.get("properties", {})
    ]
    if missing:
        raise ValueError(
            f"{len(missing)} feature(s) lack the key property {key_prop!r} (first at index {missing[0]})"
        )
    return Boundaries(
        "custom",
        feats,
        RegionIndex.for_custom(feats, key_prop, name_prop),
        "plate-carree",
    )


def projected_collection(b: Boundaries) -> dict:
    fc = {"type": "FeatureCollection", "features": b.features}
    if b.projection == "albers-usa":
        return project_collection(fc, _state_fips_of(b.level))
    return fc  # plain longitude/latitude, drawn as x/y (not equal-area)


def _state_fips_of(level: str):
    if level == "state":
        return lambda f: f["properties"]["fips"]
    return lambda f: f["properties"]["state_fips"]


# --------------------------------------------------------------------------- #
# Joining data to regions
# --------------------------------------------------------------------------- #

_CLEAN_NUMBER = r"[,\s$€£%]"


def parse_numbers(col: pl.Series) -> tuple[pl.Series, int, int]:
    """Parse a column to floats; also return (blank cells, cells that were present but not numeric)."""
    text = col.cast(pl.Utf8).str.strip_chars()
    blank = int(text.is_null().sum() + (text == "").sum())
    cleaned = text.str.replace_all(_CLEAN_NUMBER, "")
    numeric = cleaned.cast(pl.Float64, strict=False)
    # "nan"/"inf" parse to non-finite floats; treat them as absent rather than let them poison sums
    numeric = (
        numeric.to_frame("value")
        .select(
            pl.when(pl.col("value").is_finite()).then(pl.col("value")).otherwise(None)
        )
        .to_series()
    )
    unparseable = int(numeric.is_null().sum()) - blank
    return numeric.alias("value"), blank, max(unparseable, 0)


def aggregate_to_regions(
    df: pl.DataFrame, region_col: str, value_col: str, agg: str, index: RegionIndex
) -> tuple[pl.DataFrame, JoinReport]:
    """One row per region that appeared in the data: ``pos`` (feature position), ``value`` (null if all blank).

    A region whose rows are all blank aggregates to *null*, never 0: ``sum`` of nothing is
    "unknown", not "zero sales".
    """
    if agg not in AGGREGATIONS:
        raise ValueError(f"agg must be one of {AGGREGATIONS}")
    report = JoinReport(level=index.level, rows=df.height, total_regions=len(index))
    values, report.blank_values, report.unparseable_values = parse_numbers(
        df[value_col]
    )
    positions, report.unmatched = resolve_keys(df[region_col].to_list(), index)
    frame = pl.DataFrame(
        {"pos": positions, "value": values},
        schema={"pos": pl.Int64, "value": pl.Float64},
    )
    matched = frame.filter(pl.col("pos").is_not_null())
    report.matched_rows = matched.height

    non_null = pl.col("value").drop_nulls()
    exprs = {
        "sum": pl.when(non_null.len() == 0).then(None).otherwise(pl.col("value").sum()),
        "mean": pl.col("value").mean(),
        "median": pl.col("value").median(),
        "min": pl.col("value").min(),
        "max": pl.col("value").max(),
        "count": non_null.len().cast(pl.Float64),
    }
    out = (
        matched.group_by("pos")
        .agg(exprs[agg].alias("value"), pl.len().alias("rows"))
        .sort("pos")
    )
    report.matched_regions = out.height
    report.duplicate_rows_merged = matched.height - out.height
    present = set(out["pos"].to_list())
    report.regions_without_record = [
        index.names[i] for i in range(len(index)) if i not in present
    ]
    return out, report


def _regions_array(out: pl.DataFrame, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(values with NaN for missing, boolean 'has a row') aligned to feature order."""
    vals = np.full(n, np.nan)
    has_row = np.zeros(n, dtype=bool)
    pos = out["pos"].to_numpy()
    has_row[pos] = True
    v = out["value"].to_numpy(allow_copy=True).astype(float)  # nulls become NaN
    vals[pos] = v
    return vals, has_row


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


@dataclass
class Measure:
    key: str  # "raw" or "percap"
    label: str
    values: np.ndarray  # NaN where the region has no value on this measure
    status: np.ndarray  # OK / BLANK / NO_RECORD / NO_POPULATION per region


@dataclass
class View:
    key: str
    measure: str
    scheme: str
    edges: np.ndarray
    classes: (
        np.ndarray
    )  # class index per region, or -status for regions without a value
    gvf: float
    counts: list[int]


@dataclass
class Lisa:
    measure: str
    cluster: (
        np.ndarray
    )  # per region: 0 n.s., 1 HH, 2 LL, 3 HL, 4 LH, or -1 not analysed
    result: LocalMoran
    weights: Weights


@dataclass
class MapModel:
    title: str
    boundaries: Boundaries
    collection: dict  # projected features for drawing
    measures: dict[str, Measure]
    views: list[View]
    n_classes: int
    palette: str
    prefix: str
    join: JoinReport
    population_join: JoinReport | None = None
    population: np.ndarray | None = None
    aggregation: str = "sum"
    moran: dict[str, GlobalMoran | None] = field(default_factory=dict)
    lisa: Lisa | None = None
    stats: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    source_note: str = ""
    permutations: int = 999


def spearman(a, b) -> float:
    """Spearman rank correlation with average ranks for ties, over the pairs where both are finite."""
    x, y = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    ranks = pl.DataFrame({"x": x[ok], "y": y[ok]}).select(
        pl.col("x").rank("average").alias("x"), pl.col("y").rank("average").alias("y")
    )
    rx, ry = ranks["x"].to_numpy(), ranks["y"].to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def top_overlap(a, b, fraction: float = 0.1) -> float:
    """Share of the top ``fraction`` of regions by ``a`` that are also in the top ``fraction`` by ``b``."""
    x, y = np.asarray(a, float), np.asarray(b, float)
    ok = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
    k = max(round(len(ok) * fraction), 1)
    top_x = set(ok[np.argsort(-x[ok], kind="stable")[:k]].tolist())
    top_y = set(ok[np.argsort(-y[ok], kind="stable")[:k]].tolist())
    return len(top_x & top_y) / k


def _status_array(has_row: np.ndarray, values: np.ndarray) -> np.ndarray:
    status = np.full(len(values), NO_RECORD, dtype=int)
    status[has_row] = BLANK
    status[np.isfinite(values)] = OK
    return status


def make_view(measure: Measure, scheme: str, k: int) -> View | None:
    ok = measure.status == OK
    if not ok.any():
        return None
    edges = compute_edges(measure.values[ok], scheme, k)
    classes = np.where(ok, assign(measure.values, edges), -measure.status)
    return View(
        key=f"{measure.key}:{scheme}",
        measure=measure.key,
        scheme=scheme,
        edges=edges,
        classes=classes,
        gvf=gvf(measure.values[ok], classes[ok]),
        counts=class_counts(classes, len(edges) - 1),
    )


def build_model(
    boundaries: Boundaries,
    values: pl.DataFrame,
    join: JoinReport,
    *,
    title: str,
    value_label: str,
    prefix: str = "",
    population: tuple[pl.DataFrame, JoinReport] | None = None,
    aggregation: str = "sum",
    n_classes: int = 5,
    palette: str = "viridis",
    permutations: int = 999,
    lisa_permutations: int = 9999,
    seed: int = 0,
    source_note: str = "",
) -> MapModel:
    n = len(boundaries.features)
    raw_values, has_row = _regions_array(values, n)
    raw = Measure("raw", value_label, raw_values, _status_array(has_row, raw_values))
    measures = {"raw": raw}
    notes: list[str] = []
    pop_join = None
    pop_arr = None

    if population is not None and aggregation == "sum":
        pop_df, pop_join = population
        pop_arr, _ = _regions_array(pop_df, n)
        usable = np.isfinite(pop_arr) & (pop_arr > 0)
        per_cap = np.where(
            usable & np.isfinite(raw_values),
            raw_values / np.where(usable, pop_arr, 1.0),
            np.nan,
        )
        status = raw.status.copy()
        status[(raw.status == OK) & ~usable] = NO_POPULATION
        measures["percap"] = Measure(
            "percap", f"{value_label} per resident", per_cap, status
        )
    elif population is not None:
        notes.append(
            f"Per-resident view skipped: it is only meaningful for a sum, but --agg is {aggregation!r}."
        )

    views = [
        v
        for m in measures.values()
        for s in SCHEMES
        if (v := make_view(m, s, n_classes)) is not None
    ]

    stats: dict = {}
    if "percap" in measures:
        both = np.isfinite(raw.values) & np.isfinite(measures["percap"].values)
        stats["spearman_raw_population"] = spearman(
            np.where(both, raw.values, np.nan), pop_arr
        )
        stats["top_decile_overlap"] = top_overlap(
            raw.values, measures["percap"].values, 0.1
        )

    weights_source = contiguity(boundaries.features, "queen")
    moran: dict[str, GlobalMoran | None] = {}
    lisa = None
    for key, m in measures.items():
        valid = m.status == OK
        try:
            w = build_weights(weights_source, valid)
            moran[key] = morans_i(m.values, w, permutations, seed)
            if key == ("percap" if "percap" in measures else "raw"):
                res = local_moran(m.values, w, lisa_permutations, seed=seed + 1)
                cluster = np.full(n, -1, dtype=int)
                cluster[w.index] = res.cluster
                lisa = Lisa(key, cluster, res, w)
        except ValueError as exc:  # too few connected regions / constant data
            moran[key] = None
            notes.append(f"Spatial statistics for {m.label!r} skipped: {exc}.")

    return MapModel(
        title=title,
        boundaries=boundaries,
        collection=projected_collection(boundaries),
        measures=measures,
        views=views,
        n_classes=n_classes,
        palette=palette,
        prefix=prefix,
        join=join,
        population_join=pop_join,
        population=pop_arr,
        aggregation=aggregation,
        moran=moran,
        lisa=lisa,
        stats=stats,
        notes=notes,
        source_note=source_note,
        permutations=permutations,
    )


# --------------------------------------------------------------------------- #
# Bundled demo data + CLI
# --------------------------------------------------------------------------- #


def bundled_naics_titles() -> dict[str, str]:
    df = pl.read_csv(DATA / "naics_titles.csv", schema_overrides={"naics": pl.Utf8})
    return dict(zip(df["naics"].to_list(), df["title"].to_list(), strict=True))


def load_bundled_tables(level: str, naics: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Sales (in dollars) and population for one level and NAICS code, keyed by FIPS."""
    schema = {"fips": pl.Utf8, "naics": pl.Utf8, "level": pl.Utf8}
    sales = pl.read_csv(DATA / "retail_sales_2017.csv", schema_overrides=schema)
    titles = bundled_naics_titles()
    if naics not in titles:
        raise ValueError(
            f"unknown NAICS code {naics!r}; choose from {', '.join(titles)}"
        )
    sales = sales.filter((pl.col("level") == level) & (pl.col("naics") == naics))
    sales = sales.select("fips", (pl.col("sales_k") * 1000).alias("sales"))
    pop = pl.read_csv(DATA / "population_2017.csv", schema_overrides=schema).filter(
        pl.col("level") == level
    )
    return sales, pop.select("fips", "population")


def detect_level(keys: list[object]) -> str:
    """Pick 'state' or 'county' by which index resolves more of the keys (ties favour state)."""
    sample = keys[:2000]
    states = load_bundled_boundaries("state")
    counties = load_bundled_boundaries("county")
    hit_state = sum(states.index.resolve(k).index is not None for k in sample)
    hit_county = sum(counties.index.resolve(k).index is not None for k in sample)
    if hit_state == 0 and hit_county == 0:
        raise ValueError(
            "no key matched any U.S. state or county; check --region-col or use --geojson"
        )
    return "county" if hit_county > hit_state else "state"


def _read_csv(path: Path) -> pl.DataFrame:
    # Everything as text first: a numeric read would turn FIPS "01001" into 1001 before we can repair it.
    return pl.read_csv(path, infer_schema_length=0, encoding="utf8-lossy")


def _need_columns(df: pl.DataFrame, *cols: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"column(s) not found: {', '.join(missing)}. Available: {', '.join(df.columns)}"
        )


def cmd_demo(args: argparse.Namespace) -> MapModel:
    titles = bundled_naics_titles()
    boundaries = load_bundled_boundaries(args.level)
    sales, pop = load_bundled_tables(args.level, args.naics)
    values, join = aggregate_to_regions(sales, "fips", "sales", "sum", boundaries.index)
    pop_values, pop_join = aggregate_to_regions(
        pop, "fips", "population", "sum", boundaries.index
    )
    what = titles[args.naics]
    return build_model(
        boundaries,
        values,
        join,
        title=f"{what}: retail sales by {args.level}, 2017",
        value_label="Sales",
        prefix="$",
        population=(pop_values, pop_join),
        n_classes=args.classes,
        palette=args.palette,
        permutations=args.permutations,
        lisa_permutations=args.lisa_permutations,
        source_note=(
            "Sales: 2017 Economic Census, Retail Trade (NAICS "
            f"{args.naics}), U.S. Census Bureau. Population: Census Population Estimates, July 1 2017. "
            "Boundaries: Census cartographic boundary files, 1:20,000,000, 2017."
        ),
    )


def cmd_build(args: argparse.Namespace) -> MapModel:
    df = _read_csv(Path(args.data))
    _need_columns(df, args.region_col, args.value_col)
    if args.geojson:
        boundaries = load_custom_boundaries(
            Path(args.geojson), args.geo_key, args.geo_name
        )
    else:
        level = (
            args.level
            if args.level != "auto"
            else detect_level(df[args.region_col].to_list())
        )
        boundaries = load_bundled_boundaries(level)
    values, join = aggregate_to_regions(
        df, args.region_col, args.value_col, args.agg, boundaries.index
    )
    if args.multiplier != 1.0:
        values = values.with_columns(pl.col("value") * args.multiplier)

    population = None
    if args.population_file or args.population_col:
        if args.population_col and not args.population_file:
            _need_columns(df, args.population_col)
            pop_df, pop_region = df, args.region_col
        else:
            pop_df = _read_csv(Path(args.population_file))
            pop_region = args.population_region_col or args.region_col
            _need_columns(pop_df, pop_region, args.population_col or "population")
        pop_col = args.population_col or "population"
        # a population repeated on every row of a region must not be summed: take the first
        pop_first = pop_df.group_by(pop_region, maintain_order=True).first()
        population = aggregate_to_regions(
            pop_first, pop_region, pop_col, "max", boundaries.index
        )
    return build_model(
        boundaries,
        values,
        join,
        title=args.title or f"{args.value_col} by region",
        value_label=args.label or args.value_col,
        prefix=args.prefix,
        population=population,
        aggregation=args.agg,
        n_classes=args.classes,
        palette=args.palette,
        permutations=args.permutations,
        lisa_permutations=args.lisa_permutations,
        source_note=f"Data: {Path(args.data).name}.",
    )


def build_parser() -> argparse.ArgumentParser:
    from report import PALETTES

    parser = argparse.ArgumentParser(
        description="Build a self-contained choropleth map from regional data."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "-o", "--output", default="heatmap.html", help="output HTML file"
        )
        p.add_argument(
            "--classes",
            type=int,
            default=5,
            help="requested number of classes (2-9, default 5)",
        )
        p.add_argument(
            "--palette",
            choices=sorted(PALETTES),
            default="viridis",
            help="sequential palette",
        )
        p.add_argument(
            "--permutations",
            type=int,
            default=999,
            help="permutations for global Moran's I",
        )
        p.add_argument(
            "--lisa-permutations",
            type=int,
            default=9999,
            help="permutations for local Moran's I (a p-value can't go below 1/(N+1), which caps what FDR control can flag)",
        )

    demo = sub.add_parser(
        "demo", help="map the bundled 2017 Economic Census retail sales"
    )
    demo.add_argument("--level", choices=("state", "county"), default="county")
    demo.add_argument(
        "--naics",
        default="44-45",
        help="NAICS code: 44-45 (all retail) or a 3-digit subsector",
    )
    common(demo)

    build = sub.add_parser("build", help="map your own CSV")
    build.add_argument("data", help="CSV with one row per record")
    build.add_argument(
        "--region-col",
        required=True,
        help="column holding FIPS codes, names or abbreviations",
    )
    build.add_argument("--value-col", required=True, help="numeric column to map")
    build.add_argument("--level", choices=("auto", "state", "county"), default="auto")
    build.add_argument(
        "--agg",
        choices=AGGREGATIONS,
        default="sum",
        help="how to combine rows of one region",
    )
    build.add_argument(
        "--multiplier",
        type=float,
        default=1.0,
        help="scale the values (e.g. 1000 if in $ thousands)",
    )
    build.add_argument(
        "--population-file",
        help="CSV of region + population to enable the per-resident view",
    )
    build.add_argument(
        "--population-region-col",
        help="region column in the population file (default: --region-col)",
    )
    build.add_argument(
        "--population-col",
        help="population column (in the data, or in --population-file)",
    )
    build.add_argument(
        "--geojson",
        help="your own boundaries (lon/lat) instead of U.S. states/counties",
    )
    build.add_argument(
        "--geo-key", default="id", help="GeoJSON property holding the region key"
    )
    build.add_argument(
        "--geo-name", default="name", help="GeoJSON property holding the display name"
    )
    build.add_argument("--title")
    build.add_argument("--label", help="name of the measure, e.g. 'Revenue'")
    build.add_argument("--prefix", default="", help="unit prefix for numbers, e.g. '$'")
    common(build)
    return parser


def main(argv: list[str] | None = None) -> int:
    from report import PALETTES, render_html

    args = build_parser().parse_args(argv)
    if not 2 <= args.classes <= 9:
        print("error: --classes must be between 2 and 9", file=sys.stderr)
        return 2
    if args.palette not in PALETTES:
        print(f"error: unknown palette {args.palette!r}", file=sys.stderr)
        return 2
    try:
        model = cmd_demo(args) if args.command == "demo" else cmd_build(args)
    except (ValueError, OSError, pl.exceptions.PolarsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    html_text = render_html(model)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text, encoding="utf-8")
    j = model.join
    print(
        f"wrote {args.output} ({len(html_text) / 1e6:.1f} MB): {j.matched_regions}/{j.total_regions} regions have data; "
        f"{len(j.unmatched)} unmatched key(s) covering {j.unmatched_rows} row(s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
