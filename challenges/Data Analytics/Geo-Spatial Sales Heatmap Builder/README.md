# Geo-Spatial Sales Heatmap Builder

**Category:** Data Analytics
**Difficulty:** I

**Status:** Implemented (Python)

Source modules live in `src/sales_heatmap/`; the tests are in `tests/`.

Choropleth maps from regional aggregate data. Point it at a CSV with one row per
record (a state or county in some spelling, plus a number) and it writes **one
self-contained HTML map**: no server, no CDN, no map tiles. The bundled demo maps
real 2017 Economic Census retail sales for all 3,220 U.S. counties.

## What it does

- **A real map stack.** [Folium](https://python-visualization.github.io/folium/) drives
  Leaflet; Leaflet's JS/CSS are inlined from `vendor/`, the boundaries are embedded, and
  there is no tile layer, so the file works offline. The page has no external resource
  loads (a test asserts this).
- **Equal-area projection.** A choropleth encodes value as colour *over area*, so area
  must be honest. Web Mercator (Leaflet's default) inflates Alaska roughly 3x against the lower
  48; this tool projects to Albers equal-area conic itself (`projection.py`) and renders in
  Leaflet's plain `CRS.Simple`. Alaska (35% scale, as in d3's AlbersUSA), Hawaii and Puerto
  Rico are inset. Tests check the projection against spherical area, Texas/Montana against
  their real areas, and that no inset overlaps the lower 48.
- **A join that never fails silently.** Keys go through one normaliser: FIPS with Excel's
  dropped leading zero (`1001` -> `01001`), `1.0`, `'01001`; state names, USPS codes,
  `D.C.`; county `Name, State` in many spellings (`St.`/`Saint`, `Doña`/`Dona`,
  `Parish`/`Borough`). Anything that does not match is listed with *why* in the report
  (no such key, unrecognised state, county name without a state, **ambiguous** - e.g.
  `Richmond, Virginia` is both a county and an independent city, so it is refused rather
  than guessed; `Richmond city, Virginia` and `Richmond County, Virginia` resolve fine).
- **Missing is not zero.** Four states per region and measure: has a value / value blank or
  withheld / no row in the data / no population. Missing regions are grey with their own
  legend entry, never coloured as the lowest class. An all-blank region aggregates to *null*
  (a sum of nothing is unknown), and a real `0` stays in the lowest class.
- **Four classification schemes, compared.** Quantile, equal interval, standard deviation
  and exact Fisher-Jenks natural breaks (`classify.py`), switchable in the page, with a
  goodness-of-variance-fit table for each.
- **Raw totals vs per-resident.** A second measure divides by population; the report shows
  how strongly raw totals track population and how few regions survive the switch.
- **Spatial statistics.** Global Moran's *I* (queen contiguity from the polygons' shared
  vertices, permutation p-value) and local Moran's *I* (LISA) hot/cold-spot map with
  Benjamini-Hochberg false-discovery-rate control (`spatial.py`).
- **Colour-vision-safe palettes.** `viridis` (default), `cividis`, `ylorrd`; tests assert
  lightness decreases monotonically with class, so magnitude survives any hue deficiency.
  Hot/cold spots use an Okabe-Ito orange/blue pair that also differs in lightness.
- **Escaped output.** Every data-derived string goes through `html.escape` (page, tooltips)
  or `textContent` (controls); embedded JSON escapes `<`, `>`, `&`. Tests inject
  `</script><img src=x onerror=...>` through region names, keys, and labels.

## Run it

```bash
cd "challenges/Data Analytics/Geo-Spatial Sales Heatmap Builder"

# the bundled 2017 retail-sales data (county: ~2 MB file, ~15 s incl. 9,999-permutation LISA)
uv run sales-heatmap demo --level county -o retail_county.html
uv run sales-heatmap demo --level state --naics 445 -o groceries_state.html

# your own CSV; auto-detects state vs county from the keys
uv run sales-heatmap build sample_data/messy_orders.csv \
  --region-col state --value-col order_value --population-col customer_pop \
  --prefix '$' --label "Order value" -o messy.html

uv run pytest -q # 253 tests
```

`demo` options: `--level {state,county}`, `--naics` (`44-45` all retail, or a 3-digit
subsector: 441 motor vehicle, 445 food & beverage, 452 general merchandise, 454 nonstore, ...),
`--classes 2-9`, `--palette`. `build` adds `--region-col`, `--value-col`,
`--agg {sum,mean,median,min,max,count}`, `--multiplier` (e.g. 1000 if values are in $
thousands), `--population-file` / `--population-region-col` / `--population-col`,
`--geojson PATH --geo-key PROP --geo-name PROP` (your own lon/lat boundaries; drawn
plate carrée, which is *not* equal-area), `--title`, `--label`, `--prefix`. Global Moran's
permutations: `--permutations` (999); local: `--lisa-permutations` (9,999). Exit code `2`
on a bad file, missing column, or keys that match nothing. Generated `*.html` in this
folder is git-ignored.

`sample_data/messy_orders.csv` is a **hand-made** illustration of join repair, not real
data: mixed state spellings, a FIPS code, a typo (`Californa`), a territory, `$1,200.50`,
`n/a`, a blank key, and a genuine `0` next to a blank. The report shows exactly which rows
were merged, repaired, and dropped, and why.

## Real data, and what it taught

**Sales:** 2017 Economic Census, Retail Trade (sector 44-45), `EC1744BASIC` - actual sales
by state and county, in $ thousands. **Population:** Census Population Estimates
(vintage 2019), July 1 2017 - matched to the census year. **Boundaries:** Census
cartographic boundary files, 2017, 1:20,000,000. All U.S. federal public domain.
`fetch_data.py` rebuilds every vendored file from source and records SHA-256s in
`data/SOURCES.json`.

1. **The Census stores a withheld cell as `0`** with a `D` flag. Read naively, 16 counties
   (61,000+ withheld cells across all NAICS rows) become *real zero-sales* counties, the
   worst possible lie on a sales map. The loader turns `D` into null; these counties show
   as "withheld", separate from the 100 counties with no row at all (Puerto Rico's 78
   municipios among them).
2. **Raw totals are a population map.** Spearman correlation of county retail sales with
   population is **0.96** (0.99 for states), and only **25%** of the top-decile counties by
   total are also top-decile per resident. Los Angeles County ($150B), King ($89.6B),
   Maricopa, Harris and Cook lead the raw ranking; per resident it is Fairfax city, VA
   ($73.5K), Cheyenne County, NE, Iowa County, WI. At state level California leads raw
   totals; North Dakota, New Hampshire and Washington lead per resident. Retail sales are
   booked where the store is, so per-resident values in small or store-heavy counties can
   exceed anything residents could spend.
3. **Raw and per-resident have different spatial structure.** County Moran's *I* on raw
   sales is **0.32** (big counties sit next to big counties, the population map again) but
   only **0.06** per resident (z = 5.7, still significant with 3,097 counties, but weak).
4. **LISA needs permutations to have power at this scale.** At 5% uncorrected, 436 of 3,097
   counties test as clusters (14%, nearly 3x the 5% expected by chance); after
   Benjamini-Hochberg FDR control **13** remain (2 high-high, 5 low-low - five
   counties in west-central Georgia - plus 6 spatial outliers). With 999 permutations the
   smallest possible p-value, 0.001, cannot clear the FDR threshold and the count is **0**,
   so local tests default to 9,999.
5. **Schemes disagree, and GVF says how much.** On county totals (5 classes), quantile
   explains 22% of variance, equal interval 69%, Jenks 93%; equal interval puts 3,086 of
   3,104 counties in one class and standard deviation collapses to 3 classes, both correctly
   reporting a distribution too skewed for them. Per resident the four are closer
   (0.75 / 0.69 / 0.89 / 0.82). Quantile has the worst fit but the most legible map, which
   is why it is the page's default view.
6. **State-level per-resident sales *are* spatially clustered** (*I* = 0.25, p = 0.01) while
   raw state totals are not (*I* = -0.01): a population effect and an economic one,
   respectively (sales-tax-free New Hampshire and Delaware are both among the six highest per resident,
   a border-shopping effect that raw totals hide).

## Design notes

**Projection.** Albers is computed from Snyder's spherical formulas with per-territory
parameters (`ALASKA`, `HAWAII`, `PUERTO_RICO`), longitude wrapped at the antimeridian
(the Aleutians are stored as +172 degrees). Insets are laid out from the geometry actually
present, so a dataset without Alaska gets no Alaska inset. Leaflet clamps `minZoom` to 0,
but a CRS.Simple map measured in kilometres needs negative zoom levels to fit, so the
page sets `setMinZoom(-6)` itself before `fitBounds`.

**Fisher-Jenks** is the exact DP over the sorted *distinct* values, weighted by
multiplicity (ties are never split), on mean-shifted prefix sums (retail sales in billions
would make a naive sum-of-squares lose precision). Tests compare to a brute-force search
over every partition of small inputs, and to `jenkspy`'s C implementation (identical
breaks on continuous data; never worse when values tie). Above 4,000 distinct values it
runs on an evenly spaced sample, as mapclassify's `FisherJenksSampled` does.

**Contiguity from geometry.** Neighbours are found by hashing exact shared vertices (queen)
or shared segments (rook). That is only valid when neighbours carry identical coordinates
along the border, which the Census files do; the tests pin known adjacency (Missouri and
Tennessee have 8 neighbours, Maine 1, Four Corners is queen but not rook) and that the
county map averages ~6 neighbours, as a planar map must. Regions with data but no neighbour
that also has data (islands, or neighbours all missing) are dropped from the statistics and
counted in the report.

**Moran's I was cross-checked against `esda`/`libpysal`** during development (not a test
dependency): the global *I* agrees to 15 digits; local *I* is identical up to the factor
n/(n-1), because Anselin's original scales by `sum(z^2)/n` and `esda` by `/(n-1)`; the
permutation p-values correlate at 0.99 and flag 429 (ours) vs 435 (`esda`) regions at 5%, which is
permutation noise.
The tests instead verify against dense-matrix formulas (`sum(local I) == n * global I`,
checkerboard *I* = -1 exactly, planted hot/cold spots recovered).

**Local significance** uses conditional permutation (hold the region fixed, redraw *other*
regions' values into its neighbour slots, sampled without replacement by rejection), then
BH-FDR over all regions. Uncorrected counts are still reported.

**Display names.** Counties carry their designation ("Fairfax County" vs "Fairfax city",
"Orleans Parish"), otherwise six Maryland/Missouri/Virginia county-and-city pairs would repeat a name.

## Limitations

- Per-resident sales for small counties are noisy; no empirical-Bayes smoothing is applied.
- The 1:20,000,000 boundaries are generalised: fine for a national map, not for a city.
- `--geojson` maps are drawn plate carrée (fine near the equator, distorted at high
  latitude) and have no insets.
- Local Moran's *I* on a heavily skewed variable is dominated by a few extreme values; a
  log transform is worth trying on raw sales.
- The county map is a 2 MB canvas: hovering is smooth, but very old machines may lag while
  switching views.

## Files

| File               | Purpose                                                       |
| ------------------ | ------------------------------------------------------------- |
| `geo_heatmap.py`   | CLI, joining, aggregation, per-capita, model building         |
| `regions.py`       | key normalisation, alias index, `JoinReport`                  |
| `classify.py`      | quantile / equal interval / std-dev / exact Fisher-Jenks, GVF |
| `spatial.py`       | contiguity, global + local Moran's *I*, Benjamini-Hochberg    |
| `projection.py`    | Albers equal-area conic and inset layout                      |
| `report.py`        | Folium/Leaflet assembly, palettes, controls, escaped report   |
| `fetch_data.py`    | rebuilds `data/` and `vendor/` from the public sources        |
| `data/`, `vendor/` | committed Census data (~2.5 MB) and Leaflet 1.9.4 (BSD-2)     |
| `test_*.py`        | 253 tests, all offline (`uv run pytest -q`)                   |

Refresh the data with `uv run --group fetch python -m sales_heatmap.fetch_data` (needs network and `pyshp`).
