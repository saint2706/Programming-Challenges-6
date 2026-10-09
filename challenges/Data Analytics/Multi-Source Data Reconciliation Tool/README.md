# Multi-Source Data Reconciliation Tool

**Category:** Data Analytics
**Difficulty:** I

**Status:** Implemented (Python)

Source modules live in `src/data_reconciler/`; the tests are in `tests/`.

Diff two datasets on a key, and say not just *that* rows differ but *how*: which rows pair up, which
have no partner and why, and for every field of every pair whether the difference is cosmetic,
within tolerance, a near-miss, or a real disagreement. A job is one TOML file (two sources, the keys
to match on, the fields to compare), so it is not tied to any one dataset. The bundled job reconciles
two real airport files: **OpenFlights** (7,698 airports) against **OurAirports** (86,226 records).

```bash
uv run reconcile run                       # summary in the terminal + out/report.html and CSVs
uv run reconcile show LHR                  # one key: how its pair compares, or why it has none
uv run reconcile run --config my_job.toml  # any other job
uv run python -m data_reconciler.fetch_data  # download the full files into data/ (about 13 MB)
uv run pytest -q
```

The committed `sample_data/` (500 OpenFlights airports plus the OurAirports rows that carry their
codes, and 150 OurAirports airports that have no OpenFlights partner) makes it all work out of the
box. The numbers below are from the full files. A run takes about 0.3 seconds.

## How a job works

1. **Load** each source as text. Nothing is typed until a comparison needs it, so `"05"` vs `5` and a
   missing-value marker such as `\N` are decisions the report shows, not things a CSV reader hides.
   Optional `derive` columns and a `scope` filter (Polars SQL expressions) shape each side.
2. **Match** by a cascade of keys. Each stage looks only at rows the earlier ones left unresolved and
   pairs a key that appears **exactly once on each side**. A key on several rows is ambiguous, and
   guessing would silently pair the wrong rows, so every row involved is reported as `duplicate_key`.
   Keys are trimmed and upper-cased first. Every row ends in exactly one status: `matched`, `only_left`,
   `only_right`, `duplicate_key` or `unkeyed` (no key value at all).
3. **Compare** every field of every pair with the field's comparator, giving one category each:

   | Category                         | Meaning                                                                                 |
   | -------------------------------- | --------------------------------------------------------------------------------------- |
   | `match`                          | identical as written                                                                    |
   | `format_only`                    | different as written, same after normalizing (case, spaces, accents, punctuation)       |
   | `within_tolerance`               | numbers or coordinates that differ by no more than the configured tolerance             |
   | `near_match`                     | text that is clearly the same thing ("Rabah Bitat" inside "Annaba Rabah Bitat Airport") |
   | `mismatch`                       | a real disagreement                                                                     |
   | `left_missing` / `right_missing` | only the other source has a value                                                       |
   | `invalid`                        | a value that cannot be read as the type compared (text in a number, latitude 91)        |
   | `unverifiable`                   | crosswalk only: too few pairs with this value to say what it should map to              |
   | `both_missing`                   | neither has a value: nothing to reconcile                                               |

   Comparators: `text`, `code`, `number` (absolute and relative tolerance), `geo` (haversine
   distance on a latitude/longitude pair), and `crosswalk`: for fields written differently in the two
   sources (`United Kingdom` vs `GB`), the mapping is **learned from the pairs themselves** by majority,
   and pairs that disagree with it are the mismatches. No hand-written lookup table.
4. **Judge each pair.** `identical` (nothing differs), `changed`, or `different_entity`: when all the
   configured *identity* fields disagree, the key was probably reused for a different thing and the
   pair should not be counted as an update.
5. **Report**: `pairs.csv` (one row per pair), `differences.csv` (one row per differing field, with
   the values, category and a metric: km apart, absolute difference, or text similarity),
   `unmatched_left.csv` / `unmatched_right.csv`, `summary.json`, and one self-contained `report.html`.

## What the data says

OpenFlights lists `OurAirports` as the source of every one of its 7,698 rows. So this is not two
independent opinions but **a years-old copy against its live upstream**, and the differences are what
has changed since. Only OurAirports rows with an IATA or ICAO code are in scope (11,764 of 86,226),
because that is all OpenFlights tracks.

**Where every row went.** 6,597 pairs: 5,898 matched on IATA and 699 more on ICAO, for rows
OpenFlights has no IATA code for. OpenFlights has 1,100 rows with no partner and 1 with no code at
all. OurAirports has 5,163 with no partner (4,430 of them small airports; 3,034 carry an IATA code, so
those are airports OpenFlights does not have) and 4 duplicates.

**"Missing" is often "present but outside the scope".** Of the 1,100 unmatched OpenFlights rows, 880
*do* exist in OurAirports, just outside the scope filter: 637 small airports, 183 medium airports, 33
heliports, 19 marked `closed`, 8 seaplane bases. Only 220 have no trace under any key. The tool says
so per row (`found_out_of_scope`, with the other side's `type`), because "missing" and "reclassified"
call for different action.

**Half the pairs agree completely; the rest differ in few fields.** 3,230 pairs (49%) are identical
once tolerances apply; 3,359 changed; 8 are a different airport. Of the changed ones, 2,284 differ in
exactly one field, 852 in two, 155 in three, 50 in four and 18 in five.

| Field (6,589 same-airport pairs) | Exact | Cosmetic or tolerated            | Real difference                                                                |
| -------------------------------- | ----- | -------------------------------- | ------------------------------------------------------------------------------ |
| location                         | 2,175 | 3,939 within 0.5 km              | 475 mismatches: median 0.93 km, 65 over 10 km, 10 over 100 km, 4 over 1,000 km |
| name                             | 4,956 | 140 format only                  | 829 near-matches, 664 mismatches                                               |
| city                             | 4,148 | 352 format only                  | 771 near-matches, 1,100 mismatches, 15 + 179 missing on one side               |
| elevation (ft)                   | 6,164 | 86 within 5 ft                   | 247 mismatches (median 37 ft, 19 over 1,000 ft), 92 missing on OurAirports     |
| ICAO code                        | 6,405 |                                  | 184 mismatches                                                                 |
| IATA code                        | 5,890 | 533 neither side has one         | 6 mismatches, 111 only OurAirports has one, 49 only OpenFlights                |
| country (learned crosswalk)      | 6,378 | 190 unverifiable (under 5 pairs) | 21 mismatches                                                                  |

**Exact comparison would have buried the signal.** Only a third of the pairs (2,175 of 6,597) have
byte-identical coordinates. The other 3,939 differ by under half a kilometre, mostly in how many
decimals were kept (`51.4706` vs `51.470748` for Heathrow, 140 m apart). Comparing as written would
call two thirds of the file a location mismatch; with a 0.5 km tolerance it is 475 pairs, and the
tail is the interesting part.

**Keys get reused, and a key match is not an identity match.** Eight pairs agree on the code and on
nothing else: `AAP` is Andrau Airpark in Houston on one side and Aji Pangeran Tumenggung Pranoto
International in Indonesia on the other (15,248 km apart); `ACF` Brisbane Archerfield vs a Chinese
airport; `CGX` Chicago Meigs (closed in 2003) vs a Zambian airstrip. Counted as updates they would
add name, city, country, location and ICAO mismatches and thousands of kilometres to the "what
changed" statistics. Flagged as `different_entity`, they are listed on their own.

**Real changes show up as changes.** `EDDB` is Berlin-Schönefeld on one side and Berlin Brandenburg
on the other (2.4 km apart, and its IATA code moved from `SXF` to `BER`); `OSL` is "Oslo Lufthavn" vs
"Oslo-Gardermoen International"; `CUR` maps "Netherlands Antilles" (dissolved in 2010) to `CW`
(Curaçao), which the learned crosswalk flags because only 3 of its 5 pairs say `BQ`.

**The disagreement is not evenly spread.** Egypt (19 of 20 pairs differ), Taiwan (20 of 22) and
China (204 of 238) lead the per-country table (countries with 20 or more pairs); the top 15 all have
a difference in at least three quarters of their pairs.

## Design notes

**Ambiguity is reported, not resolved.** A duplicate key is never "fixed" by picking the first row.
The 4 OurAirports duplicates are real: the derived ICAO key `coalesce(icao_code, ident)` finds `WABK`
on both *Biri Airstrip* (as its ICAO code) and *Karubaga Airport* (as its ident). They are reported
with the shared key so a person can decide.

**A stricter identity rule, chosen by looking.** The first version called a pair a different entity
when 2 of 3 identity fields (name, location, country) disagreed. That caught 122 pairs, and reading
them showed most were renames and relocations: Berlin Schönefeld becoming Brandenburg, Oslo's new
airport, Algerian airports with new names. The 8 pairs where all three disagree all have a different
*country* too, and renaming or moving an airport does not change its country: that rule gives the 8 above. The cost is that a
key reused within one country (`EGSY` Sheffield City Heliport vs `MOD St Athan`, 261 km apart) stays
in `changed`, with its 261 km in the table. The threshold is `identity_min` in the config.

**The crosswalk is a majority vote and says so.** A left value is trusted once it has `min_support`
pairs and `min_share` of them agree (5 and 60% here). The vote includes the pair being judged, so it
can find the odd one out among many, and cannot judge a value it has seen twice: that is
`unverifiable`, not a guess. 88 of the 233 distinct country values are not trusted for this reason.

**Text similarity is deliberately plain.** Names are normalized (accents, case, punctuation), then
`near_match` is either token containment ("Abadan Airport" in "Abadan Ayatollah Jami International
Airport") or a token-sorted similarity of 0.85 or more (the "Ghaidah / Ghaydah" spelling variant).
Everything else is a mismatch with its similarity recorded. This is a heuristic, and the metric
column is there to tune it.

**Everything from the data is escaped in the report**, and the report fetches nothing from outside.

## Not done

- Two sources only, and one job per run: no three-way reconciliation.
- Exact and tolerance matching on keys only. No fuzzy record linkage on names, so an airport whose
  code *and* ICAO changed shows up as one `only_left` and one `only_right`, not as a pair.
- No automatic fixes or merged "golden record": the tool finds and classifies, a person decides.
- The two sources are not independent here (OpenFlights imported from OurAirports), so a mismatch
  says who has changed, not who is right.

## Sources

- OpenFlights airports database, <https://openflights.org/data.html> (Open Database License); file
  `airports.dat` from <https://github.com/jpatokal/openflights>.
- OurAirports, <https://ourairports.com/data/> (public domain); `airports.csv` from
  <https://davidmegginson.github.io/ourairports-data/>. Download SHA-256s are in
  `sample_data/SOURCES.json`.
