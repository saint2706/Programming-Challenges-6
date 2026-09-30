# Client-Side Data Table with Sort/Filter/CSV Export

**Category:** Web Development Showcase
**Difficulty:** I

**Status:** Implemented (JavaScript)

A data table that sorts, filters and exports **100,000+ rows entirely in the browser**, with no backend. Only the ~20 rows on screen exist in the DOM. Multi-column sort, per-column and global filters, keyboard navigation per the ARIA grid pattern, and a CSV export of exactly what the table currently shows.

## What it does

- **Virtualized rows.** One scroll container (`role="grid"`) with a sticky header and a spacer whose height is `rows x 36px`; `@tanstack/virtual-core` picks the window and rows are absolutely positioned with `translateY`. 100k rows and 250k rows mount in the same ~20 DOM rows.
- **Sort.** Click a header: ascending, descending, off. <kbd>Shift</kbd>+click adds a secondary key (glyphs show `▲1`, `▼2`). Sorts are stable, and missing values (`null`/empty) sort **last in both directions**.
- **Filter.** A global search box (all whitespace-separated tokens must appear somewhere in the row) plus one filter per column: substring for text, a dropdown for enums and booleans, and a small expression language for numbers and dates: `>50000`, `<=100`, `=42`, `40000..60000`, `>=2021`, `2020-03..2021-06`. Partial ISO dates cover the whole period (`>2021` means after 31 Dec 2021). A typo (`100k`) is flagged `aria-invalid` but does **not** blank the table.
- **CSV export.** Exports the whole filtered and sorted view (not just the rendered window), RFC 4180 quoted, CRLF terminated, optional UTF-8 BOM so Excel reads accents and CJK, and **formula-injection neutralised** (below).
- **Accessibility.** ARIA grid roles and counts, `aria-sort` on headers, `aria-activedescendant` for the active cell (focus stays on the grid), a polite live region announcing "Showing N of M rows", visible focus rings, light/dark via `prefers-color-scheme`, and no animation beyond a colour transition.

Keyboard, with the grid focused: arrows move the active cell; <kbd>PgUp</kbd>/<kbd>PgDn</kbd> move a screenful; <kbd>Home</kbd>/<kbd>End</kbd> go to the first/last column; <kbd>Ctrl</kbd>+<kbd>Home</kbd>/<kbd>End</kbd> go to the first/last cell.

## Design notes

**table-core owns the state, a flat engine owns the rows.** `@tanstack/table-core` is headless and gives exactly the right *state machine*: the asc → desc → off cycle, shift multi-sort, and column/global filter state. But its row models wrap every row in `Row`/`Cell` closures, which is the wrong shape for 100k rows. Measured on this machine at 100k rows (`npm run bench:baseline`):

| table-core row model | time | heap |
| --- | ---: | ---: |
| `getCoreRowModel` (wrap every row) | 512 ms | +559 MB |
| `getSortedRowModel`, salary | 546 ms | +54 MB |
| `getSortedRowModel`, name | 1,309 ms | +53 MB |
| `getFilteredRowModel`, city contains "ber" | 115 ms | +3 MB |

So the table model runs table-core with `manualSorting`/`manualFiltering` and `data: []`, and `src/engine.js` computes the view as an **`Int32Array` of indices into the raw row array**. Rows are never copied or wrapped, and the view of 100k rows is 400 KB.

**Sort keys are flat `Float64Array`s built lazily per column.** Numbers and currency use the value itself. Text and dates are ranked: the *distinct* values are sorted once with the real `Intl.Collator('en', { numeric: true, sensitivity: 'base' })`, then rows sort on integer ranks. 100k rows over 288 distinct names cost 288 collator comparisons' worth of work, not 1.7 million. `NaN` marks a missing value, which is how "missing last in both directions" falls out (a comparator alone cannot do that, because descending just negates it). A final `a - b` on the original index makes every sort stable.

**Filtering is compiled predicates over indices.** Text columns keep a lazily built lowercase copy, and the global search keeps one lowercase haystack per row (cached in a `WeakMap`). The filtered order is cached by a signature of the filter state, so re-sorting an unchanged filter never re-filters.

**CSV, and why the formula guard.** `src/csv.js` is DOM-free. A text cell whose first character is `=`, `+`, `-`, `@`, TAB or CR gets a leading `'` so Excel and Sheets show it literally instead of executing it (`=HYPERLINK(...)`, OWASP "CSV injection"). The trade-off is that a guarded cell no longer equals its source (`'=1+1`), which I accept: the alternative is running attacker-supplied formulas. Numeric, currency and boolean columns are never guarded, because they are serialized from real `number`/`boolean` values (so `-5` stays `-5`). Headers are guarded too. The sample data deliberately contains commas, quotes, LF and CRLF inside cells, leading/trailing spaces, formula prefixes, Unicode, and `null` versus empty-string notes, and a test round-trips all of them through an independent RFC 4180 parser.

**Untrusted data never touches `innerHTML`.** Every cell is created with `textContent`; a test mounts a row whose name is `<img src=x onerror=...>` and asserts no element was created.

**A DOM-free core, a thin DOM layer.** `csv`, `comparators`, `filters`, `engine`, `tableModel` and `data` have no DOM dependency and are unit-tested directly; `grid.js` is the only file that touches the document.

## Benchmarks

Measured on this machine (Node, Windows, 100,000 rows, `npm run bench`; "cold" is the first use of a column or filter, which builds its lazy sort keys or lowercase cache, "warm" is the same interaction again):

| operation | time |
| --- | ---: |
| generate 100k rows | ~110 ms |
| cold sort, salary (number) | ~50 ms |
| cold sort, name (text, 288 distinct) | 30-70 ms |
| cold sort, email (text, 100k distinct) | 210-290 ms |
| cold sort, joined (date) | ~80 ms |
| multi-sort, department then salary | ~60 ms |
| warm re-sort / toggle direction | ~45 ms / 0.3 ms to clear |
| cold global search "smith" | 125-185 ms |
| warm global search (1-2 tokens) | 9-12 ms |
| column filter salary `>=100000` | ~5 ms |
| + department filter, then sort the result | 6 ms, 3 ms |
| CSV of all 100k rows (10.5 MB) | ~170 ms |

Against table-core's own row models above, the engine is ~10x faster on the sorts, and it removes the 559 MB row-wrapping cost entirely.

In a real browser (Chrome, DPR 2, via the DevTools MCP): at **100k rows** a header sort, a multi-sort and a global search each finish, including rendering, in under ~100 ms; scrolling 200 steps of 5,000 px stays at frame rate. At **500k rows**, mounting takes ~440 ms, sorting the 500k-distinct-ish email column ~1.7 s, and a cold search ~0.5 s.

## Limitations

- **Browsers cap how tall a scroll area can be.** With 36px rows, this Chrome (DPR 2) stops scrolling at 2^24 = 16,777,216 px, which is ~466k rows. At 500k rows, the bottom ~34k rows are unreachable by scrolling. The UI therefore tops out at **250,000 rows**, which reaches the last row (verified, including <kbd>Ctrl</kbd>+<kbd>End</kbd>). Going beyond that needs scroll-position scaling (a synthetic scrollbar mapping), which I did not build.
- **Sorting is single-threaded.** A cold sort of a 100k-distinct text column blocks the main thread for ~250 ms; a Web Worker would remove that, at the cost of transferring the view back.
- **Row height is fixed** (36px), so notes containing line breaks show on one line (full text on hover via `title`).
- **Filter expressions on dates need ISO-style input** (`2021`, `2021-03`, `2021-03-04`).
- **CSV formulas are prefixed with `'`**, so a re-import of an export shows the apostrophe for those cells.
- The dataset is generated (deterministic, seeded), not fetched: the challenge is about the table, not the data.

## Run it

```bash
cd "challenges/Web Development Showcase/Client-Side Data Table with Sort-Filter-CSV Export"

npm install
npm run dev       # dev server (Vite); pick 1k / 10k / 100k / 250k rows in the page
npm run build     # production bundle in dist/ (~89 kB JS, ~27 kB gzipped)
npm test          # 130 tests
npm run bench     # engine benchmark, optional row count: npm run bench -- 250000
npm run bench:baseline   # table-core's own row models, for comparison
```

## Tests

130 tests (Vitest + jsdom), all offline and deterministic:

- `csv.test.js`: RFC 4180 quoting, CRLF, BOM, formula guard on text/date/header cells but not numbers, guard-then-quote ordering, and a round-trip of every awkward note through an independent parser.
- `comparators.test.js`: natural number ordering (`item2 < item10`), case and accent insensitivity, missing-value classification.
- `filters.test.js`: every operator for numbers and partial dates, missing values never matching, typo tolerance, global search AND semantics, debounce.
- `engine.test.js`: the fast engine cross-checked against a slow reference implementation for every column in both directions, multi-sort, combined filters, stability, and missing-last in both directions.
- `tableModel.test.js`: sort cycle, shift multi-sort, filter composition, `onChange`, reset, and a randomised property check.
- `grid.test.js`: the virtualization window (first/last rendered index computed from scroll offset, header height and overscan), ARIA structure, sorting and filtering through the real controls, keyboard navigation, hostile cell content, and the Export button producing a BOM-prefixed CSV blob.
- `perf.test.js`: 100k-row sanity ceilings (loose, to catch an accidental O(n²)).
- `data.test.js`: the seeded generator and cell formatting.

jsdom has no layout, so `grid.test.js` supplies the element sizes the virtualizer measures (and `scrollTo`, `scrollHeight`); real scrolling, sizing and download behaviour were checked in a browser as described above.
