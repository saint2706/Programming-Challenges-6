# Interactive Sales Dashboard from a Spreadsheet

**Category:** Data Analytics
**Difficulty:** B

**Status:** Implemented (JavaScript)

One HTML file. No backend, no build step. Open it, load a spreadsheet, get
filterable/drillable charts.

## What it does

- **Loads CSV or XLSX client-side** via [SheetJS](https://sheetjs.com/)
  (`XLSX.read`/`sheet_to_json`), so both a bank-style CSV export and an
  actual Excel workbook work through the same code path.
- **Auto-detects columns** (Date/Category/Region/Product/Amount, optional
  Quantity) by matching common header aliases case-insensitively, with a
  manual override `<select>` per role for anything guessed wrong or
  ambiguous. The dashboard only renders once every required role is
  mapped.
- **Filters**: Category, Region, and a date range, all recomputed
  client-side with no page reload.
- **Drill-down**: clicking a bar in the "Sales by category" chart filters
  the whole dashboard to that category (click again to clear it) — the
  clicked bar is highlighted so the active filter state is visible at a
  glance.
- **Charts** via [Observable Plot](https://observablehq.com/plot/): a
  monthly sales trend line, a category totals bar chart, and a top-10
  products table, all recomputed from the currently filtered rows.

## Design notes

Modern front-end stack chosen deliberately over the more common jQuery/
Chart.js combo: Observable Plot (D3-based, declarative, actively
developed) for charts, plain ES2022+ (no framework, no build step — the
"no backend" requirement extends naturally to "no bundler" here). All
three CDN scripts (SheetJS, D3, Observable Plot) are pinned to an exact
version with a Subresource Integrity hash, so a compromised CDN can't
silently swap in different code.

**How the drill-down click actually works**: Observable Plot doesn't
expose a click API on marks directly. The bar chart's underlying data
array is built in a fixed, explicit order (`y: { domain: categories }`),
so after `Plot.plot()` renders, `svg.querySelectorAll("[aria-label='bar']
rect")` returns the bar elements in that same order — cheap to reason
about for a handful of categories, though a chart with many more
categories or a reorder-on-hover interaction would be better served by
Plot's lower-level `render` hook.

**Why `fetch()` for the sample-data button needs a server**: opening the
file directly via `file://` blocks `fetch()` of a local file under most
browsers' CORS rules. `FileReader` (the file-picker path) has no such
restriction since it's a user-initiated read, not a network request —
that path always works, `file://` or not. See "Run it" below.

## Run it

```bash
cd "challenges/Data Analytics/Interactive Sales Dashboard from a Spreadsheet"

# Either open dashboard.html directly and use the file picker to load
# sample_data/sales_sample.csv, OR serve the folder so the
# "Load sample data" button's fetch() call works too:
python -m http.server 8000
# then visit http://localhost:8000/dashboard.html
```

## Sample data

`sample_data/sales_sample.csv` — 220 synthetic orders across 4 regions,
4 categories, and 16 products over a 6-month window.

## Where this is actually used

This is the shape of every embedded BI tool's "explore" view (Looker
Studio, Tableau Public's free tier, Metabase's question builder): load a
table, auto-suggest a chart, let the viewer filter and click into a
segment. Doing it as a single static file rather than a hosted app is
also exactly how ops teams end up sharing one-off dashboards over email
or Slack when standing up a real BI tool isn't worth it for one dataset.
