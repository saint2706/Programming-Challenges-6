# Personal Spending Categorizer & Monthly Trend Report

**Category:** Data Analytics
**Difficulty:** B

**Status:** Implemented (Python)

Parses a bank CSV export, categorizes each transaction against an editable
keyword-rule file, and emits a self-contained HTML report of monthly
spending trends by category.

## What it does

- **Column mapping, not a fixed format.** Different banks export different
  headers (`Date`/`Posted Date`, `Description`/`Memo`, `Amount`/`Debit`).
  Common aliases are auto-detected case-insensitively; `--date-col`,
  `--desc-col`, `--amount-col` override the guess, and an unmappable file
  fails with the list of columns it actually found rather than guessing
  wrong silently.
- **Amount normalization.** Handles `$1,234.56`, `(12.34)` (accounting
  parens for negative), and plain `-12.34` — all coerced to a signed float
  where negative means money out.
- **Rule-based categorization**, not ML. `rules.json` maps a category to a
  list of keywords/substrings matched case-insensitively against the
  description (`"starbucks"` → `Dining`). Anything that matches nothing
  becomes `Uncategorized`, and the report's last section lists a sample of
  those so you can extend `rules.json` — the point of keeping this
  rule-based is that closing a gap is a one-line edit, not a retrain.
- **Monthly trend report**: total spend/income/net summary, a total
  monthly spend line chart, a stacked monthly-spend-by-category bar chart,
  a category totals table, and the uncategorized sample — one HTML file,
  Plotly's JS bundled inline so it needs no internet connection to view.

## Design notes

Same stack as the CSV Profiler: **polars** for the CSV/aggregation work,
**Plotly** for interactive charts. Income (positive amounts) is summarized
separately from spend (negative amounts) rather than mixed into the same
categories — "how much did I spend on X" and "how much came in" are
different questions, and merging them would make the category chart
meaningless (a paycheck and a purchase refund look identical to a keyword
matcher).

## Run it

```bash
cd "challenges/Data Analytics/Personal Spending Categorizer & Monthly Trend Report"

uv run --with polars --with plotly --with numpy python categorizer.py sample_data/checking_export_a.csv
uv run --with polars --with plotly --with numpy python categorizer.py sample_data/checking_export_b.csv -o export_b_report.html

uv run --with pytest --with polars --with plotly --with numpy pytest -q   # 11 tests
```

Edit `rules.json` to match your own bank's merchant naming, then rerun.

## Sample data

- `sample_data/checking_export_a.csv` — `Date`/`Description`/`Amount`
  headers, plain-signed amounts, two months of transactions.
- `sample_data/checking_export_b.csv` — `Posted Date`/`Memo`/`Debit`
  headers, `MM/DD/YYYY` dates, currency-formatted and parenthesized
  amounts — exercises the column-mapping and amount-parsing paths on a
  differently-shaped export.

All transactions are synthetic.

## Where this is actually used

This is the free-tier feature of every budgeting app (Mint, YNAB, bank
"insights" tabs): pull a transaction export, bucket it by merchant
keyword, chart it by month. The rule-keyword approach is also what those
tools actually ship for the common case — full ML categorization exists,
but a maintained keyword list gets you most of the value with a
categorization decision you can explain and fix in one line.
