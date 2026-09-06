"""Personal Spending Categorizer & Monthly Trend Report.

Parses a bank CSV export, categorizes each transaction against an editable
keyword-rule file, and emits a self-contained HTML monthly trend report.

Run with:  uv run --with polars --with plotly python categorizer.py transactions.csv
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import polars as pl

DEFAULT_RULES_PATH = Path(__file__).parent / "rules.json"
UNCATEGORIZED = "Uncategorized"

# Common header aliases per logical role, tried in order (case-insensitive).
COLUMN_ALIASES = {
    "date": ["date", "transaction date", "posted date", "posting date"],
    "description": ["description", "desc", "memo", "payee", "merchant", "narrative"],
    "amount": ["amount", "transaction amount", "debit", "value"],
}


@dataclass
class ColumnMapping:
    date: str
    description: str
    amount: str


def resolve_column(df: pl.DataFrame, role: str, override: str | None) -> str:
    if override:
        if override not in df.columns:
            raise ValueError(
                f"column {override!r} not found; available columns: {df.columns}"
            )
        return override
    lowered = {c.lower(): c for c in df.columns}
    for alias in COLUMN_ALIASES[role]:
        if alias in lowered:
            return lowered[alias]
    raise ValueError(
        f"couldn't auto-detect the {role} column; pass --{role}-col explicitly. "
        f"Available columns: {df.columns}"
    )


def resolve_mapping(
    df: pl.DataFrame, date_col: str | None, desc_col: str | None, amount_col: str | None
) -> ColumnMapping:
    return ColumnMapping(
        date=resolve_column(df, "date", date_col),
        description=resolve_column(df, "description", desc_col),
        amount=resolve_column(df, "amount", amount_col),
    )


def parse_amount_expr(col: str) -> pl.Expr:
    """Normalize amounts like "$1,234.56", "(12.34)" (parens = negative), "-12.34"."""
    cleaned = pl.col(col).cast(pl.Utf8).str.strip_chars()
    is_paren_negative = cleaned.str.starts_with("(") & cleaned.str.ends_with(")")
    stripped = cleaned.str.replace_all(r"[\$,()]", "").str.strip_chars()
    value = stripped.cast(pl.Float64, strict=False)
    return pl.when(is_paren_negative).then(-value.abs()).otherwise(value)


def load_transactions(
    path: Path, mapping_overrides: tuple[str | None, str | None, str | None]
) -> tuple[pl.DataFrame, ColumnMapping]:
    raw = pl.read_csv(path, infer_schema_length=10_000)
    mapping = resolve_mapping(raw, *mapping_overrides)

    df = (
        raw.select(
            pl.col(mapping.date).alias("_date_raw"),
            pl.col(mapping.description)
            .cast(pl.Utf8)
            .fill_null("")
            .alias("description"),
            parse_amount_expr(mapping.amount).alias("amount"),
        )
        .with_columns(
            pl.coalesce(
                pl.col("_date_raw").str.to_date("%Y-%m-%d", strict=False),
                pl.col("_date_raw").str.to_date("%m/%d/%Y", strict=False),
                pl.col("_date_raw").str.to_date("%m-%d-%Y", strict=False),
            ).alias("date")
        )
        .drop("_date_raw")
    )

    df = df.filter(pl.col("amount").is_not_null())
    return df, mapping


def load_rules(path: Path) -> dict[str, list[str]]:
    return json.loads(path.read_text(encoding="utf-8"))


def compile_rules(rules: dict[str, list[str]]) -> list[tuple[str, re.Pattern[str]]]:
    compiled = []
    for category, keywords in rules.items():
        pattern = re.compile("|".join(re.escape(k) for k in keywords), re.IGNORECASE)
        compiled.append((category, pattern))
    return compiled


def categorize(
    description: str, compiled_rules: list[tuple[str, re.Pattern[str]]]
) -> str:
    for category, pattern in compiled_rules:
        if pattern.search(description):
            return category
    return UNCATEGORIZED


def categorize_transactions(
    df: pl.DataFrame, rules: dict[str, list[str]]
) -> pl.DataFrame:
    compiled_rules = compile_rules(rules)
    categories = [
        categorize(desc, compiled_rules) for desc in df["description"].to_list()
    ]
    return df.with_columns(pl.Series("category", categories))


def _fig_to_div(fig: go.Figure) -> str:
    return pio.to_html(
        fig, include_plotlyjs=False, full_html=False, config={"displaylogo": False}
    )


def build_monthly_category_chart(spend: pl.DataFrame) -> str:
    if spend.is_empty():
        return "<p>No expense transactions to chart.</p>"
    pivot = (
        spend.with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
        .group_by(["month", "category"])
        .agg(pl.col("spend").sum())
        .sort("month")
    )
    months = sorted(pivot["month"].unique().to_list())
    fig = go.Figure()
    for category in sorted(pivot["category"].unique().to_list()):
        cat_rows = pivot.filter(pl.col("category") == category)
        by_month = dict(zip(cat_rows["month"].to_list(), cat_rows["spend"].to_list()))
        fig.add_trace(
            go.Bar(name=category, x=months, y=[by_month.get(m, 0.0) for m in months])
        )
    fig.update_layout(
        barmode="stack",
        title="Monthly spend by category",
        margin=dict(l=40, r=20, t=40, b=30),
        height=380,
        legend=dict(orientation="h", y=-0.2),
    )
    return _fig_to_div(fig)


def build_total_trend_chart(spend: pl.DataFrame) -> str:
    if spend.is_empty():
        return ""
    monthly = (
        spend.with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
        .group_by("month")
        .agg(pl.col("spend").sum())
        .sort("month")
    )
    fig = go.Figure(
        go.Scatter(
            x=monthly["month"].to_list(),
            y=monthly["spend"].to_list(),
            mode="lines+markers",
        )
    )
    fig.update_layout(
        title="Total monthly spend", margin=dict(l=40, r=20, t=40, b=30), height=280
    )
    return _fig_to_div(fig)


def build_category_totals_table(spend: pl.DataFrame) -> str:
    totals = (
        spend.group_by("category")
        .agg(pl.col("spend").sum())
        .sort("spend", descending=True)
    )
    rows = "\n".join(
        f"<tr><td>{html.escape(cat)}</td><td>{amt:,.2f}</td></tr>"
        for cat, amt in zip(totals["category"].to_list(), totals["spend"].to_list())
    )
    return f"<table class='totals'><tr><th>Category</th><th>Total spend</th></tr>{rows}</table>"


def build_uncategorized_sample(spend: pl.DataFrame, limit: int = 15) -> str:
    uncategorized = spend.filter(pl.col("category") == UNCATEGORIZED).head(limit)
    if uncategorized.is_empty():
        return "<p>Everything matched a rule.</p>"
    rows = "\n".join(
        f"<tr><td>{d}</td><td>{html.escape(desc)}</td><td>{amt:,.2f}</td></tr>"
        for d, desc, amt in zip(
            uncategorized["date"].to_list(),
            uncategorized["description"].to_list(),
            uncategorized["spend"].to_list(),
        )
    )
    return f"<table class='totals'><tr><th>Date</th><th>Description</th><th>Amount</th></tr>{rows}</table>"


CSS = """
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; padding: 2rem;
       background: #0f1115; color: #e8e8e8; }
h1 { font-size: 1.6rem; }
h2 { font-size: 1.15rem; margin-top: 2rem; }
.summary { display: flex; gap: 1.5rem; flex-wrap: wrap; margin: 1rem 0 2rem; }
.summary .stat { background: #1a1d24; border-radius: 8px; padding: 0.75rem 1.25rem; }
.summary .stat .value { font-size: 1.4rem; font-weight: 600; }
.summary .stat .label { font-size: 0.8rem; opacity: 0.7; }
table.totals { border-collapse: collapse; width: 100%; max-width: 560px; font-size: 0.9rem; }
table.totals th, table.totals td { text-align: left; padding: 4px 10px; border-bottom: 1px solid #2a2d34; }
table.totals td:last-child, table.totals th:last-child { text-align: right; }
"""


def render_report(df: pl.DataFrame) -> str:
    spend = df.filter(pl.col("amount") < 0).with_columns(
        (-pl.col("amount")).alias("spend")
    )
    income_total = df.filter(pl.col("amount") > 0)["amount"].sum() or 0.0
    spend_total = spend["spend"].sum() or 0.0
    uncategorized_count = spend.filter(pl.col("category") == UNCATEGORIZED).height

    plotly_js = pio.to_html(go.Figure(), include_plotlyjs="inline", full_html=False)
    script_start = plotly_js.index("<script")
    script_end = plotly_js.index("</script>") + len("</script>")
    plotly_bundle = plotly_js[script_start:script_end]

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Spending Report</title>
<style>{CSS}</style>
</head>
<body>
{plotly_bundle}
<h1>Monthly Spending Report</h1>
<div class="summary">
  <div class="stat"><div class="value">{len(df):,}</div><div class="label">transactions</div></div>
  <div class="stat"><div class="value">{spend_total:,.2f}</div><div class="label">total spend</div></div>
  <div class="stat"><div class="value">{income_total:,.2f}</div><div class="label">total income</div></div>
  <div class="stat"><div class="value">{income_total - spend_total:,.2f}</div><div class="label">net</div></div>
  <div class="stat"><div class="value">{uncategorized_count}</div><div class="label">uncategorized</div></div>
</div>
<h2>Trend</h2>
{build_total_trend_chart(spend)}
<h2>Spend by category</h2>
{build_monthly_category_chart(spend)}
<h2>Category totals</h2>
{build_category_totals_table(spend)}
<h2>Uncategorized (refine rules.json to close these)</h2>
{build_uncategorized_sample(spend)}
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="Path to the bank transactions CSV")
    parser.add_argument("-o", "--output", type=Path, default=Path("report.html"))
    parser.add_argument(
        "--rules",
        type=Path,
        default=DEFAULT_RULES_PATH,
        help="Path to a category rules JSON file",
    )
    parser.add_argument("--date-col", type=str, default=None)
    parser.add_argument("--desc-col", type=str, default=None)
    parser.add_argument("--amount-col", type=str, default=None)
    args = parser.parse_args(argv)

    if not args.csv_path.exists():
        print(f"error: {args.csv_path} does not exist", file=sys.stderr)
        return 1

    try:
        df, mapping = load_transactions(
            args.csv_path, (args.date_col, args.desc_col, args.amount_col)
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    rules = load_rules(args.rules)
    df = categorize_transactions(df, rules)

    args.output.write_text(render_report(df), encoding="utf-8")
    print(
        f"Categorized {len(df):,} transactions "
        f"(date={mapping.date!r}, description={mapping.description!r}, amount={mapping.amount!r}) -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
