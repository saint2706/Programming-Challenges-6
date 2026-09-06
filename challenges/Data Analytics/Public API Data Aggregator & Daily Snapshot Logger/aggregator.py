"""Public API Data Aggregator & Daily Snapshot Logger.

Polls the Open-Meteo weather API (no key required) once per invocation and
upserts the result into a SQLite table, building a longitudinal dataset one
snapshot at a time. Meant to be run once a day (via cron / a scheduled CI
job -- see daily-snapshot-workflow.yml.example) rather than as a daemon.

Run with:  uv run --with httpx python aggregator.py poll --db snapshots.db
           uv run --with polars --with plotly python aggregator.py report --db snapshots.db -o report.html
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DEFAULT_DB = Path("snapshots.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    date TEXT NOT NULL,
    location TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    temperature_c REAL,
    humidity_pct REAL,
    wind_speed_kmh REAL,
    temp_max_c REAL,
    temp_min_c REAL,
    precipitation_mm REAL,
    raw_json TEXT,
    PRIMARY KEY (date, location)
);
"""

UPSERT = """
INSERT INTO snapshots
    (date, location, fetched_at, temperature_c, humidity_pct, wind_speed_kmh,
     temp_max_c, temp_min_c, precipitation_mm, raw_json)
VALUES (:date, :location, :fetched_at, :temperature_c, :humidity_pct, :wind_speed_kmh,
        :temp_max_c, :temp_min_c, :precipitation_mm, :raw_json)
ON CONFLICT(date, location) DO UPDATE SET
    fetched_at = excluded.fetched_at,
    temperature_c = excluded.temperature_c,
    humidity_pct = excluded.humidity_pct,
    wind_speed_kmh = excluded.wind_speed_kmh,
    temp_max_c = excluded.temp_max_c,
    temp_min_c = excluded.temp_min_c,
    precipitation_mm = excluded.precipitation_mm,
    raw_json = excluded.raw_json;
"""


@dataclass
class Snapshot:
    date: str
    location: str
    fetched_at: str
    temperature_c: float | None
    humidity_pct: float | None
    wind_speed_kmh: float | None
    temp_max_c: float | None
    temp_min_c: float | None
    precipitation_mm: float | None
    raw_json: str | None = None

    def as_row(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "location": self.location,
            "fetched_at": self.fetched_at,
            "temperature_c": self.temperature_c,
            "humidity_pct": self.humidity_pct,
            "wind_speed_kmh": self.wind_speed_kmh,
            "temp_max_c": self.temp_max_c,
            "temp_min_c": self.temp_min_c,
            "precipitation_mm": self.precipitation_mm,
            "raw_json": self.raw_json,
        }


def fetch_current_weather(lat: float, lon: float) -> dict[str, Any]:
    response = httpx.get(
        FORECAST_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
            "timezone": "auto",
        },
        timeout=10.0,
    )
    response.raise_for_status()
    return response.json()


def parse_forecast_response(payload: dict[str, Any], location: str) -> Snapshot:
    current = payload["current"]
    today = current["time"].split("T")[0]
    daily = payload["daily"]
    day_idx = daily["time"].index(today) if today in daily["time"] else 0

    return Snapshot(
        date=today,
        location=location,
        fetched_at=datetime.now(timezone.utc).isoformat(),
        temperature_c=current.get("temperature_2m"),
        humidity_pct=current.get("relative_humidity_2m"),
        wind_speed_kmh=current.get("wind_speed_10m"),
        temp_max_c=daily["temperature_2m_max"][day_idx],
        temp_min_c=daily["temperature_2m_min"][day_idx],
        precipitation_mm=daily["precipitation_sum"][day_idx],
        raw_json=json.dumps(payload),
    )


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(SCHEMA)
    return conn


def store_snapshot(conn: sqlite3.Connection, snapshot: Snapshot) -> None:
    conn.execute(UPSERT, snapshot.as_row())
    conn.commit()


def load_snapshots(
    conn: sqlite3.Connection, location: str | None = None
) -> list[dict[str, Any]]:
    query = "SELECT * FROM snapshots"
    params: tuple[Any, ...] = ()
    if location:
        query += " WHERE location = ?"
        params = (location,)
    query += " ORDER BY date"
    conn.row_factory = sqlite3.Row
    return [dict(row) for row in conn.execute(query, params).fetchall()]


def cmd_poll(args: argparse.Namespace) -> int:
    payload = fetch_current_weather(args.lat, args.lon)
    snapshot = parse_forecast_response(payload, args.location)
    conn = connect(args.db)
    store_snapshot(conn, snapshot)
    conn.close()
    print(
        f"{snapshot.date} {snapshot.location}: {snapshot.temperature_c}°C, "
        f"{snapshot.humidity_pct}% humidity -> {args.db}"
    )
    return 0


def cmd_seed(args: argparse.Namespace) -> int:
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    conn = connect(args.db)
    for row in rows:
        snapshot = Snapshot(
            date=row["date"],
            location=row["location"],
            fetched_at=row["fetched_at"],
            temperature_c=row.get("temperature_c"),
            humidity_pct=row.get("humidity_pct"),
            wind_speed_kmh=row.get("wind_speed_kmh"),
            temp_max_c=row.get("temp_max_c"),
            temp_min_c=row.get("temp_min_c"),
            precipitation_mm=row.get("precipitation_mm"),
            raw_json=None,
        )
        store_snapshot(conn, snapshot)
    conn.close()
    print(f"Seeded {len(rows)} snapshots from {args.input} -> {args.db}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    import polars as pl
    import plotly.graph_objects as go
    import plotly.io as pio

    conn = connect(args.db)
    rows = load_snapshots(conn, args.location)
    conn.close()

    if not rows:
        print(
            "error: no snapshots found -- run `poll` or `seed` first", file=sys.stderr
        )
        return 1

    df = pl.DataFrame(rows).sort("date")
    dates = df["date"].to_list()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=df["temp_max_c"].to_list(),
            name="Max temp (°C)",
            line=dict(color="#F58518"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=df["temperature_c"].to_list(),
            name="Reading (°C)",
            line=dict(color="#4C78A8"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=df["temp_min_c"].to_list(),
            name="Min temp (°C)",
            line=dict(color="#54A24B"),
        )
    )
    fig.update_layout(
        title="Temperature over time", margin=dict(l=50, r=20, t=40, b=30), height=320
    )
    temp_chart = pio.to_html(
        fig, include_plotlyjs=False, full_html=False, config={"displaylogo": False}
    )

    precip_fig = go.Figure(
        go.Bar(x=dates, y=df["precipitation_mm"].to_list(), marker_color="#4C78A8")
    )
    precip_fig.update_layout(
        title="Precipitation (mm)", margin=dict(l=50, r=20, t=40, b=30), height=260
    )
    precip_chart = pio.to_html(
        precip_fig,
        include_plotlyjs=False,
        full_html=False,
        config={"displaylogo": False},
    )

    plotly_js = pio.to_html(go.Figure(), include_plotlyjs="inline", full_html=False)
    bundle = plotly_js[
        plotly_js.index("<script") : plotly_js.index("</script>") + len("</script>")
    ]

    locations = df["location"].unique().to_list()
    location_label = locations[0] if len(locations) == 1 else ", ".join(locations)

    html = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Weather Snapshot Trend</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; padding: 2rem;
        background: #0f1115; color: #e8e8e8; }}
h1 {{ font-size: 1.5rem; }}
</style></head>
<body>
{bundle}
<h1>Weather Snapshot Trend &mdash; {location_label}</h1>
<p>{len(df)} snapshots from {dates[0]} to {dates[-1]}.</p>
{temp_chart}
{precip_chart}
</body>
</html>
"""
    args.output.write_text(html, encoding="utf-8")
    print(f"Report ({len(df)} snapshots) -> {args.output}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    poll = subparsers.add_parser(
        "poll", help="Fetch one snapshot and upsert it into the database"
    )
    poll.add_argument("--db", type=Path, default=DEFAULT_DB)
    poll.add_argument("--location", type=str, default="New York")
    poll.add_argument("--lat", type=float, default=40.7128)
    poll.add_argument("--lon", type=float, default=-74.0060)
    poll.set_defaults(func=cmd_poll)

    seed = subparsers.add_parser(
        "seed",
        help="Load pre-fetched real snapshots (e.g. historical data) into the database",
    )
    seed.add_argument("--db", type=Path, default=DEFAULT_DB)
    seed.add_argument("--input", type=Path, required=True)
    seed.set_defaults(func=cmd_seed)

    report = subparsers.add_parser(
        "report", help="Render a self-contained HTML trend report from the database"
    )
    report.add_argument("--db", type=Path, default=DEFAULT_DB)
    report.add_argument("--location", type=str, default=None)
    report.add_argument("-o", "--output", type=Path, default=Path("report.html"))
    report.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
