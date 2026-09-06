"""Tests for the public API data aggregator.

The real Open-Meteo endpoint is never hit here -- httpx.get is monkeypatched
with a fixed fixture response, so these run offline and deterministically.

Run with:  uv run --with pytest --with httpx --with polars --with plotly pytest -q
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from aggregator import (
    Snapshot,
    build_arg_parser,
    connect,
    fetch_current_weather,
    load_snapshots,
    main,
    parse_forecast_response,
    store_snapshot,
)

SAMPLE_DIR = Path(__file__).parent / "sample_data"

FAKE_FORECAST_RESPONSE = {
    "current": {
        "time": "2026-09-06T14:00",
        "temperature_2m": 18.2,
        "relative_humidity_2m": 87,
        "wind_speed_10m": 7.6,
    },
    "daily": {
        "time": ["2026-09-06", "2026-09-07"],
        "temperature_2m_max": [25.0, 24.0],
        "temperature_2m_min": [17.0, 16.5],
        "precipitation_sum": [0.0, 2.3],
    },
}


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


def test_parse_forecast_response_extracts_current_and_matching_day() -> None:
    snapshot = parse_forecast_response(FAKE_FORECAST_RESPONSE, "New York")
    assert snapshot.date == "2026-09-06"
    assert snapshot.temperature_c == 18.2
    assert snapshot.humidity_pct == 87
    assert snapshot.temp_max_c == 25.0
    assert snapshot.temp_min_c == 17.0
    assert snapshot.precipitation_mm == 0.0


def test_fetch_current_weather_calls_httpx_get(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_get(url, params, timeout):
        captured["url"] = url
        captured["params"] = params
        return FakeResponse(FAKE_FORECAST_RESPONSE)

    monkeypatch.setattr(httpx, "get", fake_get)
    payload = fetch_current_weather(40.7128, -74.0060)
    assert payload == FAKE_FORECAST_RESPONSE
    assert captured["params"]["latitude"] == 40.7128
    assert "open-meteo.com" in captured["url"]


def test_store_snapshot_upserts_without_duplicating(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    conn = connect(db_path)

    snap1 = Snapshot(
        date="2026-09-06",
        location="New York",
        fetched_at="t1",
        temperature_c=18.0,
        humidity_pct=80,
        wind_speed_kmh=5.0,
        temp_max_c=25.0,
        temp_min_c=17.0,
        precipitation_mm=0.0,
        raw_json=None,
    )
    store_snapshot(conn, snap1)

    snap2 = Snapshot(
        date="2026-09-06",
        location="New York",
        fetched_at="t2",
        temperature_c=19.5,
        humidity_pct=75,
        wind_speed_kmh=6.0,
        temp_max_c=25.0,
        temp_min_c=17.0,
        precipitation_mm=0.0,
        raw_json=None,
    )
    store_snapshot(conn, snap2)

    rows = load_snapshots(conn)
    assert len(rows) == 1
    assert rows[0]["temperature_c"] == 19.5
    assert rows[0]["fetched_at"] == "t2"
    conn.close()


def test_store_snapshot_distinguishes_locations(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    conn = connect(db_path)
    for loc in ("New York", "London"):
        store_snapshot(
            conn,
            Snapshot(
                date="2026-09-06",
                location=loc,
                fetched_at="t",
                temperature_c=10.0,
                humidity_pct=50,
                wind_speed_kmh=1.0,
                temp_max_c=12.0,
                temp_min_c=8.0,
                precipitation_mm=0.0,
                raw_json=None,
            ),
        )
    rows = load_snapshots(conn)
    assert len(rows) == 2
    assert {r["location"] for r in rows} == {"New York", "London"}
    conn.close()


def test_cmd_poll_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        httpx, "get", lambda *a, **k: FakeResponse(FAKE_FORECAST_RESPONSE)
    )
    db_path = tmp_path / "snapshots.db"
    exit_code = main(["poll", "--db", str(db_path), "--location", "New York"])
    assert exit_code == 0
    assert "New York" in capsys.readouterr().out

    conn = connect(db_path)
    rows = load_snapshots(conn)
    conn.close()
    assert len(rows) == 1
    assert rows[0]["date"] == "2026-09-06"


def test_cmd_seed_loads_historical_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db_path = tmp_path / "seeded.db"
    exit_code = main(
        [
            "seed",
            "--db",
            str(db_path),
            "--input",
            str(SAMPLE_DIR / "historical_nyc.json"),
        ]
    )
    assert exit_code == 0
    assert "Seeded 14 snapshots" in capsys.readouterr().out

    conn = connect(db_path)
    rows = load_snapshots(conn)
    conn.close()
    assert len(rows) == 14
    assert rows[0]["date"] == "2026-08-23"
    assert rows[-1]["date"] == "2026-09-05"


def test_cmd_report_requires_data(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db_path = tmp_path / "empty.db"
    output = tmp_path / "report.html"
    exit_code = main(["report", "--db", str(db_path), "-o", str(output)])
    assert exit_code == 1
    assert "no snapshots found" in capsys.readouterr().err


def test_cmd_report_builds_self_contained_html(tmp_path: Path) -> None:
    db_path = tmp_path / "seeded.db"
    output = tmp_path / "report.html"
    main(
        [
            "seed",
            "--db",
            str(db_path),
            "--input",
            str(SAMPLE_DIR / "historical_nyc.json"),
        ]
    )
    exit_code = main(["report", "--db", str(db_path), "-o", str(output)])
    assert exit_code == 0
    report_html = output.read_text(encoding="utf-8")
    assert "<!doctype html>" in report_html
    assert "<script" in report_html
    assert "Weather Snapshot Trend" in report_html


def test_arg_parser_requires_subcommand() -> None:
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
