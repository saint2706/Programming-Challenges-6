"""Fetch the public NYC TLC yellow taxi Parquet files and the taxi zone lookup.

Source: the TLC trip record data page's CloudFront bucket (public, no login). Files are cached in
``data/raw/`` and never committed. ``data/raw/manifest.json`` records size and SHA-256 of each file
so a later run can show it used the same bytes.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from dist_pipeline.paths import data_dir

BASE = "https://d37ci6vzurychx.cloudfront.net"
ZONE_URL = f"{BASE}/misc/taxi_zone_lookup.csv"
YEAR = 2019
MONTHS = tuple(f"{YEAR}-{m:02d}" for m in range(1, 13))


def trip_url(month: str) -> str:
    return f"{BASE}/trip-data/yellow_tripdata_{month}.parquet"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, target: Path) -> Path:
    """Download ``url`` to ``target`` unless it is already there. Writes via a temp name."""
    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "dist-pipeline-fetch/1.0"})
    print(f"downloading {url}")
    with urllib.request.urlopen(req, timeout=600) as resp, tmp.open("wb") as out:
        for block in iter(lambda: resp.read(1 << 20), b""):
            out.write(block)
    tmp.replace(target)
    return target


def download_all(months: tuple[str, ...] = MONTHS) -> dict[str, dict[str, object]]:
    raw = data_dir() / "raw"
    manifest: dict[str, dict[str, object]] = {}
    paths = [(m, trip_url(m), raw / f"yellow_tripdata_{m}.parquet") for m in months]
    paths.append(("zones", ZONE_URL, raw / "taxi_zone_lookup.csv"))
    for key, url, target in paths:
        fetch(url, target)
        manifest[key] = {
            "url": url,
            "file": target.name,
            "bytes": target.stat().st_size,
            "sha256": _sha256(target),
        }
    record = {
        "fetched_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": manifest,
    }
    (raw / "manifest.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
