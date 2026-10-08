"""Fetch more labelled series from the Numenta Anomaly Benchmark (MIT license).

    uv run python -m ts_anomaly.fetch_nab realKnownCause/machine_temperature_system_failure
    uv run ts-anomaly .cache/nab/machine_temperature_system_failure.csv --labels .cache/nab/labels.json -o out/machine.html

Downloads the series CSV and its labelled windows into `.cache/nab/` (ignored by
git). `sample_data/` already vendors four series, so this is only needed for the
other ~54: https://github.com/numenta/NAB/tree/master/data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from ts_anomaly.paths import project_root

RAW = "https://raw.githubusercontent.com/numenta/NAB/master"
CACHE = project_root() / ".cache" / "nab"
# "<category>/<name>" with no dots-only segments or slashes beyond the one separator,
# so a key can neither climb out of the cache directory nor rewrite the URL path.
_KEY = re.compile(r"^[A-Za-z]+/[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read()


def normalise_key(key: str) -> str:
    """'realKnownCause/nyc_taxi' or '.../nyc_taxi.csv' -> 'realKnownCause/nyc_taxi.csv'."""
    key = key if key.endswith(".csv") else f"{key}.csv"
    if not _KEY.match(key) or ".." in key:
        raise ValueError(
            f"not a NAB series key: {key!r} (expected e.g. realKnownCause/nyc_taxi)"
        )
    return key


def fetch(
    key: str,
    cache: Path = CACHE,
    download: Callable[[str], bytes] = _download,
) -> tuple[Path, Path, str]:
    """Download one series and merge its windows into `<cache>/labels.json`.

    Returns (csv_path, labels_path, sha256 of the csv).
    """
    key = normalise_key(key)
    cache.mkdir(parents=True, exist_ok=True)
    csv_bytes = download(f"{RAW}/data/{key}")
    if not csv_bytes.lstrip().startswith(b"timestamp,value"):
        raise ValueError(
            f"{key}: response does not look like a NAB CSV (no 'timestamp,value' header)"
        )
    all_labels = json.loads(download(f"{RAW}/labels/combined_windows.json"))
    if key not in all_labels:
        raise ValueError(f"{key}: no labelled windows in NAB's combined_windows.json")

    csv_path = cache / key.rsplit("/", 1)[1]
    csv_path.write_bytes(csv_bytes)
    labels_path = cache / "labels.json"
    existing = (
        json.loads(labels_path.read_text(encoding="utf-8"))
        if labels_path.exists()
        else {}
    )
    existing[key] = all_labels[key]
    labels_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return csv_path, labels_path, hashlib.sha256(csv_bytes).hexdigest()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "series", nargs="+", help="NAB series keys, e.g. realKnownCause/nyc_taxi"
    )
    args = p.parse_args(argv)
    for key in args.series:
        try:
            csv_path, labels_path, digest = fetch(key)
        except (ValueError, OSError, urllib.error.URLError) as exc:
            print(f"error: {key}: {exc}", file=sys.stderr)
            return 2
        print(f"{csv_path}  (sha256 {digest[:16]}...)  labels -> {labels_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
