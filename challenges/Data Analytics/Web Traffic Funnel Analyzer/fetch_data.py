"""Download a real clickstream sample (REES46 multi-category store) without downloading 14.7 GB.

The full public CSV is sorted by time, so an HTTP Range request returns a clean
contiguous slice of real traffic. We take several slices spread across the file
(different days and hours), trim each to whole lines, and write them to
`data_cache/events_raw.csv` (git-ignored). `--make-sample` then derives the small
committed sample by keeping a deterministic ~1 in N of the *sessions* (a stable
hash of `user_session`), so every kept session is complete.

Run with:
    uv run python fetch_data.py                 # download the raw slices (~64 MB)
    uv run python fetch_data.py --make-sample   # also write sample_data/rees46_sample.csv.gz

Source: REES46 Marketing Platform "eCommerce behavior data from multi-category
store" (Kaggle: mkechinov/ecommerce-behavior-data-from-multi-category-store),
mirrored on Hugging Face. Free to use for research and education with attribution.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import sys
import urllib.request
from pathlib import Path

URL = (
    "https://huggingface.co/datasets/kevykibbz/"
    "ecommerce-behavior-data-from-multi-category-store_oct-nov_2019/resolve/main/"
    "ecommerce-behavior-data-from-multi-category-store_oct-nov_2019.csv"
)
HERE = Path(__file__).parent
RAW = HERE / "data_cache" / "events_raw.csv"
SAMPLE = HERE / "sample_data" / "rees46_sample.csv.gz"
HEADER = "event_time,event_type,product_id,category_id,category_code,brand,price,user_id,user_session"


def total_size() -> int:
    """Total byte size of the remote file, from a 1-byte ranged GET's Content-Range."""
    req = urllib.request.Request(URL, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return int(resp.headers["Content-Range"].split("/")[1])


def fetch_range(start: int, length: int) -> bytes:
    req = urllib.request.Request(
        URL, headers={"Range": f"bytes={start}-{start + length - 1}"}
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.read()


def whole_lines(chunk: bytes, *, at_file_start: bool) -> list[str]:
    """Drop the partial first line (unless it starts the file) and the partial last line."""
    text = chunk.decode("utf-8", errors="replace")
    lines = text.split("\n")
    lines = lines[:-1]  # the last element is a partial (or empty) line
    if not at_file_start:
        lines = lines[1:]  # the first is (probably) cut mid-line
    return [ln for ln in lines if ln.strip() and not ln.startswith("event_time,")]


def download(chunks: int, chunk_mb: int) -> int:
    size = total_size()
    length = chunk_mb * 1024 * 1024
    RAW.parent.mkdir(exist_ok=True)
    n_rows = 0
    with RAW.open("w", encoding="utf-8", newline="\n") as f:
        f.write(HEADER + "\n")
        for i in range(chunks):
            # spread starts across the file, leaving room so the last chunk fits
            start = (size - length) * i // max(chunks - 1, 1)
            lines = whole_lines(fetch_range(start, length), at_file_start=start == 0)
            f.write("\n".join(lines) + "\n")
            n_rows += len(lines)
            print(
                f"chunk {i + 1}/{chunks}: offset {start:,}, {len(lines):,} rows "
                f"({lines[0].split(',')[0]} .. {lines[-1].split(',')[0]})"
            )
    print(f"{n_rows:,} rows -> {RAW}")
    return n_rows


def session_bucket(session_id: str, modulo: int) -> int:
    return int(hashlib.sha256(session_id.encode()).hexdigest()[:8], 16) % modulo


def make_sample(keep_one_in: int) -> None:
    kept = 0
    total = 0
    SAMPLE.parent.mkdir(exist_ok=True)
    with (
        RAW.open(encoding="utf-8") as src,
        gzip.GzipFile(SAMPLE, "wb", mtime=0) as raw_out,  # mtime=0: reproducible bytes
    ):
        header = src.readline()
        raw_out.write(header.encode())
        for line in src:
            total += 1
            session = line.rstrip("\n").rsplit(",", 1)[-1]
            if session_bucket(session, keep_one_in) == 0:
                raw_out.write(line.encode())
                kept += 1
    print(f"{kept:,}/{total:,} rows kept (1 in {keep_one_in} sessions) -> {SAMPLE}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--chunks", type=int, default=8, help="slices spread over the file")
    p.add_argument("--chunk-mb", type=int, default=8)
    p.add_argument("--make-sample", action="store_true")
    p.add_argument("--keep-one-in", type=int, default=4)
    p.add_argument(
        "--skip-download", action="store_true", help="reuse data_cache/events_raw.csv"
    )
    args = p.parse_args(argv)
    try:
        if not args.skip_download:
            download(args.chunks, args.chunk_mb)
        if args.make_sample:
            if not RAW.exists():
                print("error: run without --skip-download first", file=sys.stderr)
                return 2
            make_sample(args.keep_one_in)
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
