"""Download one contiguous slice of the real REES46 clickstream without the 14.7 GB file.

The public CSV is sorted by time, so one HTTP Range request returns a clean, gap-free run of real
traffic. That matters here: a replay needs the true arrival order and the true gaps. (DA #9 samples
whole sessions from several slices instead; that is the wrong shape for a stream.)

    uv run python -m stream_dashboard.fetch_data              # ~24 MB -> data/events_raw.csv
    uv run python -m stream_dashboard.fetch_data --make-sample

``--make-sample`` writes the small committed ``sample_data/events.csv.gz`` (seven columns, gzip with
mtime 0 so the bytes are reproducible).

Source: REES46 Marketing Platform "eCommerce behavior data from multi-category store" (Kaggle:
mkechinov/ecommerce-behavior-data-from-multi-category-store), mirrored on Hugging Face. Free to use
for research and education with attribution.
"""

from __future__ import annotations

import argparse
import gzip
import sys
import urllib.request

from stream_dashboard.paths import project_root

URL = (
    "https://huggingface.co/datasets/kevykibbz/"
    "ecommerce-behavior-data-from-multi-category-store_oct-nov_2019/resolve/main/"
    "ecommerce-behavior-data-from-multi-category-store_oct-nov_2019.csv"
)
HEADER = "event_time,event_type,product_id,category_id,category_code,brand,price,user_id,user_session"
KEEP = (
    "event_time",
    "event_type",
    "product_id",
    "brand",
    "price",
    "user_id",
    "user_session",
)


def total_size() -> int:
    req = urllib.request.Request(URL, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return int(resp.headers["Content-Range"].split("/")[1])


def fetch_range(start: int, length: int) -> bytes:
    req = urllib.request.Request(
        URL, headers={"Range": f"bytes={start}-{start + length - 1}"}
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return resp.read()


def whole_lines(chunk: bytes) -> list[str]:
    """Drop the partial first and last lines of a mid-file slice."""
    lines = chunk.decode("utf-8", errors="replace").split("\n")[1:-1]
    return [ln for ln in lines if ln.strip()]


def download(fraction: float, megabytes: int) -> int:
    size = total_size()
    length = megabytes * 1024 * 1024
    start = int((size - length) * fraction)
    lines = whole_lines(fetch_range(start, length))
    raw = project_root() / "data" / "events_raw.csv"
    raw.parent.mkdir(exist_ok=True)
    raw.write_text(
        HEADER + "\n" + "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        f"{len(lines):,} rows ({lines[0].split(',')[0]} .. {lines[-1].split(',')[0]}) -> {raw}"
    )
    return len(lines)


def make_sample(max_rows: int) -> None:
    import polars as pl

    raw = project_root() / "data" / "events_raw.csv"
    out = project_root() / "sample_data" / "events.csv.gz"
    df = pl.read_csv(raw, schema_overrides={"user_session": pl.String}).select(KEEP)
    df = df.head(max_rows)
    out.parent.mkdir(exist_ok=True)
    csv = df.write_csv().encode()
    with gzip.GzipFile(out, "wb", mtime=0) as fh:
        fh.write(csv)
    print(
        f"{df.height:,} rows ({df['event_time'][0]} .. {df['event_time'][-1]}) -> {out} "
        f"({out.stat().st_size / 1e6:.1f} MB)"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--fraction", type=float, default=0.62, help="where in the file to start (0-1)"
    )
    p.add_argument("--megabytes", type=int, default=24)
    p.add_argument("--make-sample", action="store_true")
    p.add_argument("--sample-rows", type=int, default=100_000)
    p.add_argument("--skip-download", action="store_true")
    args = p.parse_args(argv)
    try:
        if not args.skip_download:
            download(args.fraction, args.megabytes)
        if args.make_sample:
            make_sample(args.sample_rows)
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
