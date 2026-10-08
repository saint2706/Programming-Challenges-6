"""Amazon Reviews 2023 (McAuley Lab): fetch, index, de-duplicate, window, sample, extract.

The raw files are one gzip of JSON lines per category and are not time-ordered, and Software alone
holds millions of reviews, so ``prepare`` never holds a category in memory:

1. ``scan_index`` streams a file once and keeps three integers per valid review (physical line
   number, timestamp, de-duplication key);
2. ``dedupe`` drops later copies of the same normalized (title, text) across *both* categories, so
   a duplicate can never straddle two splits;
3. ``choose_windows`` counts calendar months back from the last month both categories have data;
4. ``select_rows`` samples each split uniformly with a fixed seed;
5. ``extract_rows`` streams the file a second time and parses only the sampled lines.

Reviews of one product are correlated, so ``parent_asin`` is kept for cluster bootstraps and the
unseen-product slice.
"""

from __future__ import annotations

import gzip
import json
import urllib.error
import urllib.request
from array import array
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from review_stars.config import Config
from review_stars.text import _clean, dedup_key, is_auto_title

BASE_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/"
    "{category}.jsonl.gz"
)
SPLITS = ("train", "val", "cal", "test", "ood_test", "ood_pool")
CATS = ("in", "ood")


# ---------------------------------------------------------------- download


def _total_size(resp, status: int) -> int:
    headers = resp.headers
    if status == 206:
        tail = headers.get("Content-Range", "").rsplit("/", 1)[-1]
        return int(tail) if tail.isdigit() else 0
    length = str(headers.get("Content-Length", ""))
    return int(length) if length.isdigit() else 0


def download(
    url: str, dest, *, opener=urllib.request.urlopen, chunk: int = 1 << 20
) -> Path:
    """Fetch ``url`` to ``dest`` (skipped if it exists), resuming a ``.part`` file with a Range request.

    The body is size-checked against the server's own length and renamed into place only when
    complete, so a half-downloaded archive is never mistaken for a finished one. A dropped
    connection leaves the partial file behind for the next call to continue.
    """
    if not url.startswith("https://"):
        raise ValueError(f"refusing to download a non-https url: {url!r}")
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    total = 0
    for _attempt in range(2):
        offset = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            resp = opener(urllib.request.Request(url, headers=headers), timeout=60)
        except urllib.error.HTTPError as exc:
            if exc.code != 416 or not offset:
                raise
            span = (exc.headers or {}).get("Content-Range", "")
            if (
                span == f"bytes */{offset}"
            ):  # the partial file is already the whole file
                part.replace(dest)
                return dest
            part.unlink()  # stale or oversized: start over
            continue
        with resp:
            status = getattr(resp, "status", 200)
            if (
                offset and status != 206
            ):  # the server ignored Range and is sending everything
                offset = 0
            total = _total_size(resp, status)
            with part.open("ab" if offset else "wb") as out:
                while block := resp.read(chunk):
                    out.write(block)
        break
    if total and part.stat().st_size != total:
        raise OSError(
            f"truncated download of {url}: have {part.stat().st_size} of {total} bytes; "
            "run the command again to resume"
        )
    part.replace(dest)
    return dest


def raw_path(data_dir, category: str) -> Path:
    return Path(data_dir) / "raw" / f"{category}.jsonl.gz"


def fetch(data_dir, cfg: Config, *, opener=urllib.request.urlopen) -> dict[str, Path]:
    """Download both categories into ``data_dir/raw``; returns ``{category: path}``."""
    return {
        cat: download(
            BASE_URL.format(category=cat), raw_path(data_dir, cat), opener=opener
        )
        for cat in (cfg.in_domain, cfg.out_domain)
    }


# ---------------------------------------------------------------- scanning


def iter_lines(path):
    """The physical lines of a ``.jsonl.gz``; a truncated archive is an error, not fewer rows."""
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            yield from fh
    except EOFError as exc:
        raise OSError(
            f"{path} is a truncated gzip archive; delete it and fetch again"
        ) from exc
    except gzip.BadGzipFile as exc:
        raise OSError(f"{path} is not a gzip archive: {exc}") from exc


def _valid_rating(x) -> bool:
    """A whole star count 1-5 (NaN is never ``in`` the tuple, so it is rejected too)."""
    return (
        isinstance(x, (int, float)) and not isinstance(x, bool) and x in (1, 2, 3, 4, 5)
    )


def _valid_ts(x) -> bool:
    """A positive timestamp (NaN fails ``> 0``)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x > 0


@dataclass
class Index:
    df: pl.DataFrame  # row (physical line number), ts (ms), key (63-bit dedup key)
    stats: dict


def scan_index(path) -> Index:
    """One streaming pass: keep ``(row, ts, key)`` of every usable review, count what is dropped."""
    rows, tss, keys = array("q"), array("q"), array("q")
    stats = dict.fromkeys(
        ("lines", "kept", "unparseable", "bad_rating", "bad_timestamp", "empty_text"), 0
    )
    for n, line in enumerate(iter_lines(path)):
        stats["lines"] += 1
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            r = None
        if not isinstance(r, dict):
            stats["unparseable"] += 1
        elif not _valid_rating(r.get("rating")):
            stats["bad_rating"] += 1
        elif not _valid_ts(r.get("timestamp")):
            stats["bad_timestamp"] += 1
        elif not _clean(r.get("text")):
            stats["empty_text"] += 1
        else:
            rows.append(n)
            tss.append(int(r["timestamp"]))
            keys.append(dedup_key(r.get("title"), r.get("text")))
            stats["kept"] += 1
    df = pl.DataFrame(
        {
            "row": np.frombuffer(rows, dtype=np.int64),
            "ts": np.frombuffer(tss, dtype=np.int64),
            "key": np.frombuffer(keys, dtype=np.int64),
        }
    )
    return Index(df, stats)


def dedupe(
    frames: dict[str, pl.DataFrame],
) -> tuple[dict[str, pl.DataFrame], dict[str, int]]:
    """Keep the earliest review of each ``key`` across all frames (ties: frame order, then row)."""
    order = {name: i for i, name in enumerate(frames)}
    pooled = pl.concat(
        [
            f.select("row", "ts", "key").with_columns(pl.lit(order[n]).alias("cat"))
            for n, f in frames.items()
        ]
    )
    kept = pooled.sort(["ts", "cat", "row"]).unique(
        subset="key", keep="first", maintain_order=True
    )
    out, removed = {}, {}
    for name, i in order.items():
        out[name] = (
            kept.filter(pl.col("cat") == i).select("row", "ts", "key").sort("row")
        )
        removed[name] = frames[name].height - out[name].height
    return out, removed


# ---------------------------------------------------------------- months and windows


def _mi(label: str) -> int:
    """'2023-03' -> months since 1970-01."""
    return int(np.datetime64(label, "M").astype(int))


def _label(mi: int) -> str:
    return str(np.datetime64(int(mi), "M"))


def _ms(mi: int) -> int:
    """First millisecond of month ``mi``."""
    return int(np.datetime64(int(mi), "M").astype("datetime64[ms]").astype(np.int64))


def month_counts(df: pl.DataFrame) -> dict[str, int]:
    """Reviews per calendar month ('YYYY-MM'), in order."""
    months = df["ts"].to_numpy().astype("datetime64[ms]").astype("datetime64[M]")
    names, counts = np.unique(months, return_counts=True)
    return {str(n): int(c) for n, c in zip(names, counts, strict=True)}


@dataclass(frozen=True)
class Windows:
    end: str
    test: tuple[str, str]
    cal: tuple[str, str]
    val: tuple[str, str]
    train_before: str
    cal_months: int
    val_months: int
    notes: list = field(default_factory=list)

    def bounds(self, split: str) -> tuple[int, int]:
        """Half-open ``[lo, hi)`` epoch-millisecond bounds of an in-domain split."""
        if split == "train":
            return 0, _ms(_mi(self.train_before))
        span = {"val": self.val, "cal": self.cal, "test": self.test}[split]
        return _ms(_mi(span[0])), _ms(_mi(span[1]) + 1)

    def split_of(self, ts: int) -> str | None:
        """Which split a timestamp falls in; ``None`` after the common end (used by nobody)."""
        if ts >= self.bounds("test")[1]:
            return None
        for split in ("test", "cal", "val"):
            if ts >= self.bounds(split)[0]:
                return split
        return "train"

    def to_dict(self) -> dict:
        return {
            "end": self.end,
            "test": list(self.test),
            "cal": list(self.cal),
            "val": list(self.val),
            "train_before": self.train_before,
            "cal_months": self.cal_months,
            "val_months": self.val_months,
            "notes": list(self.notes),
        }


def _widen(base: int) -> list[int]:
    return [base] + [m for m in (9, 12) if m > base]


def choose_windows(
    counts_in: dict[str, int], counts_ood: dict[str, int], cfg: Config
) -> Windows:
    """Test = last ``test_months`` months, then calibration, then validation, counted back from the
    last month in which *both* categories have at least ``min_month_rows`` reviews.

    A calibration or validation window holding fewer rows than its target is widened to 9, then 12
    months; if it is still short the shortfall is recorded in ``notes``, never hidden.
    """
    ok = [
        _mi(m)
        for m, n in counts_in.items()
        if n >= cfg.min_month_rows and counts_ood.get(m, 0) >= cfg.min_month_rows
    ]
    if not ok:
        raise ValueError(
            f"no month has at least {cfg.min_month_rows} reviews in both categories"
        )
    end = max(ok)
    by_month = {_mi(m): n for m, n in counts_in.items()}

    def rows(lo, hi):
        return sum(n for m, n in by_month.items() if lo <= m <= hi)

    notes: list[str] = []
    test_lo = end - cfg.test_months + 1

    def pick(name, base, hi, target):
        chosen = base
        for months in _widen(base):
            chosen = months
            if rows(hi - months + 1, hi) >= target:
                break
        have = rows(hi - chosen + 1, hi)
        if chosen != base:
            notes.append(
                f"{name} window widened from {base} to {chosen} months (target {target} rows)"
            )
        if have < target:
            notes.append(
                f"{name} window still short at {chosen} months: {have} of {target} rows"
            )
        return chosen

    cal_m = pick("cal", cfg.cal_months, test_lo - 1, cfg.n_cal)
    cal_lo = test_lo - cal_m
    val_m = pick("val", cfg.val_months, cal_lo - 1, cfg.n_val)
    val_lo = cal_lo - val_m
    return Windows(
        end=_label(end),
        test=(_label(test_lo), _label(end)),
        cal=(_label(cal_lo), _label(test_lo - 1)),
        val=(_label(val_lo), _label(cal_lo - 1)),
        train_before=_label(val_lo),
        cal_months=cal_m,
        val_months=val_m,
        notes=notes,
    )


# ---------------------------------------------------------------- sampling and extraction


def _split_codes(ts: np.ndarray, w: Windows) -> np.ndarray:
    """Per-row split name ('' after the common end)."""
    out = np.full(len(ts), "", dtype="<U5")
    for split in ("train", "val", "cal", "test"):
        lo, hi = w.bounds(split)
        out[(ts >= lo) & (ts < hi)] = split
    return out


def select_rows(
    index_in: pl.DataFrame, index_ood: pl.DataFrame, w: Windows, cfg: Config
) -> tuple[pl.DataFrame, dict]:
    """Uniform fixed-seed samples per split: ``(cat, row, split)`` and what was available."""
    targets = {
        "train": cfg.n_train,
        "val": cfg.n_val,
        "cal": cfg.n_cal,
        "test": cfg.n_test,
        "ood_test": cfg.n_ood,
        "ood_pool": cfg.n_ood_pool,
    }
    in_split = _split_codes(index_in["ts"].to_numpy(), w)
    ood_split = _split_codes(index_ood["ts"].to_numpy(), w)
    pools = {
        "train": ("in", index_in, in_split == "train"),
        "val": ("in", index_in, in_split == "val"),
        "cal": ("in", index_in, in_split == "cal"),
        "test": ("in", index_in, in_split == "test"),
        "ood_test": ("ood", index_ood, ood_split == "test"),
        "ood_pool": ("ood", index_ood, ood_split == "cal"),
    }
    parts, stats = [], {}
    for code, split in enumerate(SPLITS):
        cat, df, mask = pools[split]
        rows = df["row"].to_numpy()[mask]
        k = min(targets[split], len(rows))
        rng = np.random.default_rng((cfg.data_seed, code))
        chosen = np.sort(rng.choice(rows, size=k, replace=False)) if k else rows[:0]
        parts.append(
            pl.DataFrame(
                {
                    "cat": [cat] * k,
                    "row": chosen.astype(np.int64),
                    "split": [split] * k,
                },
                schema={"cat": pl.Utf8, "row": pl.Int64, "split": pl.Utf8},
            )
        )
        stats[split] = {
            "target": targets[split],
            "available": len(rows),
            "selected": int(k),
        }
    return pl.concat(parts), stats


def extract_rows(path, rows) -> pl.DataFrame:
    """Parse only the physical lines in ``rows`` (a second streaming pass)."""
    wanted = {int(r) for r in rows}
    cols = (
        "row",
        "rating",
        "title",
        "text",
        "ts",
        "parent_asin",
        "verified",
        "helpful_vote",
    )
    out = {k: [] for k in cols}
    for n, line in enumerate(iter_lines(path)):
        if not wanted:
            break
        if n not in wanted:
            continue
        r = json.loads(line)
        out["row"].append(n)
        out["rating"].append(int(r["rating"]))
        out["title"].append(r.get("title") or "")
        out["text"].append(r.get("text") or "")
        out["ts"].append(int(r["timestamp"]))
        out["parent_asin"].append(r.get("parent_asin") or r.get("asin") or "")
        out["verified"].append(bool(r.get("verified_purchase")))
        out["helpful_vote"].append(int(r.get("helpful_vote") or 0))
        wanted.discard(n)
    if wanted:
        raise ValueError(
            f"{path} no longer has {len(wanted)} of the indexed lines; was the file replaced?"
        )
    return pl.DataFrame(
        out,
        schema={
            "row": pl.Int64,
            "rating": pl.Int8,
            "title": pl.Utf8,
            "text": pl.Utf8,
            "ts": pl.Int64,
            "parent_asin": pl.Utf8,
            "verified": pl.Boolean,
            "helpful_vote": pl.Int32,
        },
    )


# ---------------------------------------------------------------- prepare

COLUMNS = (
    "id", "cat", "split", "row", "rating", "title", "text", "ts",
    "parent_asin", "verified", "helpful_vote",
)  # fmt: skip
CONFIG_KEYS = (
    "in_domain", "out_domain", "n_train", "n_val", "n_cal", "n_test", "n_ood", "n_ood_pool",
    "test_months", "cal_months", "val_months", "min_month_rows", "data_seed", "text_mode",
)  # fmt: skip


def prepare(paths: dict[str, Path], out_dir, cfg: Config) -> dict:
    """Index, de-duplicate, window, sample and extract; writes ``reviews.parquet`` and
    ``windows.json`` into ``out_dir`` and returns the summary (which ``windows.json`` also holds)."""
    out_dir = Path(out_dir)
    names = {"in": cfg.in_domain, "ood": cfg.out_domain}
    index = {cat: scan_index(paths[names[cat]]) for cat in CATS}
    deduped, removed = dedupe({cat: index[cat].df for cat in CATS})
    counts = {cat: month_counts(deduped[cat]) for cat in CATS}
    windows = choose_windows(counts["in"], counts["ood"], cfg)
    selection, split_stats = select_rows(deduped["in"], deduped["ood"], windows, cfg)

    frames = []
    for cat in CATS:
        sel = selection.filter(pl.col("cat") == cat)
        got = extract_rows(paths[names[cat]], sel["row"].to_numpy())
        frames.append(
            got.join(sel.select("row", "split"), on="row").with_columns(
                pl.lit(cat).alias("cat")
            )
        )
    order = {s: i for i, s in enumerate(SPLITS)}
    reviews = (
        pl.concat(frames)
        .with_columns((pl.col("cat") + ":" + pl.col("row").cast(pl.Utf8)).alias("id"))
        .with_columns(
            pl.col("split").replace_strict(order, return_dtype=pl.Int8).alias("_o")
        )
        .sort(["_o", "ts", "row"])
        .select(COLUMNS)
    )
    # Amazon stopped pre-filling "Five Stars" titles around 2019, so this share differs wildly by
    # era: it is recorded per category and, because splits are time windows, per split
    auto = np.array([is_auto_title(t) for t in reviews["title"]], dtype=bool)
    cats, split_col = reviews["cat"].to_numpy(), reviews["split"].to_numpy()

    def share_of(mask):
        return float(auto[mask].mean()) if mask.any() else None

    share = {names[c]: share_of(cats == c) for c in CATS}
    share_by_split = {s: share_of(split_col == s) for s in SPLITS}
    out_dir.mkdir(parents=True, exist_ok=True)
    reviews.write_parquet(out_dir / "reviews.parquet")
    summary = {
        "config": {k: getattr(cfg, k) for k in CONFIG_KEYS},
        "windows": windows.to_dict(),
        "month_counts": {names[cat]: counts[cat] for cat in CATS},
        "scan": {names[cat]: index[cat].stats for cat in CATS},
        "duplicates_removed": {names[cat]: removed[cat] for cat in CATS},
        "splits": split_stats,
        "auto_title_share": share,
        "auto_title_share_by_split": share_by_split,
    }
    (out_dir / "windows.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8"
    )
    return summary


def load_prepared(out_dir) -> pl.DataFrame:
    path = Path(out_dir) / "reviews.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist; run `review-stars prepare` first"
        )
    return pl.read_parquet(path)
