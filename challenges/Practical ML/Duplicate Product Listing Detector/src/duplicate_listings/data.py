"""Shopee Product Matching data: fetch, slice by whole duplicate group, split.

Ground truth is Shopee's ``label_group``: two listings with the same group
are duplicates of each other. Every helper here keeps groups intact, because
slicing or splitting *within* a group would silently turn real duplicate pairs
into false negatives (or leak a group across the validation/test boundary).
"""

import functools
import itertools
import random
import time
import zipfile
from pathlib import Path

import polars as pl

from duplicate_listings.paths import project_root

COMPETITION = "shopee-product-matching"
ARCHIVE_NAME = f"{COMPETITION}.zip"
DATA_DIR = project_root() / "data"
COLUMNS = ["posting_id", "image", "image_phash", "title", "label_group"]


def extract_if_zip(path: Path) -> None:
    """Unwrap `path` in place if Kaggle served it as a zip.

    Kaggle's single-file download returns ``train.csv`` as a zip archive with
    the same name (images arrive as plain JPEGs), so sniff the magic bytes
    instead of trusting the extension.
    """
    if not zipfile.is_zipfile(path):
        return
    with zipfile.ZipFile(path) as z:
        payload = z.read(z.namelist()[0])
    path.write_bytes(payload)


@functools.cache
def _kaggle_api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def fetch_catalog(data_dir: Path = DATA_DIR) -> Path:
    """Download ``train.csv`` (34,250 listings) unless it is already cached."""
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / "train.csv"
    if not target.exists():
        _kaggle_api().competition_download_file(
            COMPETITION, "train.csv", path=str(data_dir), quiet=True
        )
        extract_if_zip(target)
    return target


def load_catalog(path: Path) -> pl.DataFrame:
    df = pl.read_csv(path, schema_overrides={"label_group": pl.Int64})
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return df.select(COLUMNS)


def sample_groups(df: pl.DataFrame, n_groups: int, seed: int = 0) -> pl.DataFrame:
    """Keep `n_groups` randomly chosen duplicate groups, every listing of each."""
    groups = sorted(df["label_group"].unique().to_list())
    chosen = random.Random(seed).sample(groups, min(n_groups, len(groups)))
    return df.filter(pl.col("label_group").is_in(chosen))


def split_groups(
    df: pl.DataFrame, val_fraction: float = 0.4, seed: int = 0
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split into (validation, test) by group, so no group appears in both."""
    groups = sorted(df["label_group"].unique().to_list())
    random.Random(seed).shuffle(groups)
    n_val = min(max(1, round(len(groups) * val_fraction)), len(groups) - 1)
    val_groups = set(groups[:n_val])
    return (
        df.filter(pl.col("label_group").is_in(val_groups)),
        df.filter(~pl.col("label_group").is_in(val_groups)),
    )


def true_pairs(df: pl.DataFrame) -> set[tuple[str, str]]:
    """Every ground-truth duplicate pair as an ordered ``(a, b)`` with a < b."""
    pairs: set[tuple[str, str]] = set()
    for _, ids in df.group_by("label_group").agg(pl.col("posting_id")).iter_rows():
        pairs.update(itertools.combinations(sorted(ids), 2))
    return pairs


def _is_rate_limited(exc: Exception) -> bool:
    return getattr(getattr(exc, "response", None), "status_code", None) == 429


def _retry_after(exc: Exception) -> float:
    """Seconds the server asked us to wait (``Retry-After``), or 0 if absent/unparseable."""
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    try:
        return max(0.0, float(headers.get("Retry-After", 0)))
    except (TypeError, ValueError):
        return 0.0


def fetch_archive(
    data_dir: Path = DATA_DIR, attempts: int = 6, base_delay: float = 30.0
) -> Path:
    """Download the whole competition archive once (a single API request).

    Per-image downloads look cheaper but Kaggle rate-limits them (HTTP 429 after
    a few hundred requests with 8 workers), and the archive is one request. A
    429 -- including a cool-down left over from an earlier burst -- is retried
    with exponential backoff; any other HTTP error is raised immediately. The
    archive stays on disk so re-slicing with another seed costs nothing.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / ARCHIVE_NAME
    if archive.exists() and zipfile.is_zipfile(archive):
        return archive
    for attempt in range(1, attempts + 1):
        try:
            _kaggle_api().competition_download_files(
                COMPETITION, path=str(data_dir), quiet=False
            )
            break
        except Exception as exc:
            if not _is_rate_limited(exc) or attempt == attempts:
                raise
            time.sleep(max(base_delay * 2 ** (attempt - 1), _retry_after(exc)))
    if not zipfile.is_zipfile(archive):
        raise RuntimeError(f"{archive} is not a valid zip after download")
    return archive


def extract_images(
    archive: Path, names: list[str], data_dir: Path = DATA_DIR
) -> list[str]:
    """Extract only `names` from the archive's ``train_images/``; return names not found.

    Resumable: a non-empty existing file is left alone, an empty one (a
    truncated leftover) is re-extracted. Names that could escape the target
    directory are refused and reported as not found.
    """
    img_dir = data_dir / "train_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    with zipfile.ZipFile(archive) as z:
        members = set(z.namelist())
        for name in names:
            member = f"train_images/{name}"
            if "/" in name or "\\" in name or ".." in name or member not in members:
                missing.append(name)
                continue
            dest = img_dir / name
            if dest.exists() and dest.stat().st_size > 0:
                continue
            dest.write_bytes(z.read(member))
    return missing


def fetch_images(names: list[str], data_dir: Path = DATA_DIR) -> list[str]:
    """Make sure every image in `names` is on disk; return the ones that could not be found."""
    img_dir = data_dir / "train_images"
    if all((img_dir / n).exists() and (img_dir / n).stat().st_size > 0 for n in names):
        return []
    archive = data_dir / ARCHIVE_NAME
    if not archive.exists():
        archive = fetch_archive(data_dir)
    return extract_images(archive, names, data_dir)
