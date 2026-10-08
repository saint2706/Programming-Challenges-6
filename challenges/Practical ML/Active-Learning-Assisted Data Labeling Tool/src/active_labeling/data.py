"""Banking77 (Casanueva et al. 2020, CC-BY-4.0): fetch, parse, de-duplicate and split; plus a CSV pool loader.

HuggingFace's copy is a loading script, so the same CSVs are fetched from PolyAI's GitHub into
the gitignored ``data/``. ``train.csv`` has 10,003 rows and ``test.csv`` 3,080; ``categories.json``
lists the 77 intents.
"""

from __future__ import annotations

import json
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from active_labeling.paths import project_root

HERE = project_root()
DATA_DIR = HERE / "data"
BASE_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/"
FILES = ("train.csv", "test.csv", "categories.json")
VAL_SIZE = 2003


def download(url: str, dest: Path) -> Path:
    """Stream ``url`` to ``dest`` (skipped if it exists); size-checked, written atomically."""
    if not url.startswith("https://"):
        raise ValueError(f"refusing to download a non-https url: {url!r}")
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as resp, part.open("wb") as out:
        expected = int(resp.headers.get("Content-Length", 0))
        shutil.copyfileobj(resp, out, 1 << 20)
    if expected and part.stat().st_size != expected:
        part.unlink()
        raise OSError(f"truncated download of {url}: expected {expected} bytes")
    part.replace(dest)
    return dest


def fetch(data_dir: Path = DATA_DIR, base_url: str = BASE_URL) -> Path:
    for name in FILES:
        download(base_url + name, Path(data_dir) / name)
    return Path(data_dir)


@dataclass
class Raw:
    train_texts: list[str]
    train_y: np.ndarray
    test_texts: list[str]
    test_y: np.ndarray
    classes: list[str]


def load_raw(data_dir: Path = DATA_DIR) -> Raw:
    data_dir = Path(data_dir)
    missing = [n for n in FILES if not (data_dir / n).exists()]
    if missing:
        raise FileNotFoundError(
            f"{data_dir} lacks {missing}; run `uv run active-labeling fetch` first"
        )
    classes = json.loads((data_dir / "categories.json").read_text(encoding="utf-8"))
    index = {c: i for i, c in enumerate(classes)}

    def read(name: str):
        df = pl.read_csv(data_dir / name)
        unknown = set(df["category"].unique().to_list()) - set(index)
        if unknown:
            raise ValueError(
                f"{name} has intents not in categories.json: {sorted(unknown)[:3]}"
            )
        y = np.array([index[c] for c in df["category"].to_list()], dtype=np.int64)
        return df["text"].to_list(), y

    train_texts, train_y = read("train.csv")
    test_texts, test_y = read("test.csv")
    return Raw(train_texts, train_y, test_texts, test_y, classes)


def dedupe(texts) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the first occurrence of each distinct text, and how many rows share it."""
    first: dict[str, int] = {}
    counts: dict[str, int] = {}
    for i, t in enumerate(texts):
        if t not in first:
            first[t], counts[t] = i, 0
        counts[t] += 1
    return (
        np.array(list(first.values()), dtype=np.int64),
        np.array([counts[t] for t in first], dtype=np.int64),
    )


@dataclass
class Splits:
    pool_texts: list[str]
    pool_y: np.ndarray
    pool_count: np.ndarray  # exact-duplicate rows each pool text stands for
    val_texts: list[str]
    val_y: np.ndarray
    test_texts: list[str]
    test_y: np.ndarray
    classes: list[str]
    test_overlap: int  # test texts that also occur in train (reported, not removed)
    n_duplicates: int  # train rows collapsed into an earlier identical row


def make_splits(raw: Raw, seed: int = 0, val_size: int = VAL_SIZE) -> Splits:
    """De-duplicate train, then split it stratified by intent into pool and validation.

    Exact duplicates are collapsed first so a strategy cannot gain by picking twins and the
    validation set shares no text with the pool. Test is returned untouched.
    """
    keep, counts = dedupe(raw.train_texts)
    texts = [raw.train_texts[i] for i in keep]
    y = raw.train_y[keep]
    pool_i, val_i = train_test_split(
        np.arange(len(texts)), test_size=val_size, stratify=y, random_state=seed
    )
    pool_i, val_i = np.sort(pool_i), np.sort(val_i)
    train_set = set(texts)
    return Splits(
        pool_texts=[texts[i] for i in pool_i],
        pool_y=y[pool_i],
        pool_count=counts[pool_i],
        val_texts=[texts[i] for i in val_i],
        val_y=y[val_i],
        test_texts=list(raw.test_texts),
        test_y=raw.test_y,
        classes=raw.classes,
        test_overlap=sum(t in train_set for t in raw.test_texts),
        n_duplicates=len(raw.train_texts) - len(keep),
    )


def load_pool_csv(path, text_col: str = "text", label_col: str | None = None):
    """Texts (and optional gold labels) from a CSV; blank and exactly duplicated texts are dropped."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    df = pl.read_csv(path, infer_schema_length=0)  # every column as text
    for col in (text_col, label_col):
        if col is not None and col not in df.columns:
            raise ValueError(f"{path} has no {col!r} column; columns are {df.columns}")
    texts = df[text_col].fill_null("").str.strip_chars().to_list()
    gold = None
    if label_col is not None:
        gold = [
            g or None for g in df[label_col].fill_null("").str.strip_chars().to_list()
        ]
    keep, seen = [], set()
    for i, t in enumerate(texts):
        if t and t not in seen:
            seen.add(t)
            keep.append(i)
    if not keep:
        raise ValueError(f"{path} has no non-blank texts in {text_col!r}")
    return [texts[i] for i in keep], ([gold[i] for i in keep] if gold else None)


def read_classes(path) -> list[str]:
    """One class per line; blank lines ignored."""
    classes = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines()]
    classes = [c for c in classes if c]
    if len(classes) < 2:
        raise ValueError(f"{path}: a class list needs at least two classes")
    if len(set(classes)) != len(classes):
        raise ValueError(f"{path}: duplicate class names")
    return classes
