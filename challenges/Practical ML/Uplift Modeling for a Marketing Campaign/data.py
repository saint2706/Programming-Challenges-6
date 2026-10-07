"""Criteo Uplift Prediction Dataset v2.1 (Diemert et al., AdKDD 2018).

13,979,592 rows from randomized incrementality tests: 12 anonymized dense features,
``treatment`` (85% treated), ``visit``, ``conversion`` and ``exposure``. License
CC BY-NC-SA 4.0 (non-commercial). The ``go.criteo.net`` URL that scikit-uplift documents
returns 404 now; this uses Criteo's own HuggingFace mirror. Downloaded once into
``data/`` (gitignored), sampled to 2M rows with a fixed seed and cached as parquet.

``treatment`` is the intervention (intent-to-treat). ``exposure`` is a post-treatment
variable (did the ad actually get shown) and is never a feature.
"""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
RAW_NAME = "criteo-research-uplift-v2.1.csv.gz"
RAW_URL = (
    f"https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/{RAW_NAME}"
)
SAMPLE_NAME = "criteo_sample.parquet"
SAMPLE_ROWS = 2_000_000

FEATURES = [f"f{i}" for i in range(12)]
TREATMENT = "treatment"
OUTCOMES = ("visit", "conversion")
LEAKY = ("exposure",)
SPLIT_FRACS = (0.6, 0.2, 0.2)
MAX_ABS_SMD = 0.1


class RandomizationError(ValueError):
    """The treatment arms differ in their features: the data is not a clean RCT."""


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


def sample_raw(raw: Path, n: int, seed: int) -> pl.DataFrame:
    """Read the gz CSV with compact dtypes and take a seeded random sample of ``n`` rows."""
    dtypes = {
        **{f: pl.Float32 for f in FEATURES},
        TREATMENT: pl.Int8,
        "conversion": pl.Int8,
        "visit": pl.Int8,
        "exposure": pl.Int8,
    }
    df = pl.read_csv(raw, schema_overrides=dtypes)
    return df.sample(n=n, seed=seed) if df.height > n else df


def fetch(
    data_dir: Path = DATA_DIR, n: int = SAMPLE_ROWS, seed: int = 0, url: str = RAW_URL
) -> Path:
    """Download (once), sample and cache; returns the parquet path."""
    target = data_dir / SAMPLE_NAME
    if target.exists():
        return target
    raw = download(url, data_dir / RAW_NAME)
    sample_raw(raw, n, seed).write_parquet(target)
    return target


def load(data_dir: Path = DATA_DIR) -> pl.DataFrame:
    path = data_dir / SAMPLE_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python cli.py fetch` first")
    return pl.read_parquet(path)


def balance(df: pl.DataFrame) -> pl.DataFrame:
    """Standardized mean difference of each feature between treated and control."""
    treated = df[TREATMENT].to_numpy().astype(bool)
    rows = []
    for f in FEATURES:
        x = df[f].to_numpy().astype(np.float64)
        a, b = x[treated], x[~treated]
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        smd = 0.0 if pooled == 0 else float((a.mean() - b.mean()) / pooled)
        rows.append({"feature": f, "smd": smd})
    return pl.DataFrame(rows)


def smd_se(n1: int, n0: int, smd: float = 0.0) -> float:
    """Standard error of a standardized mean difference between groups of size ``n1`` and ``n0``."""
    return float(np.sqrt((n1 + n0) / (n1 * n0) + smd**2 / (2 * (n1 + n0))))


def assert_randomized(
    df: pl.DataFrame, max_abs_smd: float = MAX_ABS_SMD, z: float = 4.0
) -> pl.DataFrame:
    """Return the balance table, or raise on a *gross* imbalance between the arms.

    A feature fails only if ``|SMD|`` exceeds both ``max_abs_smd`` and ``z`` standard errors, so
    the check neither false-alarms on small clean samples (where 0.1 is about two standard
    errors) nor needs tuning for large ones. It is a gate against a broken assignment, not a
    proof of randomization: a feature-by-feature pass can hide a weak joint dependence, which
    ``propensity.diagnose`` measures and the evaluation then adjusts for.
    """
    bal = balance(df)
    n1 = int((df[TREATMENT] == 1).sum())
    n0 = df.height - n1
    bad = bal.filter(
        pl.col("smd").abs() > max_abs_smd,
        pl.col("smd").abs() > z * smd_se(n1, n0),
    )
    if bad.height:
        worst = ", ".join(f"{r['feature']} ({r['smd']:+.3f})" for r in bad.to_dicts())
        raise RandomizationError(
            f"treatment arms are imbalanced beyond |SMD| > {max_abs_smd} "
            f"(and {z:g} standard errors): {worst}"
        )
    return bal


def split(df: pl.DataFrame, seed: int = 0, fracs=SPLIT_FRACS, outcome: str = "visit"):
    """Random train/val/test split, stratified on treatment x outcome."""
    key = df[TREATMENT].to_numpy().astype(int) * 2 + df[outcome].to_numpy().astype(int)
    rng = np.random.default_rng(seed)
    part = np.empty(len(df), dtype=np.int8)
    cuts = np.cumsum(fracs)[:-1]
    for k in np.unique(key):
        idx = rng.permutation(np.flatnonzero(key == k))
        for p, chunk in enumerate(np.split(idx, (cuts * len(idx)).astype(int))):
            part[chunk] = p
    return tuple(df.filter(pl.Series(part == p)) for p in range(len(fracs)))


def xy(df: pl.DataFrame, outcome: str = "visit", features=None):
    """``(X, t, y)`` arrays; refuses treatment, outcomes and post-treatment columns as features."""
    features = list(FEATURES if features is None else features)
    forbidden = {TREATMENT, *OUTCOMES, *LEAKY} & set(features)
    if forbidden:
        raise ValueError(
            f"cannot use treatment/outcome/post-treatment columns as features: {sorted(forbidden)}"
        )
    return (
        df.select(features).to_numpy().astype(np.float32),
        df[TREATMENT].to_numpy().astype(np.int8),
        df[outcome].to_numpy().astype(np.int8),
    )


def propensity(df: pl.DataFrame) -> float:
    """Constant propensity of an RCT: the treated share."""
    return float(df[TREATMENT].mean())
