"""Glue shared by the CLI and the Streamlit app: slice -> embed -> split -> evaluate.

Everything is cached under ``data/`` so each stage only does work once:
``slice.csv`` (the chosen listings), ``embeddings.npz``, ``lancedb/`` and
``report.json`` (the tuned operating points the app and ``dedupe`` read back).
"""

import dataclasses
import json
import warnings
from pathlib import Path

import polars as pl

from duplicate_listings import data, embed, evaluate

SLICE_FILE = "slice.csv"
EMBEDDINGS_FILE = "embeddings.npz"
REPORT_FILE = "report.json"
DB_DIR = "lancedb"


def build_slice(
    n_groups: int, seed: int, data_dir: Path = data.DATA_DIR
) -> pl.DataFrame:
    """Pick `n_groups` whole duplicate groups, fetch their images, remember the choice."""
    catalog = data.load_catalog(data.fetch_catalog(data_dir))
    df = data.sample_groups(catalog, n_groups, seed)
    missing = data.fetch_images(df["image"].to_list(), data_dir)
    if missing:
        warnings.warn(
            f"{len(missing)} images were not found in the archive; they get no image similarity",
            stacklevel=2,
        )
    df.write_csv(data_dir / SLICE_FILE)
    return df


def load_slice(data_dir: Path = data.DATA_DIR) -> pl.DataFrame:
    path = data_dir / SLICE_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run the `fetch` command first")
    return data.load_catalog(path)


def default_encoders() -> tuple[embed.TextEncoder, embed.ImageEncoder]:
    return embed.MultilingualTextEncoder(), embed.SigLIP2ImageEncoder()


def embed_slice(
    df: pl.DataFrame,
    data_dir: Path = data.DATA_DIR,
    text_encoder: embed.TextEncoder | None = None,
    image_encoder: embed.ImageEncoder | None = None,
) -> embed.Embeddings:
    """Embeddings for `df`, from ``embeddings.npz`` if it covers exactly these listings."""
    cache = data_dir / EMBEDDINGS_FILE
    cached = embed.load_embeddings(cache, df["posting_id"].to_list())
    if cached is not None:
        return cached
    if text_encoder is None or image_encoder is None:
        text_encoder, image_encoder = default_encoders()
    emb = embed.embed_catalog(df, data_dir, text_encoder, image_encoder)
    embed.save_embeddings(cache, emb)
    return emb


def split_slice(
    df: pl.DataFrame, emb: embed.Embeddings, val_fraction: float = 0.4, seed: int = 0
) -> tuple[
    tuple[pl.DataFrame, embed.Embeddings], tuple[pl.DataFrame, embed.Embeddings]
]:
    """(validation, test), disjoint by group, each with its embeddings row-aligned."""
    row_of = {pid: i for i, pid in enumerate(emb.posting_ids)}
    parts = []
    for part in data.split_groups(df, val_fraction, seed):
        parts.append(
            (part, emb.subset([row_of[p] for p in part["posting_id"].to_list()]))
        )
    return parts[0], parts[1]


def prepare_splits(
    df: pl.DataFrame,
    emb: embed.Embeddings,
    data_dir: Path = data.DATA_DIR,
    k: int = 10,
    val_fraction: float = 0.4,
    seed: int = 0,
) -> tuple[evaluate.SplitData, evaluate.SplitData]:
    """Index both splits in LanceDB and score their candidate pairs (no tuning)."""
    (vdf, vemb), (tdf, temb) = split_slice(df, emb, val_fraction, seed)
    val = evaluate.prepare_split(vdf, vemb, data_dir / DB_DIR, "val", k=k)
    test = evaluate.prepare_split(tdf, temb, data_dir / DB_DIR, "test", k=k)
    return val, test


def run_evaluation(
    df: pl.DataFrame,
    emb: embed.Embeddings,
    data_dir: Path = data.DATA_DIR,
    k: int = 10,
    val_fraction: float = 0.4,
    seed: int = 0,
) -> tuple[evaluate.EvalReport, evaluate.SplitData, evaluate.SplitData]:
    val, test = prepare_splits(df, emb, data_dir, k, val_fraction, seed)
    return evaluate.evaluate(val, test), val, test


def save_report(path: Path, report: evaluate.EvalReport, settings: dict) -> None:
    payload = {
        "modes": {name: dataclasses.asdict(m) for name, m in report.modes.items()},
        "candidate_recall": report.candidate_recall,
        "n_listings": report.n_listings,
        "n_true_pairs": report.n_true_pairs,
        "n_candidates": report.n_candidates,
        "settings": settings,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_report(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run the `evaluate` command first to tune the thresholds"
        )
    return json.loads(path.read_text())
