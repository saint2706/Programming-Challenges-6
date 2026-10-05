"""Rank one mailbox-day of held-out mail: shared by the CLI and the Streamlit page."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from data import DATA_DIR
from explain import reasons
from features import LOCAL_OFFSET
from models import time_split
from pipeline import DATASET_FILE, Artifacts, load_artifacts

EXPLAINED_MODELS = ("lgbm_meta_text", "lgbm_meta")
REASONS_PER_MESSAGE = 4


@dataclass
class InboxItem:
    rank: int
    message_id: str
    sender: str
    subject: str
    raw_score: float
    probability: float
    reasons: list[tuple[str, float]]
    acted: bool
    kind: str | None


def load_context(data_dir: Path = DATA_DIR) -> tuple[Artifacts, pl.DataFrame]:
    """Trained models plus the prepared dataset (both written by ``train``)."""
    return load_artifacts(data_dir), pl.read_parquet(Path(data_dir) / DATASET_FILE)


def mailboxes(df: pl.DataFrame) -> list[str]:
    return sorted(set(df["mailbox"]))


def _local_day() -> pl.Expr:
    return (pl.col("date") + LOCAL_OFFSET).dt.date()


def days_with_mail(df: pl.DataFrame, mailbox: str) -> dict[date, int]:
    """Held-out (test-period) local days with received mail -> message count."""
    test = time_split(df)[2].filter(pl.col("mailbox") == mailbox)
    counts = test.group_by(_local_day().alias("day")).len().sort("day")
    return dict(zip(counts["day"].to_list(), counts["len"].to_list(), strict=True))


def rank_inbox(
    art: Artifacts,
    df: pl.DataFrame,
    mailbox: str,
    day: date,
    top: int = 10,
    model: str = "lgbm_meta_text",
) -> list[InboxItem]:
    """The day's received mail, most-likely-to-be-answered first, each with reasons."""
    if model not in EXPLAINED_MODELS:
        raise ValueError(f"model must be one of {EXPLAINED_MODELS}, not {model!r}")
    if mailbox not in mailboxes(df):
        raise ValueError(f"unknown mailbox {mailbox!r}; choose from {mailboxes(df)}")
    test = time_split(df)[2]
    sub = test.filter((pl.col("mailbox") == mailbox) & (_local_day() == day))
    if sub.is_empty():
        raise ValueError(
            f"no mail for {mailbox} on {day.isoformat()} in the test period"
        )

    raw = art.models.predict(model, sub)
    prob = art.calibrators[model].predict(raw)
    order = np.argsort(-raw, kind="stable")[:top]
    chosen = sub[order.tolist()]
    with_text = model == "lgbm_meta_text"
    names = art.models.meta_text_names if with_text else art.models.meta_names
    why = reasons(
        getattr(art.models, model),
        art.models.matrix(chosen, with_text=with_text),
        names,
        k=REASONS_PER_MESSAGE,
    )
    return [
        InboxItem(
            rank=r + 1,
            message_id=row["message_id"],
            sender=row["sender"],
            subject=row["subject"],
            raw_score=float(raw[i]),
            probability=float(prob[i]),
            reasons=why[r],
            acted=bool(row["acted"]),
            kind=row["kind"],
        )
        for r, (i, row) in enumerate(
            zip(order, chosen.iter_rows(named=True), strict=True)
        )
    ]
