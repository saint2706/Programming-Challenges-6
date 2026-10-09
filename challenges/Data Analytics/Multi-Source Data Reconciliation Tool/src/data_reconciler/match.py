"""Deciding which row of one source is the same thing as which row of the other.

Keys are tried in order (a cascade). Each stage only looks at rows no earlier stage resolved, and
only matches a key that appears **exactly once on each side**. A key that is on several rows of
either side is ambiguous: guessing would silently pair the wrong rows, so every row involved is
reported as a duplicate key instead and takes no further part.

Key values are compared after trimming and upper-casing, so ``" lhr"`` finds ``"LHR"``; an empty
value is no key at all.

Every row of each source ends with exactly one status:

``matched``        paired with one row on the other side (``stage`` says by which key)
``only_left``      has a key, nothing on the other side has it   (``only_right`` likewise)
``duplicate_key``  its key is on several rows, on this side or the other, so no pairing was made
``unkeyed``        no value for any key, so it could not be looked up at all
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from data_reconciler.config import Key
from data_reconciler.load import ROW

STATUSES = ("matched", "only_left", "only_right", "duplicate_key", "unkeyed")


@dataclass(frozen=True)
class MatchResult:
    pairs: (
        pl.DataFrame
    )  # left_row, right_row, stage (0-based key index), key (the value matched on)
    rows: pl.DataFrame  # side, row, status, stage (matched or duplicate only), key


def normalize_key(column: str) -> pl.Expr:
    key = pl.col(column).cast(pl.Utf8).str.strip_chars().str.to_uppercase()
    return pl.when(key == "").then(None).otherwise(key).alias(column)


def _candidates(frame: pl.DataFrame, column: str, resolved: set[int]) -> pl.DataFrame:
    rows = frame.select(pl.col(ROW), normalize_key(column).alias("k")).filter(
        pl.col("k").is_not_null()
    )
    if resolved:
        rows = rows.filter(~pl.col(ROW).is_in(list(resolved)))
    return rows.with_columns(n=pl.len().over("k"))


def match(
    left: pl.DataFrame, right: pl.DataFrame, keys: tuple[Key, ...]
) -> MatchResult:
    resolved = {"left": set(), "right": set()}
    own_duplicate: dict[str, dict[int, str]] = {
        "left": {},
        "right": {},
    }  # row -> key repeated on its own side
    pairs: list[pl.DataFrame] = []
    flagged: list[pl.DataFrame] = []  # side, row, stage, key for ambiguous rows

    for stage, key in enumerate(keys):
        cl = _candidates(left, key.left, resolved["left"])
        cr = _candidates(right, key.right, resolved["right"])
        for side, cand in (("left", cl), ("right", cr)):
            for row, k in cand.filter(pl.col("n") > 1).select(ROW, "k").iter_rows():
                own_duplicate[side].setdefault(row, k)
        both = cl.join(cr, on="k", how="inner", suffix="_r")
        unique = both.filter((pl.col("n") == 1) & (pl.col("n_r") == 1))
        pairs.append(
            unique.select(
                pl.col(ROW).alias("left_row"),
                pl.col(f"{ROW}_r").alias("right_row"),
                pl.lit(stage).alias("stage"),
                pl.col("k").alias("key"),
            )
        )
        resolved["left"] |= set(unique[ROW])
        resolved["right"] |= set(unique[f"{ROW}_r"])
        ambiguous_keys = both.filter((pl.col("n") > 1) | (pl.col("n_r") > 1))[
            "k"
        ].unique()
        for side, cand in (("left", cl), ("right", cr)):
            rows = cand.filter(pl.col("k").is_in(ambiguous_keys.implode()))
            flagged.append(
                rows.select(
                    pl.lit(side).alias("side"),
                    pl.col(ROW).alias("row"),
                    pl.lit(stage).alias("stage"),
                    pl.col("k").alias("key"),
                )
            )
            resolved[side] |= set(rows[ROW])

    pair_table = pl.concat(pairs) if pairs else pl.DataFrame()
    flagged_table = pl.concat(flagged)
    status_rows = []
    for side, frame, columns in (
        ("left", left, [k.left for k in keys]),
        ("right", right, [k.right for k in keys]),
    ):
        has_key = pl.any_horizontal(*(normalize_key(c).is_not_null() for c in columns))
        base = frame.select(
            pl.lit(side).alias("side"),
            pl.col(ROW).alias("row"),
            has_key.alias("has_key"),
        )
        matched = pair_table.select(
            pl.col("left_row" if side == "left" else "right_row").alias("row"),
            "stage",
            "key",
        ).with_columns(status=pl.lit("matched"))
        flags = (
            flagged_table.filter(pl.col("side") == side)
            .drop("side")
            .with_columns(status=pl.lit("duplicate_key"))
        )
        dup_own = pl.DataFrame(
            {
                "row": list(own_duplicate[side]),
                "dup_key": list(own_duplicate[side].values()),
            },
            schema={"row": pl.Int64, "dup_key": pl.Utf8},
        )
        out = (
            base.join(matched, on="row", how="left")
            .join(flags, on="row", how="left", suffix="_flag")
            .with_columns(
                status=pl.coalesce("status", "status_flag"),
                stage=pl.coalesce("stage", "stage_flag"),
                key=pl.coalesce("key", "key_flag"),
            )
            .drop("status_flag", "stage_flag", "key_flag")
            .join(dup_own, on="row", how="left")
            .with_columns(
                key=pl.coalesce("key", "dup_key"),
                status=pl.when(pl.col("status").is_not_null())
                .then(pl.col("status"))
                .when(pl.col("dup_key").is_not_null())
                .then(pl.lit("duplicate_key"))
                .when(pl.col("has_key"))
                .then(pl.lit(f"only_{side}"))
                .otherwise(pl.lit("unkeyed")),
            )
            .select("side", "row", "status", "stage", "key")
        )
        status_rows.append(out)
    return MatchResult(pair_table, pl.concat(status_rows).sort("side", "row"))
