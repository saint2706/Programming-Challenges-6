"""Run a reconciliation job: load, match, compare every field of every pair, summarize."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from data_reconciler import compare as cmp
from data_reconciler.config import Config, Field
from data_reconciler.load import ROW, load_split, require_columns
from data_reconciler.match import match, normalize_key


@dataclass(frozen=True)
class Reconciliation:
    config: Config
    left: pl.DataFrame
    right: pl.DataFrame
    rows: (
        pl.DataFrame
    )  # every in-scope row of both sources with its status (see match.STATUSES)
    pairs: (
        pl.DataFrame
    )  # one row per matched pair, with <field>__left/right/category/metric/note
    differences: pl.DataFrame  # long form: one row per field that differs on a pair
    crosswalks: dict[str, pl.DataFrame]
    summary: dict
    left_excluded: (
        pl.DataFrame
    )  # rows the left scope dropped, kept so unmatched rows can be explained
    right_excluded: pl.DataFrame


def _value(
    frame_values: dict[str, list], columns: tuple[str, ...], prefix: str, i: int
):
    vals = [frame_values[f"{prefix}.{c}"][i] for c in columns]
    return vals[0] if len(vals) == 1 else tuple(vals)


def _show(value) -> str | None:
    if isinstance(value, tuple):
        return (
            None
            if all(v is None for v in value)
            else ", ".join("" if v is None else str(v) for v in value)
        )
    return value


def _compare_pairs(
    config: Config, left: pl.DataFrame, right: pl.DataFrame, pairs: pl.DataFrame
):
    joined = pairs.join(
        left.rename(lambda c: f"L.{c}"),
        left_on="left_row",
        right_on=f"L.{ROW}",
        how="left",
    ).join(
        right.rename(lambda c: f"R.{c}"),
        left_on="right_row",
        right_on=f"R.{ROW}",
        how="left",
    )
    values = {
        c: joined[c].to_list() for c in joined.columns if c.startswith(("L.", "R."))
    }
    columns: dict[str, list] = {}
    crosswalks: dict[str, pl.DataFrame] = {}
    for spec in config.fields:
        lefts = [_value(values, spec.left, "L", i) for i in range(joined.height)]
        rights = [_value(values, spec.right, "R", i) for i in range(joined.height)]
        results, table = cmp.compare_field(spec, lefts, rights)
        columns[f"{spec.name}__left"] = [_show(v) for v in lefts]
        columns[f"{spec.name}__right"] = [_show(v) for v in rights]
        columns[f"{spec.name}__category"] = [r.category for r in results]
        columns[f"{spec.name}__metric"] = [r.metric for r in results]
        columns[f"{spec.name}__note"] = [r.note for r in results]
        if spec.compare == "crosswalk":
            crosswalks[spec.name] = cmp.crosswalk_frame(table)
    wide = pairs.with_columns(
        *(pl.Series(name, vals, dtype=_dtype(name)) for name, vals in columns.items())
    )
    return wide, crosswalks


def _dtype(name: str):
    return pl.Float64 if name.endswith("__metric") else pl.Utf8


VERDICTS = ("identical", "changed", "different_entity")


def _with_verdict(config: Config, wide: pl.DataFrame) -> pl.DataFrame:
    """``identical`` (nothing differs), ``different_entity`` (identity fields disagree, so the key was
    probably reused for another thing), otherwise ``changed``."""
    cats = [pl.col(f"{f.name}__category") for f in config.fields]
    differs = (
        pl.any_horizontal(*(c.is_in(list(cmp.DIFFERENCES)) for c in cats))
        if cats
        else pl.lit(False)
    )
    disagree = (
        pl.sum_horizontal(
            *(pl.col(f"{n}__category") == cmp.MISMATCH for n in config.identity)
        )
        if config.identity
        else pl.lit(0)
    )
    return wide.with_columns(
        verdict=pl.when(~differs)
        .then(pl.lit("identical"))
        .when(disagree >= config.identity_min if config.identity else pl.lit(False))
        .then(pl.lit("different_entity"))
        .otherwise(pl.lit("changed"))
    )


def _long_differences(config: Config, wide: pl.DataFrame) -> pl.DataFrame:
    parts = []
    for spec in config.fields:
        parts.append(
            wide.filter(
                pl.col(f"{spec.name}__category").is_in(list(cmp.DIFFERENCES))
            ).select(
                "left_row",
                "right_row",
                "stage",
                "key",
                "verdict",
                pl.lit(spec.name).alias("field"),
                pl.col(f"{spec.name}__left").alias("left_value"),
                pl.col(f"{spec.name}__right").alias("right_value"),
                pl.col(f"{spec.name}__category").alias("category"),
                pl.col(f"{spec.name}__metric").alias("metric"),
                pl.col(f"{spec.name}__note").alias("note"),
            )
        )
    return pl.concat(parts).sort("key", "field")


def _summary(config: Config, left, right, rows, wide, diffs) -> dict:
    status = {
        side: {
            r["status"]: r["n"]
            for r in rows.filter(pl.col("side") == side)
            .group_by("status")
            .agg(n=pl.len())
            .iter_rows(named=True)
        }
        for side in ("left", "right")
    }
    by_stage = {
        config.keys[r["stage"]].left: r["n"]
        for r in wide.group_by("stage")
        .agg(n=pl.len())
        .sort("stage")
        .iter_rows(named=True)
    }

    def field_counts(frame: pl.DataFrame) -> dict:
        out = {}
        for spec in config.fields:
            counts = (
                frame.group_by(f"{spec.name}__category")
                .agg(n=pl.len())
                .rename({f"{spec.name}__category": "category"})
            )
            out[spec.name] = {c: 0 for c in cmp.CATEGORIES} | dict(counts.iter_rows())
        return out

    different_pairs = wide.filter(pl.col("verdict") != "identical").height
    verdicts = {v: wide.filter(pl.col("verdict") == v).height for v in VERDICTS}
    return {
        "verdicts": verdicts,
        "fields_same_entity": field_counts(
            wide.filter(pl.col("verdict") != "different_entity")
        ),
        "left": {
            "name": config.left.name,
            "rows": left.height,
            "status": status["left"],
        },
        "right": {
            "name": config.right.name,
            "rows": right.height,
            "status": status["right"],
        },
        "matched": wide.height,
        "matched_by_key": by_stage,
        "pairs_with_differences": different_pairs,
        "pairs_identical": wide.height - different_pairs,
        "fields": field_counts(wide),
        "differences": diffs.height,
    }


def reconcile(config: Config, root: Path | None = None) -> Reconciliation:
    (left, left_out), (right, right_out) = (
        load_split(config.left, root),
        load_split(config.right, root),
    )
    require_columns(
        left,
        [k.left for k in config.keys]
        + [c for f in config.fields for c in f.left]
        + ([config.group_by] if config.group_by else []),
        config.left,
    )
    require_columns(
        right,
        [k.right for k in config.keys] + [c for f in config.fields for c in f.right],
        config.right,
    )
    result = match(left, right, config.keys)
    pairs, crosswalks = _compare_pairs(
        config, left, right, result.pairs.sort("left_row")
    )
    pairs = _with_verdict(config, pairs)
    diffs = _long_differences(config, pairs) if config.fields else pl.DataFrame()
    summary = _summary(config, left, right, result.rows, pairs, diffs)
    return Reconciliation(
        config,
        left,
        right,
        result.rows,
        pairs,
        diffs,
        crosswalks,
        summary,
        left_out,
        right_out,
    )


def group_rates(rec: Reconciliation, top: int = 15) -> pl.DataFrame:
    """Per value of ``group_by`` (a left column): matched pairs, pairs with a difference, and the rate."""
    if not rec.config.group_by:
        return pl.DataFrame()
    grouped = rec.pairs.join(
        rec.left.select(ROW, rec.config.group_by), left_on="left_row", right_on=ROW
    )
    different = (
        rec.differences.select("left_row").unique().with_columns(different=pl.lit(True))
    )
    return (
        grouped.join(different, on="left_row", how="left")
        .group_by(rec.config.group_by)
        .agg(
            pairs=pl.len(), with_differences=pl.col("different").fill_null(False).sum()
        )
        .with_columns(rate=pl.col("with_differences") / pl.col("pairs"))
        .filter(pl.col("pairs") >= 20)
        .sort("rate", "pairs", descending=True)
        .head(top)
    )


def unmatched(rec: Reconciliation, side: str) -> pl.DataFrame:
    """The source rows with this status prefix (``only_left``/``only_right``/duplicates/unkeyed), all columns."""
    frame = rec.left if side == "left" else rec.right
    status = rec.rows.filter(
        (pl.col("side") == side) & (pl.col("status") != "matched")
    ).select(pl.col("row").alias(ROW), "status", "key")
    table = status.join(frame, on=ROW, how="left").sort("status", ROW)
    return _explain_out_of_scope(rec, side, table)


def _explain_out_of_scope(
    rec: Reconciliation, side: str, table: pl.DataFrame
) -> pl.DataFrame:
    """Add ``found_out_of_scope`` (and the other source's ``explain`` columns) to unmatched rows.

    A row with no counterpart might still exist on the other side, just outside its scope (a
    closed airport, a row with no code). Saying so turns "missing" into "present but excluded".
    """
    other_cfg, other = (
        (rec.config.right, rec.right_excluded)
        if side == "left"
        else (rec.config.left, rec.left_excluded)
    )
    mine = [k.left if side == "left" else k.right for k in rec.config.keys]
    theirs = [k.right if side == "left" else k.left for k in rec.config.keys]
    explain = list(other_cfg.explain)
    lookups = []
    for column in theirs:
        index: dict[str, dict] = {}
        for row in (
            other.filter(normalize_key(column).is_not_null())
            .with_columns(_k=normalize_key(column))
            .iter_rows(named=True)
        ):
            index.setdefault(row["_k"], row)
        lookups.append(index)
    found, extra = [], {c: [] for c in explain}
    for row in table.iter_rows(named=True):
        hit = None
        for column, index in zip(mine, lookups, strict=True):
            value = row.get(column)
            if (
                value is not None
                and (hit := index.get(str(value).strip().upper())) is not None
            ):
                break
        found.append(hit is not None)
        for c in explain:
            extra[c].append(None if hit is None else hit.get(c))
    return table.with_columns(
        pl.Series("found_out_of_scope", found, dtype=pl.Boolean),
        *(pl.Series(f"other_{c}", vals, dtype=pl.Utf8) for c, vals in extra.items()),
    )


def lookup(rec: Reconciliation, value: str) -> dict:
    """Everything known about one key value: the pair (if any) and any unmatched rows carrying it."""
    target = value.strip().upper()
    pair = rec.pairs.filter(pl.col("key") == target)
    found: dict = {"pair": pair.row(0, named=True) if pair.height else None, "rows": []}
    keys = rec.config.keys
    for side, frame, cols in (
        ("left", rec.left, [k.left for k in keys]),
        ("right", rec.right, [k.right for k in keys]),
    ):
        hit = frame.filter(
            pl.any_horizontal(*(normalize_key(c) == target for c in cols))
        )
        for row in hit.iter_rows(named=True):
            status = rec.rows.filter(
                (pl.col("side") == side) & (pl.col("row") == row[ROW])
            )["status"][0]
            found["rows"].append(
                {
                    "side": side,
                    "status": status,
                    "values": {k: v for k, v in row.items() if k != ROW},
                }
            )
    return found


def field_spec(config: Config, name: str) -> Field:
    return next(f for f in config.fields if f.name == name)
