"""Adapter for the challenge's own map-shuffle-reduce pipeline (``dist_pipeline.mrpipe``)."""

from __future__ import annotations

import os

import pyarrow as pa

from dist_pipeline.engines.base import Dataset
from dist_pipeline.mrpipe import MrPipeline
from dist_pipeline.workloads import Query

LABEL = "mrpipe (own map-shuffle-reduce)"


def start(ds: Dataset, opts: dict[str, object]) -> MrPipeline:
    workers = int(opts.get("workers") or os.cpu_count() or 1)
    return MrPipeline(workers=workers)


def run(
    session: MrPipeline, query: Query, ds: Dataset, opts: dict[str, object]
) -> pa.Table:
    if ds.fmt != "parquet":
        raise NotImplementedError("mrpipe reads Parquet part files only")
    zones = ds.zones_frame() if query.needs_zones else None
    parts = ds.parts
    if opts.get("parts"):  # weak scaling: a prefix of the part files
        parts = parts[: int(opts["parts"])]
    out = session.run(query.name, parts, zones)
    return out.to_arrow()


def stop(session: MrPipeline) -> None:
    session.close()
