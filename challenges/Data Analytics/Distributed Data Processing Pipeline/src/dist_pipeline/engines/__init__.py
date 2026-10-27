"""Engine registry.

Each entry names the adapter module, the options it receives, the environment variables set in its
worker process before any import, and the top-level import names it needs. The harness checks those
imports before starting a worker, so a missing engine is recorded as "not installed" and skipped.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field

PKG = "dist_pipeline.engines"


@dataclass(frozen=True)
class EngineSpec:
    key: str
    module: str
    requires: tuple[str, ...]
    opts: dict[str, object] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    formats: tuple[str, ...] = ("parquet",)


ENGINES: dict[str, EngineSpec] = {
    "polars": EngineSpec(
        "polars",
        f"{PKG}.polars_engine",
        ("polars",),
        {"collect": "streaming"},
        formats=("parquet", "csv"),
    ),
    "polars_eager": EngineSpec(
        "polars_eager",
        f"{PKG}.polars_engine",
        ("polars",),
        {"collect": "in-memory"},
        formats=("parquet", "csv"),
    ),
    "polars_1t": EngineSpec(
        "polars_1t",
        f"{PKG}.polars_engine",
        ("polars",),
        {"collect": "in-memory"},
        env={"POLARS_MAX_THREADS": "1"},
    ),
    "duckdb": EngineSpec(
        "duckdb",
        f"{PKG}.duckdb_engine",
        ("duckdb",),
        formats=("parquet", "csv"),
    ),
    "mrpipe": EngineSpec("mrpipe", f"{PKG}.mrpipe_engine", ("polars", "pyarrow")),
    "dask": EngineSpec(
        "dask",
        f"{PKG}.dask_engine",
        ("dask", "distributed", "pandas"),
        {"threads": 2},
    ),
    "spark": EngineSpec(
        "spark",
        f"{PKG}.spark_engine",
        ("pyspark",),
        {"driver_memory": "6g"},
    ),
    "daft": EngineSpec("daft", f"{PKG}.daft_engine", ("daft",)),
}


def missing_imports(spec: EngineSpec) -> list[str]:
    return [name for name in spec.requires if importlib.util.find_spec(name) is None]


def load(spec: EngineSpec):
    import importlib

    return importlib.import_module(spec.module)
