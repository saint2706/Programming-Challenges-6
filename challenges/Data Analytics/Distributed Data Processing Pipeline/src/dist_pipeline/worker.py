"""Fresh-process worker: one engine, one query, in its own interpreter.

The parent (``bench.py``) starts this module with ``python -m dist_pipeline.worker`` for every
engine-query pair, so no engine state, caches or memory carries over between measurements. The
worker:

1. sets the engine's environment variables (for example ``POLARS_MAX_THREADS=1``) before any
   engine library is imported;
2. times ``start`` (session or cluster start-up) separately from the queries;
3. runs ``warmup`` untimed-in-the-record runs, then ``repeats`` timed runs in the same process;
4. writes the first timed output as Arrow IPC for the parent to verify, and a JSON record.

Peak memory is not measured here: the parent samples this process tree, which is the only place
that sees every child (Spark's JVM, Dask's workers).
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

from dist_pipeline.engines import ENGINES, load, missing_imports


def _phase_stats(session: object) -> dict[str, object] | None:
    """Phase timing for engines that expose it (mrpipe keeps its last run's stage stats)."""
    last = getattr(session, "last", None)
    if last is None:
        return None
    return {
        "map_s": last.map_s,
        "reduce_s": sum(last.reduce_s),
        "reduce_max_s": max(last.reduce_s, default=0.0),
        "shuffle_bytes": last.shuffle_bytes,
        "rows_in": last.rows_in,
        "straggler_ratio": last.straggler_ratio,
        "group_skew": last.group_skew,
    }


def _write_arrow(table, path: Path) -> None:
    import pyarrow as pa

    with (
        pa.OSFile(str(path), "wb") as sink,
        pa.ipc.new_file(sink, table.schema) as writer,
    ):
        writer.write_table(table)


def measure(
    engine: str,
    query: str,
    scale: str,
    fmt: str,
    repeats: int,
    warmup: int,
    opts: dict[str, object],
    result_path: Path,
) -> dict[str, object]:
    from dist_pipeline.engines.base import dataset, has_data
    from dist_pipeline.workloads import get

    spec = ENGINES[engine]
    module = load(spec)
    q = get(query)
    ds = dataset(scale, fmt)
    record: dict[str, object] = {
        "engine": engine,
        "query": query,
        "scale": scale,
        "fmt": fmt,
        "status": "ok",
        "error": None,
        "start_s": None,
        "warmup_s": [],
        "times_s": [],
        "rows": None,
        "phases": [],
    }
    if not has_data(ds):
        record.update(status="no_data", error=f"no {fmt} data for {scale}")
        return record
    session = None
    try:
        record["stage"] = "start"  # a failure here skips the engine's remaining queries
        t0 = time.perf_counter()
        session = module.start(ds, dict(spec.opts) | opts)
        record["start_s"] = time.perf_counter() - t0
        record["stage"] = "run"
        out = None
        for i in range(warmup + repeats):
            t1 = time.perf_counter()
            out = module.run(session, q, ds, dict(spec.opts) | opts)
            elapsed = time.perf_counter() - t1
            if i < warmup:
                record["warmup_s"].append(elapsed)  # type: ignore[union-attr]
            else:
                record["times_s"].append(elapsed)  # type: ignore[union-attr]
                phases = _phase_stats(session)
                if phases is not None:
                    record["phases"].append(phases)  # type: ignore[union-attr]
                if len(record["times_s"]) == 1:  # type: ignore[arg-type]
                    record["rows"] = out.num_rows
                    _write_arrow(out, result_path)
    except NotImplementedError as exc:
        record.update(status="not_supported", error=str(exc))
    except Exception as exc:  # noqa: BLE001 - recorded in the JSON, the parent moves on
        record.update(
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            traceback="".join(traceback.format_exception(exc)[-4:])[-1500:],
        )
    finally:
        if session is not None:
            try:
                module.stop(session)
            except Exception as exc:  # noqa: BLE001 - a cleanup failure must not hide the timings
                record["stop_error"] = f"{type(exc).__name__}: {exc}"
    return record


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="dist_pipeline.worker")
    p.add_argument("--engine", required=True, choices=sorted(ENGINES))
    p.add_argument("--query", required=True)
    p.add_argument("--scale", required=True)
    p.add_argument("--fmt", default="parquet")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--opts", default="{}", help="JSON object of engine options")
    p.add_argument(
        "--result", required=True, type=Path, help="Arrow IPC file for the output"
    )
    p.add_argument("--out", required=True, type=Path, help="JSON record for the parent")
    args = p.parse_args(argv)

    spec = ENGINES[args.engine]
    missing = missing_imports(spec)
    if missing:
        record = {
            "engine": args.engine,
            "query": args.query,
            "scale": args.scale,
            "fmt": args.fmt,
            "status": "not_installed",
            "error": f"missing: {', '.join(missing)}",
        }
    else:
        # Set before any engine library is imported (POLARS_MAX_THREADS is read at import time).
        os.environ.update(spec.env)
        record = measure(
            args.engine,
            args.query,
            args.scale,
            args.fmt,
            args.repeats,
            args.warmup,
            json.loads(args.opts),
            args.result,
        )
    args.out.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
