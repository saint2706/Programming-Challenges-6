"""Parent-side benchmark driver.

For every engine-query pair this module:

* starts a fresh interpreter (``dist_pipeline.worker``), so nothing is shared between measurements;
* samples the resident memory (RSS) of that process tree every 50 ms with psutil, which covers
  Spark's JVM and Dask's worker processes; the peak of the sum is the reported peak;
* checks the worker's output against the exact DuckDB oracle (``verify.compare``);
* kills any child process the engine left behind, so the next measurement starts clean.

Scaling runs (strong and weak) use the Map-Shuffle-Reduce pipeline, whose worker count and input
size are options.
"""

from __future__ import annotations

import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pyarrow as pa
import pyarrow.ipc

from dist_pipeline.engines import ENGINES, missing_imports
from dist_pipeline.paths import project_root
from dist_pipeline.stats import summarise
from dist_pipeline.verify import Verdict, compare, duckdb_oracle
from dist_pipeline.workloads import QUERIES, get

SAMPLE_S = 0.05
TIMEOUT_S = 1800
VERSION_PACKAGES = (
    "polars",
    "duckdb",
    "pyarrow",
    "dask",
    "distributed",
    "pyspark",
    "daft",
    "psutil",
)


def machine() -> dict[str, object]:
    """Hardware and software the numbers were measured on, for the report."""
    vm = psutil.virtual_memory()
    versions: dict[str, str] = {}
    for pkg in VERSION_PACKAGES:
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    me = os.getpid()
    others = sum(
        1
        for p in psutil.process_iter(["name"])
        if p.info["name"]
        and p.info["name"].lower().startswith("python")
        and p.pid != me
    )
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_physical": psutil.cpu_count(logical=False),
        "cpu_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(vm.total / 2**30, 1),
        "ram_available_gb_at_start": round(vm.available / 2**30, 1),
        "other_python_processes": others,
        "package_versions": versions,
    }


def _tree_rss(root: psutil.Process, seen: dict[int, psutil.Process]) -> int:
    """Sum of RSS over the process tree; every child seen is remembered for the leftover check."""
    try:
        procs = [root, *root.children(recursive=True)]
    except psutil.Error:
        return 0
    total = 0
    for p in procs:
        if p.pid != root.pid:
            seen.setdefault(p.pid, p)
        try:
            total += p.memory_info().rss
        except psutil.Error:
            pass
    return total


def _kill_tree(root: psutil.Process) -> None:
    try:
        victims = [*root.children(recursive=True), root]
    except psutil.Error:
        return
    for p in victims:
        try:
            p.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(victims, timeout=10)


def run_worker(
    engine: str,
    query: str,
    scale: str,
    fmt: str,
    repeats: int,
    warmup: int,
    opts: dict[str, object],
    workdir: Path,
) -> dict[str, object]:
    """Run one engine-query in a fresh process and return its record with memory and wall time."""
    tag = f"{engine}-{query}-{scale}-{fmt}-w{opts.get('workers', 'd')}-p{opts.get('parts', 'all')}"
    out_json = workdir / f"{tag}.json"
    result = workdir / f"{tag}.arrow"
    log = workdir / f"{tag}.log"
    for stale in (out_json, result):
        stale.unlink(missing_ok=True)
    cmd = [
        sys.executable,
        "-I",  # isolated: the cwd is not put on sys.path, so a stray polars.py there is not imported
        "-m",
        "dist_pipeline.worker",
        "--engine",
        engine,
        "--query",
        query,
        "--scale",
        scale,
        "--fmt",
        fmt,
        "--repeats",
        str(repeats),
        "--warmup",
        str(warmup),
        "--opts",
        json.dumps(opts),
        "--result",
        str(result),
        "--out",
        str(out_json),
    ]
    t0 = time.perf_counter()
    with log.open("w", encoding="utf-8") as err:
        proc = subprocess.Popen(
            cmd, cwd=project_root(), stdout=subprocess.DEVNULL, stderr=err
        )
        root = psutil.Process(proc.pid)
        seen: dict[int, psutil.Process] = {}
        peak = 0
        while proc.poll() is None:
            peak = max(peak, _tree_rss(root, seen))
            if time.perf_counter() - t0 > TIMEOUT_S:
                _kill_tree(root)
                break
            time.sleep(SAMPLE_S)
        proc.wait(timeout=30)
    wall = time.perf_counter() - t0
    # Children the engine left running after the worker exited: stop them so the next run is clean.
    leftovers = [p for p in seen.values() if p.is_running()]
    for p in leftovers:
        try:
            p.kill()
        except psutil.Error:
            pass

    if out_json.exists():
        record = json.loads(out_json.read_text(encoding="utf-8"))
    else:
        tail = log.read_text(encoding="utf-8", errors="replace")[-800:]
        record = {
            "engine": engine,
            "query": query,
            "scale": scale,
            "fmt": fmt,
            "status": "crashed",
            "error": f"worker exited with code {proc.returncode} without a record",
            "traceback": tail,
            "times_s": [],
            "phases": [],
        }
    record["peak_rss_mb"] = round(peak / 2**20, 1)
    record["wall_s"] = round(wall, 3)
    record["leftover_children"] = len(leftovers)
    record["opts"] = opts
    for f in (out_json, log):
        f.unlink(missing_ok=True)
    return record


def verify_record(
    record: dict[str, object],
    scale: str,
    oracle_cache: dict[tuple[str, str], pa.Table],
    workdir: Path,
) -> Verdict | None:
    """Compare a worker's output with the exact oracle. ``None`` when there is no output."""
    query = get(str(record["query"]))
    result = workdir / (
        f"{record['engine']}-{record['query']}-{scale}-{record['fmt']}-"
        f"w{record['opts'].get('workers', 'd')}-p{record['opts'].get('parts', 'all')}.arrow"  # type: ignore[union-attr]
    )
    if not result.exists():
        return None
    try:
        actual = pa.ipc.open_file(str(result)).read_all()
    finally:
        result.unlink(missing_ok=True)
    key = (scale, query.name)
    if key not in oracle_cache:
        from dist_pipeline.engines.base import dataset

        oracle_cache[key] = duckdb_oracle(query, dataset(scale))
    return compare(actual, oracle_cache[key], query)


def _row(record: dict[str, object], verdict: Verdict | None) -> dict[str, object]:
    times = [float(t) for t in record.get("times_s", [])]  # type: ignore[union-attr]
    phases = record.get("phases") or []
    row: dict[str, object] = {
        "engine": record["engine"],
        "label": _label(str(record["engine"])),
        "query": record["query"],
        "scale": record["scale"],
        "fmt": record["fmt"],
        "status": record["status"],
        "error": record.get("error"),
        "stage": record.get("stage"),
        "opts": record.get("opts", {}),
        "start_s": record.get("start_s"),
        "time": summarise(times),
        "rows": record.get("rows"),
        "peak_rss_mb": record.get("peak_rss_mb"),
        "wall_s": record.get("wall_s"),
        "leftover_children": record.get("leftover_children", 0),
        "verified": None if verdict is None else verdict.ok,
        "verify_detail": None if verdict is None else verdict.detail,
        "max_rel_error": None if verdict is None else verdict.max_rel_error,
        "phases": phases,
    }
    if phases:
        row["phase_median"] = {
            k: summarise([float(p[k]) for p in phases])["median"]
            for k in (
                "map_s",
                "reduce_s",
                "shuffle_bytes",
                "straggler_ratio",
                "group_skew",
            )
            if all(p.get(k) is not None for p in phases)
        }
    return row


def _label(engine: str) -> str:
    module = ENGINES[engine].module.rsplit(".", 1)[-1]
    from importlib import import_module

    try:
        return str(import_module(ENGINES[engine].module).LABEL)
    except Exception:  # noqa: BLE001 - an engine that cannot import still needs a row
        return module


def run_matrix(
    scales: tuple[str, ...],
    engines: tuple[str, ...],
    queries: tuple[str, ...],
    repeats: int,
    warmup: int,
    workdir: Path,
    csv_scales: tuple[str, ...] = ("1m",),
    csv_engines: tuple[str, ...] = ("polars", "duckdb"),
    progress=None,
) -> list[dict[str, object]]:
    """The full comparison: every engine on every query, for each scale, plus the CSV leg."""
    rows: list[dict[str, object]] = []
    oracle_cache: dict[tuple[str, str], pa.Table] = {}
    plan: list[tuple[str, str, str]] = []  # (scale, fmt, engine)
    for scale in scales:
        plan += [(scale, "parquet", e) for e in engines]
    for scale in csv_scales:
        plan += [(scale, "csv", e) for e in csv_engines]
    for scale, fmt, engine in plan:
        spec = ENGINES[engine]
        if fmt not in spec.formats:
            continue
        missing = missing_imports(spec)
        if missing:
            for q in queries:
                rows.append(
                    _row(
                        {
                            "engine": engine,
                            "query": q,
                            "scale": scale,
                            "fmt": fmt,
                            "status": "not_installed",
                            "error": f"missing: {', '.join(missing)}",
                        },
                        None,
                    )
                )
            continue
        start_failed: str | None = None
        for q in queries:
            if start_failed is not None:
                rec = {
                    "engine": engine,
                    "query": q,
                    "scale": scale,
                    "fmt": fmt,
                    "status": "skipped",
                    "error": f"not run: start failed on an earlier query ({start_failed})",
                }
                rows.append(_row(rec, None))
                continue
            record = run_worker(engine, q, scale, fmt, repeats, warmup, {}, workdir)
            verdict = None
            if record["status"] == "ok":
                verdict = verify_record(record, scale, oracle_cache, workdir)
            if record.get("stage") == "start" and record["status"] == "error":
                start_failed = str(record["error"])
            row = _row(record, verdict)
            rows.append(row)
            if progress:
                progress(row)
    return rows


def run_scaling(
    scale: str,
    queries: tuple[str, ...],
    strong_workers: tuple[int, ...],
    weak_workers: tuple[int, ...],
    parts_per_worker: int,
    repeats: int,
    warmup: int,
    workdir: Path,
) -> dict[str, object]:
    """Strong scaling (fixed input, more workers) and weak scaling (input grows with workers).

    Scaling runs are timed but not verified: strong runs use the full input and the matrix
    already verifies that configuration; weak runs use a prefix of the part files, so there is
    no full-input oracle for them.
    """
    strong: list[dict[str, object]] = []
    weak: list[dict[str, object]] = []
    for q in queries:
        for w in strong_workers:
            rec = run_worker(
                "mrpipe", q, scale, "parquet", repeats, warmup, {"workers": w}, workdir
            )
            strong.append(_scale_row(rec, w, None))
        for w in weak_workers:
            parts = parts_per_worker * w
            rec = run_worker(
                "mrpipe",
                q,
                scale,
                "parquet",
                repeats,
                warmup,
                {"workers": w, "parts": parts},
                workdir,
            )
            weak.append(_scale_row(rec, w, parts))
    return {"scale": scale, "strong": strong, "weak": weak}


def _scale_row(
    rec: dict[str, object], workers: int, parts: int | None
) -> dict[str, object]:
    times = [float(t) for t in rec.get("times_s", [])]  # type: ignore[union-attr]
    return {
        "query": rec["query"],
        "workers": workers,
        "parts": parts,
        "status": rec["status"],
        "rows": rec.get("rows"),
        "time": summarise(times),
        "peak_rss_mb": rec.get("peak_rss_mb"),
    }


def default_workdir() -> Path:
    path = project_root() / "data" / "runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


ALL_ENGINES = tuple(ENGINES)
ALL_QUERIES = tuple(q.name for q in QUERIES)
