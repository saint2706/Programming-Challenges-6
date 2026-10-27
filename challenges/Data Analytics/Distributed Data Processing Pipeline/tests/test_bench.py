"""Harness mechanics: process-tree sampling, cleanup, and one fresh-process run on the fixture."""

import subprocess
import sys
import time

import psutil
from dist_pipeline.bench import _kill_tree, _tree_rss, run_worker, verify_record
from helpers import FIXTURE_SCALE


def test_tree_sampling_sees_a_child_and_kill_removes_it():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        root = psutil.Process()
        seen: dict[int, psutil.Process] = {}
        deadline = time.time() + 10
        while time.time() < deadline and child.pid not in seen:
            _tree_rss(root, seen)
            time.sleep(0.05)
        assert child.pid in seen
        assert _tree_rss(root, seen) > 0
        _kill_tree(psutil.Process(child.pid))
        child.wait(timeout=10)
        assert (
            not psutil.pid_exists(child.pid)
            or psutil.Process(child.pid).status() == "zombie"
        )
    finally:
        if child.poll() is None:
            child.kill()


def test_fresh_worker_run_is_verified_and_leaves_no_children(mini_home, tmp_path):
    record = run_worker(
        "duckdb",
        "q4_moments_distinct",
        FIXTURE_SCALE,
        "parquet",
        repeats=2,
        warmup=1,
        opts={},
        workdir=tmp_path,
    )
    assert record["status"] == "ok", record.get("error")
    assert len(record["times_s"]) == 2
    assert record["leftover_children"] == 0
    assert record["peak_rss_mb"] > 0
    verdict = verify_record(record, FIXTURE_SCALE, {}, tmp_path)
    assert verdict is not None and verdict.ok, verdict and verdict.detail
    assert (
        list(tmp_path.iterdir()) == []
    )  # the worker's files are removed after verification
