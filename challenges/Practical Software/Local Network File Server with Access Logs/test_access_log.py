from __future__ import annotations

import json
import threading
from pathlib import Path

from access_log import AccessLogger


def test_log_appends_one_json_line(tmp_path: Path) -> None:
    logger = AccessLogger(tmp_path / "access.log.jsonl")
    logger.log(
        identity="token",
        ip="127.0.0.1",
        method="GET",
        path="/browse/",
        action="list",
        status=200,
    )

    lines = (tmp_path / "access.log.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["identity"] == "token"
    assert entry["ip"] == "127.0.0.1"
    assert entry["method"] == "GET"
    assert entry["path"] == "/browse/"
    assert entry["action"] == "list"
    assert entry["status"] == 200
    assert "ts" in entry


def test_tail_empty_when_log_file_does_not_exist(tmp_path: Path) -> None:
    logger = AccessLogger(tmp_path / "does_not_exist.jsonl")
    assert logger.tail() == []


def test_tail_returns_newest_first(tmp_path: Path) -> None:
    logger = AccessLogger(tmp_path / "access.log.jsonl")
    for i in range(5):
        logger.log(
            identity="token",
            ip="127.0.0.1",
            method="GET",
            path=f"/f{i}",
            action="download",
            status=200,
        )

    entries = logger.tail(3)
    assert [e["path"] for e in entries] == ["/f4", "/f3", "/f2"]


def test_tail_respects_n_limit(tmp_path: Path) -> None:
    logger = AccessLogger(tmp_path / "access.log.jsonl")
    for i in range(10):
        logger.log(
            identity="token",
            ip="1.1.1.1",
            method="GET",
            path=f"/{i}",
            action="list",
            status=200,
        )

    assert len(logger.tail(4)) == 4
    assert len(logger.tail(100)) == 10


def test_tail_skips_corrupt_lines(tmp_path: Path) -> None:
    log_path = tmp_path / "access.log.jsonl"
    logger = AccessLogger(log_path)
    logger.log(
        identity="token",
        ip="1.1.1.1",
        method="GET",
        path="/good1",
        action="list",
        status=200,
    )
    with log_path.open("a", encoding="utf-8") as f:
        f.write("{not valid json\n")
    logger.log(
        identity="token",
        ip="1.1.1.1",
        method="GET",
        path="/good2",
        action="list",
        status=200,
    )

    entries = logger.tail(10)
    paths = [e["path"] for e in entries]
    assert paths == ["/good2", "/good1"]


def test_concurrent_writes_never_interleave_or_corrupt(tmp_path: Path) -> None:
    logger = AccessLogger(tmp_path / "access.log.jsonl")
    barrier = threading.Barrier(20)

    def write_one(i: int) -> None:
        barrier.wait()
        logger.log(
            identity="token",
            ip="1.1.1.1",
            method="GET",
            path=f"/concurrent/{i}",
            action="list",
            status=200,
        )

    threads = [threading.Thread(target=write_one, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = (tmp_path / "access.log.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 20
    parsed_paths = {json.loads(line)["path"] for line in lines}
    assert parsed_paths == {f"/concurrent/{i}" for i in range(20)}
