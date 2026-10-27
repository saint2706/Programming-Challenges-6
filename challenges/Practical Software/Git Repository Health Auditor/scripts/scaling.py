"""Synthetic history generator and timing harness for the scaling check.

Builds linear histories with `git fast-import` (one process, no per-commit subprocess), then times
`run_audit` on each size. Writes results/scaling.json and results/scaling.md.

    uv run python scripts/scaling.py                 # commits 250, 1000, 4000
    uv run python scripts/scaling.py --sizes 250 1000
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from gitaudit.audit import run_audit
from gitaudit.config import Settings
from helpers import fake_aws_key_id

RESULTS = ROOT / "results"
FILES = 40
BIG_EVERY = 400  # one 2 MiB random blob every N commits (exercises BL001/BL002 and LFS)
SECRET_EVERY = (
    900  # one committed fake key every N commits (exercises SEC001 in history)
)
NOTE_MARK = 1  # fast-import mark bases; each kind gets its own integer range
BIG_MARK = 10_000_000
COMMIT_MARK = 20_000_000


def _blob(mark: int, payload: bytes) -> bytes:
    return f"blob\nmark :{mark}\ndata {len(payload)}\n".encode() + payload + b"\n"


def _stream(commits: int) -> bytes:
    """Yield one fast-import stream for `commits` commits on refs/heads/main."""
    import os

    parts: list[bytes] = []
    for i in range(commits):
        note = f"# revision {i}\n"
        if i % SECRET_EVERY == 0:
            note += f"KEY = '{fake_aws_key_id()}'\n"
        parts.append(_blob(NOTE_MARK + i, note.encode()))
        changes = [
            f"M 100644 :{NOTE_MARK + i} src/module_{i % FILES:03d}.py\n".encode()
        ]
        if i % BIG_EVERY == 0:
            parts.append(_blob(BIG_MARK + i, os.urandom(2 * 1024**2)))
            changes.append(
                f"M 100644 :{BIG_MARK + i} assets/bundle_{i:05d}.bin\n".encode()
            )
        msg = f"commit {i}".encode()
        when = 1_600_000_000 + i * 3600
        header = (
            (
                f"commit refs/heads/main\nmark :{COMMIT_MARK + i}\n"
                f"author Dev <dev@example.org> {when} +0000\n"
                f"committer Dev <dev@example.org> {when} +0000\n"
                f"data {len(msg)}\n"
            ).encode()
            + msg
            + b"\n"
        )
        if i:
            header += f"from :{COMMIT_MARK + i - 1}\n".encode()
        parts.append(header + b"".join(changes) + b"\n")
    return b"".join(parts)


def generate(repo: Path, commits: int) -> None:
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    proc = subprocess.run(
        ["git", "-C", str(repo), "fast-import", "--quiet"],
        input=_stream(commits),
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode(errors="replace")[:500])
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", "main"], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[250, 1000, 4000])
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="gitaudit-scale-"))
    rows = []
    try:
        for n in args.sizes:
            repo = workdir / f"synthetic-{n}"
            t0 = time.perf_counter()
            generate(repo, n)
            gen_s = time.perf_counter() - t0
            t1 = time.perf_counter()
            result = run_audit(Settings(repo=repo, history=True))
            audit_s = time.perf_counter() - t1
            commits = int(result.stats.get("commits", 0))
            rows.append(
                {
                    "commits": commits,
                    "generate_seconds": round(gen_s, 2),
                    "audit_seconds": round(audit_s, 2),
                    "seconds_per_1000_commits": round(
                        audit_s / max(commits, 1) * 1000, 2
                    ),
                    "findings": len(result.findings),
                }
            )
            print(
                f"commits={commits} audit={audit_s:.2f}s findings={len(result.findings)}",
                flush=True,
            )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    (RESULTS / "scaling.json").write_text(
        json.dumps(rows, indent=4) + "\n", encoding="utf-8", newline="\n"
    )
    header = "| Commits | Generate (s) | Audit (s) | s per 1000 commits | Findings |"
    sep = "| ------- | ------------ | --------- | ------------------ | -------- |"
    body = [
        f"| {r['commits']} | {r['generate_seconds']} | {r['audit_seconds']} | "
        f"{r['seconds_per_1000_commits']} | {r['findings']} |"
        for r in rows
    ]
    (RESULTS / "scaling.md").write_text(
        "# Scaling check\n\nSynthetic fast-import histories: 40 files, a 2 MiB blob every 400 "
        "commits, a fake key every 900. Generated by `scripts/scaling.py`.\n\n"
        + "\n".join([header, sep, *body])
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
