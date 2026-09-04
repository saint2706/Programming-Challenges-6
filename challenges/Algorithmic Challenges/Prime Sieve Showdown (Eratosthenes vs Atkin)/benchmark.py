"""Benchmark harness for the sieve showdown.

Two things make in-process timing loops lie about sieves, and both are fixed
the same way:

* **Memory.** Peak RSS is a high-water mark for the whole process. Run two
  sieves in one process and the second inherits the first's watermark, so a
  segmented sieve looks exactly as hungry as the array it was designed to
  avoid.
* **Allocator reuse.** A 100 MB buffer freed by run 1 gets handed straight back
  to run 2, which then never touches the kernel and looks unfairly fast.

So every measurement runs in a fresh child process, which measures its own
footprint over a baseline taken right after importing -- which also makes
NumPy's ~30 MB import cost cancel out instead of being charged to the
algorithm.

That child uses two methods and keeps the larger: a background thread sampling
``/proc/self/statm`` every millisecond, and the ``ru_maxrss`` delta. Neither
alone is enough. ``ru_maxrss`` is blind to anything smaller than CPython's own
startup peak, so a 2.6 MB sieve reads as exactly zero; sampling is blind to a
spike shorter than its interval.

    uv run --with numpy python benchmark.py                 # default: 10^8
    uv run --with numpy python benchmark.py --limit 1e7 --repeat 3
    uv run --with numpy python benchmark.py --markdown      # table for the README
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import sieves

HERE = Path(__file__).resolve()


@dataclass
class Measurement:
    key: str
    label: str
    family: str
    limit: int
    count: int
    seconds: float
    rss_bytes: int
    memory_class: str
    correct: bool
    skipped: str = ""

    @property
    def bytes_per_int(self) -> float:
        return self.rss_bytes / self.limit if self.limit else 0.0

    @property
    def primes_per_second(self) -> float:
        return self.count / self.seconds if self.seconds else 0.0


# ---------------------------------------------------------------------------
# Child process: one sieve, one measurement
# ---------------------------------------------------------------------------


_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def _peak_rss_bytes() -> int:
    """The process high-water mark. Kilobytes on Linux, bytes on macOS."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw if sys.platform == "darwin" else raw * 1024


def _current_rss_bytes() -> int | None:
    """*Current* resident set size, or None where it cannot be read cheaply."""
    try:
        with open("/proc/self/statm", "rb") as fh:
            return int(fh.read().split()[1]) * _PAGE_SIZE
    except (OSError, IndexError, ValueError):  # pragma: no cover - non-Linux
        return None


class _RssSampler(threading.Thread):
    """Poll current RSS while the sieve runs.

    ``ru_maxrss`` alone is not enough here. It is a high-water mark for the
    whole process, and CPython's own startup peak is a few tens of megabytes --
    so a sieve that allocates 2.6 MB never moves it, and the delta reads as
    exactly zero. Sampling the *current* RSS and subtracting the pre-run
    baseline measures what the sieve actually held, at any size.
    """

    def __init__(self, interval: float = 0.001) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.baseline = _current_rss_bytes() or 0
        self.peak = self.baseline
        self._done = threading.Event()

    def run(self) -> None:
        while not self._done.is_set():
            cur = _current_rss_bytes()
            if cur is not None and cur > self.peak:
                self.peak = cur
            self._done.wait(self.interval)

    def stop(self) -> int:
        self._done.set()
        self.join(timeout=1.0)
        cur = _current_rss_bytes()
        if cur is not None and cur > self.peak:
            self.peak = cur
        return max(0, self.peak - self.baseline)


def run_worker(key: str, limit: int, repeat: int) -> int:
    impl = sieves.IMPLEMENTATIONS[key]

    # Let the sampler thread actually get scheduled between bytecodes.
    sys.setswitchinterval(0.0005)
    rusage_baseline = _peak_rss_bytes()
    sampler = _RssSampler()
    have_sampler = _current_rss_bytes() is not None
    if have_sampler:
        sampler.start()

    best = float("inf")
    count = 0
    for _ in range(repeat):
        start = time.perf_counter()
        count = impl.fn(limit)
        best = min(best, time.perf_counter() - start)

    sampled = sampler.stop() if have_sampler else 0
    payload = {
        "key": key,
        "limit": limit,
        "count": count,
        "seconds": best,
        # Whichever method saw more: sampling catches small allocations that
        # never move the process high-water mark, ru_maxrss catches spikes
        # shorter than the sampling interval.
        "rss_bytes": max(sampled, _peak_rss_bytes() - rusage_baseline, 0),
        "rss_method": "sampled+rusage" if have_sampler else "rusage",
    }
    print(json.dumps(payload))
    return 0


# ---------------------------------------------------------------------------
# Parent process
# ---------------------------------------------------------------------------


def measure(key: str, limit: int, repeat: int) -> Measurement:
    impl = sieves.IMPLEMENTATIONS[key]
    ok, why = impl.suitable(limit)
    if not ok:
        return Measurement(
            key, impl.label, impl.family, limit, 0, 0.0, 0, impl.memory, False, why
        )

    proc = subprocess.run(
        [sys.executable, str(HERE), "--worker", key,
         "--limit", str(limit), "--repeat", str(repeat)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return Measurement(
            key, impl.label, impl.family, limit, 0, 0.0, 0, impl.memory, False,
            f"worker failed: {proc.stderr.strip().splitlines()[-1:] or ['?']}",
        )

    data = json.loads(proc.stdout.strip().splitlines()[-1])
    expected = sieves.PI_REFERENCE.get(limit)
    correct = expected is None or data["count"] == expected
    return Measurement(
        key, impl.label, impl.family, limit, data["count"], data["seconds"],
        data["rss_bytes"], impl.memory, correct,
    )


def _fmt_bytes(n: int) -> str:
    if n <= 0:
        return "-"
    for unit, scale in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= scale:
            return f"{n / scale:.1f} {unit}"
    return f"{n} B"


def render_table(results: list[Measurement], markdown: bool) -> str:
    header = ["Implementation", "Family", "Time", "Peak RSS", "bytes/int", "Memory", "pi(N)"]
    rows = []
    for m in results:
        if m.skipped:
            rows.append([m.label, m.family, f"skipped ({m.skipped})", "-", "-", m.memory_class, "-"])
            continue
        rows.append([
            m.label,
            m.family,
            f"{m.seconds:.2f} s",
            _fmt_bytes(m.rss_bytes),
            f"{m.bytes_per_int:.3f}",
            m.memory_class,
            f"{m.count:,}" + ("" if m.correct else "  WRONG"),
        ])

    if markdown:
        out = ["| " + " | ".join(header) + " |",
               "| " + " | ".join("---" for _ in header) + " |"]
        out += ["| " + " | ".join(r) + " |" for r in rows]
        return "\n".join(out)

    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]
    lines = ["  ".join(h.ljust(w) for h, w in zip(header, widths)),
             "  ".join("-" * w for w in widths)]
    lines += ["  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows]
    return "\n".join(lines)


def summarize(results: list[Measurement]) -> str:
    ran = [m for m in results if not m.skipped]
    if not ran:
        return ""
    best_era = min((m for m in ran if m.family == "eratosthenes"),
                   key=lambda m: m.seconds, default=None)
    best_atkin = min((m for m in ran if m.family == "atkin"),
                     key=lambda m: m.seconds, default=None)
    lines = []
    if best_era and best_atkin:
        ratio = best_atkin.seconds / best_era.seconds
        lines.append(
            f"Best Eratosthenes ({best_era.label}) is {ratio:.1f}x faster than "
            f"best Atkin ({best_atkin.label})."
        )
    leanest = min(ran, key=lambda m: m.rss_bytes or 1 << 62)
    if leanest.rss_bytes:
        lines.append(
            f"Leanest: {leanest.label} at {_fmt_bytes(leanest.rss_bytes)} "
            f"({leanest.bytes_per_int:.4f} bytes per integer sieved)."
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", default="1e8", help="upper bound N (accepts 1e8)")
    ap.add_argument("--repeat", type=int, default=1, help="runs per implementation; best wins")
    ap.add_argument("--only", help="comma-separated implementation keys")
    ap.add_argument("--markdown", action="store_true", help="emit a Markdown table")
    ap.add_argument("--json", action="store_true", help="emit raw JSON")
    ap.add_argument("--list", action="store_true", help="list implementation keys")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.list:
        for key, impl in sieves.IMPLEMENTATIONS.items():
            print(f"{key:15s} {impl.label}  [{impl.memory}]")
        return 0

    limit = int(float(args.limit))
    if args.worker:
        return run_worker(args.worker, limit, args.repeat)

    keys = list(sieves.IMPLEMENTATIONS)
    if args.only:
        keys = [k.strip() for k in args.only.split(",")]
        unknown = [k for k in keys if k not in sieves.IMPLEMENTATIONS]
        if unknown:
            ap.error(f"unknown implementation(s): {', '.join(unknown)}")

    results = [measure(k, limit, args.repeat) for k in keys]

    if args.json:
        json.dump([asdict(m) for m in results], sys.stdout, indent=2)
        print()
        return 0

    print(f"N = {limit:,}   repeat = {args.repeat}   "
          f"python {platform.python_version()} on {platform.machine()}")
    if limit in sieves.PI_REFERENCE:
        print(f"expected pi(N) = {sieves.PI_REFERENCE[limit]:,}")
    print()
    print(render_table(results, args.markdown))
    print()
    print(summarize(results))
    return 0 if all(m.correct or m.skipped for m in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
