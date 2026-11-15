"""spc-stream: monitor a stream, reproduce the ARL table, score NAB, or serve the live page."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import TextIO

from spc_stream import arl, nab
from spc_stream.charts import CHARTS, make_chart
from spc_stream.monitor import Monitor
from spc_stream.paths import project_root


def read_values(f: TextIO) -> Iterator[tuple[str, float]]:
    """Lines of `value` or `timestamp,value` (a `timestamp,value` header is skipped)."""
    for n, line in enumerate(f, 1):
        line = line.strip()
        if not line or line.lower() == "timestamp,value":
            continue
        label, _, v = line.rpartition(",")
        try:
            yield label or str(n), float(v)
        except ValueError:
            raise SystemExit(f"line {n}: not a number: {v!r}") from None


def cmd_monitor(a: argparse.Namespace) -> int:
    mon = Monitor(make_chart(a.chart), a.baseline, a.season, a.ar1, a.cooldown)
    f = sys.stdin if a.file == "-" else open(a.file, encoding="utf-8")  # noqa: SIM115
    alarms = 0
    with f:
        for label, x in read_values(f):
            e = mon.update(x)
            if e.alarm:
                alarms += 1
                print(f"{label}\t{x:g}\t{e.alarm}", flush=True)
    if mon.base is None:
        raise SystemExit(f"stream ended inside Phase I ({a.baseline} points needed)")
    print(
        f"# {alarms} alarm(s); baseline mean {mon.base.mean:g}, sigma {mon.base.sigma:g}"
    )
    return 0


def _md_table(head: list[str], rows: list[list[str]]) -> str:
    w = [max(len(r[i]) for r in [head, *rows]) for i in range(len(head))]
    fmt = lambda r: "| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(r)) + " |"
    return "\n".join([fmt(head), fmt(["-" * x for x in w]), *map(fmt, rows)]) + "\n"


def _write(out: Path, stem: str, data: object, md: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    utf8 = {"encoding": "utf-8", "newline": "\n"}
    (out / f"{stem}.json").write_text(json.dumps(data, indent=4) + "\n", **utf8)
    (out / f"{stem}.md").write_text(md, **utf8)
    print(md)


def cmd_arl(a: argparse.Namespace) -> int:
    res = arl.simulate(a.reps, a.seed)
    rows = []
    for name, by_shift in res.items():
        pub = arl.PUBLISHED.get(name, {})
        for d, (m, se) in by_shift.items():
            ref = arl.shewhart_exact(d) if name == "shewhart 3 sigma" else pub.get(d)
            rows.append(
                [name, f"{d:g}", f"{m:.2f} ± {se:.2f}", f"{ref:.2f}" if ref else "-"]
            )
    md = f"# ARL by simulation ({a.reps} runs per cell, seed {a.seed})\n\n"
    md += _md_table(["Chart", "Shift (sigma)", "Simulated ARL", "Published"], rows)
    data = {
        n: {str(d): {"arl": m, "se": se} for d, (m, se) in v.items()}
        for n, v in res.items()
    }
    _write(a.out, "arl", data, md)
    return 0


def cmd_nab(a: argparse.Namespace) -> int:
    scores = nab.benchmark()
    rows = [
        [
            s.stream,
            s.chart,
            "yes" if s.prewhiten else "no",
            f"{s.windows_hit}/{s.windows}",
            str(s.false_alarms),
            f"{s.false_per_1k:g}",
            "-" if s.mean_delay is None else f"{s.mean_delay:g}",
        ]
        for s in scores
    ]
    md = "# NAB replay (first 15% is Phase I, 12-point alarm cooldown)\n\n"
    md += _md_table(
        [
            "Stream",
            "Chart",
            "Prewhiten",
            "Windows hit",
            "False alarms",
            "Per 1k",
            "Delay",
        ],
        rows,
    )
    _write(
        a.out,
        "nab",
        [asdict(s) for s in scores],
        md,
    )
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("spc_stream.server:app", host=a.host, port=a.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="spc-stream", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser(
        "monitor", help="watch a CSV or stdin stream and print alarms live"
    )
    m.add_argument("file", help="CSV of value or timestamp,value; - for stdin")
    m.add_argument("--chart", choices=sorted(CHARTS), default="ewma")
    m.add_argument(
        "--baseline", type=int, default=200, help="Phase I points (default 200)"
    )
    m.add_argument(
        "--season", type=int, default=0, help="seasonal difference lag in points"
    )
    m.add_argument("--ar1", action="store_true", help="chart AR(1) residuals")
    m.add_argument(
        "--cooldown", type=int, default=12, help="points to hold back repeat alarms"
    )
    m.set_defaults(fn=cmd_monitor)

    out = project_root() / "results"
    r = sub.add_parser(
        "arl", help="simulate ARL0/ARL1 and compare with published tables"
    )
    r.add_argument("--reps", type=int, default=2000)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", type=Path, default=out)
    r.set_defaults(fn=cmd_arl)

    n = sub.add_parser("nab", help="score every chart on the vendored NAB streams")
    n.add_argument("--out", type=Path, default=out)
    n.set_defaults(fn=cmd_nab)

    s = sub.add_parser("serve", help="live control-chart page at http://127.0.0.1:8000")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
