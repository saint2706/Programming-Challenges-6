"""Replays a NAB stream through a chart and pushes each point to the browser over SSE."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable
from enum import StrEnum
from importlib.resources import files
from typing import Literal

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent

from spc_stream.nab import STREAMS, load, monitor_for

app = FastAPI(title="SPC stream")
# An enum path parameter: FastAPI rejects any other name with 422 before it can reach a file path.
StreamName = StrEnum("StreamName", {k: k for k in STREAMS})


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return files("spc_stream").joinpath("static/index.html").read_text(encoding="utf-8")


@app.get("/api/streams")
def streams() -> list[str]:
    return sorted(STREAMS)


@app.get("/api/stream/{name}", response_class=EventSourceResponse)
async def stream(
    name: StreamName,
    chart: Literal["shewhart", "ewma", "cusum"] = "ewma",
    prewhiten: bool = True,
    speed: int = Query(200, ge=1, le=5000, description="points per second"),
    limit: int = Query(
        0, ge=0, description="stop after this many points (0 = whole stream)"
    ),
) -> AsyncIterable[ServerSentEvent]:
    ts, xs, windows = load(name.value)
    mon = monitor_for(name.value, chart, prewhiten, len(xs))
    yield ServerSentEvent(
        data={"baseline_n": mon.baseline_n, "windows": windows, "n": len(xs)},
        event="meta",
    )
    for t, x in zip(ts, xs[: limit or None], strict=False):
        e = mon.update(x)
        p = e.point
        yield ServerSentEvent(
            data={
                "i": e.i,
                "t": t.isoformat(),
                "x": x,
                "stat": p.stat if p else None,
                "lcl": p.lcl if p else None,
                "ucl": p.ucl if p else None,
                "alarm": e.alarm,
            },
            event="point",
            id=str(e.i),
        )
        await asyncio.sleep(1 / speed)
    yield ServerSentEvent(raw_data="end", event="end")
