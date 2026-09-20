"""Small FastAPI web UI for the multi-timezone meeting scheduler.

Reuses the exact same `scheduler.py` core as the CLI -- this file only does
HTTP plumbing and HTML rendering. Run:

    uv run --with fastapi --with "uvicorn[standard]" --with python-multipart python web.py
"""

from __future__ import annotations

import html
from datetime import date

import uvicorn
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse
from scheduler import find_overlaps, format_window, parse_participant_spec

app = FastAPI(title="Multi-Timezone Meeting Scheduler")

CSS = """
:root { color-scheme: light dark; }
body { font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }
textarea { width: 100%; min-height: 8rem; font-family: ui-monospace, monospace; box-sizing: border-box; }
input, select, button { font-size: 1rem; padding: 0.4rem 0.6rem; }
.row { margin: 0.75rem 0; display: flex; gap: 1rem; flex-wrap: wrap; align-items: center; }
table { border-collapse: collapse; width: 100%; margin-top: 1rem; }
th, td { border: 1px solid #8884; padding: 0.4rem 0.6rem; text-align: left; }
.hint { color: #666; font-size: 0.85rem; }
"""


def page(title: str, body: str) -> HTMLResponse:
    return HTMLResponse(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>{CSS}</style></head>
<body><h1>Multi-Timezone Meeting Scheduler</h1>{body}</body></html>""")


FORM = """
<form method="post" action="/find">
  <div class="row">
    <label>Date <input type="date" name="date_str" required></label>
    <label>Reference timezone <input type="text" name="ref_tz" placeholder="America/New_York" required></label>
    <label>Display timezone <input type="text" name="display_tz" placeholder="(defaults to reference)"></label>
    <label>Min participants <input type="number" name="min_participants" min="1" placeholder="(everyone)"></label>
  </div>
  <p class="hint">One participant per line: <code>Name|Area/City|HH:MM-HH:MM</code>
     (add <code>|0,1,2,3,4,5</code> etc. to include weekend days; IANA zone names, e.g. Asia/Kolkata).</p>
  <textarea name="participants" placeholder="Asha|Asia/Kolkata|09:00-18:00&#10;Ben|Europe/London|09:00-17:00&#10;Cara|America/New_York|09:00-17:00" required></textarea>
  <div class="row"><button type="submit">Find overlaps</button></div>
</form>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return page("Scheduler", FORM)


@app.post("/find", response_class=HTMLResponse)
def find(
    date_str: str = Form(...),
    ref_tz: str = Form(...),
    participants: str = Form(...),
    display_tz: str | None = Form(None),
    min_participants: int | None = Form(None),
) -> HTMLResponse:
    try:
        ref_date = date.fromisoformat(date_str)
    except ValueError as exc:
        raise HTTPException(400, f"Invalid date {date_str!r}: {exc}") from exc

    lines: list[str] = [
        line.strip() for line in participants.splitlines() if line.strip()
    ]
    if not lines:
        raise HTTPException(400, "At least one participant is required")

    try:
        people = [parse_participant_spec(line) for line in lines]
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    display = display_tz.strip() if display_tz and display_tz.strip() else ref_tz

    try:
        windows = find_overlaps(people, ref_date, ref_tz, min_participants or None)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    if not windows:
        body = FORM + "<p>No overlapping window found for the given constraints.</p>"
        return page("Scheduler", body)

    rows = "".join(
        f"<tr><td>{s}</td><td>{e}</td><td>{w.duration_minutes():.0f} min</td>"
        f"<td>{html.escape(', '.join(w.available))}</td></tr>"
        for w in windows
        for s, e in [format_window(w, display)]
    )
    table = f"""
<table>
<thead><tr><th>Start</th><th>End</th><th>Duration</th><th>Available</th></tr></thead>
<tbody>{rows}</tbody>
</table>
"""
    body = (
        FORM
        + f"<h2>Overlaps on {html.escape(date_str)} ({html.escape(ref_tz)}), shown in {html.escape(display)}</h2>"
        + table
    )
    return page("Scheduler", body)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
