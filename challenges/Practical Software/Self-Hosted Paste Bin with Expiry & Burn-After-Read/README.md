# Self-Hosted Paste Bin with Expiry & Burn-After-Read

**Category:** Practical Software
**Difficulty:** B (brief: "One-time-view links, TTL cleanup job, syntax highlighting.")

**Status:** Implemented (Python)

A small, self-hostable pastebin: create a paste, get a link, optionally make
it a one-time-view ("burn after read") link or give it a TTL, and view it
with syntax highlighting. No external services -- one SQLite file, one
Python process.

## The concurrency bug most from-scratch pastebins have

The obvious implementation of burn-after-read is "SELECT the paste, render
it, then DELETE it." That has a race: two nearly-simultaneous requests for
the same link (a person double-clicking, or a chat client generating a link
preview alongside the human opening it) can both SELECT before either
DELETEs, and both see content that was supposed to be shown exactly once.

`storage.PasteStore.get_and_consume` closes that race by opening a manual
`BEGIN IMMEDIATE` transaction *before* the SELECT. SQLite's own locking means
a second connection's `BEGIN IMMEDIATE` blocks until the first transaction
commits -- so the SELECT-then-maybe-DELETE is effectively atomic across
processes/threads even though each request opens its own connection. This is
covered by a real concurrency test (`test_concurrent_burn_after_read_only_one_winner`)
that fires two threads at the same link through a `threading.Barrier` and
asserts exactly one of them ever sees the content.

## Design

- **`storage.py`** — all persistence and the expiry/burn semantics, with zero
  web-framework dependency (so it's testable and reusable on its own).
  `create` / `peek` (non-consuming lookup, used by the TTL countdown) /
  `get_and_consume` (the one atomic read) / `delete_expired`.
- **`app.py`** — FastAPI routes, HTML rendering, and syntax highlighting.
  Pages are small server-rendered HTML fragments built with `f"""..."""`
  strings and `html.escape()` rather than a template-file directory, to keep
  this a single self-contained script per this repo's convention — the only
  place a template engine would earn its keep is the highlighted-code block,
  and Pygments already produces safe, pre-escaped HTML for that.
- **Background TTL sweep.** A daemon thread (started/stopped via a FastAPI
  `lifespan` context manager) calls `delete_expired()` every 60 seconds. This
  is a *courtesy* cleanup for disk usage, not the correctness mechanism —
  expiry is also checked at read time inside `get_and_consume`/`peek`, so a
  paste can never be served late even if the sweep hasn't run yet.
- **HTMX, used for one real thing.** The "expires in Ns" line on a paste's
  view page polls `/p/{id}/countdown` every 5 seconds via
  `hx-get`/`hx-trigger`, so the countdown updates without a page reload or
  any hand-written JS polling loop. It's the only place a live server value
  needs to reach the page after load, so it's the only place HTMX is used.
- **Two-part links stay separate.** Creating a paste never redirects you to
  view it — that would burn a one-time link before you had a chance to copy
  it. `/pastes` (POST) returns a "here's your link" confirmation page;
  opening the link is the view.

## Usage

```bash
cd "challenges/Practical Software/Self-Hosted Paste Bin with Expiry & Burn-After-Read"

uv run --with fastapi --with "uvicorn[standard]" --with pygments --with python-multipart python app.py
# -> http://127.0.0.1:8000

# or, with auto-reload during development:
uv run --with fastapi --with "uvicorn[standard]" --with pygments --with python-multipart \
    uvicorn app:app --reload

uv run --with fastapi --with "uvicorn[standard]" --with pygments --with python-multipart \
    --with httpx2 --with pytest pytest -q      # 21 tests
```

Open `http://127.0.0.1:8000`, paste some text, pick a language (or leave it
on `auto` for Pygments' lexer guesser), pick an expiry, optionally check
"burn after read", and submit. `GET /health` reports `{"status": "ok",
"pastes": <count>}` for a load balancer or uptime check. `GET /raw/{id}`
returns the plain-text content (and is subject to the same one-time-view
rule as the HTML view -- fetching it also counts as the view).

## Deploying it for real

It's a normal ASGI app, so anything that runs `uvicorn`/`gunicorn` works:

```bash
# systemd-style long-running process
uv run --with fastapi --with "uvicorn[standard]" --with pygments --with python-multipart \
    uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```

**Use exactly one worker process**, or put a reverse proxy with sticky
routing in front, if you need more: each worker would run its own
`lifespan`-managed cleanup thread against the *same* SQLite file, which is
safe (SQLite handles the concurrent writers fine, see above), but you'd get
`N` redundant cleanup sweeps. That's harmless, just wasteful — this is
explicitly a small self-hosted tool, not a design for a multi-tenant SaaS.
Put it behind Caddy/Nginx/Cloudflare Tunnel for TLS if it's reachable outside
your LAN, since it currently serves plain HTTP.

## What it deliberately doesn't do

- No accounts, no paste history/listing, no password-protected pastes — the
  brief asked for expiry, burn-after-read, and highlighting, not a full
  Pastebin.com clone.
- No rate limiting or max-size enforcement — appropriate for a self-hosted
  tool behind your own reverse proxy, not for an anonymous public instance.

## Tests

21 pytest cases across two files. `test_storage.py` (11 cases) covers create
→ consume round trips, burn-after-read allowing exactly one read, non-burn
pastes surviving repeated reads, TTL expiry enforced at read time (via a
paste created already-expired), the background sweep removing only expired
rows, `peek` never consuming a burn-after-read paste, unique URL-safe ids,
and the concurrency test described above. `test_app.py` (10 cases) drives
the real FastAPI app through `TestClient`: the create → link → view flow,
burn-after-read over real HTTP returning 404 on the second GET, empty
content rejected with 400, the raw endpoint also respecting burn semantics,
the countdown fragment for both a live and an already-gone paste, the health
endpoint, and that Python content actually comes back Pygments-highlighted.
A separate manual smoke test against a live running server (not just
`TestClient`) confirmed the same burn-after-read behavior over real HTTP.

The suite runs warning-free: Starlette's `TestClient` prefers the `httpx2`
package over `httpx` (hence it's in the test command above, not `httpx`),
and `pytest.ini` filters one remaining warning that Starlette 1.6.0's own
`testclient.py` emits at import time against anyio 4.15 or newer (a
reference to a deprecated `anyio.abc.BlockingPortal` alias) — an unfixed
upstream version interaction that nothing in this test suite triggers or
can work around.
