# Local Network File Server with Access Logs

**Category:** Practical Software
**Difficulty:** B (brief: "Serve a folder over LAN, log who fetched what and when.")

**Status:** Implemented (Python)

A small, token-protected file server for sharing a folder with other devices
on your LAN: directory browsing, downloads, and a full accounting of who
fetched what and when. No database, no accounts system — one shared secret,
one JSON-lines log file, one Python process.

## The actually-hard part: a URL is not a filesystem path

The brief sounds like `os.listdir()` behind a web page, but "serve a folder
over the network" is exactly the shape of bug that produces a CVE: any code
that turns an attacker-controlled URL into a filesystem path is a
path-traversal vulnerability unless it's proven otherwise. `security.py`'s
`resolve_safe_path` is the one place that's allowed to turn a URL fragment
into a `Path`, and it has to survive more than the obvious `../../etc/passwd`:

- **Percent-encoding.** `..%2Fsecret.txt` and `%2e%2e%2fsecret.txt` both
  decode to a literal `..` segment — closed by decoding before splitting on
  `/`, then rejecting any segment that equals `..`.
- **Double percent-encoding.** `%252e%252e%252f` decodes *once* (by the ASGI
  layer, before our code ever sees it) to `%2e%2e%2f` — still not a literal
  `..`. A second `unquote()` inside `resolve_safe_path` closes this; a naive
  "check the raw string for `..`" implementation would not catch it. Verified
  empirically (see Tests) against what uvicorn/Starlette actually hand our
  route, not assumed.
- **Absolute-path injection.** `Path.joinpath` silently *discards* everything
  before an absolute argument — `Path("/root").joinpath("/etc/passwd")` is
  `/etc/passwd`, not `/root/etc/passwd`. Every segment is validated
  individually before joining, so this class of bug can't reach the join at
  all (and splitting first turns out to neutralize it structurally — see the
  `test_double_slash_collapses_harmlessly` test and its comment).
- **Symlink escapes.** `.resolve()` follows symlinks, and the containment
  check (`root in resolved.parents`) runs *after* resolution, so a symlink
  planted inside the served root pointing outside it is still caught.
- **Null bytes and backslashes.** Rejected outright — neither has a
  legitimate reason to appear in a URL path segment on this server.

The policy is deliberately blunt: **any** `..` segment is rejected, even one
that would mathematically cancel out and stay inside root (`sub/../top.txt`).
Proving containment after normalization is a harder property to get right
than "never allow the suspicious token to begin with," and this is a small
self-hosted tool, not a general-purpose path library.

## Design

- **`security.py`** — `resolve_safe_path` (the traversal defense above) and
  constant-time shared-secret hashing/verification (`hash_token`/
  `verify_token`, SHA-256 + `hmac.compare_digest`). Zero web-framework
  dependency, so it's the most heavily unit-tested file in the challenge.
- **`access_log.py`** — `AccessLogger`: append-only JSON-lines writer behind
  a `threading.Lock` (FastAPI runs sync handlers in a thread pool, so
  concurrent requests can genuinely race to append at once — unsynchronized
  writes can interleave two requests' bytes into one corrupt line) plus a
  `tail()` for the in-app log viewer.
- **`server.py`** — FastAPI routes, HTML rendering, and auth. Pages are
  small server-rendered HTML fragments (`f"""..."""` + `html.escape()`),
  matching this repo's convention of no template-file directory for a
  single self-contained script.
- **Two auth mechanisms, one shared secret.** A bearer token
  (`Authorization: Bearer <token>`) for scripts/`curl`, or a password-login
  form that issues an HttpOnly session cookie for normal browsing. Sessions
  live only in server memory (a `dict`, reset on restart or on
  `configure()`) — no database, and restarting the server logs everyone out,
  which is the right trade for a small self-hosted tool.
- **Every field in the access log is attacker-influenced, so every field is
  escaped on the way back out.** `path` is the most obvious case — it's a
  raw URL an attacker chose — but `identity` can also carry arbitrary text
  (an invalid bearer token isn't validated as "safe" before being logged;
  only `hash_token`'d comparison happens, the raw candidate itself never
  reaches the log, but a crafted `Authorization` header could still smuggle
  markup into other fields in principle). The admin log page renders these
  server-side with `html.escape()` on every interpolated value — otherwise
  this is a textbook stored-XSS setup: attacker picks a URL, it gets logged,
  an admin later views `/logs` in their own authenticated browser session.
  Directory listings (filenames are attacker-influenceable if anyone else
  can write into the served folder) and the login page's `next` redirect
  parameter get the same treatment, and `next` is additionally validated to
  be a same-site path (rejecting `//evil.example.com`-style open redirects)
  on both the page that displays it and the page that redirects to it.
- **`FILESERVER_ROOT` / `FILESERVER_TOKEN` / `FILESERVER_LOG` env vars**
  configure the server at import time (so `uvicorn server:app --reload`
  works for development), with `--root`/`--token`/`--log`/`--host`/`--port`
  CLI flags as the equivalent for the `python server.py` entry point. If no
  token is given either way, one is generated and printed once — the same
  pattern Jupyter uses for its own token auth.

## Usage

```bash
cd "challenges/Practical Software/Local Network File Server with Access Logs"

uv run --with fastapi --with "uvicorn[standard]" --with python-multipart \
    python server.py --root /path/to/folder --token "a-secret-only-you-know"
# -> Serving <root> on http://0.0.0.0:8000

# or, with auto-reload during development:
FILESERVER_ROOT=. FILESERVER_TOKEN=devsecret \
    uv run --with fastapi --with "uvicorn[standard]" --with python-multipart \
    uvicorn server:app --reload

uv run --with fastapi --with "uvicorn[standard]" --with python-multipart \
    --with httpx2 --with pytest pytest -q      # 66 tests
```

**Connecting from another device on your LAN:** find this machine's LAN IP
(`ipconfig` on Windows, `ip addr`/`ifconfig` on Linux/macOS — look for
something like `192.168.x.x` or `10.x.x.x`), then from the other device
browse to `http://<that-ip>:8000/browse/` and log in with the token, or
`curl -H "Authorization: Bearer a-secret-only-you-know" http://<that-ip>:8000/browse/`.
`GET /health` is unauthenticated (for a load balancer or uptime check) and
reports `{"status": "ok", "uptime_seconds": N}` without leaking the served
path. `GET /logs` (authenticated) shows the most recent 200 access-log
entries in a browser; visiting it isn't itself logged, to avoid the log
growing every time someone reads it.

## What it deliberately doesn't do

- **No upload** — read-only sharing was the brief; a write path is a very
  different (and much larger) threat model.
- **No HTTPS/TLS** — it serves plain HTTP. The bearer token and session
  cookie are both sniffable by anything else on the same LAN segment; this
  is appropriate for a trusted home/office network, not an untrusted one.
  Put it behind a reverse proxy with TLS termination if that matters to you.
- **No byte-range / resume support** — downloads are whole-file streams.
  Real for large-file resume, but out of scope for a Beginner-tier brief
  that's really about auth + traversal safety + logging.
- **No rate limiting or account lockout** — a brute-force guesser can hammer
  `/login` as fast as the network allows. Fine behind your own router; not
  something you'd expose past it without a reverse proxy in front.

## Tests

66 pytest cases across three files.

`test_security.py` (27 cases) unit-tests `resolve_safe_path` directly: valid
nested lookups, `.`-segment normalization, a 12-payload traversal battery
(literal `..`, percent-encoded, **double**-percent-encoded, mixed with
legitimate segments, backslash variants, null bytes, a Windows drive-letter
segment), plus the shared-secret hashing/verification behavior. Also
`test_double_slash_collapses_harmlessly`, which documents — with a passing
test, not just a comment — the empirical finding that `Path.joinpath`'s
"absolute argument discards everything before it" pitfall can't actually
fire here, because segments are split and validated individually before any
join happens.

`test_access_log.py` (6 cases) covers the JSON-lines format, `tail()`
ordering and limits, skipping a corrupted line rather than failing the
whole read, and — the one that actually matters — 20 threads racing to
`log()` at once via a `threading.Barrier`, asserting the resulting file is
still exactly 20 valid, distinct JSON lines (proving the lock genuinely
prevents interleaved writes, not just "usually works").

`test_server.py` (33 cases) drives the real FastAPI app through
`TestClient`: public vs. authenticated routes, both auth mechanisms
(bearer token and session-cookie login, including wrong-password and
open-redirect-`next` rejection), directory listing and byte-identical
downloads (including a nested file), a 5-payload traversal battery at the
HTTP level (distinct from the unit-level battery — this one proves the
route layer's integration with `resolve_safe_path` is wired correctly, not
just the function in isolation), access-log correctness for downloads,
denials, traversal attempts, and logins, and three adversarial XSS tests
(a filename containing HTML-significant characters, a crafted log
`identity` field, and a crafted `next` query parameter) asserting the
dangerous markup never appears unescaped in the rendered response.

A separate manual verification against a **live** server on `127.0.0.1`
(not `TestClient`) confirmed all of the above over real HTTP with `curl`,
plus one thing the test suite can't show: `curl` doesn't pre-normalize URLs
the way `httpx`'s test client does, so a literal `/browse/../secret.txt`
sent raw over the wire never even reaches our route — uvicorn/Starlette's
own path handling collapses it first, producing a generic 404 before our
code runs at all. The percent-encoded and double-percent-encoded variants
*do* reach `_browse_impl` and are caught there (400). Either way the secret
file's content never appeared in a response, which is the property that
actually matters — confirmed by diffing real downloaded files (including a
50KB binary) byte-for-byte against their source, checking the access log
file afterward recorded every request accurately, and confirming a wrong
token and a missing token both get a real 401.

The suite runs warning-free: Starlette's `TestClient` prefers the `httpx2`
package over `httpx` (hence it's in the test command above, not `httpx`),
and `pytest.ini` filters one remaining warning that Starlette 1.6.0's own
`testclient.py` emits at import time against anyio 4.15 or newer — an
unfixed upstream version interaction that nothing in this test suite
triggers or can work around.
