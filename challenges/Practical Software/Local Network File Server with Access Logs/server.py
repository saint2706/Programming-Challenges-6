"""Local Network File Server: serve a folder read-only over the LAN, gated
by a shared-secret token, with every list/download/denial recorded to a
JSON-lines access log.

Run directly:
    uv run --with fastapi --with "uvicorn[standard]" --with python-multipart \\
        python server.py --root . --token mysecret

Or with a reload server (reads config from env vars instead of CLI flags,
since `uvicorn server:app --reload` never calls our `__main__` block):
    FILESERVER_ROOT=. FILESERVER_TOKEN=mysecret \\
        uv run --with fastapi --with "uvicorn[standard]" --with python-multipart \\
        uvicorn server:app --reload

Pages are rendered as small server-side HTML fragments (no template engine,
no static file directory), per this repo's convention -- see the sibling
paste bin / scheduler challenges for the same pattern.
"""

from __future__ import annotations

import html
import mimetypes
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import uvicorn
from access_log import AccessLogger
from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from security import PathTraversalError, hash_token, resolve_safe_path, verify_token
from starlette.responses import Response

BASE_DIR = Path(__file__).parent
SESSION_COOKIE_NAME = "fs_session"
SESSION_TTL_SECONDS = 12 * 3600
_START_TIME = time.time()


@dataclass
class Session:
    created_at: float
    expires_at: float


app = FastAPI(title="Local Network File Server")

# Populated by configure(); declared here so type checkers and tests can see
# them as module attributes. Real values are set immediately below (from env
# vars) and can be re-set at any time -- by the __main__ CLI block, or by a
# test fixture pointing the server at a fresh tmp_path root/token/log.
SERVE_ROOT: Path
_TOKEN_HASH: bytes
_access_logger: AccessLogger
_SESSIONS: dict[str, Session]


def configure(*, root: Path, token: str, log_path: Path) -> None:
    """(Re)point the server at a served root, shared secret, and log file.

    Also resets in-memory sessions, so a reconfigure (as tests do, once per
    test, against a fresh tmp_path) never leaks a session across tests.
    """
    global SERVE_ROOT, _TOKEN_HASH, _access_logger, _SESSIONS
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(
            f"served root does not exist or is not a directory: {root}"
        )
    SERVE_ROOT = root.resolve()
    _TOKEN_HASH = hash_token(token)
    _access_logger = AccessLogger(Path(log_path))
    _SESSIONS = {}


_default_token = os.environ.get("FILESERVER_TOKEN")
if _default_token is None:
    _default_token = secrets.token_urlsafe(24)
    print(
        f"[fileserver] No FILESERVER_TOKEN set -- generated a one-time token "
        f"for this run:\n\n    {_default_token}\n"
    )
configure(
    root=Path(os.environ.get("FILESERVER_ROOT", ".")),
    token=_default_token,
    log_path=Path(os.environ.get("FILESERVER_LOG", str(BASE_DIR / "access.log.jsonl"))),
)


# --------------------------------------------------------------------------
# Auth: a bearer token for API-style clients, or a signed-by-possession
# session cookie (set on password login) for browsing normally. Sessions
# live only in this process's memory -- restarting the server logs
# everyone out, which is the right trade for a small self-hosted tool with
# no database.
# --------------------------------------------------------------------------


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _create_session() -> str:
    session_id = secrets.token_urlsafe(32)
    now = time.time()
    _SESSIONS[session_id] = Session(
        created_at=now, expires_at=now + SESSION_TTL_SECONDS
    )
    return session_id


def _session_valid(session_id: str | None) -> bool:
    if session_id is None:
        return False
    session = _SESSIONS.get(session_id)
    if session is None:
        return False
    if session.expires_at <= time.time():
        del _SESSIONS[session_id]
        return False
    return True


def _identify(request: Request) -> tuple[str, bool]:
    """Return (identity label for logging, authenticated?)."""
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        candidate = auth[7:].strip()
        if verify_token(candidate, _TOKEN_HASH):
            return "token", True
        return "invalid-token", False
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if _session_valid(session_id):
        assert session_id is not None
        return f"session:{session_id[:8]}", True
    return "anonymous", False


def require_auth(request: Request) -> str:
    identity, ok = _identify(request)
    if ok:
        return identity

    _access_logger.log(
        identity=identity,
        ip=_client_ip(request),
        method=request.method,
        path=request.url.path,
        action="denied",
        status=401,
    )
    if "text/html" in request.headers.get("accept", ""):
        next_q = quote(request.url.path, safe="")
        raise HTTPException(
            status_code=303, headers={"Location": f"/login?next={next_q}"}
        )
    raise HTTPException(
        status_code=401,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

CSS = """
:root { color-scheme: light dark; }
body { font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }
header { margin-bottom: 1.5rem; display: flex; justify-content: space-between; align-items: center; }
header a.brand { font-weight: 700; font-size: 1.25rem; text-decoration: none; }
table { width: 100%; border-collapse: collapse; }
th, td { text-align: left; padding: 0.35rem 0.5rem; border-bottom: 1px solid #8883; }
th { font-size: 0.85rem; text-transform: uppercase; opacity: 0.7; }
.dir::before { content: "\\1F4C1  "; }
.file::before { content: "\\1F4C4  "; }
.crumbs { margin-bottom: 1rem; opacity: 0.8; }
.crumbs a { text-decoration: none; }
input[type=password], button { font-size: 1rem; padding: 0.4rem 0.6rem; }
.error { color: #b91c1c; }
.muted { opacity: 0.7; font-size: 0.9rem; }
code { font-family: ui-monospace, monospace; }
"""


def page(
    title: str, body: str, identity_html: str = "", status_code: int = 200
) -> HTMLResponse:
    return HTMLResponse(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{CSS}</style>
</head>
<body>
<header><a class="brand" href="/">Local File Server</a>{identity_html}</header>
<main>{body}</main>
</body>
</html>""",
        status_code=status_code,
    )


def _nav_html(request: Request) -> str:
    identity, ok = _identify(request)
    if not ok:
        return '<a href="/login">Log in</a>'
    return (
        f'<span class="muted">{identity}</span> '
        '<form method="post" action="/logout" style="display:inline">'
        '<button type="submit">Log out</button></form>'
    )


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def render_login(next_path: str, error: str | None = None) -> str:
    error_html = f'<p class="error">{html.escape(error)}</p>' if error else ""
    return f"""
<h1>Log in</h1>
{error_html}
<form method="post" action="/login">
  <input type="hidden" name="next" value="{html.escape(next_path)}">
  <p><input type="password" name="password" placeholder="Shared token" required autofocus></p>
  <button type="submit">Log in</button>
</form>
"""


def render_listing(rel_path: str, entries: list[tuple[str, bool, int, float]]) -> str:
    crumb_parts = [p for p in rel_path.split("/") if p]
    crumbs = ['<a href="/browse/">root</a>']
    built = ""
    for part in crumb_parts:
        built = f"{built}/{part}" if built else part
        crumbs.append(
            f'<a href="/browse/{quote(built, safe="/")}">{html.escape(part)}</a>'
        )
    crumbs_html = f'<p class="crumbs">{" / ".join(crumbs)}</p>'

    rows = []
    for name, is_dir, size, mtime in entries:
        entry_rel = f"{rel_path}/{name}" if rel_path else name
        href = f"/browse/{quote(entry_rel, safe='/')}" + ("/" if is_dir else "")
        css_class = "dir" if is_dir else "file"
        size_html = "-" if is_dir else _human_size(size)
        mtime_html = time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))
        rows.append(
            f'<tr><td class="{css_class}"><a href="{href}">{html.escape(name)}</a></td>'
            f"<td>{size_html}</td><td>{mtime_html}</td></tr>"
        )
    if not rows:
        rows.append('<tr><td colspan="3" class="muted">(empty)</td></tr>')

    return f"""
<h1>Index of /{html.escape(rel_path)}</h1>
{crumbs_html}
<table>
<tr><th>Name</th><th>Size</th><th>Modified</th></tr>
{"".join(rows)}
</table>
"""


def render_logs(entries: list[dict]) -> str:
    if not entries:
        rows = '<tr><td colspan="6" class="muted">(no access log entries yet)</td></tr>'
    else:
        # Every field here can contain attacker-chosen text (`path` most of
        # all -- it's a raw URL an attacker picked), so this is a stored-XSS
        # spot if unescaped: an attacker requests a path containing markup,
        # it gets logged, and later renders in an admin's browser here.
        rows = "".join(
            f"<tr><td>{html.escape(str(e['ts']))}</td><td>{html.escape(str(e['identity']))}</td>"
            f"<td>{html.escape(str(e['ip']))}</td><td>{html.escape(str(e['method']))}</td>"
            f"<td><code>{html.escape(str(e['path']))}</code></td>"
            f"<td>{html.escape(str(e['action']))} ({e['status']})</td></tr>"
            for e in entries
        )
    return f"""
<h1>Access log</h1>
<p class="muted">Most recent {len(entries)} entries, newest first.</p>
<table>
<tr><th>Time (UTC)</th><th>Identity</th><th>IP</th><th>Method</th><th>Path</th><th>Action</th></tr>
{rows}
</table>
"""


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    body = """
<h1>Local Network File Server</h1>
<p>A read-only, token-protected share of a folder on this machine's LAN.</p>
<p><a href="/browse/">Browse files</a> &middot; <a href="/logs">Access log</a></p>
"""
    return page("Local File Server", body, _nav_html(request))


def _safe_next(next_path: str) -> str:
    """Only ever accept a same-site path as a post-login redirect target.

    `next` is user input (query string on GET, form field on POST) echoed
    back into the login page and later used as a redirect Location -- an
    unvalidated `//evil.com` or `https://evil.com` would be an open
    redirect. Applied on both the GET (where it's just displayed) and POST
    (where it's actually redirected to) paths, so a crafted `next` is never
    trusted regardless of which one a client hits.
    """
    if next_path.startswith("/") and not next_path.startswith("//"):
        return next_path
    return "/browse/"


@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request, next: str = "/browse/") -> HTMLResponse:
    return page("Log in", render_login(_safe_next(next)))


@app.post("/login", response_class=HTMLResponse)
def login_submit(
    request: Request, password: str = Form(...), next: str = Form("/browse/")
) -> Response:
    ip = _client_ip(request)
    safe_next = _safe_next(next)
    if not verify_token(password, _TOKEN_HASH):
        _access_logger.log(
            identity="anonymous",
            ip=ip,
            method="POST",
            path="/login",
            action="login_failed",
            status=401,
        )
        return page(
            "Log in", render_login(safe_next, error="Incorrect token."), status_code=401
        )

    session_id = _create_session()
    _access_logger.log(
        identity=f"session:{session_id[:8]}",
        ip=ip,
        method="POST",
        path="/login",
        action="login_success",
        status=303,
    )
    resp = RedirectResponse(safe_next, status_code=303)
    resp.set_cookie(
        SESSION_COOKIE_NAME,
        session_id,
        httponly=True,
        samesite="lax",
        max_age=SESSION_TTL_SECONDS,
    )
    return resp


@app.post("/logout")
def logout(request: Request) -> RedirectResponse:
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if session_id is not None:
        _SESSIONS.pop(session_id, None)
    resp = RedirectResponse("/", status_code=303)
    resp.delete_cookie(SESSION_COOKIE_NAME)
    return resp


def _browse_impl(file_path: str, request: Request, identity: str) -> Response:
    ip = _client_ip(request)
    try:
        target = resolve_safe_path(SERVE_ROOT, file_path)
    except PathTraversalError:
        _access_logger.log(
            identity=identity,
            ip=ip,
            method="GET",
            path=request.url.path,
            action="denied",
            status=400,
        )
        raise HTTPException(status_code=400, detail="Invalid path") from None

    if not target.exists():
        _access_logger.log(
            identity=identity,
            ip=ip,
            method="GET",
            path=request.url.path,
            action="not_found",
            status=404,
        )
        raise HTTPException(status_code=404, detail="Not found")

    if target.is_dir():
        _access_logger.log(
            identity=identity,
            ip=ip,
            method="GET",
            path=request.url.path,
            action="list",
            status=200,
        )
        entries = sorted(
            (
                (
                    p.name,
                    p.is_dir(),
                    0 if p.is_dir() else p.stat().st_size,
                    p.stat().st_mtime,
                )
                for p in target.iterdir()
            ),
            key=lambda e: (not e[1], e[0].lower()),
        )
        rel = file_path.strip("/")
        return page(
            f"Index of /{rel}", render_listing(rel, entries), _nav_html(request)
        )

    _access_logger.log(
        identity=identity,
        ip=ip,
        method="GET",
        path=request.url.path,
        action="download",
        status=200,
    )
    media_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    return FileResponse(path=target, media_type=media_type, filename=target.name)


@app.get("/browse", response_class=HTMLResponse)
@app.get("/browse/", response_class=HTMLResponse)
def browse_root(request: Request, identity: str = Depends(require_auth)) -> Response:
    return _browse_impl("", request, identity)


@app.get("/browse/{file_path:path}")
def browse_path(
    file_path: str, request: Request, identity: str = Depends(require_auth)
) -> Response:
    return _browse_impl(file_path, request, identity)


@app.get("/logs", response_class=HTMLResponse)
def logs_page(request: Request, identity: str = Depends(require_auth)) -> HTMLResponse:
    # Deliberately not logged as its own access-log entry -- logging every
    # visit to the log viewer would make the log grow every time someone
    # reads it, for no benefit.
    return page("Access log", render_logs(_access_logger.tail(200)), _nav_html(request))


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "uptime_seconds": int(time.time() - _START_TIME)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Serve a folder read-only over the LAN, with token auth and access logging."
    )
    parser.add_argument(
        "--root",
        default=os.environ.get("FILESERVER_ROOT", "."),
        help="folder to serve (default: cwd)",
    )
    parser.add_argument("--host", default=os.environ.get("FILESERVER_HOST", "0.0.0.0"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("FILESERVER_PORT", "8000"))
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("FILESERVER_TOKEN"),
        help="shared secret; a random one is generated and printed if omitted",
    )
    parser.add_argument(
        "--log",
        default=os.environ.get("FILESERVER_LOG", str(BASE_DIR / "access.log.jsonl")),
        help="path to the JSON-lines access log (default: access.log.jsonl next to this script)",
    )
    args = parser.parse_args()

    cli_token = args.token or secrets.token_urlsafe(24)
    if not args.token:
        print(
            f"[fileserver] No --token given -- generated a one-time token:\n\n    {cli_token}\n"
        )
    configure(root=Path(args.root), token=cli_token, log_path=Path(args.log))
    print(
        f"[fileserver] Serving {SERVE_ROOT} on http://{args.host}:{args.port} "
        f"-- from another device on your LAN, use this machine's LAN IP instead of {args.host}."
    )
    uvicorn.run(app, host=args.host, port=args.port)
