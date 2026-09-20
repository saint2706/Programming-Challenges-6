"""Self-hosted paste bin: TTL expiry, burn-after-read, syntax highlighting.

Run directly:
    uv run --with fastapi --with uvicorn[standard] --with pygments --with python-multipart python app.py

Or with a reload server:
    uv run --with fastapi --with uvicorn[standard] --with pygments --with python-multipart \\
        uvicorn app:app --reload

Pages are rendered as small server-side HTML fragments (no template engine,
no static file directory) to keep this a single self-contained script, per
this repo's convention. HTMX handles the one place where a live server value
matters without a full page reload: the "expires in Ns" countdown on a
paste's view page polls `/p/{id}/countdown` every 5 seconds.
"""

from __future__ import annotations

import html
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import TextLexer, get_lexer_by_name, guess_lexer
from pygments.util import ClassNotFound
from storage import Paste, PasteStore

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "pastebin.db"
CLEANUP_INTERVAL_SECONDS = 60

LANGUAGES = [
    "auto",
    "text",
    "python",
    "javascript",
    "typescript",
    "bash",
    "json",
    "yaml",
    "sql",
    "html",
    "css",
    "go",
    "rust",
    "java",
    "c",
    "cpp",
    "csharp",
    "ruby",
    "php",
    "markdown",
]
EXPIRY_CHOICES: dict[str, int | None] = {
    "never": None,
    "10m": 600,
    "1h": 3600,
    "1d": 86400,
    "7d": 604800,
}

store = PasteStore(DB_PATH)
formatter = HtmlFormatter(style="friendly", nowrap=False, cssclass="highlight")

_stop_event = threading.Event()
_cleanup_thread: threading.Thread | None = None


def _cleanup_loop() -> None:
    while not _stop_event.wait(CLEANUP_INTERVAL_SECONDS):
        store.delete_expired()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global _cleanup_thread
    _stop_event.clear()
    _cleanup_thread = threading.Thread(target=_cleanup_loop, daemon=True)
    _cleanup_thread.start()
    try:
        yield
    finally:
        _stop_event.set()
        if _cleanup_thread is not None:
            _cleanup_thread.join(timeout=2)


app = FastAPI(title="Self-Hosted Paste Bin", lifespan=lifespan)


# --------------------------------------------------------------------------
# Rendering (plain string templates, everything is html.escape()'d except
# pygments output, which escapes the source itself before wrapping it).
# --------------------------------------------------------------------------

CSS = """
:root { color-scheme: light dark; }
body { font-family: system-ui, sans-serif; max-width: 860px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }
header { margin-bottom: 1.5rem; }
header a.brand { font-weight: 700; font-size: 1.25rem; text-decoration: none; }
textarea { width: 100%; min-height: 16rem; font-family: ui-monospace, monospace; font-size: 0.9rem; box-sizing: border-box; }
select, input[type=text], button { font-size: 1rem; padding: 0.4rem 0.6rem; }
form .row { margin: 0.75rem 0; display: flex; gap: 1rem; align-items: center; flex-wrap: wrap; }
.link-box { display: flex; gap: 0.5rem; margin: 1rem 0; }
.link-box input { flex: 1; font-family: ui-monospace, monospace; }
.warning { color: #b45309; }
.meta { color: #666; font-size: 0.9rem; margin-bottom: 0.5rem; }
button { cursor: pointer; }
"""


def page(title: str, body: str, extra_head: str = "") -> HTMLResponse:
    return HTMLResponse(f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<script src="https://unpkg.com/htmx.org@2.0.3/dist/htmx.min.js"
        integrity="sha384-0895/pl2MU10Hqc6jd4RvrthNlDiE9U1tWmX7WRESftEDRosgxNsQG/Ze9YMRzHq"
        crossorigin="anonymous"></script>
<style>{CSS}</style>
{extra_head}
</head>
<body>
<header><a class="brand" href="/">Paste Bin</a></header>
<main>{body}</main>
</body>
</html>""")


def _options(values: list[str], selected: str) -> str:
    return "".join(
        f'<option value="{html.escape(v)}"{" selected" if v == selected else ""}>{html.escape(v)}</option>'
        for v in values
    )


def render_index() -> HTMLResponse:
    body = f"""
<h1>New paste</h1>
<form method="post" action="/pastes">
  <textarea name="content" placeholder="Paste your text or code here..." required></textarea>
  <div class="row">
    <label>Language <select name="language">{_options(LANGUAGES, "auto")}</select></label>
    <label>Expires <select name="expiry">{_options(list(EXPIRY_CHOICES), "1d")}</select></label>
    <label><input type="checkbox" name="burn_after_read" value="true"> Burn after read (one-time view)</label>
  </div>
  <button type="submit">Create paste</button>
</form>
"""
    return page("New paste", body)


def render_created(link: str, burn: bool, expiry: str) -> HTMLResponse:
    warning = (
        '<p class="warning">This link works exactly once. Refreshing or sharing it twice will show "gone" '
        "the second time.</p>"
        if burn
        else f"<p>This link expires: <strong>{html.escape(expiry)}</strong>.</p>"
    )
    body = f"""
<h1>Paste created</h1>
<div class="link-box">
  <input type="text" id="link" value="{html.escape(link)}" readonly>
  <button onclick="navigator.clipboard.writeText(document.getElementById('link').value)">Copy</button>
</div>
{warning}
<p><a href="{html.escape(link)}">Open it</a> (opening it now counts as the view).</p>
"""
    return page("Paste created", body)


def _highlight(content: str, language: str) -> tuple[str, str]:
    try:
        lexer = (
            guess_lexer(content) if language == "auto" else get_lexer_by_name(language)
        )
    except ClassNotFound:
        lexer = TextLexer()
    highlighted = highlight(content, lexer, formatter)
    detected = lexer.aliases[0] if lexer.aliases else "text"
    return highlighted, detected


def render_view(paste: Paste, paste_id: str) -> HTMLResponse:
    highlighted, detected_language = _highlight(paste.content, paste.language)
    created = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(paste.created_at))

    if paste.burn_after_read:
        ttl_html = '<p class="warning">This paste has now been burned: the link is dead and cannot be viewed again.</p>'
    elif paste.expires_at is None:
        ttl_html = f'<p><span id="ttl" hx-get="/p/{paste_id}/countdown" hx-trigger="every 5s" hx-swap="outerHTML">never expires</span></p>'
    else:
        remaining = max(0, int(paste.expires_at - time.time()))
        ttl_html = (
            f'<p><span id="ttl" hx-get="/p/{paste_id}/countdown" hx-trigger="every 5s" hx-swap="outerHTML">'
            f"expires in {remaining}s</span></p>"
        )

    body = f"""
<h1>Paste</h1>
<p class="meta">language: {html.escape(detected_language)} &middot; created: {created} &middot;
<a href="/raw/{paste_id}">raw</a></p>
{ttl_html}
{highlighted}
"""
    return page(
        "Paste",
        body,
        extra_head=f"<style>{formatter.get_style_defs('.highlight')}</style>",
    )


def render_gone() -> HTMLResponse:
    return page(
        "Not found",
        "<h1>Gone</h1><p>This paste does not exist, has expired, or was already burned.</p>",
    )


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return render_index()


@app.post("/pastes", response_class=HTMLResponse)
def create_paste(
    request: Request,
    content: str = Form(...),
    language: str = Form("auto"),
    expiry: str = Form("never"),
    burn_after_read: bool = Form(False),
) -> HTMLResponse:
    if not content.strip():
        raise HTTPException(400, "Content must not be empty")
    if expiry not in EXPIRY_CHOICES:
        raise HTTPException(400, f"Unknown expiry choice: {expiry!r}")
    paste = store.create(
        content=content,
        language=language,
        ttl_seconds=EXPIRY_CHOICES[expiry],
        burn_after_read=burn_after_read,
    )
    link = str(request.url_for("view_paste", paste_id=paste.id))
    return render_created(link, burn_after_read, expiry)


@app.get("/p/{paste_id}", response_class=HTMLResponse, name="view_paste")
def view_paste(paste_id: str) -> HTMLResponse:
    paste = store.get_and_consume(paste_id)
    if paste is None:
        return HTMLResponse(render_gone().body, status_code=404)
    return render_view(paste, paste_id)


@app.get("/raw/{paste_id}", response_class=PlainTextResponse)
def raw_paste(paste_id: str) -> str:
    paste = store.get_and_consume(paste_id)
    if paste is None:
        raise HTTPException(404, "Paste not found, expired, or already burned")
    return paste.content


@app.get("/p/{paste_id}/countdown", response_class=HTMLResponse)
def countdown(paste_id: str) -> HTMLResponse:
    paste = store.peek(paste_id)
    if paste is None:
        return HTMLResponse('<span id="ttl">expired</span>')
    if paste.expires_at is None:
        return HTMLResponse(
            f'<span id="ttl" hx-get="/p/{paste_id}/countdown" hx-trigger="every 5s" hx-swap="outerHTML">'
            "never expires</span>"
        )
    remaining = max(0, int(paste.expires_at - time.time()))
    if remaining == 0:
        return HTMLResponse('<span id="ttl">expired</span>')
    return HTMLResponse(
        f'<span id="ttl" hx-get="/p/{paste_id}/countdown" hx-trigger="every 5s" hx-swap="outerHTML">'
        f"expires in {remaining}s</span>"
    )


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "pastes": store.count()}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
