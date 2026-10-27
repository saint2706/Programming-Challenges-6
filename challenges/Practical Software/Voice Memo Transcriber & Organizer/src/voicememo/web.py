"""Local web UI (Litestar + Jinja2 + HTMX). Loopback by default.

Security measures, each covered by a regression test:
- Host header must name loopback (127.0.0.1, localhost, ::1): blocks DNS rebinding.
- State-changing requests need a matching Origin (when sent) and a CSRF token (double-submit:
  the cookie value must equal the form field, compared in constant time).
- Every memo field, transcript, tag and file name is HTML-escaped by Jinja autoescape.
- Audio is served by memo id. The path comes from a hash-derived name in the library row, never
  from the request, so there is no path traversal to exploit.
"""

import hmac
import re
import secrets
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit

from jinja2 import Environment, FileSystemLoader, select_autoescape
from litestar import Litestar, Request, Response, get, post
from litestar.datastructures import Cookie
from litestar.enums import MediaType
from litestar.params import FromPath, QueryParameter
from litestar.response import Stream
from litestar.static_files import create_static_files_router
from markupsafe import Markup, escape

from voicememo.library import Library, LibraryError, Memo

HERE = Path(__file__).resolve().parent
TEMPLATES = HERE / "templates"
STATIC = HERE / "static"
LOOPBACK_NAMES = frozenset({"127.0.0.1", "localhost", "::1"})
CSRF_COOKIE = "vm_csrf"
UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
MAX_QUERY = 200
CHUNK = 64 * 1024
AUDIO_TYPES = {
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".aac": "audio/aac",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".opus": "audio/ogg",
    ".flac": "audio/flac",
    ".amr": "audio/amr",
    ".3gp": "audio/3gpp",
    ".caf": "audio/x-caf",
    ".webm": "audio/webm",
}
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "media-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}

_env = Environment(
    loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"])
)


def host_name(host_header: str) -> str:
    """'localhost:8000' -> 'localhost', '[::1]:8000' -> '::1'."""
    value = host_header.strip().lower()
    if value.startswith("["):
        end = value.find("]")
        return value[1:end] if end != -1 else ""
    if value.count(":") == 1:
        return value.split(":", 1)[0]
    return value


def highlight(snippet: str) -> Markup:
    """Escape first, then turn the FTS snippet markers into <mark> tags."""
    escaped = str(escape(snippet))
    return Markup(escaped.replace("[", "<mark>").replace("]", "</mark>"))


def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """Return (start, end) inclusive for a single byte range, None for no header.

    Raises ValueError for malformed or unsatisfiable ranges (the handler answers 416).
    """
    if not header:
        return None
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
    if not match or (not match.group(1) and not match.group(2)):
        raise ValueError("malformed range")
    first, last = match.group(1), match.group(2)
    if size == 0:
        raise ValueError("empty file")
    if not first:  # suffix range: last N bytes
        length = int(last)
        if length == 0:
            raise ValueError("empty suffix")
        return max(0, size - length), size - 1
    start = int(first)
    end = min(int(last), size - 1) if last else size - 1
    if start > end or start >= size:
        raise ValueError("unsatisfiable range")
    return start, end


def _iter_file(path: Path, start: int, length: int) -> Iterator[bytes]:
    with path.open("rb") as fh:
        fh.seek(start)
        remaining = length
        while remaining > 0:
            block = fh.read(min(CHUNK, remaining))
            if not block:
                break
            remaining -= len(block)
            yield block


def _render(template: str, **ctx: object) -> str:
    return _env.get_template(template).render(**ctx)


def _html(body: str, *, token: str, status: int = 200) -> Response:
    return Response(
        content=body,
        media_type=MediaType.HTML,
        status_code=status,
        cookies=[
            Cookie(
                key=CSRF_COOKIE, value=token, httponly=True, samesite="strict", path="/"
            )
        ],
    )


def _csrf_ok(request: Request, form_token: str | None) -> bool:
    cookie = request.cookies.get(CSRF_COOKIE, "")
    return bool(cookie and form_token and hmac.compare_digest(cookie, form_token))


def create_app(library: Library) -> Litestar:
    def guard(request: Request) -> Response | None:
        """Runs before every handler: Host allow-list, then Origin check for state changes."""
        host = request.headers.get("host", "")
        if host_name(host) not in LOOPBACK_NAMES:
            return Response(
                content="bad host", status_code=400, media_type=MediaType.TEXT
            )
        if request.method in UNSAFE_METHODS:
            origin = request.headers.get("origin")
            if origin is not None and urlsplit(origin).netloc.lower() != host.lower():
                return Response(
                    content="cross-origin request refused",
                    status_code=403,
                    media_type=MediaType.TEXT,
                )
        return None

    def harden(response: Response) -> Response:
        for key, value in SECURITY_HEADERS.items():
            response.headers[key] = value
        return response

    def _memo_or_none(memo_id: FromPath[int]) -> Memo | None:
        try:
            return library.get(memo_id)
        except LibraryError:
            return None

    @get("/", sync_to_thread=False)
    def index(
        request: Request,
        q: Annotated[str, QueryParameter()] = "",
        tag: Annotated[str, QueryParameter()] = "",
    ) -> Response:
        q = q[:MAX_QUERY]
        token = request.cookies.get(CSRF_COOKIE) or secrets.token_urlsafe(32)
        hits = []
        memos: list[Memo] = []
        if q.strip():
            hits = [
                {
                    "memo_id": h.memo_id,
                    "title": h.title,
                    "start": h.start,
                    "snippet": highlight(h.snippet),
                }
                for h in library.search(q)
            ]
        else:
            memos = library.list_memos(tag=tag[:MAX_QUERY] or None)
        body = _render(
            "index.html",
            q=q,
            tag=tag,
            hits=hits,
            memos=memos,
            tag_counts=library.tag_counts(),
        )
        return _html(body, token=token)

    @get("/memos/{memo_id:int}", sync_to_thread=False)
    def detail(request: Request, memo_id: FromPath[int]) -> Response:
        memo = _memo_or_none(memo_id)
        token = request.cookies.get(CSRF_COOKIE) or secrets.token_urlsafe(32)
        if memo is None:
            return _html(
                _render("error.html", message="No such memo."), token=token, status=404
            )
        body = _render(
            "memo.html",
            memo=memo,
            segments=library.segments(memo.id),
            tags=library.tags(memo.id),
            csrf=token,
            has_audio=_audio_file(memo) is not None,
        )
        return _html(body, token=token)

    def _audio_file(memo: Memo) -> Path | None:
        try:
            path = library.audio_path(memo)
        except LibraryError:
            return None
        return path if path.is_file() else None

    @get("/memos/{memo_id:int}/audio", sync_to_thread=False)
    def audio(request: Request, memo_id: FromPath[int]) -> Response:
        memo = _memo_or_none(memo_id)
        path = _audio_file(memo) if memo else None
        if path is None:
            return Response(
                content="not found", status_code=404, media_type=MediaType.TEXT
            )
        size = path.stat().st_size
        content_type = AUDIO_TYPES.get(path.suffix.lower(), "application/octet-stream")
        try:
            span = parse_range(request.headers.get("range"), size)
        except ValueError:
            return Response(
                content="range not satisfiable",
                status_code=416,
                media_type=MediaType.TEXT,
                headers={"Content-Range": f"bytes */{size}"},
            )
        headers = {"Accept-Ranges": "bytes"}
        if span is None:
            status, start, end = 200, 0, size - 1
        else:
            status, (start, end) = 206, span
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        length = max(0, end - start + 1)
        headers["Content-Length"] = str(length)
        return Stream(
            _iter_file(path, start, length),
            status_code=status,
            media_type=content_type,
            headers=headers,
        )

    def _tags_fragment(request: Request, memo: Memo, token: str) -> Response:
        body = _render("_tags.html", memo=memo, tags=library.tags(memo.id), csrf=token)
        return _html(body, token=token)

    async def _form(request: Request) -> dict[str, str]:
        form = await request.form()
        return {key: str(form.get(key, "")) for key in ("csrf_token", "tag")}

    @post("/memos/{memo_id:int}/tags")
    async def add_tag(request: Request, memo_id: FromPath[int]) -> Response:
        fields = await _form(request)
        if not _csrf_ok(request, fields["csrf_token"]):
            return Response(
                content="csrf check failed", status_code=403, media_type=MediaType.TEXT
            )
        memo = _memo_or_none(memo_id)
        if memo is None:
            return Response(
                content="not found", status_code=404, media_type=MediaType.TEXT
            )
        token = fields["csrf_token"]
        try:
            library.add_tag(memo.id, fields["tag"])
        except LibraryError as exc:
            body = _render(
                "_tags.html",
                memo=memo,
                tags=library.tags(memo.id),
                csrf=token,
                error=str(exc),
            )
            return _html(body, token=token, status=400)
        if request.headers.get("hx-request"):
            return _tags_fragment(request, memo, token)
        return Response(
            content="", status_code=303, headers={"Location": f"/memos/{memo.id}"}
        )

    @post("/memos/{memo_id:int}/tags/remove")
    async def remove_tag(request: Request, memo_id: FromPath[int]) -> Response:
        fields = await _form(request)
        if not _csrf_ok(request, fields["csrf_token"]):
            return Response(
                content="csrf check failed", status_code=403, media_type=MediaType.TEXT
            )
        memo = _memo_or_none(memo_id)
        if memo is None:
            return Response(
                content="not found", status_code=404, media_type=MediaType.TEXT
            )
        token = fields["csrf_token"]
        try:
            library.remove_tag(memo.id, fields["tag"])
        except LibraryError as exc:
            return _html(
                _render("error.html", message=str(exc)), token=token, status=400
            )
        if request.headers.get("hx-request"):
            return _tags_fragment(request, memo, token)
        return Response(
            content="", status_code=303, headers={"Location": f"/memos/{memo.id}"}
        )

    return Litestar(
        route_handlers=[
            index,
            detail,
            audio,
            add_tag,
            remove_tag,
            create_static_files_router(path="/static", directories=[STATIC]),
        ],
        before_request=guard,
        after_request=harden,
        debug=False,
    )
