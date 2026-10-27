"""Export memos as Markdown, SRT, WebVTT, or JSON (single memo or the whole library)."""

import json
import re
import time
from typing import Any

from voicememo.library import Library, Memo

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MD_SPECIAL = re.compile(r"([\\`*_\[\]<>|#~{}])")


def clean_text(text: str) -> str:
    """One line of printable text: control characters removed, whitespace collapsed."""
    return re.sub(r"\s+", " ", _CONTROL.sub("", text)).strip()


def timestamp(seconds: float, sep: str = ",") -> str:
    """HH:MM:SS<sep>mmm. SRT uses a comma, WebVTT a dot. Negative times clamp to zero."""
    total_ms = max(0, round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{ms:03d}"


def _vtt_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def to_srt(segments: list[dict[str, Any]]) -> str:
    blocks = []
    for i, seg in enumerate(segments, start=1):
        text = clean_text(seg["text"]).replace("-->", "-&gt;")
        blocks.append(
            f"{i}\n{timestamp(seg['start'])} --> {timestamp(seg['end'])}\n{text}\n"
        )
    return "\n".join(blocks)


def to_vtt(segments: list[dict[str, Any]]) -> str:
    blocks = ["WEBVTT\n"]
    for seg in segments:
        text = _vtt_escape(clean_text(seg["text"]))
        blocks.append(
            f"{timestamp(seg['start'], '.')} --> {timestamp(seg['end'], '.')}\n{text}\n"
        )
    return "\n".join(blocks)


def md_escape(text: str) -> str:
    return _MD_SPECIAL.sub(r"\\\1", clean_text(text))


def memo_record(library: Library, memo: Memo) -> dict[str, Any]:
    return {
        "id": memo.id,
        "title": memo.title,
        "original_name": memo.original_name,
        "duration": memo.duration,
        "language": memo.language,
        "backend": memo.backend,
        "created_at": memo.created_at,
        "tags": [tag for tag, _kind in library.tags(memo.id)],
        "segments": library.segments(memo.id),
    }


def to_markdown(library: Library, memo: Memo) -> str:
    tags = [tag for tag, _kind in library.tags(memo.id)]
    lines = [
        f"# {md_escape(memo.title)}",
        "",
        f"- Duration: {memo.duration:.1f} s",
        f"- Language: {md_escape(memo.language or 'unknown')}",
        f"- Tags: {', '.join(md_escape(t) for t in tags) or 'none'}",
        "",
        "## Transcript",
        "",
    ]
    for seg in library.segments(memo.id):
        lines.append(
            f"- **[{timestamp(seg['start'], '.')[:8]}]** {md_escape(seg['text'])}"
        )
    return "\n".join(lines) + "\n"


def to_json(library: Library, memo: Memo) -> str:
    return json.dumps(memo_record(library, memo), indent=2, ensure_ascii=False) + "\n"


def library_json(library: Library) -> str:
    records = [memo_record(library, library.get(mid)) for mid in library.all_ids()]
    payload = {
        "exported_at": round(time.time(), 3),
        "count": len(records),
        "memos": records,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def render(library: Library, memo: Memo, fmt: str) -> str:
    segments = library.segments(memo.id)
    if fmt == "md":
        return to_markdown(library, memo)
    if fmt == "srt":
        return to_srt(segments)
    if fmt == "vtt":
        return to_vtt(segments)
    if fmt == "json":
        return to_json(library, memo)
    raise ValueError(f"unknown export format {fmt!r}")
