"""A stack-based balanced-delimiter validator for user-defined grammars.

The classic exercise is one stack and three hard-coded pairs. That version
falls over on every input a real editor or linter has to survive, so this
module keeps the stack and generalizes everything around it:

* **Multi-character delimiters** -- ``/* */``, ``<!-- -->``, ``\\begin``/``\\end``,
  ` ``` ` -- resolved with leftmost-longest matching so ``<`` never shadows
  ``<!--``.
* **Opaque regions** -- inside a string or a comment, a stray ``)`` is text,
  not a delimiter. Opaque pairs suppress the whole delimiter set until they
  close, with optional escape sequences (``"a\\"b"`` is one string).
* **Self-pairing delimiters** -- ``"``, ``$``, ` ``` ` open and close with the
  same lexeme, so the role of a token depends on the stack.
* **Nesting rules** -- C block comments do not nest; a LaTeX ``$`` cannot open
  inside another ``$``; ``may_contain`` restricts which pairs may sit directly
  inside which.
* **Multi-error recovery** -- one pass reports *every* fault with line/column
  and a caret, not just the first, using the same "pop to the matching frame"
  heuristic real parsers use.
* **Streaming** -- :class:`Validator` accepts chunks, so a 10 GB file costs
  O(max nesting depth) memory rather than O(file).

Run directly for a self-check and a demo:

    uv run python brackets.py --self-check
    uv run python brackets.py --spec c somefile.c
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import sys
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

__all__ = [
    "Pair",
    "BracketSpec",
    "Diagnostic",
    "Span",
    "Report",
    "Validator",
    "validate",
    "validate_stream",
    "matching_index",
    "longest_balanced_span",
    "auto_close",
    "SPECS",
]


# ---------------------------------------------------------------------------
# Grammar definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pair:
    """One delimiter pair.

    ``open == close`` makes the pair *self-pairing* (``"``, ``$``): the role of
    a token is then decided by the stack rather than by the lexeme.

    Attributes:
        open: Opening lexeme. May be more than one character.
        close: Closing lexeme. Equal to ``open`` for self-pairing delimiters.
        name: Stable identifier, used by ``may_contain`` and in diagnostics.
        opaque: Suppress all other delimiters until this pair closes. Use for
            strings and comments.
        escape: Inside an opaque region, this lexeme plus the character after
            it are consumed verbatim. ``None`` disables escaping.
        nestable: If false, the pair may not appear inside itself. C block
            comments and LaTeX math mode are the usual examples.
        optional_close: If true, reaching end of input with this pair open is
            not an error. Line comments (``close="\\n"``) need this.
        may_contain: If not ``None``, only these pair names may open directly
            inside this pair. ``frozenset()`` forbids all nesting.
    """

    open: str
    close: str
    name: str = ""
    opaque: bool = False
    escape: str | None = None
    nestable: bool = True
    optional_close: bool = False
    may_contain: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if not self.open or not self.close:
            raise ValueError("delimiter lexemes must be non-empty")
        if self.escape == "":
            raise ValueError("escape lexeme must be non-empty or None")
        if self.escape is not None and not self.opaque:
            raise ValueError(
                f"pair {self.name or self.open!r}: escape only applies inside an "
                "opaque region; set opaque=True or drop the escape"
            )
        if self.escape is not None and self.escape == self.close:
            # The scanner would read every closer as an escape, so the region
            # could never end -- a silently unterminated string for the whole
            # rest of the file.
            raise ValueError(
                f"pair {self.name or self.open!r}: escape {self.escape!r} equals "
                "the closer, which would make the region unterminatable"
            )
        if not self.name:
            object.__setattr__(self, "name", f"{self.open}{self.close}")

    @property
    def self_pairing(self) -> bool:
        return self.open == self.close

    def to_dict(self) -> dict:
        d = {"open": self.open, "close": self.close, "name": self.name}
        if self.opaque:
            d["opaque"] = True
        if self.escape is not None:
            d["escape"] = self.escape
        if not self.nestable:
            d["nestable"] = False
        if self.optional_close:
            d["optional_close"] = True
        if self.may_contain is not None:
            d["may_contain"] = sorted(self.may_contain)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Pair":
        may = d.get("may_contain")
        return cls(
            open=d["open"],
            close=d["close"],
            name=d.get("name", ""),
            opaque=bool(d.get("opaque", False)),
            escape=d.get("escape"),
            nestable=bool(d.get("nestable", True)),
            optional_close=bool(d.get("optional_close", False)),
            may_contain=None if may is None else frozenset(may),
        )


class BracketSpec:
    """A compiled set of :class:`Pair` definitions.

    Compilation builds one alternation over every lexeme, ordered longest
    first. Python's ``re`` picks the leftmost match and, at that position, the
    first alternative that matches -- so the ordering buys leftmost-*longest*
    semantics while the scan itself runs at C speed and skips uninteresting
    text without a Python-level loop over characters.
    """

    def __init__(self, pairs: Sequence[Pair], name: str = "custom") -> None:
        self.name = name
        self.pairs: tuple[Pair, ...] = tuple(pairs)
        if not self.pairs:
            raise ValueError("a spec needs at least one pair")

        by_name: dict[str, Pair] = {}
        for p in self.pairs:
            if p.name in by_name:
                raise ValueError(f"duplicate pair name: {p.name!r}")
            by_name[p.name] = p
        self.by_name = by_name

        for p in self.pairs:
            if p.may_contain is None:
                continue
            unknown = p.may_contain - by_name.keys()
            if unknown:
                raise ValueError(
                    f"pair {p.name!r} may_contain names no pair defines: {sorted(unknown)}"
                )

        # lexeme -> ordered candidate roles. Two pairs may legitimately share a
        # lexeme (``[`` opens a list, ``]`` closes it and also closes a slice),
        # so a lexeme maps to a list and the stack decides.
        #
        # An opaque pair's closer is deliberately *not* registered globally:
        # it is only ever recognized by that pair's own scanner, from inside
        # the region. Otherwise a line comment (``#`` .. ``\n``) would make
        # every newline in the file a stray closer.
        roles: dict[str, list[tuple[Pair, str]]] = {}
        for p in self.pairs:
            if p.self_pairing:
                roles.setdefault(p.open, []).append((p, "toggle"))
            else:
                roles.setdefault(p.open, []).append((p, "open"))
                if not p.opaque:
                    roles.setdefault(p.close, []).append((p, "close"))
        self.roles = roles

        lexemes = sorted(roles, key=lambda s: (-len(s), s))
        self.scanner = re.compile("|".join(re.escape(s) for s in lexemes))
        self.max_lexeme = max(len(s) for s in lexemes)

        # Per-pair scanner used while inside an opaque region: only that pair's
        # closer and escape can end or extend it.
        self._opaque_scanners: dict[str, re.Pattern[str]] = {}
        for p in self.pairs:
            if not p.opaque:
                continue
            alts = (
                [p.close]
                if p.escape is None
                else sorted({p.close, p.escape}, key=lambda s: (-len(s), s))
            )
            self._opaque_scanners[p.name] = re.compile(
                "|".join(re.escape(s) for s in alts)
            )

        # Longest lookahead an incremental feed must hold back before it can
        # safely decide a token: any delimiter -- opaque closers included, even
        # though they are not globally registered -- or an escape plus the
        # character it protects.
        hold = self.max_lexeme
        for p in self.pairs:
            hold = max(hold, len(p.open), len(p.close))
            if p.escape:
                hold = max(hold, len(p.escape) + 1)
        self.hold = hold

    def opaque_scanner(self, pair: Pair) -> re.Pattern[str]:
        return self._opaque_scanners[pair.name]

    def with_pairs(self, *extra: Pair, name: str | None = None) -> "BracketSpec":
        """Derive a spec by appending pairs -- the cheap way to customize."""
        return BracketSpec(self.pairs + extra, name or f"{self.name}+")

    def to_json(self, **kwargs) -> str:
        return json.dumps(
            {"name": self.name, "pairs": [p.to_dict() for p in self.pairs]}, **kwargs
        )

    @classmethod
    def from_dict(cls, d: dict) -> "BracketSpec":
        return cls([Pair.from_dict(p) for p in d["pairs"]], d.get("name", "custom"))

    @classmethod
    def from_json(cls, text: str) -> "BracketSpec":
        return cls.from_dict(json.loads(text))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<BracketSpec {self.name!r} pairs={len(self.pairs)}>"


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

UNCLOSED = "unclosed"
UNEXPECTED_CLOSE = "unexpected-close"
MISMATCHED = "mismatched"
FORBIDDEN_NESTING = "forbidden-nesting"
DEPTH_EXCEEDED = "depth-exceeded"


@dataclass
class Diagnostic:
    kind: str
    message: str
    offset: int
    line: int
    column: int
    lexeme: str
    pair: str
    related_offset: int | None = None
    related_line: int | None = None
    related_column: int | None = None

    def to_dict(self) -> dict:
        d = {
            "kind": self.kind,
            "message": self.message,
            "offset": self.offset,
            "line": self.line,
            "column": self.column,
            "lexeme": self.lexeme,
            "pair": self.pair,
        }
        if self.related_offset is not None:
            d["related"] = {
                "offset": self.related_offset,
                "line": self.related_line,
                "column": self.related_column,
            }
        return d

    def __str__(self) -> str:
        base = f"{self.line}:{self.column}: {self.kind}: {self.message}"
        if self.related_line is not None:
            base += f" (opened at {self.related_line}:{self.related_column})"
        return base


@dataclass(frozen=True)
class Span:
    """A delimiter pair that was successfully matched.

    Both sides carry their own half-open range because delimiters can be more
    than one character long -- an editor jumping to the match of ``<!--`` needs
    to know that ``-->`` occupies three columns, not one.
    """

    open_start: int
    open_end: int
    close_start: int
    close_end: int
    name: str

    def __contains__(self, offset: int) -> bool:
        return self.open_start <= offset < self.close_end


@dataclass
class Report:
    """Result of a validation pass."""

    ok: bool
    diagnostics: list[Diagnostic] = field(default_factory=list)
    spans: list[Span] = field(default_factory=list)
    max_depth: int = 0
    length: int = 0
    # Faults found, which is not the same as faults *listed*: ``max_diagnostics``
    # caps the list. ``ok`` follows this count, never the list -- otherwise
    # capping the output at zero would silently report broken input as valid.
    fault_count: int = 0
    truncated: bool = False

    def __bool__(self) -> bool:
        return self.ok

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "max_depth": self.max_depth,
            "length": self.length,
            "fault_count": self.fault_count,
            "truncated": self.truncated,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }

    def render(self, text: str, context: bool = True) -> str:
        """Human-readable diagnostics, optionally with a caret under each."""
        if self.ok:
            return "ok"
        lines = text.split("\n")
        out: list[str] = []
        if self.truncated:
            out.append(
                f"({self.fault_count} faults found, showing the first "
                f"{len(self.diagnostics)})"
            )
        for d in self.diagnostics:
            out.append(str(d))
            if context and 1 <= d.line <= len(lines):
                src = lines[d.line - 1]
                out.append(f"  {src}")
                caret = "^" * max(1, len(d.lexeme))
                out.append("  " + " " * (d.column - 1) + caret)
        return "\n".join(out)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


@dataclass
class _Frame:
    pair: Pair
    offset: int
    lexeme: str


class Validator:
    """Incremental, single-pass validator.

    Feed text in any chunking; positions reported are absolute over the
    concatenation. Memory is O(nesting depth + line count), never O(input).
    """

    def __init__(
        self,
        spec: BracketSpec,
        *,
        max_depth: int | None = None,
        collect_spans: bool = True,
        max_diagnostics: int | None = 100,
    ) -> None:
        self.spec = spec
        self.max_depth = max_depth
        self.collect_spans = collect_spans
        self.max_diagnostics = max_diagnostics

        self._stack: list[_Frame] = []
        self._buf = ""
        self._base = 0  # absolute offset of _buf[0]
        self._line_starts = [0]  # absolute offsets of line starts, ascending
        self._scanned_to = 0  # absolute offset up to which newlines are indexed
        self._diagnostics: list[Diagnostic] = []
        self._spans: list[Span] = []
        self._max_depth_seen = 0
        self._faults = 0
        self._abandoned = False
        self._finished = False

    # -- position bookkeeping ------------------------------------------------

    def _index_newlines(self, upto: int) -> None:
        """Record line starts for absolute offsets below ``upto``."""
        if upto <= self._scanned_to:
            return
        chunk = self._buf[self._scanned_to - self._base : upto - self._base]
        pos = 0
        while True:
            nl = chunk.find("\n", pos)
            if nl < 0:
                break
            self._line_starts.append(self._scanned_to + nl + 1)
            pos = nl + 1
        self._scanned_to = upto

    def _line_col(self, offset: int) -> tuple[int, int]:
        self._index_newlines(min(offset + 1, self._base + len(self._buf)))
        idx = bisect.bisect_right(self._line_starts, offset) - 1
        return idx + 1, offset - self._line_starts[idx] + 1

    def _emit(
        self,
        kind: str,
        message: str,
        offset: int,
        lexeme: str,
        pair: str,
        related: int | None = None,
    ) -> None:
        # Count first, list second. ``ok`` is derived from the count, so a
        # capped (or zero) diagnostic budget hides detail without ever turning
        # a broken document into a valid one.
        self._faults += 1
        if (
            self.max_diagnostics is not None
            and len(self._diagnostics) >= self.max_diagnostics
        ):
            return
        line, column = self._line_col(offset)
        rl = rc = None
        if related is not None:
            rl, rc = self._line_col(related)
        self._diagnostics.append(
            Diagnostic(
                kind, message, offset, line, column, lexeme, pair, related, rl, rc
            )
        )

    # -- core scan -----------------------------------------------------------

    def feed(self, text: str) -> None:
        if self._finished:
            raise RuntimeError("cannot feed after finish()")
        if not text or self._abandoned:
            return
        self._buf += text
        self._consume(final=False)

    def finish(self) -> Report:
        if not self._finished:
            self._consume(final=True)
            self._finished = True
            self._flush_stack()
        # Report in source order. Unclosed frames are discovered at end of
        # input but belong where their opener is, which is what every compiler
        # does and what makes the caret output readable top to bottom.
        diagnostics = sorted(self._diagnostics, key=lambda d: (d.offset, d.kind))
        return Report(
            ok=self._faults == 0,
            diagnostics=diagnostics,
            spans=list(self._spans),
            max_depth=self._max_depth_seen,
            length=self._base + len(self._buf),
            fault_count=self._faults,
            truncated=self._faults > len(diagnostics),
        )

    def _consume(self, final: bool) -> None:
        """Scan ``self._buf``, retiring everything that can be decided now."""
        buf = self._buf
        # Without the whole input in hand, a lexeme could straddle the chunk
        # boundary, so hold back the longest lookahead the spec can need.
        limit = len(buf) if final else max(0, len(buf) - self.spec.hold)
        i = 0  # index into buf

        while i <= limit:
            if self._abandoned:
                i = len(buf)
                break
            frame = self._stack[-1] if self._stack else None
            if frame is not None and frame.pair.opaque:
                i, stop = self._scan_opaque(buf, i, limit, frame, final)
                if stop:
                    break
                continue

            m = self.spec.scanner.search(buf, i)
            if m is None or m.start() > limit:
                i = max(i, limit)
                break
            i = self._handle_token(buf, m)

        consumed = min(i, len(buf))
        if consumed:
            self._index_newlines(self._base + consumed)
            self._buf = buf[consumed:]
            self._base += consumed

    def _scan_opaque(
        self, buf: str, i: int, limit: int, frame: _Frame, final: bool
    ) -> tuple[int, bool]:
        """Skip through an opaque region.

        Returns ``(resume_index, stop)``. ``resume_index`` is always a position
        the region can be re-entered at safely -- never in the middle of an
        escape sequence, because re-scanning from inside ``\\"`` would read the
        quote as a closer and desynchronize the whole rest of the file.
        """
        pair = frame.pair
        scanner = self.spec.opaque_scanner(pair)
        while True:
            m = scanner.search(buf, i)
            if m is None or m.start() > limit:
                # Everything below the hold limit is inert text; drop it.
                return max(i, limit), True
            lex = m.group()
            if pair.escape is not None and lex == pair.escape:
                nxt = m.end() + 1
                if nxt > len(buf):
                    if not final:
                        return m.start(), True  # decide it once the next chunk lands
                    nxt = len(buf)
                i = nxt
                continue
            self._close_frame(self._base + m.start(), lex, pair)
            return m.end(), False

    def _handle_token(self, buf: str, m: re.Match[str]) -> int:
        lex = m.group()
        offset = self._base + m.start()
        candidates = self.spec.roles[lex]
        top = self._stack[-1] if self._stack else None

        # Self-pairing lexemes close the frame they opened; a lexeme shared by
        # several pairs resolves against the stack top before falling back.
        for pair, role in candidates:
            if role == "toggle" and top is not None and top.pair is pair:
                self._close_frame(offset, lex, pair)
                return m.end()
        for pair, role in candidates:
            if role == "close" and top is not None and top.pair is pair:
                self._close_frame(offset, lex, pair)
                return m.end()

        for pair, role in candidates:
            if role in ("open", "toggle"):
                self._open_frame(offset, lex, pair)
                return m.end()

        # Only closers left, and none matches the top of the stack.
        pair = candidates[0][0]
        self._close_unmatched(
            offset, lex, [p for p, r in candidates if r == "close"] or [pair]
        )
        return m.end()

    def _open_frame(self, offset: int, lex: str, pair: Pair) -> None:
        top = self._stack[-1] if self._stack else None
        if not pair.nestable and any(f.pair is pair for f in self._stack):
            owner = next(f for f in reversed(self._stack) if f.pair is pair)
            self._emit(
                FORBIDDEN_NESTING,
                f"{pair.name!r} does not nest",
                offset,
                lex,
                pair.name,
                owner.offset,
            )
            return
        if top is not None and top.pair.may_contain is not None:
            if pair.name not in top.pair.may_contain:
                self._emit(
                    FORBIDDEN_NESTING,
                    f"{pair.name!r} is not allowed directly inside {top.pair.name!r}",
                    offset,
                    lex,
                    pair.name,
                    top.offset,
                )
                return
        if self.max_depth is not None and len(self._stack) >= self.max_depth:
            # Fatal, not recoverable. Dropping the frame and carrying on turns
            # every subsequent closer into a spurious "unexpected close": a
            # 3-deep limit on 20 nested parens produced 1 real diagnostic and
            # 17 pieces of noise. The limit exists to bound work, so stop.
            self._emit(
                DEPTH_EXCEEDED,
                f"nesting deeper than {self.max_depth}; scan abandoned here",
                offset,
                lex,
                pair.name,
            )
            self._abandoned = True
            return
        self._stack.append(_Frame(pair, offset, lex))
        self._max_depth_seen = max(self._max_depth_seen, len(self._stack))

    def _close_frame(self, offset: int, lex: str, pair: Pair) -> None:
        frame = self._stack.pop()
        if self.collect_spans:
            self._spans.append(
                Span(
                    frame.offset,
                    frame.offset + len(frame.lexeme),
                    offset,
                    offset + len(lex),
                    pair.name,
                )
            )

    def _close_unmatched(self, offset: int, lex: str, wanted: list[Pair]) -> None:
        """A closer arrived that the stack top does not want.

        Recovery mirrors what production parsers do: if the closer matches a
        frame further down, everything above it was left open, so report those
        and resynchronize at that depth. Otherwise the closer itself is the
        stray token -- drop it and keep the stack intact, which keeps the rest
        of the file's diagnostics meaningful instead of cascading.
        """
        wanted_ids = {id(p) for p in wanted}
        depth = None
        for idx in range(len(self._stack) - 1, -1, -1):
            if id(self._stack[idx].pair) in wanted_ids:
                depth = idx
                break

        if depth is None:
            names = " or ".join(sorted({p.name for p in wanted}))
            if self._stack:
                top = self._stack[-1]
                self._emit(
                    MISMATCHED,
                    f"{lex!r} closes {names} but {top.pair.name!r} is open",
                    offset,
                    lex,
                    names,
                    top.offset,
                )
            else:
                self._emit(
                    UNEXPECTED_CLOSE,
                    f"{lex!r} closes {names}, which is not open",
                    offset,
                    lex,
                    names,
                )
            return

        for frame in self._stack[depth + 1 :]:
            self._emit(
                UNCLOSED,
                f"{frame.lexeme!r} is never closed; {lex!r} closed "
                f"{self._stack[depth].pair.name!r} first",
                frame.offset,
                frame.lexeme,
                frame.pair.name,
                offset,
            )
        closing = self._stack[depth]
        del self._stack[depth:]
        if self.collect_spans:
            self._spans.append(
                Span(
                    closing.offset,
                    closing.offset + len(closing.lexeme),
                    offset,
                    offset + len(lex),
                    closing.pair.name,
                )
            )

    def _flush_stack(self) -> None:
        if self._abandoned:
            # The stack is whatever it was when scanning stopped; reporting it
            # as "unclosed" would be a second wave of noise about one fault.
            self._stack.clear()
            return
        for frame in self._stack:
            if frame.pair.optional_close:
                continue
            self._emit(
                UNCLOSED,
                f"{frame.lexeme!r} is never closed (expected {frame.pair.close!r})",
                frame.offset,
                frame.lexeme,
                frame.pair.name,
            )
        self._stack.clear()

    # -- introspection -------------------------------------------------------

    def pending_closers(self, final: bool = False) -> list[str]:
        """Closers needed, innermost first, to balance what is open right now.

        ``final=True`` first decides the held-back tail of the buffer, which
        matters when the last few characters could still begin a delimiter --
        without it, a buffer ending in ``/*`` would not yet know a comment is
        open. Unlike :meth:`finish` it leaves the stack intact and reports no
        unclosed-frame diagnostics, because a partial buffer is not an error.
        """
        if final and not self._finished:
            self._consume(final=True)
        return [
            f.pair.close for f in reversed(self._stack) if not f.pair.optional_close
        ]

    @property
    def pending(self) -> list[str]:
        """Closers needed for what is open right now, innermost first."""
        return self.pending_closers()


def validate(
    text: str,
    spec: "BracketSpec | str" = "plain",
    *,
    max_depth: int | None = None,
    max_diagnostics: int | None = 100,
) -> Report:
    """Validate a whole string. ``spec`` may be a preset name or a spec."""
    v = Validator(_resolve(spec), max_depth=max_depth, max_diagnostics=max_diagnostics)
    v.feed(text)
    return v.finish()


def validate_stream(
    chunks: Iterable[str],
    spec: "BracketSpec | str" = "plain",
    *,
    max_depth: int | None = None,
    max_diagnostics: int | None = 100,
) -> Report:
    """Validate an iterable of chunks -- O(depth) memory, not O(input)."""
    v = Validator(
        _resolve(spec),
        max_depth=max_depth,
        collect_spans=False,
        max_diagnostics=max_diagnostics,
    )
    for chunk in chunks:
        v.feed(chunk)
    return v.finish()


# ---------------------------------------------------------------------------
# Editor-flavored helpers built on the same pass
# ---------------------------------------------------------------------------


def matching_index(
    text: str, offset: int, spec: "BracketSpec | str" = "plain"
) -> int | None:
    """Start offset of the delimiter matching the one at ``offset``.

    Accepts either side of a pair and any offset *within* a multi-character
    delimiter, the way an editor's jump-to-match does. Returns ``None`` if the
    caret is not on a delimiter or that delimiter never matched.
    """
    report = validate(text, spec, max_diagnostics=0)
    for span in report.spans:
        if span.open_start <= offset < span.open_end:
            return span.close_start
        if span.close_start <= offset < span.close_end:
            return span.open_start
    return None


def longest_balanced_span(
    text: str, spec: "BracketSpec | str" = "plain"
) -> tuple[int, int]:
    """Longest substring that is itself balanced, as a half-open ``(start, end)``.

    The single-pair-type version of this is the classic "longest valid
    parentheses" problem. Generalizing to many pair types needs no new
    machinery -- the validator already reports every matched span plus the
    offset of every fault, so the answer is the longest chain of adjacent
    top-level spans with no fault between them.
    """
    report = validate(text, spec, max_diagnostics=None)
    if not report.spans:
        return (0, 0)

    tops: list[Span] = []
    for span in sorted(report.spans, key=lambda s: (s.open_start, -s.close_end)):
        if tops and span.open_start < tops[-1].close_end:
            continue  # nested inside the previous top-level span
        tops.append(span)

    faults = sorted(d.offset for d in report.diagnostics)
    best = (0, 0)
    run_start: int | None = None
    prev_end: int | None = None
    for span in tops:
        adjacent = prev_end is not None and not text[prev_end : span.open_start].strip()
        clean = prev_end is None or bisect.bisect_left(
            faults, prev_end
        ) == bisect.bisect_left(faults, span.open_start)
        if run_start is None or not adjacent or not clean:
            run_start = span.open_start
        prev_end = span.close_end
        if prev_end - run_start > best[1] - best[0]:
            best = (run_start, prev_end)
    return best


def auto_close(text: str, spec: "BracketSpec | str" = "plain") -> str:
    """The closers that would balance ``text``, innermost first.

    This is the "type ``{`` and get ``}``" editor feature, applied to a whole
    buffer -- and it works on a partial buffer because :class:`Validator` never
    needs to see the end of the input to know what is open.
    """
    v = Validator(_resolve(spec), collect_spans=False, max_diagnostics=0)
    v.feed(text)
    return "".join(v.pending_closers(final=True))


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

_ROUND = Pair("(", ")", "paren")
_SQUARE = Pair("[", "]", "square")
_CURLY = Pair("{", "}", "curly")
_ANGLE = Pair("<", ">", "angle")

PLAIN = BracketSpec([_ROUND, _SQUARE, _CURLY], "plain")

C_LIKE = BracketSpec(
    [
        _ROUND,
        _SQUARE,
        _CURLY,
        # Block comments do not nest in C, C++, Java, or JavaScript. Getting
        # this wrong is the classic "/* /* */" bug.
        Pair("/*", "*/", "block-comment", opaque=True, nestable=False),
        Pair("//", "\n", "line-comment", opaque=True, optional_close=True),
        Pair('"', '"', "string", opaque=True, escape="\\"),
        Pair("'", "'", "char", opaque=True, escape="\\"),
    ],
    "c",
)

PYTHON = BracketSpec(
    [
        _ROUND,
        _SQUARE,
        _CURLY,
        Pair('"""', '"""', "triple-double", opaque=True, escape="\\"),
        Pair("'''", "'''", "triple-single", opaque=True, escape="\\"),
        Pair('"', '"', "string", opaque=True, escape="\\"),
        Pair("'", "'", "string-single", opaque=True, escape="\\"),
        Pair("#", "\n", "comment", opaque=True, optional_close=True),
    ],
    "python",
)

# JSON has no comments and no bare parentheses; ``may_contain`` encodes the
# grammar's shape well enough to catch ``{[}]``-style damage.
JSON_SPEC = BracketSpec(
    [
        Pair("{", "}", "object"),
        Pair("[", "]", "array"),
        Pair('"', '"', "string", opaque=True, escape="\\"),
    ],
    "json",
)

HTML = BracketSpec(
    [
        Pair("<!--", "-->", "comment", opaque=True, nestable=False),
        Pair("<script", "</script>", "script", opaque=True),
        Pair("<style", "</style>", "style", opaque=True),
        _ROUND,
        _SQUARE,
        _CURLY,
    ],
    "html",
)

# TeX forbids math mode inside math mode, whichever spelling you use, so the
# math pairs whitelist what may appear directly inside them.
_MATH_BODY = frozenset({"curly", "square", "environment", "comment"})

LATEX = BracketSpec(
    [
        _CURLY,
        _SQUARE,
        Pair("\\begin", "\\end", "environment"),
        Pair("\\[", "\\]", "display-math", nestable=False, may_contain=_MATH_BODY),
        Pair("\\(", "\\)", "inline-math", nestable=False, may_contain=_MATH_BODY),
        # ``$`` and ``$$`` are self-pairing: the stack decides their role.
        Pair("$$", "$$", "display-dollar", nestable=False, may_contain=_MATH_BODY),
        Pair("$", "$", "math", nestable=False, may_contain=_MATH_BODY),
        Pair("%", "\n", "comment", opaque=True, optional_close=True),
    ],
    "latex",
)

MARKDOWN = BracketSpec(
    [
        Pair("```", "```", "fence", opaque=True),
        Pair("`", "`", "code", opaque=True),
        _ROUND,
        _SQUARE,
    ],
    "markdown",
)

# Unicode brackets people actually hit: CJK, guillemets, math angle brackets.
UNICODE = BracketSpec(
    PLAIN.pairs
    + (
        Pair("「", "」", "corner"),
        Pair("『", "』", "white-corner"),
        Pair("（", "）", "fullwidth-paren"),
        Pair("［", "］", "fullwidth-square"),
        Pair("｛", "｝", "fullwidth-curly"),
        Pair("«", "»", "guillemet"),
        Pair("‹", "›", "single-guillemet"),
        Pair("⟨", "⟩", "math-angle"),
    ),
    "unicode",
)

SPECS: dict[str, BracketSpec] = {
    s.name: s
    for s in (PLAIN, C_LIKE, PYTHON, JSON_SPEC, HTML, LATEX, MARKDOWN, UNICODE)
}
SPECS["angle"] = BracketSpec(PLAIN.pairs + (_ANGLE,), "angle")


def _resolve(spec: "BracketSpec | str") -> BracketSpec:
    if isinstance(spec, BracketSpec):
        return spec
    try:
        return SPECS[spec]
    except KeyError:
        raise ValueError(
            f"unknown spec {spec!r}; choose from {', '.join(sorted(SPECS))}"
        ) from None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _iter_chunks(fh, size: int = 1 << 16) -> Iterator[str]:
    while True:
        chunk = fh.read(size)
        if not chunk:
            return
        yield chunk


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="brackets",
        description="Validate balanced delimiters against a configurable grammar.",
    )
    ap.add_argument("files", nargs="*", help="files to check; omit or '-' for stdin")
    ap.add_argument(
        "--spec", default="plain", help=f"preset: {', '.join(sorted(SPECS))}"
    )
    ap.add_argument("--spec-file", help="JSON file describing a custom spec")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument(
        "--stream",
        action="store_true",
        help="chunked scan; O(depth) memory, no source context in output",
    )
    ap.add_argument("--auto-close", action="store_true", help="print missing closers")
    ap.add_argument("--dump-spec", action="store_true", help="print the spec as JSON")
    ap.add_argument("--self-check", action="store_true", help="run the built-in demo")
    args = ap.parse_args(argv)

    if args.self_check:
        return _self_check()

    if args.spec_file:
        with open(args.spec_file, encoding="utf-8") as fh:
            spec = BracketSpec.from_json(fh.read())
    else:
        try:
            spec = _resolve(args.spec)
        except ValueError as exc:
            ap.error(str(exc))

    if args.dump_spec:
        print(spec.to_json(indent=2))
        return 0

    paths = args.files or ["-"]
    if args.auto_close and args.stream:
        ap.error("--auto-close needs the whole buffer; drop --stream")

    failed = False
    results = []
    for path in paths:
        label = "<stdin>" if path == "-" else path
        try:
            if args.stream:
                # Never materializes the file; source context is unavailable,
                # so diagnostics print without carets.
                opener = _stdin_chunks() if path == "-" else _file_chunks(path)
                report = validate_stream(opener, spec, max_depth=args.max_depth)
                text = None
            else:
                text = sys.stdin.read() if path == "-" else _read_text(path)
                report = validate(text, spec, max_depth=args.max_depth)
        except OSError as exc:
            print(f"{label}: cannot read: {exc.strerror or exc}", file=sys.stderr)
            failed = True
            continue

        if args.auto_close:
            print(auto_close(text or "", spec))
            continue

        failed |= not report.ok
        if args.json:
            results.append({"file": label, **report.to_dict()})
        else:
            print(
                f"{label}: {'ok' if report.ok else 'FAILED'} "
                f"(max depth {report.max_depth})"
            )
            if not report.ok:
                rendered = (
                    report.render(text, context=True)
                    if text is not None
                    else "\n".join(str(d) for d in report.diagnostics)
                )
                print("\n".join("  " + ln for ln in rendered.split("\n")))

    if args.json:
        json.dump(results, sys.stdout, indent=2)
        print()
    return 1 if failed else 0


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def _file_chunks(path: str, size: int = 1 << 16) -> Iterator[str]:
    with open(path, encoding="utf-8", errors="replace") as fh:
        yield from _iter_chunks(fh, size)


def _stdin_chunks(size: int = 1 << 16) -> Iterator[str]:
    yield from _iter_chunks(sys.stdin, size)


def _self_check() -> int:
    cases: list[tuple[str, str, bool, str]] = [
        ("plain", "([{}])", True, "textbook nesting"),
        ("plain", "([)]", False, "interleaved pairs"),
        ("plain", "", True, "empty input"),
        ("c", '{ char *s = ")"; }', True, "bracket inside a string"),
        ("c", "/* ) */ ( )", True, "bracket inside a comment"),
        ("c", "/* /* */", True, "C comments do not nest"),
        ("c", '"a\\"b" ( )', True, "escaped quote does not end the string"),
        ("c", "// unterminated ( comment", True, "line comment may end at EOF"),
        ("python", "f('''(''')", True, "triple quote beats single quote"),
        ("python", "# ( \nx = (1,)", True, "comment then real code"),
        ("latex", "$x + (y)$", True, "self-pairing math mode"),
        ("latex", "$ $ $", False, "odd number of dollars"),
        ("markdown", "```\n) unclosed\n```\n[a](b)", True, "fenced block is opaque"),
        ("html", "<!-- ( --> ( )", True, "html comment is opaque"),
        ("unicode", "「(（）)」", True, "cjk + fullwidth"),
        ("json", '{"a": [1, 2], "b": "]"}', True, "json shape"),
        ("json", "{[}]", False, "crossed object/array"),
    ]
    failures = 0
    for spec_name, text, expected, why in cases:
        got = validate(text, spec_name).ok
        status = "ok " if got == expected else "FAIL"
        if got != expected:
            failures += 1
        print(f"  [{status}] {spec_name:9s} {why}")

    # Streaming must agree with the whole-string pass at every chunk size.
    samples = [
        '{ /* ) */ "a\\"b" ([{}]) } // x (',
        'x = """a"b""" + \'\'\'(\'\'\' # )\n([)]',
        "$ (a) $$ b $$ % ) \n {c}",
    ]
    sizes = (1, 2, 3, 5, 7, 64)
    stream_ok = True
    for spec_name, sample in zip(("c", "python", "latex"), samples):
        whole = validate(sample, spec_name)
        for size in sizes:
            chunks = [sample[i : i + size] for i in range(0, len(sample), size)]
            streamed = validate_stream(chunks, spec_name)
            if streamed.ok != whole.ok or [
                (d.kind, d.offset) for d in streamed.diagnostics
            ] != [(d.kind, d.offset) for d in whole.diagnostics]:
                print(f"  [FAIL] streaming disagrees: {spec_name} at chunk size {size}")
                failures += 1
                stream_ok = False
    if stream_ok:
        print(
            f"  [ok ] streaming matches whole-string scan "
            f"({len(samples)} samples x {len(sizes)} chunk sizes)"
        )

    span = longest_balanced_span("())((()))", "plain")
    if span != (3, 9):
        print(f"  [FAIL] longest_balanced_span -> {span}, expected (3, 9)")
        failures += 1
    else:
        print("  [ok ] longest balanced span")

    closers = auto_close('int f() { g("a", [1,', "c")
    if closers != "])}":
        print(f"  [FAIL] auto_close -> {closers!r}, expected '])}}'")
        failures += 1
    else:
        print("  [ok ] auto-close of a partial buffer")

    # Jump-to-match, including from inside a multi-character delimiter.
    doc = "a <!-- b --> c"
    jumps = {0: None, 2: 9, 4: 9, 9: 2, 11: 2, 13: None}
    bad = {
        o: matching_index(doc, o, "html")
        for o in jumps
        if matching_index(doc, o, "html") != jumps[o]
    }
    if bad:
        print(f"  [FAIL] matching_index -> {bad}")
        failures += 1
    else:
        print("  [ok ] jump-to-match across multi-character delimiters")

    print("all self-checks passed" if not failures else f"{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
