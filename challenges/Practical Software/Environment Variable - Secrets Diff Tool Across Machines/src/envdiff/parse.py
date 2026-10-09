"""Parsers for the env-file dialects that disagree with each other.

Three dialects, because the same line means different things in each:

* ``dotenv``: ``KEY=value``, optional ``export``, quotes with escapes, inline ``# comments``.
* ``shell``: what ``export -p`` / ``declare -x`` print, parsed as shell words (adjacent quoted
  segments, ``$'...'``), with ``$VAR`` left unexpanded.
* ``raw``: Docker ``--env-file`` and ``env`` output: everything after the first ``=`` is the value,
  verbatim. ``KEY="x"`` therefore keeps its quote characters, which is exactly the surprise that
  makes the same file behave differently under ``docker run`` and ``source``.

Nothing here ever puts file *content* into an error or issue: only line numbers and key names,
because a malformed line is often a secret pasted in the wrong place.
"""

import re
from dataclasses import dataclass, field

KEY = r"[A-Za-z_][A-Za-z0-9_.-]*"
_KEY_RE = re.compile(KEY)
# A line without ``=`` declares a pass-through variable only if it looks like an environment variable
# name. Anything else is far more likely to be a continuation line of a multi-line value (a PEM body
# is a column of alphanumeric lines) or a secret pasted on its own, and a variable name is printed.
_BARE_NAME = re.compile(r"[A-Z_][A-Z0-9_]*")
_DOTENV_LINE = re.compile(rf"[ \t]*(?:export[ \t]+)?({KEY})[ \t]*(=|$)(.*)", re.DOTALL)
_SHELL_PREFIX = re.compile(
    r"(?:(?:declare|typeset)[ \t]+(?:-[A-Za-z-]+[ \t]+)*|export[ \t]+(?:-[A-Za-z]+[ \t]+)*)"
)
_ANSI_C_ESCAPES = {
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "\\": "\\",
    "'": "'",
    '"': '"',
    "a": "\a",
    "b": "\b",
    "e": "\x1b",
}
_DQ_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "$": "$"}


@dataclass(frozen=True)
class Issue:
    kind: str  # duplicate-key | invalid-line | unterminated-quote | trailing-garbage | bom | crlf | unquoted-space
    line: int  # 1-based; 0 for whole-file issues
    key: str | None = None  # only ever a syntactically valid variable name


@dataclass
class Parsed:
    values: dict[str, str | None] = field(
        default_factory=dict
    )  # None: declared without a value
    lines: dict[str, int] = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)

    def add(self, key: str, value: str | None, line: int) -> None:
        if key in self.values:
            self.issues.append(
                Issue("duplicate-key", line, key)
            )  # last definition wins, as in shells
        self.values[key] = value
        self.lines[key] = line


def _prepare(text: str, parsed: Parsed) -> list[str]:
    if text.startswith("﻿"):
        text = text[1:]
        parsed.issues.append(Issue("bom", 1))
    if "\r\n" in text:
        parsed.issues.append(Issue("crlf", 0))
        text = text.replace("\r\n", "\n")
    return text.split("\n")


def parse_dotenv(text: str) -> Parsed:
    parsed = Parsed()
    lines = _prepare(text, parsed)
    index = 0
    while index < len(lines):
        raw, line_no = lines[index], index + 1
        index += 1
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        match = _DOTENV_LINE.fullmatch(raw)
        if not match:
            parsed.issues.append(Issue("invalid-line", line_no))
            continue
        key, equals, rest = match.group(1), match.group(2), match.group(3)
        if not equals:
            if _BARE_NAME.fullmatch(key):
                parsed.add(key, None, line_no)
            else:
                parsed.issues.append(Issue("invalid-line", line_no))
            continue
        rest = rest.lstrip(" \t")
        if rest[:1] in {'"', "'"}:
            quoted = _read_dotenv_quoted(rest, lines, index)
            if quoted is None:
                parsed.issues.append(Issue("unterminated-quote", line_no, key))
                continue
            value, remainder, index = quoted
            tail = remainder.strip()
            if tail and not tail.startswith("#"):
                parsed.issues.append(Issue("trailing-garbage", line_no, key))
            parsed.add(key, value, line_no)
            continue
        value = re.split(r"[ \t]#", rest, maxsplit=1)[0].strip()
        if re.search(r"\s", value):
            parsed.issues.append(Issue("unquoted-space", line_no, key))
        parsed.add(key, value, line_no)
    return parsed


def _read_dotenv_quoted(
    first: str, lines: list[str], next_index: int
) -> tuple[str, str, int] | None:
    """Value, text after the closing quote, and the next unread line index; None if never closed."""
    quote = first[0]
    out: list[str] = []
    current, index = first[1:], next_index
    while True:
        position = 0
        while position < len(current):
            char = current[position]
            if char == quote:
                return "".join(out), current[position + 1 :], index
            if quote == '"' and char == "\\" and position + 1 < len(current):
                following = current[position + 1]
                out.append(_DQ_ESCAPES.get(following, "\\" + following))
                position += 2
                continue
            out.append(char)
            position += 1
        if index >= len(lines):
            return None
        out.append("\n")
        current, index = lines[index], index + 1


def parse_raw(text: str) -> Parsed:
    parsed = Parsed()
    for number, raw in enumerate(_prepare(text, parsed), start=1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        name, equals, value = raw.lstrip().partition("=")
        bare_ok = equals or _BARE_NAME.fullmatch(name)
        if not _KEY_RE.fullmatch(name) or not bare_ok:
            parsed.issues.append(Issue("invalid-line", number))
            continue
        parsed.add(name, value if equals else None, number)
    return parsed


class _ShellScanner:
    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    @property
    def line(self) -> int:
        return self.text.count("\n", 0, self.pos) + 1

    def at_end(self) -> bool:
        return self.pos >= len(self.text)

    def skip_blanks(self, newlines: bool) -> None:
        while not self.at_end() and (
            self.text[self.pos] in " \t" or (newlines and self.text[self.pos] == "\n")
        ):
            self.pos += 1

    def skip_line(self) -> None:
        end = self.text.find("\n", self.pos)
        self.pos = len(self.text) if end < 0 else end + 1

    def read_word(self) -> str | None:
        """One shell word, or None if a quote is never closed."""
        out: list[str] = []
        text = self.text
        while not self.at_end():
            char = text[self.pos]
            if char in " \t\n;":
                break
            if char == "'":
                end = text.find("'", self.pos + 1)
                if end < 0:
                    return None
                out.append(text[self.pos + 1 : end])
                self.pos = end + 1
            elif char == "$" and text.startswith("$'", self.pos):
                if (chunk := self._read_ansi_c()) is None:
                    return None
                out.append(chunk)
            elif char == '"':
                if (chunk := self._read_double()) is None:
                    return None
                out.append(chunk)
            elif char == "\\" and self.pos + 1 < len(text):
                following = text[self.pos + 1]
                if following != "\n":  # a backslash-newline is a line continuation
                    out.append(following)
                self.pos += 2
            else:
                out.append(char)
                self.pos += 1
        return "".join(out)

    def _read_double(self) -> str | None:
        text, out = self.text, []
        self.pos += 1
        while self.pos < len(text):
            char = text[self.pos]
            if char == '"':
                self.pos += 1
                return "".join(out)
            if char == "\\" and self.pos + 1 < len(text):
                following = text[self.pos + 1]
                if following in '"\\$`':
                    out.append(following)
                elif following != "\n":
                    out.append("\\" + following)
                self.pos += 2
                continue
            out.append(char)
            self.pos += 1
        return None

    def _read_ansi_c(self) -> str | None:
        text, out = self.text, []
        self.pos += 2
        while self.pos < len(text):
            char = text[self.pos]
            if char == "'":
                self.pos += 1
                return "".join(out)
            if char == "\\" and self.pos + 1 < len(text):
                following = text[self.pos + 1]
                out.append(_ANSI_C_ESCAPES.get(following, "\\" + following))
                self.pos += 2
                continue
            out.append(char)
            self.pos += 1
        return None


def parse_shell(text: str) -> Parsed:
    """``export -p`` / ``declare -x`` output, and hand-written ``export KEY=value`` scripts."""
    parsed = Parsed()
    scanner = _ShellScanner("\n".join(_prepare(text, parsed)))
    while True:
        scanner.skip_blanks(newlines=True)
        if scanner.at_end():
            break
        line_no = scanner.line
        if scanner.text[scanner.pos] == "#":
            scanner.skip_line()
            continue
        prefix = _SHELL_PREFIX.match(scanner.text, scanner.pos)
        if prefix:
            scanner.pos = prefix.end()
        _read_assignments(scanner, parsed, line_no, bare_allowed=prefix is not None)
    return parsed


def _read_assignments(
    scanner: _ShellScanner, parsed: Parsed, line_no: int, bare_allowed: bool
) -> None:
    """One or more ``KEY[=word]`` on the current line (``export A=1 B=2``).

    A bare ``KEY`` is only a declaration after ``export``/``declare``; on its own it is a command.
    """
    start = True
    while True:
        scanner.skip_blanks(newlines=False)
        if scanner.at_end() or scanner.text[scanner.pos] in "\n#":
            break
        key = _KEY_RE.match(scanner.text, scanner.pos)
        if not key:
            if start:
                parsed.issues.append(Issue("invalid-line", line_no))
            else:
                parsed.issues.append(Issue("trailing-garbage", line_no))
            scanner.skip_line()
            return
        start = False
        scanner.pos = key.end()
        if scanner.at_end() or scanner.text[scanner.pos] != "=":
            if not bare_allowed:
                parsed.issues.append(Issue("invalid-line", line_no))
                scanner.skip_line()
                return
            parsed.add(key.group(), None, line_no)
            continue
        scanner.pos += 1
        line_end = scanner.text.find("\n", scanner.pos)
        value = scanner.read_word()
        if value is None:
            parsed.issues.append(Issue("unterminated-quote", line_no, key.group()))
            scanner.pos = len(scanner.text) if line_end < 0 else line_end
            return
        parsed.add(key.group(), value, line_no)
    scanner.skip_line()


FORMATS = {"dotenv": parse_dotenv, "shell": parse_shell, "raw": parse_raw}


def detect_format(text: str, name: str = "") -> str:
    """Shell when the content shows ``declare -x`` / ``export -p`` output or the name says script."""
    lowered = name.lower()
    if lowered.endswith((".sh", ".bash", ".zsh")):
        return "shell"
    if re.search(r"^(?:declare|typeset)[ \t]+-[A-Za-z-]*x", text, re.MULTILINE):
        return "shell"
    return "dotenv"


def parse(text: str, fmt: str = "auto", name: str = "") -> tuple[Parsed, str]:
    """Parse ``text``; returns the result and the dialect actually used."""
    chosen = detect_format(text, name) if fmt == "auto" else fmt
    if chosen not in FORMATS:
        raise ValueError(
            f"unknown format {fmt!r}; choose from auto, {', '.join(FORMATS)}"
        )
    return FORMATS[chosen](text), chosen
