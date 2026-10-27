"""Text hygiene for untrusted input and for anything printed to a terminal."""

import re

# ANSI/VT escape sequences (CSI and OSC) are removed whole, so their parameters do not survive.
_ESCAPE_SEQ = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]"
)
# Control characters other than tab, LF and CR, plus DEL and the C1 range.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
# Everything a terminal could interpret (CR and ESC included) must be shown, not sent.
_TERMINAL_UNSAFE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
# Zero-width and bidirectional control characters: they hide or reorder text on screen
# (Trojan Source style). Removed everywhere; ZWJ and ZWNJ are kept for emoji and scripts.
_INVISIBLE = re.compile(r"[\u200b\u200e\u200f\u202a-\u202e\u2060\u2066-\u2069\ufeff]")


def strip_controls(text: str) -> str:
    """Remove escape sequences, control and invisible bidi characters, keeping tabs and newlines."""
    return _INVISIBLE.sub("", _CONTROL.sub("", _ESCAPE_SEQ.sub("", text)))


def safe_text(text: str) -> str:
    """Make untrusted text safe to print: control characters become visible \\xNN escapes."""
    return _TERMINAL_UNSAFE.sub(
        lambda m: f"\\x{ord(m.group(0)):02x}", _INVISIBLE.sub("", text)
    )


def one_line(text: str) -> str:
    """Collapse whitespace (including newlines) to single spaces."""
    return " ".join(text.split())


def is_utf8_safe(text: str) -> bool:
    """False for lone surrogates, which cannot be encoded as UTF-8 (they crash writers)."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True
