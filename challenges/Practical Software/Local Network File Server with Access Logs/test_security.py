from __future__ import annotations

import os
from pathlib import Path

import pytest
from security import PathTraversalError, hash_token, resolve_safe_path, verify_token


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "nested.txt").write_text("nested")
    (tmp_path / "top.txt").write_text("top")
    outside = tmp_path.parent / "outside_secret.txt"
    outside.write_text("should never be reachable")
    return tmp_path


# --------------------------------------------------------------------------
# Legitimate paths
# --------------------------------------------------------------------------


def test_empty_path_resolves_to_root(root: Path) -> None:
    assert resolve_safe_path(root, "") == root.resolve()


def test_simple_file_resolves(root: Path) -> None:
    assert resolve_safe_path(root, "top.txt") == (root / "top.txt").resolve()


def test_nested_file_resolves(root: Path) -> None:
    assert (
        resolve_safe_path(root, "sub/nested.txt")
        == (root / "sub" / "nested.txt").resolve()
    )


def test_dot_segments_are_harmless(root: Path) -> None:
    assert (
        resolve_safe_path(root, "./sub/./nested.txt")
        == (root / "sub" / "nested.txt").resolve()
    )


def test_trailing_slash_on_directory(root: Path) -> None:
    assert resolve_safe_path(root, "sub/") == (root / "sub").resolve()


def test_double_slash_collapses_harmlessly(root: Path) -> None:
    # A leading "//" (seen as "/browse//etc/passwd" over HTTP) does NOT
    # smuggle an absolute path through -- see the empirical probe in the
    # session notes: segments are split first, so a leading "/" just
    # contributes an empty segment that gets filtered out, and the
    # (nonexistent) result stays contained under root.
    resolved = resolve_safe_path(root, "/etc/passwd")
    assert root.resolve() in resolved.parents or resolved == root.resolve()


# --------------------------------------------------------------------------
# Attacks that must all be rejected
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "attack",
    [
        "..",
        "../outside_secret.txt",
        "../../outside_secret.txt",
        "sub/../../outside_secret.txt",
        "sub/../../../outside_secret.txt",
        "%2e%2e/outside_secret.txt",  # single percent-encoded ".."
        "%2e%2e%2foutside_secret.txt",  # single percent-encoded "../"
        "%252e%252e%252foutside_secret.txt",  # double percent-encoded "../"
        "sub/%2e%2e/%2e%2e/outside_secret.txt",
        "..\\outside_secret.txt",  # backslash traversal (Windows-style)
        "sub\\..\\..\\outside_secret.txt",
        "top.txt\x00.png",  # null-byte trick
    ],
)
def test_traversal_attacks_rejected(root: Path, attack: str) -> None:
    with pytest.raises(PathTraversalError):
        resolve_safe_path(root, attack)


@pytest.mark.parametrize(
    "attack",
    [
        "/etc/shadow" if os.name != "nt" else "/Windows/System32/cmd.exe",
    ],
)
def test_result_never_escapes_root_even_when_not_rejected(
    root: Path, attack: str
) -> None:
    # Whether or not a specific absolute-looking string is rejected outright,
    # the *result* must never resolve outside root -- this is the invariant
    # that actually matters, checked independently of the rejection path.
    try:
        resolved = resolve_safe_path(root, attack)
    except PathTraversalError:
        return
    root_resolved = root.resolve()
    assert resolved == root_resolved or root_resolved in resolved.parents


def test_windows_drive_letter_segment_rejected(root: Path) -> None:
    with pytest.raises(PathTraversalError):
        resolve_safe_path(root, "C:/Windows/System32")


def test_traversal_that_stays_under_root_is_allowed(root: Path) -> None:
    # "sub/../top.txt" contains ".." but never actually leaves root -- our
    # implementation takes the stricter, simpler stance of rejecting *any*
    # ".." segment rather than trying to prove containment after the fact,
    # so this is intentionally also rejected. Documented here so the
    # decision is visible, not just implicit.
    with pytest.raises(PathTraversalError):
        resolve_safe_path(root, "sub/../top.txt")


# --------------------------------------------------------------------------
# Token hashing
# --------------------------------------------------------------------------


def test_verify_token_accepts_correct_token() -> None:
    token_hash = hash_token("correct-horse-battery-staple")
    assert verify_token("correct-horse-battery-staple", token_hash) is True


def test_verify_token_rejects_wrong_token() -> None:
    token_hash = hash_token("correct-horse-battery-staple")
    assert verify_token("wrong-guess", token_hash) is False


def test_verify_token_rejects_empty_string() -> None:
    token_hash = hash_token("correct-horse-battery-staple")
    assert verify_token("", token_hash) is False


def test_hash_token_is_not_reversible_plaintext() -> None:
    token_hash = hash_token("my-secret")
    assert b"my-secret" not in token_hash
    assert len(token_hash) == 32  # sha256 digest size


def test_hash_token_is_deterministic() -> None:
    assert hash_token("same-input") == hash_token("same-input")


def test_hash_token_differs_for_different_input() -> None:
    assert hash_token("aaa") != hash_token("bbb")
