import io
import os
from pathlib import Path

import pytest
from actionitems import ingest
from actionitems.errors import InputError
from actionitems.ingest import check_output_path, decode_bytes, read_documents
from actionitems.text import one_line, safe_text, strip_controls


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("\x1b[31mred\x1b[0m text", "red text"),
        ("bell\x07 and nul\x00 removed", "bell and nul removed"),
        ("keep\ttabs\nand newlines", "keep\ttabs\nand newlines"),
    ],
)
def test_strip_controls(raw, expected):
    assert strip_controls(raw) == expected


def test_safe_text_escapes_control_characters_and_keeps_unicode():
    assert safe_text("a\x1b b") == "a\\x1b b"
    assert safe_text("café 東京") == "café 東京"


def test_one_line_collapses_whitespace():
    assert one_line("  a\n\n b\t c ") == "a b c"


def test_decode_utf8_strips_bom():
    assert decode_bytes("﻿hello".encode()) == "hello"


def test_decode_falls_back_to_cp1252():
    assert decode_bytes("café".encode("cp1252")) == "café"


def test_decode_rejects_binary_input():
    with pytest.raises(InputError, match="binary"):
        decode_bytes(b"abc\x00def")


def test_decode_rejects_oversized_input():
    with pytest.raises(InputError, match="larger"):
        decode_bytes(b"a" * (ingest.MAX_INPUT_BYTES + 1))


def test_decode_rejects_too_many_lines():
    with pytest.raises(InputError, match="lines"):
        decode_bytes(b"\n" * (ingest.MAX_LINES + 1))


def test_directory_input_reads_only_text_suffixes_sorted(tmp_path: Path):
    (tmp_path / "b.md").write_text("two", encoding="utf-8")
    (tmp_path / "a.vtt").write_text("one", encoding="utf-8")
    (tmp_path / "c.exe").write_bytes(b"MZ")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "d.txt").write_text("nested", encoding="utf-8")
    docs = read_documents([str(tmp_path)])
    assert [d.name for d in docs] == ["a.vtt", "b.md"]


def test_empty_directory_is_an_input_error(tmp_path: Path):
    with pytest.raises(InputError, match="no transcript files"):
        read_documents([str(tmp_path)])


def test_missing_path_is_an_input_error(tmp_path: Path):
    with pytest.raises(InputError, match="no such file"):
        read_documents([str(tmp_path / "missing.txt")])


def test_file_input_keeps_name_and_mtime(tmp_path: Path):
    path = tmp_path / "notes.txt"
    path.write_text("Priya: hi", encoding="utf-8")
    (doc,) = read_documents([str(path)])
    assert doc.name == "notes.txt"
    assert doc.text == "Priya: hi"
    assert doc.mtime is not None


def test_stdin_input(monkeypatch):
    class FakeStdin:
        buffer = io.BytesIO(b"Dana: book the room")

    monkeypatch.setattr("sys.stdin", FakeStdin())
    (doc,) = read_documents(["-"])
    assert doc.name == "<stdin>"
    assert doc.text == "Dana: book the room"
    assert doc.mtime is None


def test_output_path_refuses_directory(tmp_path: Path):
    with pytest.raises(InputError, match="not a directory"):
        check_output_path(tmp_path, [])


def test_output_path_refuses_to_overwrite_an_input(tmp_path: Path):
    source = tmp_path / "notes.txt"
    source.write_text("x", encoding="utf-8")
    with pytest.raises(InputError, match="overwrite an input"):
        check_output_path(source, [str(source)])


def test_output_path_needs_an_existing_parent(tmp_path: Path):
    with pytest.raises(InputError, match="does not exist"):
        check_output_path(tmp_path / "missing" / "out.md", [])


def test_output_path_resolves_a_new_file(tmp_path: Path):
    target = tmp_path / "out.json"
    assert check_output_path(target, []) == target.resolve()


def test_output_path_refuses_symlink(tmp_path: Path):
    real = tmp_path / "real.md"
    real.write_text("x", encoding="utf-8")
    link = tmp_path / "link.md"
    try:
        os.symlink(real, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks need privileges on this platform")
    with pytest.raises(InputError, match="symbolic link"):
        check_output_path(link, [])
