import json

import pytest
from helpers import ink, make_blank, make_pdf, page_texts, render
from pdftoolkit.cli import main
from pypdf import PdfReader


def run(*argv):
    return main([str(a) for a in argv])


def test_merge_with_glob_and_page_suffix(tmp_path, capsys):
    make_pdf(tmp_path / "p1.pdf", 2)
    make_pdf(tmp_path / "p2.pdf", 3)
    out = tmp_path / "all.pdf"
    assert (
        run("merge", tmp_path / "p*.pdf", f"{tmp_path / 'p2.pdf'}[1]", "-o", out) == 0
    )
    assert page_texts(out) == ["p1 1", "p1 2", "p2 1", "p2 2", "p2 3", "p2 1"]
    assert "6 pages" in capsys.readouterr().out


def test_merge_requires_output(tmp_path, three):
    with pytest.raises(SystemExit) as exc:
        run("merge", three)
    assert exc.value.code != 0


def test_errors_are_one_line_with_status_1(tmp_path, capsys):
    assert run("merge", tmp_path / "missing.pdf", "-o", tmp_path / "o.pdf") == 1
    err = capsys.readouterr().err
    assert err.startswith("error: no such file")
    assert "Traceback" not in err


def test_split_default_folder_and_out_dir(tmp_path, five, capsys):
    assert run("split", five, "--every", 2) == 0
    assert sorted(p.name for p in (tmp_path / "five_split").iterdir()) == [
        "five_p1-2.pdf",
        "five_p3-4.pdf",
        "five_p5.pdf",
    ]
    assert run("split", five, "--each", "--out-dir", tmp_path / "pages") == 0
    assert len(list((tmp_path / "pages").iterdir())) == 5


def test_split_rejects_two_modes(five, capsys):
    assert run("split", five, "--every", 2, "--each") == 1
    assert "exactly one" in capsys.readouterr().err


def test_extract_reorder_rotate_to_default_names(tmp_path, three):
    assert run("extract", three, "--pages", "3,1") == 0
    assert page_texts(tmp_path / "three_extract.pdf") == ["three 3", "three 1"]
    assert run("reorder", three, "--order", "3-1") == 0
    assert page_texts(tmp_path / "three_reordered.pdf") == [
        "three 3",
        "three 2",
        "three 1",
    ]
    assert run("rotate", three, "--degrees", 180, "--pages", "1") == 0
    assert PdfReader(tmp_path / "three_rotated.pdf").pages[0].rotation == 180


def test_batch_of_inputs_into_out_dir(tmp_path, three, five):
    out = tmp_path / "out"
    assert run("extract", three, five, "--pages", "1", "--out-dir", out) == 0
    assert sorted(p.name for p in out.iterdir()) == [
        "five_extract.pdf",
        "three_extract.pdf",
    ]


def test_output_file_with_many_inputs_is_an_error(tmp_path, three, five, capsys):
    assert run("extract", three, five, "--pages", "1", "-o", tmp_path / "x.pdf") == 1
    assert "exactly one input" in capsys.readouterr().err


def test_never_overwrites_inputs_or_existing_files(tmp_path, three, capsys):
    assert run("extract", three, "--pages", "1", "-o", three) == 1
    assert "also an input" in capsys.readouterr().err
    out = tmp_path / "o.pdf"
    assert run("extract", three, "--pages", "1", "-o", out) == 0
    assert run("extract", three, "--pages", "2", "-o", out) == 1
    assert run("extract", three, "--pages", "2", "-o", out, "--force") == 0
    assert page_texts(out) == ["three 2"]


def test_watermark_command(tmp_path, three):
    out = tmp_path / "w.pdf"
    assert run("watermark", three, "--text", "DRAFT", "--opacity", 0.5, "-o", out) == 0
    assert ink(render(out)) > ink(render(three))
    assert (
        run(
            "watermark",
            three,
            "--text",
            "DRAFT",
            "--position",
            "tile",
            "--under",
            "-f",
            "-o",
            out,
        )
        == 0
    )


def test_watermark_validation_surfaces_as_error(three, capsys):
    assert run("watermark", three) == 1
    assert "exactly one" in capsys.readouterr().err
    with pytest.raises(SystemExit):  # Cyclopts rejects an unknown --position itself
        run("watermark", three, "--text", "x", "--position", "middle")


def test_encrypt_prompts_never_in_ci_but_reads_env(
    tmp_path, three, monkeypatch, capsys
):
    monkeypatch.delenv("PDFTOOL_USER_PASSWORD", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False, raising=False)
    assert run("encrypt", three, "-o", tmp_path / "e.pdf") == 1
    assert "required" in capsys.readouterr().err
    monkeypatch.setenv("PDFTOOL_USER_PASSWORD", "pw")
    assert run("encrypt", three, "-o", tmp_path / "e.pdf", "--no-allow-print") == 0
    assert PdfReader(tmp_path / "e.pdf").is_encrypted


def test_encrypt_prompt_on_terminal(tmp_path, three, monkeypatch):
    monkeypatch.delenv("PDFTOOL_USER_PASSWORD", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    monkeypatch.setattr("getpass.getpass", lambda prompt: "typed")
    assert run("encrypt", three, "-o", tmp_path / "e.pdf") == 0
    assert PdfReader(tmp_path / "e.pdf").decrypt("typed")


def test_password_option_and_env_for_encrypted_inputs(
    tmp_path, three, monkeypatch, capsys
):
    enc = tmp_path / "e.pdf"
    monkeypatch.setenv("PDFTOOL_USER_PASSWORD", "pw")
    run("encrypt", three, "-o", enc)
    monkeypatch.delenv("PDFTOOL_PASSWORD", raising=False)
    assert run("extract", enc, "--pages", "1", "-o", tmp_path / "x.pdf") == 1
    assert "password required" in capsys.readouterr().err
    assert (
        run(
            "extract", enc, "--pages", "1", "--password", "pw", "-o", tmp_path / "x.pdf"
        )
        == 0
    )
    monkeypatch.setenv("PDFTOOL_PASSWORD", "pw")
    assert run("decrypt", enc, "-o", tmp_path / "plain.pdf") == 0
    assert not PdfReader(tmp_path / "plain.pdf").is_encrypted


def test_meta_and_info(tmp_path, three, capsys):
    out = tmp_path / "m.pdf"
    assert run("meta", three, "--title", "Hello", "--author", "Me", "-o", out) == 0
    capsys.readouterr()
    assert run("info", out) == 0
    text = capsys.readouterr().out
    assert "3 pages" in text and "Title: Hello" in text and "300x400" in text
    assert run("info", out, "--json") == 0
    (data,) = json.loads(capsys.readouterr().out)
    assert data["pages"] == 3 and data["metadata"]["Title"] == "Hello"


def test_info_on_blank_pages_and_corrupt_file(tmp_path, capsys):
    assert run("info", make_blank(tmp_path / "b.pdf", 2)) == 0
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"junk")
    assert run("info", bad) == 1


def test_module_entry_point_and_help(tmp_path):
    import subprocess
    import sys

    done = subprocess.run(
        [sys.executable, "-m", "pdftoolkit", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0
    for command in ("merge", "split", "watermark", "encrypt", "batch"):
        assert command in done.stdout
