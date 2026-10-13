"""Regression tests for problems found in the security review of the first version."""

import math

import pytest
from helpers import make_pdf
from pdftoolkit import ops
from pdftoolkit.cli import main
from pdftoolkit.errors import PdfToolError
from pdftoolkit.files import Source
from pdftoolkit.watermark import watermark
from PIL import Image
from pypdf import PasswordType, PdfReader


def S(path, pages=None):
    return Source(path, pages)


def test_restricted_encrypt_has_a_separate_owner_key(tmp_path, three):
    """An owner password equal to the user password would make the restrictions meaningless."""
    enc = tmp_path / "e.pdf"
    ops.encrypt(S(three), enc, user_password="u", allow_copy=False)
    assert PdfReader(enc).decrypt("u") == PasswordType.USER_PASSWORD


def test_unrestricted_encrypt_keeps_one_password(tmp_path, three):
    enc = tmp_path / "e.pdf"
    ops.encrypt(S(three), enc, user_password="u")
    assert PdfReader(enc).decrypt("u") == PasswordType.OWNER_PASSWORD


@pytest.mark.parametrize("size", [0, -5, math.nan, math.inf])
def test_bad_sizes_are_errors_not_tracebacks(tmp_path, three, size):
    with pytest.raises(PdfToolError):
        watermark(S(three), tmp_path / "w.pdf", text="x", size=size, position="tile")


@pytest.mark.parametrize("name", ["angle", "opacity", "scale"])
def test_non_finite_numbers_are_refused(tmp_path, three, name):
    with pytest.raises(PdfToolError, match="finite"):
        watermark(S(three), tmp_path / "w.pdf", text="x", **{name: math.nan})


def test_huge_image_is_refused(tmp_path, three):
    huge = tmp_path / "huge.png"
    Image.new("1", (20000, 10000)).save(huge)  # tiny on disk, 200 megapixels decoded
    with pytest.raises(PdfToolError, match="too large"):
        watermark(S(three), tmp_path / "w.pdf", image=huge)


def test_fonts_with_the_same_name_in_different_folders_do_not_collide(tmp_path, three):
    from pathlib import Path

    arial = Path("C:/Windows/Fonts/arial.ttf")
    if not arial.exists():
        pytest.skip("no system TrueType font")
    copy = tmp_path / "other" / "arial.ttf"
    copy.parent.mkdir()
    copy.write_bytes(b"not a font")
    watermark(S(three), tmp_path / "a.pdf", text="a", font=arial)
    with pytest.raises(PdfToolError, match="cannot load font"):
        watermark(S(three), tmp_path / "b.pdf", text="a", font=copy)


def test_split_cannot_overwrite_another_input_of_the_same_run(tmp_path, capsys):
    a = make_pdf(tmp_path / "a.pdf", 2)
    victim = make_pdf(
        tmp_path / "a_p1.pdf", 1, label="victim"
    )  # a.pdf --each writes this name
    before = victim.read_bytes()
    assert (
        main(["split", str(a), str(victim), "--each", "--out-dir", str(tmp_path), "-f"])
        == 1
    )
    assert "also an input" in capsys.readouterr().err
    assert victim.read_bytes() == before


def test_split_of_same_named_inputs_into_one_folder_is_refused_up_front(
    tmp_path, capsys
):
    a = make_pdf(tmp_path / "x" / "same.pdf", 2)
    b = make_pdf(tmp_path / "y" / "same.pdf", 2)
    out = tmp_path / "out"
    assert main(["split", str(a), str(b), "--each", "--out-dir", str(out)]) == 1
    assert "share a file name" in capsys.readouterr().err
    assert not out.exists()


def test_unwritable_destination_is_a_clean_error(tmp_path, three, capsys):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a folder")
    assert main(["extract", str(three), "--pages", "1", "--out-dir", str(blocker)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: cannot write") and "Traceback" not in err
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("force = true", "force cannot be set"),
        ('password = "hunter2"', "password_env"),
        ('user_password = "hunter2"', "user_password_env"),
        ('title_env = "AWS_SECRET_ACCESS_KEY"', "only for passwords"),
        ('text_env = "HOME"', "only for passwords"),
    ],
)
def test_batch_files_cannot_force_or_leak_secrets(
    tmp_path, three, monkeypatch, capsys, line, message
):
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "topsecret")
    job = tmp_path / "jobs.toml"
    job.write_text(
        f'[[job]]\nop = "meta"\ninputs = ["three.pdf"]\noutput = "out.pdf"\n{line}\n',
        encoding="utf-8",
    )
    assert main(["batch", str(job)]) == 1
    assert message in capsys.readouterr().err
    assert not (tmp_path / "out.pdf").exists()


def test_batch_force_comes_only_from_the_command_line(tmp_path, three):
    (tmp_path / "out.pdf").write_bytes(b"precious")
    job = tmp_path / "jobs.toml"
    job.write_text(
        '[[job]]\nop = "extract"\ninputs = ["three.pdf"]\noutput = "out.pdf"\npages = "1"\n',
        encoding="utf-8",
    )
    assert main(["batch", str(job)]) == 1
    assert (tmp_path / "out.pdf").read_bytes() == b"precious"
    assert main(["batch", str(job), "--force"]) == 0
