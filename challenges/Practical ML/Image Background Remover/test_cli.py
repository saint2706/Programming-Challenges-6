from __future__ import annotations

import io
from pathlib import Path

import remove_bg
from cli import app
from PIL import Image
from typer.testing import CliRunner

runner = CliRunner()


def _make_png_bytes(
    size: tuple[int, int] = (8, 8), color: tuple[int, int, int] = (0, 255, 0)
) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _write_png(path: Path, **kwargs) -> None:
    path.write_bytes(_make_png_bytes(**kwargs))


def _stub_model(monkeypatch) -> None:
    """Redirect the real rembg calls used inside remove_bg.process_batch so
    CLI tests never need a network round trip or a model download."""
    monkeypatch.setattr(remove_bg, "new_session", lambda name: object())
    monkeypatch.setattr(
        remove_bg, "remove_background_bytes", lambda data, session: _make_png_bytes()
    )


def test_nonexistent_input_path_is_a_usage_error(tmp_path):
    result = runner.invoke(
        app, [str(tmp_path / "does-not-exist.png"), str(tmp_path / "out")]
    )
    assert result.exit_code == 2


def test_invalid_model_choice_is_rejected(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    _write_png(src / "a.png")

    result = runner.invoke(
        app, [str(src), str(tmp_path / "out"), "--model", "not-a-real-model"]
    )

    assert result.exit_code == 2
    assert "Unknown model" in result.output


def test_empty_input_directory_exits_with_no_images_found(tmp_path):
    src = tmp_path / "in"
    src.mkdir()

    result = runner.invoke(app, [str(src), str(tmp_path / "out")])

    assert result.exit_code == 1
    assert "No supported images found" in result.output


def test_successful_batch_reports_summary_and_writes_output(tmp_path, monkeypatch):
    _stub_model(monkeypatch)
    src = tmp_path / "in"
    src.mkdir()
    _write_png(src / "good.png")
    out_dir = tmp_path / "out"

    result = runner.invoke(app, [str(src), str(out_dir)])

    assert result.exit_code == 0
    assert "Processed" in result.output
    assert (out_dir / "good.png").exists()


def test_batch_with_a_corrupted_file_still_succeeds_overall(tmp_path, monkeypatch):
    _stub_model(monkeypatch)
    src = tmp_path / "in"
    src.mkdir()
    _write_png(src / "good.png")
    (src / "corrupted.png").write_bytes(b"not a real png")
    out_dir = tmp_path / "out"

    result = runner.invoke(app, [str(src), str(out_dir)])

    assert result.exit_code == 0
    assert (out_dir / "good.png").exists()
    assert not (out_dir / "corrupted.png").exists()
    assert "skip" in result.output
    assert "decode_error" in result.output


def test_batch_where_everything_is_skipped_exits_with_error_code(tmp_path, monkeypatch):
    _stub_model(monkeypatch)
    src = tmp_path / "in"
    src.mkdir()
    (src / "corrupted.png").write_bytes(b"not a real png")

    result = runner.invoke(app, [str(src), str(tmp_path / "out")])

    assert result.exit_code == 1


def test_recursive_flag_reaches_nested_files(tmp_path, monkeypatch):
    _stub_model(monkeypatch)
    src = tmp_path / "in"
    (src / "sub").mkdir(parents=True)
    _write_png(src / "sub" / "nested.png")
    out_dir = tmp_path / "out"

    result = runner.invoke(app, [str(src), str(out_dir), "--recursive"])

    assert result.exit_code == 0
    assert (out_dir / "sub" / "nested.png").exists()


def test_without_recursive_flag_nested_files_are_not_found(tmp_path, monkeypatch):
    _stub_model(monkeypatch)
    src = tmp_path / "in"
    (src / "sub").mkdir(parents=True)
    _write_png(src / "sub" / "nested.png")

    result = runner.invoke(app, [str(src), str(tmp_path / "out")])

    assert result.exit_code == 1
    assert "No supported images found" in result.output


def test_single_file_input_is_supported(tmp_path, monkeypatch):
    _stub_model(monkeypatch)
    src = tmp_path / "photo.png"
    _write_png(src)
    out_dir = tmp_path / "out"

    result = runner.invoke(app, [str(src), str(out_dir)])

    assert result.exit_code == 0
    assert (out_dir / "photo.png").exists()
