from __future__ import annotations

import io
from pathlib import Path

import pytest
import remove_bg
from PIL import Image, ImageDraw


def _make_png_bytes(
    size: tuple[int, int] = (8, 8), color: tuple[int, int, int] = (255, 0, 0)
) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _write_png(path: Path, **kwargs) -> None:
    path.write_bytes(_make_png_bytes(**kwargs))


class TestResolveInputFiles:
    def test_single_file_returned_as_is(self, tmp_path):
        f = tmp_path / "a.png"
        _write_png(f)
        assert remove_bg.resolve_input_files(f, recursive=False) == [f]

    def test_single_file_returned_even_with_unsupported_extension(self, tmp_path):
        # Discovery doesn't filter a directly-named single file -- process_one
        # is what flags an unsupported extension, so it still shows up as one
        # explicit result rather than silently vanishing from the batch.
        f = tmp_path / "notes.txt"
        f.write_text("not an image")
        assert remove_bg.resolve_input_files(f, recursive=False) == [f]

    def test_directory_non_recursive_finds_top_level_only(self, tmp_path):
        (tmp_path / "sub").mkdir()
        top = tmp_path / "top.png"
        _write_png(top)
        _write_png(tmp_path / "sub" / "nested.png")

        assert remove_bg.resolve_input_files(tmp_path, recursive=False) == [top]

    def test_directory_recursive_finds_nested_files(self, tmp_path):
        (tmp_path / "sub").mkdir()
        top = tmp_path / "top.png"
        nested = tmp_path / "sub" / "nested.png"
        _write_png(top)
        _write_png(nested)

        assert remove_bg.resolve_input_files(tmp_path, recursive=True) == sorted(
            [top, nested]
        )

    def test_ignores_unsupported_extensions_in_directory_scan(self, tmp_path):
        img = tmp_path / "a.png"
        _write_png(img)
        (tmp_path / "notes.txt").write_text("hi")

        assert remove_bg.resolve_input_files(tmp_path, recursive=False) == [img]

    def test_empty_directory_returns_empty_list(self, tmp_path):
        assert remove_bg.resolve_input_files(tmp_path, recursive=False) == []


class TestProcessOne:
    def test_unsupported_extension_is_skipped_without_touching_removal(
        self, tmp_path, monkeypatch
    ):
        calls = []
        monkeypatch.setattr(
            remove_bg,
            "remove_background_bytes",
            lambda data, session: calls.append(1) or data,
        )

        f = tmp_path / "notes.txt"
        f.write_text("hi")
        result = remove_bg.process_one(f, tmp_path / "out", tmp_path, session=None)

        assert result.status == "skipped"
        assert result.reason == "unsupported_extension"
        assert calls == []

    def test_corrupted_file_with_image_extension_is_skipped_not_crashed(self, tmp_path):
        f = tmp_path / "bad.png"
        f.write_bytes(b"this is not a real png file")

        result = remove_bg.process_one(f, tmp_path / "out", tmp_path, session=None)

        assert result.status == "skipped"
        assert result.reason.startswith("decode_error")

    def test_successful_removal_writes_output_and_returns_processed(
        self, tmp_path, monkeypatch
    ):
        fake_output = _make_png_bytes()
        monkeypatch.setattr(
            remove_bg, "remove_background_bytes", lambda data, session: fake_output
        )

        src = tmp_path / "photo.jpg"
        _write_png(
            src
        )  # PIL detects real format from content, extension is just the name
        out_dir = tmp_path / "out"

        result = remove_bg.process_one(src, out_dir, tmp_path, session=object())

        assert result.status == "processed"
        assert result.output_path == out_dir / "photo.png"
        assert result.output_path.read_bytes() == fake_output

    def test_removal_error_is_caught_and_leaves_no_partial_output(
        self, tmp_path, monkeypatch
    ):
        def boom(data, session):
            raise RuntimeError("onnxruntime exploded")

        monkeypatch.setattr(remove_bg, "remove_background_bytes", boom)

        src = tmp_path / "photo.png"
        _write_png(src)
        out_dir = tmp_path / "out"

        result = remove_bg.process_one(src, out_dir, tmp_path, session=object())

        assert result.status == "skipped"
        assert result.reason.startswith("removal_error")
        assert not out_dir.exists()

    def test_nested_relative_path_is_preserved_in_output(self, tmp_path, monkeypatch):
        fake_output = _make_png_bytes()
        monkeypatch.setattr(
            remove_bg, "remove_background_bytes", lambda data, session: fake_output
        )

        base_dir = tmp_path / "in"
        sub = base_dir / "sub"
        sub.mkdir(parents=True)
        src = sub / "nested.jpeg"
        _write_png(src)
        out_dir = tmp_path / "out"

        result = remove_bg.process_one(src, out_dir, base_dir, session=object())

        assert result.output_path == out_dir / "sub" / "nested.png"
        assert result.output_path.exists()


class TestProcessBatch:
    def test_invalid_model_raises_before_any_processing(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown model"):
            remove_bg.process_batch(
                tmp_path, tmp_path / "out", model_name="not-a-real-model"
            )

    def test_empty_input_never_creates_a_session(self, tmp_path, monkeypatch):
        def fail_session(name):
            raise AssertionError("should not create a session with nothing to process")

        monkeypatch.setattr(remove_bg, "new_session", fail_session)

        result = remove_bg.process_batch(
            tmp_path, tmp_path / "out", model_name=remove_bg.DEFAULT_MODEL
        )

        assert result.results == []

    def test_mixed_batch_processes_valid_and_skips_invalid(self, tmp_path, monkeypatch):
        monkeypatch.setattr(remove_bg, "new_session", lambda name: object())
        fake_output = _make_png_bytes()
        monkeypatch.setattr(
            remove_bg, "remove_background_bytes", lambda data, session: fake_output
        )

        good = tmp_path / "good.png"
        _write_png(good)
        (tmp_path / "corrupted.png").write_bytes(b"not a real png")
        (tmp_path / "notes.txt").write_text("ignored by directory scan entirely")

        out_dir = tmp_path / "out"
        batch = remove_bg.process_batch(
            tmp_path, out_dir, model_name=remove_bg.DEFAULT_MODEL
        )

        assert len(batch.processed) == 1
        assert batch.processed[0].path == good
        assert len(batch.skipped) == 1
        assert batch.skipped[0].reason.startswith("decode_error")
        assert (out_dir / "good.png").exists()

    def test_progress_callback_invoked_once_per_discovered_file(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(remove_bg, "new_session", lambda name: object())
        fake_output = _make_png_bytes()
        monkeypatch.setattr(
            remove_bg, "remove_background_bytes", lambda data, session: fake_output
        )

        for i in range(3):
            _write_png(tmp_path / f"img{i}.png")

        calls = []
        remove_bg.process_batch(
            tmp_path,
            tmp_path / "out",
            model_name=remove_bg.DEFAULT_MODEL,
            progress_callback=calls.append,
        )

        assert len(calls) == 3

    def test_recursive_flag_preserves_subdirectory_structure_in_output(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(remove_bg, "new_session", lambda name: object())
        fake_output = _make_png_bytes()
        monkeypatch.setattr(
            remove_bg, "remove_background_bytes", lambda data, session: fake_output
        )

        (tmp_path / "sub").mkdir()
        _write_png(tmp_path / "sub" / "nested.png")

        out_dir = tmp_path / "out"
        batch = remove_bg.process_batch(tmp_path, out_dir, recursive=True)

        assert len(batch.processed) == 1
        assert (out_dir / "sub" / "nested.png").exists()

    def test_non_recursive_by_default_skips_nested_files_entirely(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(remove_bg, "new_session", lambda name: object())
        monkeypatch.setattr(
            remove_bg,
            "remove_background_bytes",
            lambda data, session: _make_png_bytes(),
        )

        (tmp_path / "sub").mkdir()
        _write_png(tmp_path / "sub" / "nested.png")

        batch = remove_bg.process_batch(tmp_path, tmp_path / "out")

        assert batch.results == []


@pytest.mark.network
def test_remove_background_real_model_produces_varying_alpha(tmp_path):
    """End-to-end against a real (small) rembg model.

    Deliberately uses u2netp (~4.7MB) rather than DEFAULT_MODEL
    (isnet-general-use, ~179MB) so this one network-dependent test stays fast
    and light on bandwidth; the CLI's actual default is exercised manually
    (see README) rather than in every CI run. Skips cleanly rather than
    failing if the model can't be downloaded/run in this environment.
    """
    try:
        session = remove_bg.new_session("u2netp")
    except Exception as exc:  # noqa: BLE001 -- any download/runtime failure here means "skip", not "fail"
        pytest.skip(f"model unavailable in this environment: {exc}")

    img = Image.new("RGB", (64, 64), (10, 10, 10))
    ImageDraw.Draw(img).ellipse((16, 16, 48, 48), fill=(255, 220, 180))
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    try:
        out_bytes = remove_bg.remove_background_bytes(buf.getvalue(), session)
    except Exception as exc:  # noqa: BLE001 -- any inference failure here means "skip", not "fail"
        pytest.skip(f"model inference unavailable in this environment: {exc}")

    out_img = Image.open(io.BytesIO(out_bytes))
    assert out_img.mode == "RGBA"
    alpha_min, alpha_max = out_img.getchannel("A").getextrema()
    assert (
        alpha_min != alpha_max
    )  # genuinely varying alpha, not uniformly opaque/transparent
