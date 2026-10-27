import json
import re
from pathlib import Path

import pytest
from helpers import tone, write_wav
from voicememo.cli import build_parser, main

FIXTURES = Path(__file__).parent / "fixtures"
BOX_DRAWING = re.compile("[─-╿▀-▟]")


@pytest.fixture
def script(tmp_path, monkeypatch):
    path = tmp_path / "script.json"
    path.write_text(
        json.dumps(
            {
                "walk": "Walked the dog by the river. The water was cold.",
                "oven": "Preheat the oven. Bake the bread for forty minutes.",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("VOICEMEMO_FAKE_SCRIPT", str(path))
    return path


@pytest.fixture
def audio(tmp_path):
    return write_wav(tmp_path / "walk.wav", tone(2.0))


def ingest(path, *extra):
    return main(["ingest", str(path), "--backend", "fake", *extra])


def test_parser_requires_a_command():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_ingest_list_show_round_trip(script, audio, capsys):
    assert ingest(audio) == 0
    out = capsys.readouterr().out
    assert "walk.wav: added #1" in out

    assert main(["list"]) == 0
    assert "#1 done" in capsys.readouterr().out

    assert main(["show", "1"]) == 0
    shown = capsys.readouterr().out
    assert "The water was cold." in shown
    assert "tags: " in shown  # the pipeline auto-tags every memo it finishes


def test_ingest_directory_takes_only_audio_files(script, tmp_path, capsys):
    folder = tmp_path / "inbox"
    folder.mkdir()
    write_wav(folder / "walk.wav", tone(1.0))
    (folder / "notes.txt").write_text("not audio", encoding="utf-8")
    assert main(["ingest", str(folder), "--backend", "fake"]) == 0
    out = capsys.readouterr().out
    assert "walk.wav" in out
    assert "notes.txt" not in out


def test_ingest_reports_failed_file_with_exit_code_one(script, tmp_path, capsys):
    bad = tmp_path / "bad.m4a"
    bad.write_bytes(b"garbage" * 200)
    assert ingest(bad) == 1
    assert "failed" in capsys.readouterr().out


def test_ingest_missing_file_is_an_error_not_a_crash(script, tmp_path, capsys):
    assert ingest(tmp_path / "nope.wav") == 1
    assert "error" in capsys.readouterr().out


def test_second_ingest_is_skipped(script, audio, capsys):
    ingest(audio)
    capsys.readouterr()
    assert ingest(audio) == 0
    assert "skipped" in capsys.readouterr().out


def test_retry_failed_flag_is_accepted(script, audio):
    assert ingest(audio, "--retry-failed") == 0


def test_tag_add_then_list_filter_then_remove(script, audio, capsys):
    ingest(audio)
    capsys.readouterr()
    assert main(["tag", "add", "1", "River Walk"]) == 0
    assert "river walk" in capsys.readouterr().out
    assert main(["list", "--tag", "river walk"]) == 0
    assert "#1" in capsys.readouterr().out
    assert main(["tag", "rm", "1", "river walk"]) == 0
    assert "removed" in capsys.readouterr().out
    assert main(["tag", "rm", "1", "river walk"]) == 0
    assert "no such tag" in capsys.readouterr().out


def test_tag_on_missing_memo_exits_one(script, capsys):
    assert main(["tag", "add", "42", "x"]) == 1
    assert "42" in capsys.readouterr().out


def test_search_finds_transcript_words(script, audio, capsys):
    ingest(audio)
    capsys.readouterr()
    assert main(["search", "river"]) == 0
    out = capsys.readouterr().out
    assert "#1" in out and "river" in out.lower()


def test_search_without_matches(script, audio, capsys):
    ingest(audio)
    capsys.readouterr()
    main(["search", "zeppelin"])
    assert "no matches" in capsys.readouterr().out


def test_show_missing_memo_exits_one(script, capsys):
    assert main(["show", "9"]) == 1


def test_rm_deletes_memo_and_audio(script, audio, capsys):
    ingest(audio)
    capsys.readouterr()
    assert main(["rm", "1"]) == 0
    assert "removed #1" in capsys.readouterr().out
    assert main(["show", "1"]) == 1


def test_export_markdown_to_file(script, audio, tmp_path):
    ingest(audio)
    target = tmp_path / "out.md"
    assert main(["export", "1", "--format", "md", "--output", str(target)]) == 0
    text = target.read_text(encoding="utf-8")
    assert text.startswith("# ")
    assert "The water was cold." in text


@pytest.mark.parametrize(
    ("fmt", "marker"),
    [("srt", "00:00:00,000 --> "), ("vtt", "WEBVTT"), ("json", '"segments"')],
)
def test_export_formats_to_stdout(script, audio, capsys, fmt, marker):
    ingest(audio)
    capsys.readouterr()
    assert main(["export", "1", "--format", fmt]) == 0
    assert marker in capsys.readouterr().out


def test_export_library_writes_all_finished_memos(script, audio, tmp_path, capsys):
    ingest(audio)
    target = tmp_path / "library.json"
    assert main(["export-library", str(target)]) == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["count"] == 1


def test_retranscribe_uses_original_name_for_engine(script, audio, capsys):
    ingest(audio)
    capsys.readouterr()
    assert main(["retranscribe", "1", "--backend", "fake"]) == 0
    assert "re-transcribed" in capsys.readouterr().out


def test_retag_with_lsa_reports_method(script, audio, capsys):
    ingest(audio)
    capsys.readouterr()
    assert main(["retag", "--embedder", "lsa"]) == 0
    assert "retagged with" in capsys.readouterr().out


def test_retag_model2vec_without_model_says_how_to_fix(script, audio, capsys):
    ingest(audio)
    capsys.readouterr()
    assert main(["retag", "--embedder", "model2vec"]) == 1
    assert "models pull --embedder" in capsys.readouterr().out


def test_eval_tags_prints_scores_for_labelled_file(capsys):
    assert main(["eval-tags", str(FIXTURES / "topics.json")]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["memos"] == 48
    assert "adjusted_rand" in report and "purity" in report


def test_models_status_lists_every_model(capsys):
    assert main(["models", "status"]) == 0
    out = capsys.readouterr().out
    for name in ["parakeet", "silero-vad", "whisper-base", "model2vec"]:
        assert f"{name}: missing" in out


def test_serve_refuses_non_loopback(capsys):
    assert main(["serve", "--host", "0.0.0.0", "--port", "8765"]) == 2
    assert "loopback-only" in capsys.readouterr().out


def test_allow_remote_flag_is_gone():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["serve", "--allow-remote"])


@pytest.mark.parametrize(
    "argv",
    [
        ["ingest", "x.m4a", "--whisper", "../../evil"],
        ["retranscribe", "1", "--whisper", "huge"],
        ["models", "pull", "--whisper", "huge"],
    ],
)
def test_whisper_names_are_limited_to_the_allow_list(argv):
    with pytest.raises(SystemExit):
        build_parser().parse_args(argv)


def test_ingest_reports_unreadable_file_instead_of_crashing(tmp_path, capsys, script):
    missing = tmp_path / "gone.m4a"
    assert main(["ingest", str(missing), "--backend", "fake"]) == 1
    assert "error" in capsys.readouterr().out


def test_watch_once_ingests_then_exits(script, tmp_path, capsys):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    write_wav(inbox / "walk.wav", tone(1.0))
    assert (
        main(["watch", str(inbox), "--backend", "fake", "--settle", "0", "--once"]) == 0
    )
    assert main(["list"]) == 0
    assert "walk" in capsys.readouterr().out


def test_output_has_no_box_drawing_under_github_actions(
    script, audio, capsys, monkeypatch
):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    ingest(audio)
    main(["list"])
    main(["show", "1"])
    out = capsys.readouterr().out
    assert out
    assert not BOX_DRAWING.search(out)


def test_control_characters_in_titles_are_neutralised(library, capsys):
    # Windows refuses control characters in file names, so the title goes in directly.
    library.create_pending("ctl1", "walk\x1b[2J.m4a", "walk.m4a")
    assert main(["list"]) == 0
    assert "\x1b" not in capsys.readouterr().out
