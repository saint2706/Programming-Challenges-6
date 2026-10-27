import json
import re

import pytest
from voicememo.asr import Segment, Word
from voicememo.export import (
    clean_text,
    library_json,
    md_escape,
    render,
    timestamp,
    to_json,
    to_markdown,
    to_srt,
    to_vtt,
)
from voicememo.library import Library


def seg_dict(start, end, text):
    return {"start": start, "end": end, "text": text, "words": []}


@pytest.mark.parametrize(
    ("seconds", "sep", "expected"),
    [
        (0, ",", "00:00:00,000"),
        (0.0014, ",", "00:00:00,001"),
        (1.5, ",", "00:00:01,500"),
        (61.25, ".", "00:01:01.250"),
        (3661.5, ",", "01:01:01,500"),
        (36000.0, ".", "10:00:00.000"),
        (-5.0, ",", "00:00:00,000"),
    ],
)
def test_timestamp_format(seconds, sep, expected):
    assert timestamp(seconds, sep) == expected


def test_timestamp_millisecond_carry():
    assert timestamp(0.9996) == "00:00:01,000"
    assert timestamp(59.9996) == "00:01:00,000"


def test_clean_text_removes_escape_sequences_and_collapses_space():
    assert clean_text("a\x1b[31mred\x1b[0m\tb\n\nc") == "a[31mred[0m b c"


def test_clean_text_strips_null_and_bell():
    assert clean_text("x\x00y\x07z") == "xyz"


def test_srt_numbers_blocks_and_uses_comma_timestamps():
    out = to_srt([seg_dict(0, 1.5, "Hello."), seg_dict(1.5, 3, "World.")])
    assert out == (
        "1\n00:00:00,000 --> 00:00:01,500\nHello.\n\n"
        "2\n00:00:01,500 --> 00:00:03,000\nWorld.\n"
    )


def test_srt_escapes_arrow_that_would_break_cue_parsing():
    out = to_srt([seg_dict(0, 1, "go --> there")])
    assert "-->" in out.splitlines()[1]
    assert out.splitlines()[2] == "go -&gt; there"


def test_srt_drops_control_characters_from_text():
    out = to_srt([seg_dict(0, 1, "fine\x1b[2Jtext")])
    assert "\x1b" not in out


def test_srt_empty_is_empty():
    assert to_srt([]) == ""


def test_vtt_header_dot_timestamps_and_escapes():
    out = to_vtt([seg_dict(0, 2, "Tom & <Jerry>")])
    lines = out.splitlines()
    assert lines[0] == "WEBVTT"
    assert lines[2] == "00:00:00.000 --> 00:00:02.000"
    assert lines[3] == "Tom &amp; &lt;Jerry&gt;"


def test_vtt_empty_still_has_header():
    assert to_vtt([]).startswith("WEBVTT")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("*bold*", r"\*bold\*"),
        ("_under_", r"\_under\_"),
        ("[link](x)", r"\[link\](x)"),
        ("# head", r"\# head"),
        ("a|b", r"a\|b"),
        ("<img src=x>", r"\<img src=x\>"),
        ("back\\slash", r"back\\slash"),
        ("`code`", r"\`code\`"),
    ],
)
def test_md_escape_neutralises_markdown(raw, expected):
    assert md_escape(raw) == expected


def test_md_escape_flattens_newlines_so_headings_cannot_form():
    assert md_escape("ok\n# injected") == r"ok \# injected"


@pytest.fixture
def memo_with_hostile_text(library):
    memo = library.create_pending("hx1", "<img src=x onerror=alert(1)>.m4a", "x.m4a")
    library.complete(
        memo.id,
        [
            Segment(0.0, 1.0, "# not a heading [x](y)", (Word("#", 0.0, 0.5),)),
            Segment(1.0, 2.0, "plain words"),
        ],
        duration=2.0,
        language="en",
        backend="fake",
    )
    library.add_tag(memo.id, "*star*")
    return library.get(memo.id)


def test_markdown_escapes_hostile_title_transcript_and_tags(
    library, memo_with_hostile_text
):
    text = to_markdown(library, memo_with_hostile_text)
    assert text.startswith("# \\<img src=x onerror=alert(1)\\>")
    assert "\\# not a heading \\[x\\](y)" in text
    assert "\\*star\\*" in text
    assert re.search(r"(?<!\\)<img", text) is None


def test_markdown_timestamps_and_tags_listing(library, memo_with_hostile_text):
    text = to_markdown(library, memo_with_hostile_text)
    assert "- **[00:00:00]** " in text
    assert "- **[00:00:01]** plain words" in text
    assert "- Duration: 2.0 s" in text


def test_markdown_says_none_and_unknown_when_empty(library):
    memo = library.create_pending("hx2", "a.m4a", "a.m4a")
    library.complete(memo.id, [], duration=0.0, language="", backend="fake")
    text = to_markdown(library, library.get(memo.id))
    assert "- Language: unknown" in text
    assert "- Tags: none" in text


def test_json_round_trips_segments_and_words(library, memo_with_hostile_text):
    record = json.loads(to_json(library, memo_with_hostile_text))
    assert record["language"] == "en"
    assert record["tags"] == ["*star*"]
    assert record["segments"][0]["words"][0] == {"text": "#", "start": 0.0, "end": 0.5}


def test_json_keeps_unicode_unescaped(library):
    memo = library.create_pending("hx3", "café.m4a", "café.m4a")
    library.complete(
        memo.id,
        [Segment(0, 1, "Grüße aus Köln")],
        duration=1.0,
        language="de",
        backend="fake",
    )
    raw = to_json(library, library.get(memo.id))
    assert "Grüße aus Köln" in raw
    assert json.loads(raw)["title"] == "café"


def test_library_json_counts_only_finished_memos(library):
    done = library.create_pending("l1", "a.m4a", "a.m4a")
    library.complete(
        done.id, [Segment(0, 1, "one")], duration=1.0, language="en", backend="fake"
    )
    library.create_pending("l2", "b.m4a", "b.m4a")
    payload = json.loads(library_json(library))
    assert payload["count"] == 1
    assert [m["id"] for m in payload["memos"]] == [done.id]
    assert isinstance(payload["exported_at"], float)


def test_render_dispatches_each_format(library, memo_with_hostile_text):
    assert render(library, memo_with_hostile_text, "md").startswith("# ")
    assert render(library, memo_with_hostile_text, "srt").startswith("1\n")
    assert render(library, memo_with_hostile_text, "vtt").startswith("WEBVTT")
    assert (
        json.loads(render(library, memo_with_hostile_text, "json"))["id"]
        == memo_with_hostile_text.id
    )


def test_render_rejects_unknown_format(library, memo_with_hostile_text):
    with pytest.raises(ValueError, match="unknown export format"):
        render(library, memo_with_hostile_text, "docx")


def test_export_of_library_with_no_memos_is_valid_json(tmp_path):
    with Library(tmp_path / "empty") as lib:
        assert json.loads(library_json(lib))["memos"] == []
