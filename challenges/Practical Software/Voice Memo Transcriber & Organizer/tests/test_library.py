import pytest
from helpers import tone, write_wav
from voicememo.asr import Segment, Word
from voicememo.library import (
    MAX_TAG_LENGTH,
    Library,
    LibraryError,
    file_sha256,
    fts_query,
    normalize_tag,
)


def seg(start: float, end: float, text: str) -> Segment:
    words = tuple(
        Word(w, start + i * 0.1, start + i * 0.1 + 0.1)
        for i, w in enumerate(text.split())
    )
    return Segment(start, end, text, words)


def add_done(lib: Library, sha: str, name: str, texts: list[str]) -> int:
    memo = lib.create_pending(sha, name, ".wav")
    segments = [seg(i * 2.0, i * 2.0 + 2.0, t) for i, t in enumerate(texts)]
    lib.complete(
        memo.id,
        segments,
        duration=len(texts) * 2.0,
        language="en",
        backend="fake",
    )
    return memo.id


# ----- pure helpers ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  Dinner  ", "dinner"),
        ("Two   Words", "two words"),
        ("tab\there", "tab here"),
        ("zero\u200bwidth", "zerowidth"),
        ("CAFÉ", "café"),
    ],
)
def test_normalize_tag(raw, expected):
    assert normalize_tag(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "\u200b\u200b"])
def test_empty_tag_rejected(raw):
    with pytest.raises(LibraryError):
        normalize_tag(raw)


def test_overlong_tag_rejected():
    with pytest.raises(LibraryError):
        normalize_tag("x" * (MAX_TAG_LENGTH + 1))


def test_fts_query_quotes_operators_and_punctuation():
    q = fts_query('oven OR NEAR(( "quoted" col:x -y')
    assert q is not None
    # every term is wrapped in double quotes, so FTS5 sees plain strings
    for term in q.split(" "):
        assert term.startswith('"') and term.endswith('"'), term


def test_fts_query_prefix_star_is_kept_outside_quotes():
    assert fts_query("walk*") == '"walk"*'


def test_fts_query_doubles_embedded_quotes_and_never_returns_empty_string():
    assert fts_query('say "hi"') == '"say" "hi"'
    assert fts_query("!!! ??? ") is None
    assert fts_query("") is None


def test_fts_query_caps_term_count():
    q = fts_query(" ".join(f"w{i}" for i in range(500)))
    assert q is not None
    assert len(q.split(" ")) == 32


def test_file_sha256_matches_hashlib(tmp_path):
    import hashlib

    path = write_wav(tmp_path / "a.wav", tone(0.5))
    assert file_sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()


# ----- memo lifecycle --------------------------------------------------------------------------


def test_create_pending_then_complete_marks_done(library):
    memo = library.create_pending("abc123", "Morning walk.m4a", "Morning walk.m4a")
    assert memo.status == "pending"
    assert memo.title == "Morning walk"
    assert memo.audio_file == "abc123.m4a"
    assert library.all_ids() == []

    library.complete(
        memo.id, [seg(0, 1, "hello there")], duration=1.0, language="en", backend="fake"
    )
    done = library.get(memo.id)
    assert done.status == "done"
    assert done.language == "en"
    assert done.backend == "fake"
    assert library.all_ids() == [memo.id]


def test_find_by_sha_and_missing_lookups(library):
    memo = library.create_pending("deadbeef", "x.wav", ".wav")
    assert library.find_by_sha("deadbeef").id == memo.id
    assert library.find_by_sha("nope") is None
    with pytest.raises(LibraryError):
        library.get(9999)


def test_sha_is_unique(library):
    import sqlite3

    library.create_pending("same", "a.wav", ".wav")
    with pytest.raises(sqlite3.IntegrityError):
        library.create_pending("same", "b.wav", ".wav")


def test_fail_records_message_and_truncates(library):
    memo = library.create_pending("f1", "bad.wav", ".wav")
    library.fail(memo.id, "x" * 2000)
    got = library.get(memo.id)
    assert got.status == "failed"
    assert len(got.error) == 500
    assert library.all_ids() == []


def test_recompleting_replaces_segments_not_appends(library):
    memo_id = add_done(library, "r1", "a.wav", ["first version"])
    library.complete(
        memo_id,
        [seg(0, 1, "second"), seg(1, 2, "third")],
        duration=2.0,
        language="en",
        backend="fake",
    )
    assert [s["text"] for s in library.segments(memo_id)] == ["second", "third"]
    assert library.search("first") == []


def test_segments_keep_word_timings(library):
    memo_id = add_done(library, "w1", "a.wav", ["one two three"])
    words = library.segments(memo_id)[0]["words"]
    assert [w["text"] for w in words] == ["one", "two", "three"]
    assert words[1]["start"] == pytest.approx(0.1)


def test_transcript_joins_segments_in_order(library):
    memo_id = add_done(library, "t1", "a.wav", ["alpha", "beta", "gamma"])
    assert library.transcript(memo_id) == "alpha beta gamma"


def test_delete_removes_row_segments_index_and_audio(library, tmp_path):
    src = write_wav(tmp_path / "src.wav", tone(0.2))
    memo = library.create_pending("d1", "src.wav", ".wav")
    library.store_audio(src, memo)
    library.complete(
        memo.id, [seg(0, 1, "findme here")], duration=1.0, language="en", backend="fake"
    )
    audio = library.audio_path(memo)
    assert audio.exists()

    library.delete(memo.id)
    assert not audio.exists()
    assert library.find_by_sha("d1") is None
    assert library.search("findme") == []
    assert (
        library.conn.execute(
            "SELECT COUNT(*) FROM segments WHERE memo_id = ?", (memo.id,)
        ).fetchone()[0]
        == 0
    )


def test_delete_unknown_memo_raises(library):
    with pytest.raises(LibraryError):
        library.delete(4242)


def test_store_audio_is_content_addressed_and_idempotent(library, tmp_path):
    src = write_wav(tmp_path / "src.wav", tone(0.2))
    memo = library.create_pending("c1", "src.wav", "src.wav")
    first = library.store_audio(src, memo)
    second = library.store_audio(src, memo)
    assert first == second
    assert first.name == "c1.wav"


@pytest.mark.parametrize(
    "bad_name",
    ["../escape.wav", "..\\escape.wav", "sub/x.wav"],
)
def test_audio_path_refuses_traversal_in_stored_name(library, bad_name):
    memo = library.create_pending("p1", "x.wav", ".wav")
    library.conn.execute(
        "UPDATE memos SET audio_file = ? WHERE id = ?", (bad_name, memo.id)
    )
    library.conn.commit()
    with pytest.raises(LibraryError):
        library.audio_path(library.get(memo.id))


def test_hostile_extension_collapses_to_bin(library):
    memo = library.create_pending("e1", "weird.wav", "../../etc/passwd")
    assert memo.audio_file == "e1.bin"


def test_title_falls_back_when_name_is_only_extension(library):
    memo = library.create_pending("n1", "", "")
    assert memo.title == "memo"
    assert memo.audio_file == "n1.bin"


def test_title_is_capped_at_200(library):
    memo = library.create_pending("n2", "x" * 500 + ".wav", ".wav")
    assert len(memo.title) == 200


# ----- search ----------------------------------------------------------------------------------


def test_search_finds_word_and_returns_snippet(library):
    add_done(library, "s1", "kitchen.m4a", ["preheat the oven to 180", "mix the flour"])
    hits = library.search("oven")
    assert len(hits) == 1
    assert hits[0].title == "kitchen"
    assert hits[0].start == pytest.approx(0.0)
    assert "[oven]" in hits[0].snippet


def test_search_prefix_match(library):
    add_done(library, "s2", "walk.m4a", ["walked the dog by the river"])
    assert len(library.search("walk*")) == 1
    # Without '*' the term must match a whole token, so "walk" does not hit "walked".
    assert library.search("walk") == []
    assert len(library.search("walked")) == 1


def test_search_is_case_and_diacritic_insensitive(library):
    add_done(library, "s3", "cafe.m4a", ["Café au lait"])
    assert len(library.search("CAFE")) == 1


def test_search_operator_words_are_literal(library):
    add_done(library, "s4", "ops.m4a", ["cats and dogs"])
    # OR/NEAR would change meaning if unquoted; quoted they are just words, so no match here
    assert library.search("cats OR birds") == []
    assert library.search("NEAR(cats dogs)") == []
    assert len(library.search("cats dogs")) == 1


@pytest.mark.parametrize(
    "hostile",
    ['"', "'", "(", ")", "*", "col:x", "^", "-", ";DROP TABLE memos;--", "\x00", "%_%"],
)
def test_search_never_raises_on_hostile_input(library, hostile):
    add_done(library, "s5", "a.m4a", ["ordinary words"])
    assert isinstance(library.search(hostile), list)


def test_search_empty_or_punctuation_only_returns_nothing(library):
    add_done(library, "s6", "a.m4a", ["anything"])
    assert library.search("") == []
    assert library.search("?!") == []


def test_search_ranks_across_memos_and_respects_limit(library):
    for i in range(5):
        add_done(library, f"lim{i}", f"m{i}.m4a", ["repeat repeat repeat"] * (i + 1))
    hits = library.search("repeat", limit=3)
    assert len(hits) == 3


def test_search_ignores_unfinished_memos(library):
    library.create_pending("pend", "p.m4a", ".m4a")
    assert library.search("anything") == []


# ----- tags ------------------------------------------------------------------------------------


def test_user_tag_add_list_and_remove(library):
    memo_id = add_done(library, "g1", "a.m4a", ["x"])
    assert library.add_tag(memo_id, "  Dinner ") == "dinner"
    assert library.tags(memo_id) == [("dinner", "user")]
    assert library.remove_tag(memo_id, "DINNER") is True
    assert library.remove_tag(memo_id, "dinner") is False


def test_adding_same_tag_twice_is_idempotent(library):
    memo_id = add_done(library, "g2", "a.m4a", ["x"])
    library.add_tag(memo_id, "work")
    library.add_tag(memo_id, "WORK")
    assert library.tags(memo_id) == [("work", "user")]


def test_add_tag_to_unknown_memo_raises(library):
    with pytest.raises(LibraryError):
        library.add_tag(999, "x")


def test_list_memos_filters_by_tag(library):
    a = add_done(library, "g3", "a.m4a", ["x"])
    b = add_done(library, "g4", "b.m4a", ["y"])
    library.add_tag(a, "Work")
    assert [m.id for m in library.list_memos(tag="work")] == [a]
    assert {m.id for m in library.list_memos()} == {a, b}


def test_auto_tags_are_replaced_but_user_tags_survive(library):
    memo_id = add_done(library, "g5", "a.m4a", ["x"])
    library.add_tag(memo_id, "keep me")
    library.set_auto_tags({memo_id: ["old auto"]})
    library.set_auto_tags({memo_id: ["new auto"]})
    kinds = dict(library.tags(memo_id))
    assert kinds == {"keep me": "user", "new auto": "auto"}


def test_user_tag_promotes_auto_tag_to_user(library):
    memo_id = add_done(library, "g6", "a.m4a", ["x"])
    library.set_auto_tags({memo_id: ["cooking"]})
    library.add_tag(memo_id, "cooking")
    assert library.tags(memo_id) == [("cooking", "user")]
    library.set_auto_tags({})
    assert library.tags(memo_id) == [("cooking", "user")]


def test_tag_counts_group_across_memos(library):
    a = add_done(library, "h1", "a.m4a", ["x"])
    b = add_done(library, "h2", "b.m4a", ["y"])
    library.add_tag(a, "food")
    library.add_tag(b, "food")
    library.add_tag(b, "travel")
    assert library.tag_counts() == [("food", 2), ("travel", 1)]


def test_texts_by_memo_skips_unfinished(library):
    done = add_done(library, "k1", "a.m4a", ["hello world"])
    library.create_pending("k2", "b.m4a", ".m4a")
    assert library.texts_by_memo() == {done: "hello world"}


# ----- cascades ---------------------------------------------------------------------------------


def test_deleting_memo_cascades_tags(library):
    memo_id = add_done(library, "c9", "a.m4a", ["x"])
    library.add_tag(memo_id, "gone")
    library.delete(memo_id)
    assert library.tag_counts() == []


def test_library_reopens_with_data_intact(tmp_path):
    root = tmp_path / "lib"
    with Library(root) as lib:
        memo_id = add_done(lib, "persist", "a.m4a", ["remember this"])
        lib.add_tag(memo_id, "saved")
    with Library(root) as lib:
        assert lib.transcript(memo_id) == "remember this"
        assert lib.tags(memo_id) == [("saved", "user")]
        assert len(lib.search("remember")) == 1


@pytest.mark.parametrize("bad_name", ["C:x.m4a", "c:evil.wav"])
def test_audio_path_refuses_drive_relative_names(library, bad_name):
    memo = library.create_pending("p9", "x.wav", ".wav")
    library.conn.execute(
        "UPDATE memos SET audio_file = ? WHERE id = ?", (bad_name, memo.id)
    )
    library.conn.commit()
    with pytest.raises(LibraryError):
        library.audio_path(library.get(memo.id))


def test_list_memos_with_unusable_tag_returns_nothing(library):
    add_done(library, "t1", "a.m4a", ["x"])
    assert library.list_memos(tag="   ") == []
    assert library.list_memos(tag="a" * 41) == []
