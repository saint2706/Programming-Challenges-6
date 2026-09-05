"""Tests for the configurable bracket validator.

Run with:  uv run --with pytest pytest -q
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from brackets import (
    SPECS,
    BracketSpec,
    Pair,
    Validator,
    auto_close,
    longest_balanced_span,
    main,
    matching_index,
    validate,
    validate_stream,
)

HERE = Path(__file__).parent


# ---------------------------------------------------------------------------
# The textbook cases still have to pass
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,ok",
    [
        ("", True),
        ("()", True),
        ("([{}])", True),
        ("()[]{}", True),
        ("no brackets at all", True),
        ("(", False),
        (")", False),
        ("([)]", False),
        ("{[}", False),
        ("(()", False),
        ("())", False),
    ],
)
def test_plain(text, ok):
    assert validate(text, "plain").ok is ok


def test_deeply_nested_does_not_recurse():
    depth = 200_000
    text = "(" * depth + ")" * depth
    assert validate(text, "plain").ok
    assert validate(text, "plain").max_depth == depth


def test_max_depth_guard():
    report = validate("(" * 50 + ")" * 50, "plain", max_depth=10)
    assert not report.ok
    assert any(d.kind == "depth-exceeded" for d in report.diagnostics)


def test_depth_limit_is_fatal_and_does_not_cascade():
    """Dropping the frame and continuing made every later closer a fake fault."""
    report = validate("(" * 20 + ")" * 20, "plain", max_depth=3)
    assert [d.kind for d in report.diagnostics] == ["depth-exceeded"]
    assert report.fault_count == 1


def test_depth_limit_is_fatal_when_streaming_too():
    text = "(" * 20 + ")" * 20
    report = validate_stream((text[i : i + 3] for i in range(0, len(text), 3)),
                             "plain", max_depth=3)
    assert [d.kind for d in report.diagnostics] == ["depth-exceeded"]


def test_depth_limit_below_the_real_depth_is_not_triggered():
    assert validate("((()))", "plain", max_depth=3).ok


# ---------------------------------------------------------------------------
# Opaque regions, escapes, self-pairing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec,text,ok",
    [
        ("c", 'char *s = "))))";', True),
        ("c", "/* ( */", True),
        ("c", "/* /* */", True),  # C block comments do not nest
        ("c", "/* unterminated (", False),
        ("c", '"\\"" ()', True),
        ("c", '"\\\\" ()', True),  # escaped backslash, then a real closing quote
        ("c", "'\\''", True),
        ("c", "// ( \n)", False),  # comment hides the opener, not the closer
        ("python", '"""(""" )', False),
        ("python", "'''(''' ()", True),
        ("python", "x = '(' + \")\"", True),
        ("latex", "$x$ \\(y\\) \\[z\\]", True),
        ("latex", "$ \\(y\\) $", False),  # a second math mode inside $..$
        ("latex", "$$ {a} $$", True),
        ("latex", "$$ { $$ }", False),  # math mode does not nest
        ("latex", "% } \n {}", True),
        ("markdown", "```\n(((\n```", True),
        ("markdown", "`)` ()", True),
        ("html", "<!-- ) --> ()", True),
        ("html", "<!-- <!-- -->", True),
    ],
)
def test_opaque_and_escapes(spec, text, ok):
    assert validate(text, spec).ok is ok, validate(text, spec).render(text)


def test_self_pairing_uses_the_stack():
    report = validate("$a$ $b$", "latex")
    assert report.ok
    assert len(report.spans) == 2


def test_non_nestable_reports_the_opener():
    report = validate("$ $$ $", "latex")
    assert not report.ok
    kinds = {d.kind for d in report.diagnostics}
    assert kinds & {"forbidden-nesting", "unclosed", "mismatched"}


# ---------------------------------------------------------------------------
# Custom specs
# ---------------------------------------------------------------------------


def test_custom_pairs_from_dict():
    spec = BracketSpec.from_dict(
        {
            "name": "pascal",
            "pairs": [
                {"open": "begin", "close": "end", "name": "block"},
                {"open": "(", "close": ")", "name": "paren"},
                {"open": "{", "close": "}", "name": "comment", "opaque": True},
            ],
        }
    )
    assert validate("begin begin (x) end end", spec).ok
    assert not validate("begin (x end)", spec).ok
    assert validate("{ ) unbalanced (( } begin end", spec).ok


def test_spec_round_trips_through_json():
    spec = BracketSpec.from_json(BracketSpec.from_dict(
        {"pairs": [{"open": "<<", "close": ">>", "name": "heredoc"}]}
    ).to_json())
    assert validate("<< << >> >>", spec).ok
    assert not validate("<< >> >>", spec).ok


def test_may_contain_restricts_direct_children():
    spec = BracketSpec(
        [
            Pair("{", "}", "object", may_contain=frozenset({"array", "object"})),
            Pair("[", "]", "array"),
            Pair("(", ")", "paren"),
        ],
        "restricted",
    )
    assert validate("{[()]}", spec).ok  # paren is nested inside array, not object
    assert not validate("{()}", spec).ok  # paren directly inside object


def test_longest_lexeme_wins():
    spec = BracketSpec([Pair("<", ">", "angle"), Pair("<!--", "-->", "comment")])
    # If ``<`` shadowed ``<!--``, the leading ``<`` would be an unclosed angle
    # bracket and the trailing ``>`` would be a stray closer.
    assert validate("<!--x-->", spec).ok
    assert validate("<a> <!--x--> <b>", spec).ok
    assert not validate("<!--x->", spec).ok


def test_opaque_closers_are_context_only():
    # A line comment's terminator is a newline. If that were a globally
    # registered closer, every newline in every file would be a stray token.
    assert validate("a\nb\nc\n", "python").ok
    assert validate("x = 1  # note\ny = (2)\n", "python").ok
    # ``*/`` outside a comment is a lexer problem, not a nesting problem.
    assert validate("a */ b", "c").ok


def test_a_lexeme_shared_by_two_pairs_resolves_against_the_stack():
    """``|`` closes pair b and opens pair a; only the stack can say which."""
    spec = BracketSpec([Pair("|", "#", "a"), Pair("@", "|", "b")])
    assert validate("@|", spec).ok  # closes b
    assert validate("|#", spec).ok  # opens a
    assert not validate("|", spec).ok


def test_an_opener_that_is_a_prefix_of_its_own_closer():
    spec = BracketSpec([Pair("<", "<<", "x")])
    assert validate("< <<", spec).ok
    assert not validate("<<<", spec).ok  # longest-first makes this close, open


def test_crlf_line_endings():
    report = validate("ok()\r\nbad(\r\n", "plain")
    (d,) = report.diagnostics
    assert (d.line, d.column) == (2, 4)


def test_finish_is_idempotent_and_feeding_after_it_raises():
    v = Validator(SPECS["plain"])
    v.feed("(")
    first, second = v.finish(), v.finish()
    assert (first.ok, len(first.diagnostics)) == (second.ok, len(second.diagnostics))
    with pytest.raises(RuntimeError, match="cannot feed after finish"):
        v.feed("x")


def test_one_enormous_line_does_not_break_position_tracking():
    text = "(" * 5 + "x" * 500_000 + ")" * 4
    report = validate(text, "plain")
    assert not report.ok
    (d,) = report.diagnostics
    # The four closers match the four innermost parens, so the outermost is the
    # one left open -- at column 1, half a megabyte earlier.
    assert (d.line, d.column, d.kind) == (1, 1, "unclosed")


def test_unknown_preset_is_a_clear_error():
    with pytest.raises(ValueError, match="unknown spec"):
        validate("()", "klingon")


def test_duplicate_pair_names_rejected():
    with pytest.raises(ValueError, match="duplicate pair name"):
        BracketSpec([Pair("(", ")", "x"), Pair("[", "]", "x")])


def test_may_contain_must_reference_real_pairs():
    with pytest.raises(ValueError, match="may_contain"):
        BracketSpec([Pair("(", ")", "p", may_contain=frozenset({"ghost"}))])


def test_escape_equal_to_the_closer_is_rejected():
    """Otherwise every closer reads as an escape and the region never ends."""
    with pytest.raises(ValueError, match="unterminatable"):
        Pair('"', '"', "s", opaque=True, escape='"')


def test_escape_outside_an_opaque_region_is_rejected():
    with pytest.raises(ValueError, match="opaque"):
        Pair("(", ")", "p", escape="\\")


def test_empty_lexemes_are_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        Pair("", ")", "p")
    with pytest.raises(ValueError, match="non-empty"):
        Pair("(", "", "p")
    with pytest.raises(ValueError, match="non-empty"):
        Pair("(", ")", "p", opaque=True, escape="")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_reports_every_fault_not_just_the_first():
    report = validate("(] [) {>", "angle")
    assert len(report.diagnostics) >= 2


def test_positions_are_one_based_line_and_column():
    report = validate("ok()\nbad(\n", "plain")
    (d,) = report.diagnostics
    assert (d.line, d.column, d.offset) == (2, 4, 8)
    assert d.kind == "unclosed"


def test_mismatch_points_back_at_the_opener():
    # ``]`` wants a square bracket; none is open, so the diagnostic names the
    # frame that actually is open and where it started.
    report = validate("(\n  ]", "plain")
    d = next(d for d in report.diagnostics if d.kind == "mismatched")
    assert (d.related_line, d.related_column) == (1, 1)


def test_diagnostics_come_back_in_source_order():
    """Unclosed frames are found at EOF but belong where their opener is."""
    report = validate("( { ] ", "plain")
    offsets = [d.offset for d in report.diagnostics]
    assert offsets == sorted(offsets)
    assert [d.kind for d in report.diagnostics] == ["unclosed", "unclosed", "mismatched"]


def test_recovery_pops_to_the_frame_the_closer_matches():
    # ``}`` matches the outer brace, so the inner ``[`` is the thing that was
    # left open -- which is what clang and rustc report here too.
    report = validate("{\n  [\n}", "plain")
    (d,) = report.diagnostics
    assert d.kind == "unclosed"
    assert (d.line, d.column) == (2, 3)  # the '[' itself
    assert (d.related_line, d.related_column) == (3, 1)  # the '}' that closed first


def test_stray_closer_does_not_cascade():
    # The lone ``)`` is the only problem; the surrounding braces still match.
    report = validate("{ ) }", "plain")
    assert len(report.diagnostics) == 1
    assert report.spans[-1].name == "curly"


def test_unexpected_close_on_an_empty_stack():
    report = validate(") ()", "plain")
    assert report.diagnostics[0].kind == "unexpected-close"
    assert len(report.diagnostics) == 1


def test_render_draws_a_caret():
    text = "a(b"
    out = validate(text, "plain").render(text)
    assert "a(b" in out and "^" in out


def test_max_diagnostics_caps_output():
    report = validate(")" * 1000, "plain", max_diagnostics=5)
    assert len(report.diagnostics) == 5


def test_capping_diagnostics_never_makes_broken_input_look_valid():
    """The bug this guards: ``ok`` derived from the *listed* diagnostics."""
    for cap in (0, 1, 5, None):
        report = validate("((((", "plain", max_diagnostics=cap)
        assert report.ok is False, cap
        assert report.fault_count == 4
    assert validate("()", "plain", max_diagnostics=0).ok is True


def test_truncated_flag_and_render_say_how_many_were_hidden():
    report = validate(")" * 100, "plain", max_diagnostics=3)
    assert report.truncated and report.fault_count == 100
    assert "100 faults found, showing the first 3" in report.render(")" * 100)
    assert validate(")))", "plain").truncated is False


def test_report_to_dict_is_json_serializable():
    json.dumps(validate("([)]", "plain").to_dict())


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


CHUNK_SAMPLES = [
    ("c", '{ /* ) */ "a\\"b" ([{}]) } // trailing ('),
    ("c", '"\\\\" /* /* */ ( ]'),
    ("python", 'x = """a"b""" + \'\'\'(\'\'\' # )\n([)]'),
    ("latex", "$ (a) $$ b $$ % ) \n {c} $"),
    ("markdown", "``` ) ``` `x` [a](b) ("),
    ("html", "<!-- <!-- --> ( <script ) </script> )"),
]


@pytest.mark.parametrize("spec,text", CHUNK_SAMPLES)
@pytest.mark.parametrize("size", [1, 2, 3, 4, 5, 8, 13, 1024])
def test_streaming_matches_whole_string(spec, text, size):
    whole = validate(text, spec)
    chunks = [text[i : i + size] for i in range(0, len(text), size)]
    streamed = validate_stream(chunks, spec)
    assert streamed.ok == whole.ok
    assert [(d.kind, d.offset) for d in streamed.diagnostics] == [
        (d.kind, d.offset) for d in whole.diagnostics
    ]


def test_streaming_memory_is_bounded_by_depth():
    v = Validator(BracketSpec.from_dict({"pairs": [{"open": "(", "close": ")"}]}),
                  collect_spans=False)
    for _ in range(1000):
        v.feed("()" * 500)
    assert v.finish().ok
    # The buffer never accumulates: it is trimmed on every feed.
    assert len(v._buf) <= v.spec.hold


# ---------------------------------------------------------------------------
# Editor helpers
# ---------------------------------------------------------------------------


def test_matching_index_both_directions():
    text = "a(b[c]d)e"
    assert matching_index(text, 1) == 7
    assert matching_index(text, 7) == 1
    assert matching_index(text, 3) == 5
    assert matching_index(text, 0) is None


def test_matching_index_inside_multichar_delimiter():
    text = "x <!-- y --> z"
    assert matching_index(text, 3, "html") == 9  # inside "<!--"
    assert matching_index(text, 10, "html") == 2  # inside "-->"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("", (0, 0)),
        (")(", (0, 0)),
        ("()", (0, 2)),
        ("(()", (1, 3)),
        ("())((()))", (3, 9)),
        ("()()()", (0, 6)),
        (")()())", (1, 5)),
    ],
)
def test_longest_balanced_span(text, expected):
    assert longest_balanced_span(text, "plain") == expected


def test_longest_balanced_span_mixes_pair_types():
    assert longest_balanced_span("]{}[]", "plain") == (1, 5)


def test_auto_close_completes_a_partial_buffer():
    partial = 'def f():\n    return g("a", [1, {2:'
    assert auto_close(partial, "python") == "}])"
    assert validate(partial + auto_close(partial, "python"), "python").ok


def test_auto_close_ignores_optional_closers():
    assert auto_close("x = (1  # comment", "python") == ")"


# ---------------------------------------------------------------------------
# Differential test against a deliberately naive oracle
# ---------------------------------------------------------------------------


def naive_plain(text: str) -> bool:
    """The three-line version everybody writes first. Valid oracle for `plain`."""
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    for ch in text:
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack


def test_differential_against_naive_oracle():
    rng = random.Random(20260904)
    alphabet = "([{}])xy "
    for _ in range(20_000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 24)))
        assert validate(text, "plain").ok == naive_plain(text), text


def random_balanced(rng: random.Random, depth: int = 6) -> str:
    if depth == 0 or rng.random() < 0.3:
        return rng.choice(["", "x", "hello world"])
    o, c = rng.choice([("(", ")"), ("[", "]"), ("{", "}")])
    parts = [random_balanced(rng, depth - 1) for _ in range(rng.randint(1, 3))]
    return o + "".join(parts) + c


def test_generated_balanced_strings_validate():
    rng = random.Random(7)
    for _ in range(2000):
        text = "".join(random_balanced(rng) for _ in range(rng.randint(1, 4)))
        assert validate(text, "plain").ok, text


def test_single_deletion_from_a_balanced_string_breaks_it():
    rng = random.Random(11)
    for _ in range(2000):
        text = random_balanced(rng, depth=4)
        positions = [i for i, ch in enumerate(text) if ch in "([{}])"]
        if not positions:
            continue
        i = rng.choice(positions)
        assert not validate(text[:i] + text[i + 1 :], "plain").ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_self_check():
    assert main(["--self-check"]) == 0


def test_cli_exit_codes(tmp_path, capsys):
    good = tmp_path / "good.c"
    good.write_text('int main(void) { char *s = ")"; return 0; } // (\n')
    bad = tmp_path / "bad.c"
    bad.write_text("int main(void) { return 0;\n")

    assert main(["--spec", "c", str(good)]) == 0
    assert main(["--spec", "c", str(bad)]) == 1
    out = capsys.readouterr().out
    assert "FAILED" in out and "unclosed" in out


def test_cli_json_output(tmp_path, capsys):
    path = tmp_path / "x.txt"
    path.write_text("([)]")
    main(["--json", str(path)])
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["ok"] is False
    assert payload[0]["diagnostics"]


def test_cli_reports_unreadable_files_without_a_traceback(tmp_path, capsys):
    missing = tmp_path / "nope.txt"
    assert main([str(missing)]) == 1
    assert "cannot read" in capsys.readouterr().err


def test_cli_keeps_going_after_an_unreadable_file(tmp_path, capsys):
    good = tmp_path / "good.txt"
    good.write_text("()")
    assert main([str(tmp_path / "nope.txt"), str(good)]) == 1
    captured = capsys.readouterr()
    assert "cannot read" in captured.err
    assert "ok" in captured.out  # the readable file was still checked


def test_cli_rejects_auto_close_with_stream():
    with pytest.raises(SystemExit):
        main(["--auto-close", "--stream", "-"])


def test_cli_stream_matches_non_stream(tmp_path, capsys):
    path = tmp_path / "x.py"
    path.write_text('s = "(" \n# )\nt = [1, 2\n')
    assert main(["--spec", "python", str(path)]) == 1
    assert main(["--spec", "python", "--stream", str(path)]) == 1


def test_module_runs_as_a_script():
    proc = subprocess.run(
        [sys.executable, str(HERE / "brackets.py"), "--self-check"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "all self-checks passed" in proc.stdout
