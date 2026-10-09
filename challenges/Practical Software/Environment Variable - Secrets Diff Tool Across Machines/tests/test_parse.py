import pytest
from envdiff.parse import (
    Issue,
    detect_format,
    parse,
    parse_dotenv,
    parse_raw,
    parse_shell,
)


def values(parsed):
    return parsed.values


def test_dotenv_basics_quotes_export_and_comments():
    text = (
        "# header\nA=1\nexport B=2\nC = spaced\nD=\"quoted # not a comment\"\nE='single $HOME'\n"
        "F=plain # trailing comment\nG=\nH\n\n   # indented comment\n"
    )
    parsed = parse_dotenv(text)
    assert values(parsed) == {
        "A": "1", "B": "2", "C": "spaced", "D": "quoted # not a comment", "E": "single $HOME",
        "F": "plain", "G": "", "H": None,
    }  # fmt: skip
    assert parsed.issues == []


def test_dotenv_double_quote_escapes_and_literal_single_quotes():
    parsed = parse_dotenv(
        'A="line1\\nline2\\t\\"q\\" \\\\ \\$x \\z"\nB=\'no\\nescape\'\n'
    )
    assert parsed.values["A"] == 'line1\nline2\t"q" \\ $x \\z'
    assert parsed.values["B"] == "no\\nescape"


def test_dotenv_multiline_quoted_values():
    parsed = parse_dotenv('KEY="-----BEGIN\nabc\n-----END"\nAFTER=1\n')
    assert parsed.values == {"KEY": "-----BEGIN\nabc\n-----END", "AFTER": "1"}
    assert parsed.lines["AFTER"] == 4


def test_dotenv_hash_needs_preceding_whitespace_to_start_a_comment():
    assert parse_dotenv("URL=http://x/#frag\nB=v #c\n").values == {
        "URL": "http://x/#frag",
        "B": "v",
    }


def test_dotenv_duplicates_keep_the_last_value_and_are_reported():
    parsed = parse_dotenv("A=1\nB=x\nA=2\n")
    assert parsed.values["A"] == "2"
    assert parsed.issues == [Issue("duplicate-key", 3, "A")]


def test_dotenv_issues_never_carry_line_content():
    secret = "hunter2-SECRET-VALUE"
    parsed = parse_dotenv(
        f'this is not valid {secret}\nK="never closed {secret}\nA=1\n'
    )
    assert Issue("invalid-line", 1) in parsed.issues
    assert any(i.kind == "unterminated-quote" and i.key == "K" for i in parsed.issues)
    assert secret not in repr(parsed.issues)


def test_dotenv_unterminated_quote_drops_that_variable_only():
    parsed = parse_dotenv('A=1\nB="open\nC=3\n')
    assert "B" not in parsed.values and parsed.values["A"] == "1"


def test_dotenv_text_after_closing_quote_is_flagged_but_value_kept():
    parsed = parse_dotenv("A='it''s'\nB=\"x\" # fine\n")
    assert parsed.values == {"A": "it", "B": "x"}
    assert [i.kind for i in parsed.issues] == ["trailing-garbage"]


def test_dotenv_unquoted_whitespace_is_flagged():
    parsed = parse_dotenv("A=hello world\n")
    assert parsed.values["A"] == "hello world"
    assert parsed.issues == [Issue("unquoted-space", 1, "A")]


def test_bom_and_crlf_are_normalised_and_reported():
    parsed = parse_dotenv("\ufeffA=1\r\nB=2\r\n")
    assert parsed.values == {"A": "1", "B": "2"}
    assert {i.kind for i in parsed.issues} == {"bom", "crlf"}


def test_raw_keeps_quotes_and_whitespace_verbatim():
    parsed = parse_raw('A="quoted"\nB= padded \nC\n#comment\nD=a=b\nnot a key=1\n')
    assert parsed.values == {"A": '"quoted"', "B": " padded ", "C": None, "D": "a=b"}
    assert parsed.issues == [Issue("invalid-line", 6)]


def test_shell_declare_export_and_ansi_c_quoting():
    text = (
        'declare -x A="q\\"x\\$y"\n'
        "declare -x B=$'a\\nb'\n"
        "export C='it'\\''s'\n"
        "declare -x D\n"
        'declare -rx E="1"\n'
        'F="multi\nline"\n'
    )
    parsed = parse_shell(text)
    assert parsed.values == {
        "A": 'q"x$y',
        "B": "a\nb",
        "C": "it's",
        "D": None,
        "E": "1",
        "F": "multi\nline",
    }
    assert parsed.issues == []


def test_shell_several_assignments_on_one_line_and_comments():
    parsed = parse_shell('# c\nexport A=1 B="two words" C=3 # trailing\nD=4;\n')
    assert parsed.values == {"A": "1", "B": "two words", "C": "3", "D": "4"}


def test_shell_leaves_variable_references_unexpanded():
    assert parse_shell('export A="$HOME/x"\n').values["A"] == "$HOME/x"


def test_shell_unterminated_quote_is_reported_without_content():
    parsed = parse_shell("export A='never closed secret-xyz\nexport B=2\n")
    assert Issue("unterminated-quote", 1, "A") in parsed.issues
    assert "secret-xyz" not in repr(parsed.issues)


def test_shell_garbage_after_an_assignment_is_reported():
    parsed = parse_shell("export A=1 !!!\n")
    assert parsed.values["A"] == "1"
    assert [i.kind for i in parsed.issues] == ["trailing-garbage"]


def test_format_detection():
    assert detect_format('declare -x A="1"\n') == "shell"
    assert detect_format("A=1\n", "deploy.sh") == "shell"
    assert detect_format("A=1\n", ".env") == "dotenv"
    assert (
        detect_format("export A=1\n", ".env") == "dotenv"
    )  # ambiguous: dotenv accepts `export`


def test_parse_reports_the_dialect_it_used_and_rejects_unknown_ones():
    assert parse("A=1\n")[1] == "dotenv"
    assert parse("A=1\n", "raw")[1] == "raw"
    with pytest.raises(ValueError, match="unknown format"):
        parse("A=1", "yaml")


def test_same_line_means_different_things_per_dialect():
    line = 'A="x"\n'
    assert parse_dotenv(line).values["A"] == "x"
    assert parse_raw(line).values["A"] == '"x"'


def test_bare_lines_only_count_as_variables_when_they_look_like_variable_names():
    """Continuation lines of a multi-line value must never be promoted to variable names."""
    pem_body = "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC"
    for parse_fn in (parse_dotenv, parse_raw):
        parsed = parse_fn(
            f"KEY=-----BEGIN\n{pem_body}\nlowercase_secret\nPASS_THROUGH\n"
        )
        assert set(parsed.values) == {"KEY", "PASS_THROUGH"}
        assert [i.kind for i in parsed.issues] == ["invalid-line", "invalid-line"]
        assert pem_body not in repr(parsed)


def test_shell_bare_word_without_export_is_a_command_not_a_declaration():
    parsed = parse_shell("ls\nexport DECLARED\nsecretword\n")
    assert parsed.values == {"DECLARED": None}
    assert [i.kind for i in parsed.issues] == ["invalid-line", "invalid-line"]
