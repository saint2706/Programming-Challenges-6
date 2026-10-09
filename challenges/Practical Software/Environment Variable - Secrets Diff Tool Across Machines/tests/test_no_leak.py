"""The tool's whole promise: no value reaches any output, whatever the input looks like."""

import json

import pytest
from envdiff.cli import app
from typer.testing import CliRunner

runner = CliRunner()

# Unique, recognisable strings so a substring check cannot pass by coincidence.
PLANTED = {
    "plain": "ZQ-plain-value-7F3A",
    "quoted": "ZQ-quoted-value-91BC",
    "secret": "ZQ-sk_live_0123456789abcdef",
    "url": "ZQ-pw-in-url-55D0",
    "multiline": "ZQ-multiline-line-two-AA11",
    "malformed": "ZQ-malformed-line-content-C0DE",
    "unterminated": "ZQ-unterminated-quote-body-BEEF",
    "garbage": "ZQ-after-the-closing-quote-D00D",
    "whitespace": "ZQ-padded-value-1234",
    "dupe_old": "ZQ-overwritten-first-value-0001",
    "placeholder_neighbour": "ZQ-line-after-bom-0042",
}

ENV_A = (
    "﻿"
    f"PLAIN={PLANTED['plain']}\r\n"
    f'QUOTED="{PLANTED["quoted"]}"\r\n'
    f"API_SECRET={PLANTED['secret']}\r\n"
    f"DATABASE_URL=postgres://user:{PLANTED['url']}@db/app\r\n"
    f'CERT="-----BEGIN\r\n{PLANTED["multiline"]}\r\n-----END"\r\n'
    f"this line is malformed {PLANTED['malformed']}\r\n"
    f'UNCLOSED="{PLANTED["unterminated"]}\r\n'
    f'AFTER_QUOTE="ok" {PLANTED["garbage"]}\r\n'
    f'PADDED=" {PLANTED["whitespace"]} "\r\n'
    f"DUP={PLANTED['dupe_old']}\r\nDUP=second\r\n"
    f"NEXT={PLANTED['placeholder_neighbour']}\r\n"
)
ENV_B = ENV_A.replace(PLANTED["plain"], "other").replace(PLANTED["secret"], "different")
SHELL = f'declare -x SHELL_SECRET="{PLANTED["secret"]}"\ndeclare -x BROKEN=\'{PLANTED["unterminated"]}\n'


def all_planted_values():
    return list(PLANTED.values())


@pytest.fixture
def files(tmp_path):
    (tmp_path / "a.env").write_text(ENV_A, encoding="utf-8", newline="")
    (tmp_path / "b.env").write_text(ENV_B, encoding="utf-8", newline="")
    (tmp_path / "c.sh").write_text(SHELL)
    return tmp_path


COMMANDS = [
    ["diff"],
    ["diff", "--all"],
    ["diff", "--all", "--fingerprints"],
    ["diff", "--as", "json", "--fingerprints"],
    ["diff", "--as", "markdown", "--all"],
    ["diff", "--strict", "--format", "raw"],
    ["diff", "--no-lint"],
]


@pytest.mark.parametrize("flags", COMMANDS, ids=lambda f: " ".join(f))
def test_no_planted_value_appears_in_any_output_mode(files, flags):
    result = runner.invoke(
        app,
        [
            flags[0],
            str(files / "a.env"),
            str(files / "b.env"),
            str(files / "c.sh"),
            *flags[1:],
        ],
    )
    assert result.exit_code in (0, 1), result.output
    for name, value in PLANTED.items():
        assert value not in result.output, f"{name} leaked in: {' '.join(flags)}"


def test_snapshot_output_holds_no_planted_value(files):
    result = runner.invoke(
        app, ["snapshot", str(files / "a.env")], env={"ENVDIFF_PASSPHRASE": "pw"}
    )
    assert result.exit_code == 0
    json.loads(result.output)
    for value in all_planted_values():
        assert value not in result.output


def test_error_paths_do_not_echo_content(files, tmp_path):
    bad_snapshot = tmp_path / "bad.json"
    bad_snapshot.write_text(
        f'{{"envdiff_snapshot": 1, "entries": {{"K": "{PLANTED["secret"]}"}}}}'
    )
    result = runner.invoke(app, ["diff", str(bad_snapshot), str(files / "a.env")])
    assert result.exit_code == 2
    assert PLANTED["secret"] not in result.output


def test_reveal_shows_exactly_what_was_asked_and_nothing_else(files):
    result = runner.invoke(
        app, ["diff", str(files / "a.env"), str(files / "b.env"), "--reveal", "PLAIN"]
    )
    assert PLANTED["plain"] in result.output and "other" in result.output
    for name, value in PLANTED.items():
        if name != "plain":
            assert value not in result.output, name


def test_lengths_and_prefixes_of_secrets_are_not_inferable_from_the_output(tmp_path):
    (tmp_path / "a.env").write_text("API_SECRET=short\n")
    (tmp_path / "b.env").write_text("API_SECRET=" + "x" * 300 + "\n")
    out = runner.invoke(
        app, ["diff", str(tmp_path / "a.env"), str(tmp_path / "b.env"), "--as", "json"]
    ).output
    data = json.loads(out)
    row = data["keys"][0]
    assert row["shapes"] == {"a.env": "redacted", "b.env": "redacted"}
    assert "short" not in out and "xxx" not in out
