"""End-to-end tests on synthetic repositories. These need git and are skipped without it."""

from __future__ import annotations

import json
import re

from conftest import requires_git
from gitaudit.audit import run_audit
from gitaudit.cli import main
from gitaudit.config import build_settings
from gitaudit.reporters import render_html, render_json, render_sarif
from helpers import fake_aws_key_id, secret_fixtures

pytestmark = requires_git

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
BOX = re.compile(r"[─-╿▀-▟]")


def _settings(repo, tmp_path, **overrides):
    return build_settings(repo.path, None, tmp_path, overrides)


def _rules(result):
    return {f.rule_id for f in result.findings}


def test_clean_repo_has_no_failing_findings(repo_factory, tmp_path):
    repo = repo_factory("clean")
    repo.write("README.md", "hello\n")
    repo.write("src/app.py", "print('hi')\n")
    repo.commit("init", when=1_700_000_000)
    result = run_audit(_settings(repo, tmp_path))
    assert result.failing(_settings(repo, tmp_path).fail_threshold()) == []
    assert result.stats["commits"] == 1


def test_secret_in_history_is_found_even_after_removal(repo_factory, tmp_path):
    repo = repo_factory("history")
    repo.write("config.py", f"KEY = '{secret_fixtures()['github_classic']}'\n")
    repo.commit("add key", when=1_700_000_000)
    repo.write("config.py", "KEY = None\n")
    repo.commit("remove key", when=1_700_000_100)
    result = run_audit(_settings(repo, tmp_path))
    hits = [f for f in result.findings if f.rule_id.startswith("SEC")]
    assert hits, (
        "a secret deleted in a later commit must still be reported from history"
    )
    assert all(f.path == "config.py" and f.commit for f in hits)


def test_history_disabled_limits_scan_to_head(repo_factory, tmp_path):
    repo = repo_factory("nohist")
    repo.write("config.py", f"KEY = '{secret_fixtures()['github_classic']}'\n")
    repo.commit("add key", when=1_700_000_000)
    repo.write("config.py", "KEY = None\n")
    repo.commit("remove key", when=1_700_000_100)
    result = run_audit(_settings(repo, tmp_path, history=False))
    assert not [f for f in result.findings if f.rule_id.startswith("SEC003")]


def test_secret_in_working_tree_is_reported_with_line(repo_factory, tmp_path):
    repo = repo_factory("line")
    repo.write("a.txt", "one\ntwo\n" + f"AWS = '{fake_aws_key_id()}'\n")
    repo.commit("c", when=1_700_000_000)
    result = run_audit(_settings(repo, tmp_path))
    aws = [f for f in result.findings if f.rule_id == "SEC001" and f.path == "a.txt"]
    assert aws and aws[0].line == 3


def test_stale_and_merged_branches_are_classified(repo_factory, tmp_path):
    repo = repo_factory("branches")
    repo.write("a.txt", "a\n")
    base = repo.commit("base", when=1_600_000_000)
    repo.branch("old-feature", base)
    repo.checkout("old-feature")
    repo.write("b.txt", "b\n")
    repo.commit("old work", when=1_600_000_100)
    repo.checkout("main")
    result = run_audit(_settings(repo, tmp_path, stale_days=30))
    assert any("old-feature" in f.title + f.evidence for f in result.findings)
    assert _rules(result) & {"BR001", "BR002", "BR003", "BR004"}


def test_oversized_blob_reported_in_head(repo_factory, tmp_path):
    repo = repo_factory("big")
    repo.write("data/blob.dat", "x" * 4096)
    repo.commit("big", when=1_700_000_000)
    result = run_audit(_settings(repo, tmp_path, max_blob_size=1024, lfs_threshold=512))
    assert "BL001" in _rules(result)


def test_binary_file_is_not_content_scanned(repo_factory, tmp_path):
    repo = repo_factory("binary")
    repo.write("blob.bin", b"\x00\x01" + f"AWS={fake_aws_key_id()}".encode())
    repo.commit("bin", when=1_700_000_000)
    result = run_audit(_settings(repo, tmp_path))
    assert result.stats["binary_blobs"] >= 1
    assert "SEC001" not in _rules(result)


def test_baseline_suppresses_known_findings(repo_factory, tmp_path):
    repo = repo_factory("baseline")
    repo.write("leak.py", f"K = '{fake_aws_key_id()}'\n")
    repo.commit("leak", when=1_700_000_000)
    base_file = tmp_path / "baseline.json"
    settings = _settings(repo, tmp_path)
    first = run_audit(settings)
    assert first.failing(settings.fail_threshold())
    assert main([str(repo.path), "--write-baseline", str(base_file)]) == 0
    assert main([str(repo.path), "--baseline", str(base_file)]) == 0


def test_reporters_render_real_result(repo_factory, tmp_path):
    repo = repo_factory("reports")
    repo.write("leak.py", f"K = '{fake_aws_key_id()}'\n")
    repo.commit("leak", when=1_700_000_000)
    result = run_audit(_settings(repo, tmp_path))
    doc = json.loads(render_json(result))
    assert doc["schema"] == 1 and doc["findings"]
    sarif = json.loads(render_sarif(result))
    assert sarif["runs"][0]["results"]
    page = render_html(result)
    assert "<html" in page.lower() and fake_aws_key_id() not in page


def test_cli_exit_zero_on_clean_repo(repo_factory, tmp_path):
    repo = repo_factory("cli-clean")
    repo.write("a.txt", "a\n")
    repo.commit("c", when=1_700_000_000)
    assert (
        main(
            [
                str(repo.path),
                "--quiet",
                "--format",
                "json",
                "-o",
                str(tmp_path / "o.json"),
            ]
        )
        == 0
    )


def test_cli_exit_one_on_high_finding(repo_factory, tmp_path):
    repo = repo_factory("cli-leak")
    repo.write("leak.py", f"K = '{fake_aws_key_id()}'\n")
    repo.commit("c", when=1_700_000_000)
    assert (
        main(
            [
                str(repo.path),
                "--quiet",
                "--format",
                "json",
                "-o",
                str(tmp_path / "o.json"),
            ]
        )
        == 1
    )


def test_cli_fail_on_none_never_fails(repo_factory, tmp_path):
    repo = repo_factory("cli-none")
    repo.write("leak.py", f"K = '{fake_aws_key_id()}'\n")
    repo.commit("c", when=1_700_000_000)
    assert (
        main(
            [
                str(repo.path),
                "--quiet",
                "--fail-on",
                "none",
                "--format",
                "json",
                "-o",
                str(tmp_path / "o.json"),
            ]
        )
        == 0
    )


def test_cli_exit_two_on_bad_argument(repo_factory, tmp_path, capsys):
    repo = repo_factory("cli-bad")
    repo.write("a.txt", "a\n")
    repo.commit("c", when=1_700_000_000)
    assert main([str(repo.path), "--stale-days", "-4"]) == 2
    assert "stale_days" in capsys.readouterr().err


def test_cli_exit_two_on_missing_repo(tmp_path, capsys):
    assert main([str(tmp_path / "does-not-exist")]) == 2


def test_cli_github_actions_output_is_plain(
    repo_factory, tmp_path, monkeypatch, capsys
):
    repo = repo_factory("cli-gha")
    repo.write("leak.py", f"K = '{fake_aws_key_id()}'\n")
    repo.commit("c", when=1_700_000_000)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    main([str(repo.path), "--format", "table", "--quiet"])
    text = ANSI.sub("", capsys.readouterr().out)
    text = BOX.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    assert "gitaudit:" in text and "findings" in text and "\x1b" not in text
