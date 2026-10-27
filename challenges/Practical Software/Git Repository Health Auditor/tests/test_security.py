"""Regression tests for hostile repository content and configuration."""

from __future__ import annotations

import pytest
from conftest import requires_git
from gitaudit.audit import run_audit
from gitaudit.blobs import remediation_for
from gitaudit.cli import main
from gitaudit.config import audited_worktree, build_settings
from gitaudit.safety import cli_arg, sanitize
from helpers import secret_fixtures

# Ref names that git accepts on Windows (no spaces; NTFS rejects | and " in file names) but that
# a shell would misread if pasted.
HOSTILE_NAMES = [
    "feat$(id)",
    "feat`id`",
    "feat;id",
    "feat'q",
]


def test_cli_arg_accepts_plain_names():
    assert cli_arg("feature/login-2", "<x>") == "feature/login-2"


@pytest.mark.parametrize("name", [*HOSTILE_NAMES, "-x", "a..b", "with space"])
def test_cli_arg_rejects_metacharacters(name):
    assert cli_arg(name, "<x>") == "<x>"


def test_blob_remediation_uses_placeholder_for_hostile_path():
    text = remediation_for("evil dir/$(id).bin", in_head=True)
    assert "$(id)" not in text
    assert "--path <path>" in text and "<filename>" in text


def test_blob_remediation_keeps_plain_path():
    text = remediation_for("assets/logo.png", in_head=True)
    assert "--path assets/logo.png" in text


def test_sanitize_escapes_escape_sequences():
    out = sanitize("\x1b[31mred\x1b[0m")
    assert "\x1b" not in out
    assert "\\x1b" in out


@requires_git
@pytest.mark.parametrize("name", HOSTILE_NAMES)
def test_branch_remediation_never_embeds_hostile_name(repo_factory, name, tmp_path):

    repo = repo_factory()
    repo.write("a.txt", "a\n")
    base = repo.commit("base", when=1_600_000_000)
    repo.branch("feature/ok", base)
    repo.checkout("feature/ok")
    repo.write("b.txt", "b\n")
    repo.commit("feature work", when=1_600_000_100)
    repo.checkout("main")
    # Hostile ref name, created with update-ref so the shell is never involved. It points at the
    # base commit, which main contains, so it is classified as merged (BR001) and gets a command.
    repo.set_ref(f"refs/heads/{name}", base)
    settings = build_settings(repo.path, None, tmp_path, {})
    result = run_audit(settings)
    texts = [
        f.remediation
        for f in result.findings
        if f.rule_id in ("BR001", "BR002", "BR003")
    ]
    assert texts, "expected a branch finding for the hostile ref"
    for text in texts:
        assert "$(" not in text and "`" not in text and ";" not in text
        assert name not in text or "<branch>" in text


@requires_git
def test_repo_committed_permissive_config_is_ignored(
    repo_factory, tmp_path, monkeypatch
):
    """A repository that ships its own gitaudit.toml must not weaken its own audit."""
    s = secret_fixtures()
    repo = repo_factory("shipped")
    repo.write(
        "gitaudit.toml",
        '[gitaudit]\nfail_on = "none"\n\n[gitaudit.secrets]\nallow_regexes = [".*"]\n'
        'allow_paths = ["*"]\n',
    )
    repo.write("leak.py", f"KEY = '{s['aws_key']}'\n")
    repo.commit("add", when=1_700_000_000)
    monkeypatch.chdir(repo.path)
    settings = build_settings(repo.path, None, repo.path, {})
    assert settings.config_source is None
    assert settings.fail_on == "high"
    code = main([str(repo.path), "--format", "json", "-o", str(tmp_path / "out.json")])
    assert code == 1
    assert "SEC001" in (tmp_path / "out.json").read_text(encoding="utf-8")


@requires_git
def test_explicit_config_still_applies_inside_repo(repo_factory, tmp_path, monkeypatch):
    repo = repo_factory("explicit")
    repo.write("gitaudit.toml", '[gitaudit]\nfail_on = "critical"\n')
    repo.write("a.txt", "a\n")
    repo.commit("c", when=1_700_000_000)
    monkeypatch.chdir(repo.path)
    settings = build_settings(repo.path, repo.path / "gitaudit.toml", repo.path, {})
    assert settings.fail_on == "critical"


@requires_git
def test_config_in_cwd_outside_repo_is_loaded(repo_factory, tmp_path):
    repo = repo_factory("outside")
    repo.write("a.txt", "a\n")
    repo.commit("c", when=1_700_000_000)
    elsewhere = tmp_path / "config-dir"
    elsewhere.mkdir()
    (elsewhere / "gitaudit.toml").write_text(
        '[gitaudit]\nfail_on = "low"\n', encoding="utf-8"
    )
    settings = build_settings(repo.path, None, elsewhere, {})
    assert settings.fail_on == "low"


def test_audited_worktree_climbs_to_dot_git(tmp_path):
    root = tmp_path / "proj"
    (root / ".git").mkdir(parents=True)
    sub = root / "pkg" / "deep"
    sub.mkdir(parents=True)
    assert audited_worktree(sub) == root.resolve()


def test_audited_worktree_falls_back_to_repo_path(tmp_path):
    lonely = tmp_path / "lonely"
    lonely.mkdir()
    assert audited_worktree(lonely) == lonely.resolve()


def test_cli_refuses_output_inside_dot_git(repo_factory, tmp_path, capsys):
    repo = repo_factory("gitout")
    repo.write("a.txt", "a\n")
    repo.commit("c", when=1_700_000_000)
    target = repo.path / ".git" / "report.json"
    code = main([str(repo.path), "--format", "json", "-o", str(target)])
    assert code == 2
    assert not target.exists()
