"""Synthetic repository builder and runtime-assembled secret fixtures.

Secret-shaped strings are built by concatenating fragments at runtime so the repository never
contains a literal that GitHub push protection or CodeQL would flag.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

GIT = shutil.which("git")

_CLEAN_ENV = {
    **{k: v for k, v in os.environ.items() if not k.startswith("GIT_")},
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_TERMINAL_PROMPT": "0",
    "LC_ALL": "C",
}

DEFAULT_AUTHOR = ("Test Author", "author@example.org")


def git(repo: Path, *args: str, env: dict[str, str] | None = None) -> str:
    """Run git in a test repository with a clean, global-config-free environment."""
    proc = subprocess.run(
        [GIT, "-C", str(repo), "-c", "commit.gpgsign=false", *args],
        capture_output=True,
        env={**_CLEAN_ENV, **(env or {})},
        check=True,
    )
    return proc.stdout.decode("utf-8", "replace").strip()


class RepoBuilder:
    """Builds a repository one commit at a time. Dates are explicit so tests are reproducible."""

    def __init__(self, path: Path, branch: str = "main") -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        git(self.path, "init", "-q", "-b", branch)
        git(self.path, "config", "user.name", DEFAULT_AUTHOR[0])
        git(self.path, "config", "user.email", DEFAULT_AUTHOR[1])
        git(self.path, "config", "core.autocrlf", "false")

    def write(self, rel: str, content: str | bytes) -> None:
        target = self.path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode("utf-8") if isinstance(content, str) else content
        target.write_bytes(data)

    def remove(self, rel: str) -> None:
        (self.path / rel).unlink()

    def commit(
        self,
        message: str = "change",
        *,
        when: int | None = None,
        author: tuple[str, str] | None = None,
    ) -> str:
        git(self.path, "add", "-A")
        env: dict[str, str] = {}
        if when is not None:
            env = {
                "GIT_AUTHOR_DATE": f"@{when} +0000",
                "GIT_COMMITTER_DATE": f"@{when} +0000",
            }
        args = ["commit", "-q", "--allow-empty", "-m", message]
        if author is not None:
            args += ["--author", f"{author[0]} <{author[1]}>"]
        git(self.path, *args, env=env)
        return self.sha("HEAD")

    def sha(self, rev: str = "HEAD") -> str:
        return git(self.path, "rev-parse", rev)

    def branch(self, name: str, at: str | None = None) -> None:
        args = ["branch", name] + ([at] if at else [])
        git(self.path, *args)

    def checkout(self, name: str) -> None:
        git(self.path, "checkout", "-q", name)

    def merge(self, name: str, when: int | None = None) -> str:
        env = (
            {
                "GIT_AUTHOR_DATE": f"@{when} +0000",
                "GIT_COMMITTER_DATE": f"@{when} +0000",
            }
            if when
            else {}
        )
        git(self.path, "merge", "-q", "--no-ff", "-m", f"merge {name}", name, env=env)
        return self.sha("HEAD")

    def tag(self, name: str, rev: str = "HEAD") -> None:
        git(self.path, "tag", name, rev)

    def set_ref(self, ref: str, sha: str) -> None:
        git(self.path, "update-ref", ref, sha)

    def config(self, key: str, value: str) -> None:
        git(self.path, "config", key, value)


def make_repo(path: Path, branch: str = "main") -> RepoBuilder:
    return RepoBuilder(path, branch)


def secret_fixtures() -> dict[str, str]:
    """Secret-shaped strings assembled at runtime. Each matches exactly one rule family."""
    return {
        "aws_key": "AKIA" + "Q3XJ7K2M9PLR4T8N",
        "aws_secret": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYq8Zt2NvLwA",
        "github_classic": "ghp_" + "9fK3mQ2vR7xT1bN8cW4eH6yU0sA5dG3jL2pZ",
        "github_fine": "github_pat_"
        + (
            "11ABCDEFG0"
            + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8s9T0u1V2w3X4y5Z6a7b8c9d0e1f2g3h4i5j6k7l8"
        )[:82],
        "gitlab": "glpat-" + "x9Qz4LmP2vR8wT5yB1nK",
        "slack": "xoxb-" + "1234567890-" + "AbCdEfGhIjKlMnOp",
        "slack_hook": "https://hooks.slack.com/services/T0"
        + "ABCDEFGH/B0"
        + "ABCDEFGH/"
        + "abcdefghijklmnopqrstuvwx",
        "stripe": "sk_" + "live_" + "4eC39HqLyjWDarjtT1zdp7dc",
        "google": "AIza" + "SyD-9tSrke72PouQMnMX-a7eZSW0jkFMBWx",
        "jwt": "eyJhbGciOiJIUzI1NiJ9"
        + "."
        + "eyJzdWIiOiIxMjM0NTY3ODkwIn0"
        + "."
        + "dBjftJeZ4CVP-mB92K27uhbUJU1p1r",
        "url_password": "postgres://app:"
        + "Zx9vQ2mP7wL4"
        + "@db.internal.example:5432/app",
        "generic": 'password = "' + "Vq8Lr2Nw5Zt7Kx3p" + '"',
        "npm_token": "//registry.npmjs.org/:_authToken="
        + "npm_"
        + "Tq7Lp2Vx9Wm4Rz8Yc3Bn6Hk1Ds5Fg0Ja2Qe",
        "private_key_header": "-----BEGIN " + "RSA PRIVATE KEY-----",
    }


def fake_aws_key_id() -> str:
    return "AKIA" + "Q3XJ7K2M9PLR4T8N"
