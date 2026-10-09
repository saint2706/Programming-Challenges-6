from pathlib import Path

import pytest
from helpers import make_repo
from readme_generator.cli import app, write_atomic
from typer.testing import CliRunner

runner = CliRunner()

REPO = {
    "pyproject.toml": '[project]\nname = "demo"\ndescription = "A demo."\nrequires-python = ">=3.12"\ndependencies = ["typer"]\n'
    '[project.scripts]\ndemo = "demo.cli:app"\n',
    "uv.lock": "",
    "src/demo/__init__.py": "",
    "tests/test_demo.py": "",
    ".env.example": "# Where to listen\nPORT=8000\nSECRET_KEY=hunter2\n",
    "LICENSE": "MIT License",
}


@pytest.fixture
def repo(tmp_path):
    return make_repo(tmp_path / "demo", REPO)


def run(*args):
    return runner.invoke(app, [str(a) for a in args])


def test_writes_a_readme_with_every_inferred_section(repo):
    result = run(repo)
    assert result.exit_code == 0, result.output
    text = (repo / "README.md").read_text()
    for expected in [
        "# demo",
        "A demo.",
        "**Package manager:** uv",
        "uv sync",
        "`uv run demo`",
        "`PORT`",
        "## License\n\nMIT.",
    ]:
        assert expected in text
    assert (
        "hunter2" not in text
    )  # secret-looking template values never reach the README


def test_regenerating_is_idempotent_and_does_not_list_its_own_output(repo):
    run(repo)
    first = (repo / "README.md").read_text()
    assert "README.md" not in first.split("## Project structure")[1]
    result = run(repo)
    assert "up to date" in result.output
    assert (repo / "README.md").read_text() == first


def test_refuses_to_overwrite_a_hand_written_readme(repo):
    (repo / "README.md").write_text("# Mine\n\nHand-written.\n")
    result = run(repo)
    assert result.exit_code == 2
    assert "--force" in result.output
    assert (repo / "README.md").read_text() == "# Mine\n\nHand-written.\n"


def test_force_replaces_a_hand_written_readme(repo):
    (repo / "README.md").write_text("# Mine\n")
    assert run(repo, "--force").exit_code == 0
    assert "<!-- readme-gen:begin header -->" in (repo / "README.md").read_text()


def test_refresh_updates_only_generated_blocks(repo):
    run(repo)
    path = repo / "README.md"
    path.write_text(
        "Badges go here.\n\n" + path.read_text() + "\n## Roadmap\n\nMine.\n"
    )
    (repo / "Dockerfile").write_text("FROM python:3.12\n")
    result = run(repo)
    assert result.exit_code == 0 and "added docker" in result.output
    text = path.read_text()
    assert text.startswith("Badges go here.") and text.rstrip().endswith("Mine.")
    assert "docker build -t demo ." in text


def test_stale_blocks_are_removed_on_refresh(repo):
    (repo / "Dockerfile").write_text("FROM x\n")
    run(repo)
    (repo / "Dockerfile").unlink()
    result = run(repo)
    assert "removed docker" in result.output
    assert "docker build" not in (repo / "README.md").read_text()


def test_check_mode_reports_drift_without_writing(repo):
    assert run(repo, "--check").exit_code == 1  # missing
    run(repo)
    assert run(repo, "--check").exit_code == 0
    (repo / "Makefile").write_text("test: ## Run tests\n\tpytest\n")
    before = (repo / "README.md").read_text()
    result = run(repo, "--check")
    assert result.exit_code == 1 and "out of date" in result.output
    assert (repo / "README.md").read_text() == before


def test_check_on_an_unmanaged_readme_fails(repo):
    (repo / "README.md").write_text("# Mine\n")
    assert run(repo, "--check").exit_code == 1


def test_stdout_prints_without_touching_disk(repo):
    result = run(repo, "--stdout")
    assert result.exit_code == 0 and "# demo" in result.output
    assert not (repo / "README.md").exists()


def test_output_option_writes_elsewhere(repo, tmp_path):
    target = tmp_path / "out.md"
    assert run(repo, "-o", target).exit_code == 0
    assert target.exists() and not (repo / "README.md").exists()


def test_missing_directory_is_a_usage_error(tmp_path):
    result = run(tmp_path / "nope")
    assert result.exit_code == 2 and "not a directory" in result.output


def test_unparseable_manifest_warns_but_still_generates(tmp_path):
    root = make_repo(tmp_path / "r", {"package.json": "{broken", "main.py": "print(1)"})
    result = run(root)
    assert result.exit_code == 0
    assert "warning: Could not parse package.json" in result.output
    assert "Python (100%)" in (root / "README.md").read_text()


def test_repo_with_nothing_recognisable_still_gets_a_header_with_todo(tmp_path):
    root = make_repo(tmp_path / "bare", {"notes.txt": "hi"})
    run(root)
    text = (root / "README.md").read_text()
    assert "# bare" in text and "_TODO: describe" in text
    assert "## Getting started" not in text


def test_write_atomic_leaves_no_temp_files(tmp_path):
    write_atomic(tmp_path / "x.md", "hello")
    assert [p.name for p in tmp_path.iterdir()] == ["x.md"]


def test_write_atomic_cleans_up_when_the_write_fails(tmp_path, monkeypatch):
    def boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("readme_generator.cli.os.replace", boom)
    with pytest.raises(OSError):
        write_atomic(tmp_path / "x.md", "hello")
    assert list(Path(tmp_path).iterdir()) == []
