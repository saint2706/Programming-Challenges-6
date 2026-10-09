import os

from helpers import make_repo
from readme_generator.scan import scan_repo


def test_always_ignored_directories_are_skipped(tmp_path):
    make_repo(
        tmp_path,
        {
            "a.py": "x",
            "node_modules/x/index.js": "x",
            ".git/config": "x",
            "__pycache__/a.pyc": "x",
        },
    )
    scan = scan_repo(tmp_path)
    assert scan.files == ["a.py"]
    assert scan.dirs == []


def test_root_gitignore_is_honoured_including_directory_patterns(tmp_path):
    make_repo(
        tmp_path,
        {
            ".gitignore": "*.log\nsecrets/\n",
            "app.py": "x",
            "run.log": "x",
            "secrets/key.txt": "x",
        },
    )
    scan = scan_repo(tmp_path)
    assert "run.log" not in scan.files
    assert "secrets/key.txt" not in scan.files
    assert "secrets" not in scan.dirs
    assert "app.py" in scan.files


def test_nested_gitignore_applies_only_to_its_own_subtree(tmp_path):
    make_repo(
        tmp_path,
        {
            "sub/.gitignore": "generated.py\n",
            "sub/generated.py": "x",
            "sub/keep.py": "x",
            "generated.py": "x",
        },
    )
    scan = scan_repo(tmp_path)
    assert "sub/generated.py" not in scan.files
    assert "generated.py" in scan.files  # the nested rule must not leak upward
    assert "sub/keep.py" in scan.files


def test_language_shares_count_bytes_of_programming_languages_only(tmp_path):
    make_repo(
        tmp_path,
        {
            "a.py": "x" * 75,
            "b.js": "x" * 25,
            "notes.md": "x" * 1000,
            "data.json": "x" * 1000,
        },
    )
    shares = dict(scan_repo(tmp_path).language_shares())
    assert shares == {"Python": 0.75, "JavaScript": 0.25}


def test_no_source_files_means_no_languages(tmp_path):
    make_repo(tmp_path, {"notes.md": "hello"})
    assert scan_repo(tmp_path).language_shares() == []


def test_symlink_cycle_does_not_hang_or_recurse(tmp_path):
    make_repo(tmp_path, {"real/a.py": "x"})
    os.symlink(tmp_path, tmp_path / "real" / "loop", target_is_directory=True)
    scan = scan_repo(tmp_path)
    assert scan.files == ["real/a.py"]
    assert "real/loop" in scan.dirs


def test_scan_stops_at_the_file_cap_and_says_so(tmp_path):
    make_repo(tmp_path, {f"f{i}.py": "x" for i in range(10)})
    scan = scan_repo(tmp_path, max_files=4)
    assert scan.truncated
    assert len(scan.files) == 4


def test_lockfiles_stay_visible_to_the_scan(tmp_path):
    make_repo(tmp_path, {"uv.lock": "x", "package-lock.json": "{}"})
    assert scan_repo(tmp_path).has_file("uv.lock")
