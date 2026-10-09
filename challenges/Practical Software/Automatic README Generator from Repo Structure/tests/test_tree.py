from helpers import make_repo
from readme_generator.scan import scan_repo
from readme_generator.tree import MAX_ENTRIES_PER_DIR, render_tree


def _tree(root, files, **kwargs):
    make_repo(root, files)
    return render_tree(scan_repo(root), "proj", **kwargs).splitlines()


def test_directories_come_first_and_known_paths_are_annotated(tmp_path):
    lines = _tree(
        tmp_path, {"b.txt": "", "src/a.py": "", "tests/t.py": "", "pyproject.toml": ""}
    )
    assert lines == [
        "proj/",
        "├── src/  # source code",
        "│   └── a.py",
        "├── tests/  # test suite",
        "│   └── t.py",
        "├── b.txt",
        "└── pyproject.toml  # Python project metadata",
    ]


def test_depth_limits_how_far_the_tree_descends(tmp_path):
    files = {"a/b/c/d.txt": ""}
    assert _tree(tmp_path / "one", files, max_depth=1) == ["proj/", "└── a/"]
    assert _tree(tmp_path / "three", files, max_depth=3) == [
        "proj/",
        "└── a/",
        "    └── b/",
        "        └── c/",
    ]


def test_lockfiles_and_excluded_paths_are_hidden(tmp_path):
    lines = _tree(
        tmp_path,
        {"uv.lock": "", "README.md": "", "main.py": ""},
        exclude=frozenset({"README.md"}),
    )
    assert lines == ["proj/", "└── main.py"]


def test_crowded_directories_are_truncated_with_a_count(tmp_path):
    total = MAX_ENTRIES_PER_DIR + 5
    lines = _tree(tmp_path, {f"f{i:02}.txt": "" for i in range(total)})
    assert lines[-1] == "└── … and 5 more"
    assert len(lines) == 1 + MAX_ENTRIES_PER_DIR + 1
