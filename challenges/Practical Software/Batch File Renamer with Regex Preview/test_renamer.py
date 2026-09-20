from __future__ import annotations

from pathlib import Path

import pytest
import renamer as renamer_module
from renamer import (
    RenameError,
    apply_renames,
    build_plans,
    compile_pattern,
    find_conflicts,
    load_log,
    record_batch,
    undo_batch,
)
from typer.testing import CliRunner

runner = CliRunner()


def _touch(path: Path, content: str = "x") -> Path:
    path.write_text(content, encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# build_plans
# --------------------------------------------------------------------------


def test_build_plans_basic_replace(tmp_path: Path) -> None:
    f1 = _touch(tmp_path / "img_001.jpg")
    f2 = _touch(tmp_path / "img_002.jpg")
    other = _touch(tmp_path / "notes.txt")

    pattern = compile_pattern(r"^img_(\d+)\.jpg$")
    plans = build_plans([f1, f2, other], pattern, r"photo_\1.jpg")

    assert {p.src.name for p in plans} == {"img_001.jpg", "img_002.jpg"}
    dst_by_src = {p.src.name: p.dst.name for p in plans}
    assert dst_by_src["img_001.jpg"] == "photo_001.jpg"
    assert dst_by_src["img_002.jpg"] == "photo_002.jpg"


def test_build_plans_skips_non_matching_files(tmp_path: Path) -> None:
    f1 = _touch(tmp_path / "keep.txt")
    pattern = compile_pattern(r"^nomatch")
    plans = build_plans([f1], pattern, "renamed")
    assert plans == []


def test_build_plans_named_group_backreference(tmp_path: Path) -> None:
    f1 = _touch(tmp_path / "2024-01-15_report.csv")
    pattern = compile_pattern(r"^(?P<date>\d{4}-\d{2}-\d{2})_(?P<name>.+)\.csv$")
    plans = build_plans([f1], pattern, r"\g<name>_\g<date>.csv")
    assert plans[0].dst.name == "report_2024-01-15.csv"


def test_ignore_case_flag(tmp_path: Path) -> None:
    f1 = _touch(tmp_path / "IMG_001.JPG")
    pattern_sensitive = compile_pattern(r"^img_")
    assert build_plans([f1], pattern_sensitive, "photo_") == []

    pattern_insensitive = compile_pattern(r"^img_", ignore_case=True)
    plans = build_plans([f1], pattern_insensitive, "photo_")
    assert plans[0].dst.name == "photo_001.JPG"


# --------------------------------------------------------------------------
# find_conflicts
# --------------------------------------------------------------------------


def test_duplicate_target_is_a_conflict(tmp_path: Path) -> None:
    f1 = _touch(tmp_path / "a1.txt")
    f2 = _touch(tmp_path / "a2.txt")
    pattern = compile_pattern(r"^a\d\.txt$")
    plans = build_plans([f1, f2], pattern, "merged.txt")
    conflicts = find_conflicts(plans)
    assert len(conflicts) == 2
    assert all(c.kind == "duplicate_target" for c in conflicts)


def test_target_exists_outside_batch_is_a_conflict(tmp_path: Path) -> None:
    src = _touch(tmp_path / "old.txt")
    _touch(tmp_path / "new.txt")  # already present, not part of this batch
    pattern = compile_pattern(r"^old\.txt$")
    plans = build_plans([src], pattern, "new.txt")
    conflicts = find_conflicts(plans)
    assert len(conflicts) == 1
    assert conflicts[0].kind == "target_exists"


def test_case_only_rename_is_not_a_conflict(tmp_path: Path) -> None:
    src = _touch(tmp_path / "readme.txt")
    pattern = compile_pattern(r"^readme\.txt$")
    plans = build_plans([src], pattern, "README.txt")
    # On a case-insensitive filesystem plan.dst.exists() would be True here;
    # it must not be reported as a conflict.
    assert find_conflicts(plans) == []


def test_swap_within_batch_is_not_a_conflict(tmp_path: Path) -> None:
    a = _touch(tmp_path / "a.txt", "A")
    b = _touch(tmp_path / "b.txt", "B")

    from renamer import RenamePlan

    # A single regex substitution can't branch per-match, so build the swap directly.
    plans = [
        RenamePlan(a, tmp_path / "b.txt"),
        RenamePlan(b, tmp_path / "a.txt"),
    ]
    assert find_conflicts(plans) == []


# --------------------------------------------------------------------------
# apply_renames
# --------------------------------------------------------------------------


def test_apply_simple_batch(tmp_path: Path) -> None:
    f1 = _touch(tmp_path / "img_001.jpg", "one")
    f2 = _touch(tmp_path / "img_002.jpg", "two")
    pattern = compile_pattern(r"^img_(\d+)\.jpg$")
    plans = build_plans([f1, f2], pattern, r"photo_\1.jpg")

    applied = apply_renames(plans)

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "photo_001.jpg",
        "photo_002.jpg",
    ]
    assert (tmp_path / "photo_001.jpg").read_text() == "one"
    assert (tmp_path / "photo_002.jpg").read_text() == "two"
    assert len(applied) == 2


def test_apply_case_only_rename(tmp_path: Path) -> None:
    src = _touch(tmp_path / "readme.txt", "content")
    plans = build_plans([src], compile_pattern(r"^readme\.txt$"), "README.txt")

    apply_renames(plans)

    names = [p.name for p in tmp_path.iterdir()]
    assert names == ["README.txt"]
    assert (tmp_path / "README.txt").read_text() == "content"


def test_apply_three_way_rotation(tmp_path: Path) -> None:
    _touch(tmp_path / "a.txt", "A")
    _touch(tmp_path / "b.txt", "B")
    _touch(tmp_path / "c.txt", "C")

    from renamer import RenamePlan

    # a -> b, b -> c, c -> a (a full rotation; every dst is also a src)
    plans = [
        RenamePlan(tmp_path / "a.txt", tmp_path / "b.txt"),
        RenamePlan(tmp_path / "b.txt", tmp_path / "c.txt"),
        RenamePlan(tmp_path / "c.txt", tmp_path / "a.txt"),
    ]
    assert find_conflicts(plans) == []
    apply_renames(plans)

    assert (tmp_path / "a.txt").read_text() == "C"
    assert (tmp_path / "b.txt").read_text() == "A"
    assert (tmp_path / "c.txt").read_text() == "B"


def test_apply_partial_failure_reports_detail_and_keeps_successes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    f1 = _touch(tmp_path / "ok.txt", "ok")
    f2 = _touch(tmp_path / "bad.txt", "bad")
    from renamer import RenamePlan

    plans = [
        RenamePlan(f1, tmp_path / "ok2.txt"),
        RenamePlan(f2, tmp_path / "bad2.txt"),
    ]

    real_rename = Path.rename

    def flaky_rename(self: Path, target):  # type: ignore[no-untyped-def]
        if self.name.startswith(".__rename_tmp_") and "bad2" in str(target):
            raise OSError("simulated failure")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", flaky_rename)

    with pytest.raises(RenameError, match="1 of 2"):
        apply_renames(plans)

    # ok.txt succeeded and landed at its destination.
    assert (tmp_path / "ok2.txt").read_text() == "ok"
    # bad.txt's temp file survives for manual recovery; original name is gone.
    assert not (tmp_path / "bad.txt").exists()
    assert not (tmp_path / "bad2.txt").exists()
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".__rename_tmp_")]
    assert len(leftovers) == 1
    assert leftovers[0].read_text() == "bad"


# --------------------------------------------------------------------------
# log / undo
# --------------------------------------------------------------------------


def test_apply_then_undo_restores_original_names(tmp_path: Path) -> None:
    f1 = _touch(tmp_path / "img_001.jpg", "one")
    f2 = _touch(tmp_path / "img_002.jpg", "two")
    pattern = compile_pattern(r"^img_(\d+)\.jpg$")
    plans = build_plans([f1, f2], pattern, r"photo_\1.jpg")
    applied = apply_renames(plans)

    log_path = tmp_path / ".batch_rename_log.json"
    batch = record_batch(log_path, tmp_path, pattern.pattern, r"photo_\1.jpg", applied)

    undone = undo_batch(log_path)
    assert undone.id == batch.id
    assert undone.undone is True

    assert sorted(p.name for p in tmp_path.iterdir() if not p.name.startswith(".")) == [
        "img_001.jpg",
        "img_002.jpg",
    ]

    batches = load_log(log_path)
    assert batches[0].undone is True


def test_undo_missing_target_raises_clear_error(tmp_path: Path) -> None:
    f1 = _touch(tmp_path / "img_001.jpg", "one")
    pattern = compile_pattern(r"^img_(\d+)\.jpg$")
    plans = build_plans([f1], pattern, r"photo_\1.jpg")
    applied = apply_renames(plans)

    log_path = tmp_path / ".batch_rename_log.json"
    record_batch(log_path, tmp_path, pattern.pattern, r"photo_\1.jpg", applied)

    (tmp_path / "photo_001.jpg").unlink()  # simulate the user moving/deleting it since

    with pytest.raises(RenameError, match="no longer exist"):
        undo_batch(log_path)


def test_undo_with_no_batches_raises(tmp_path: Path) -> None:
    log_path = tmp_path / ".batch_rename_log.json"
    with pytest.raises(RenameError, match="No undoable batch"):
        undo_batch(log_path)


def test_undo_specific_batch_by_id(tmp_path: Path) -> None:
    f1 = _touch(tmp_path / "a.txt")
    pattern = compile_pattern(r"^a\.txt$")
    plans = build_plans([f1], pattern, "a1.txt")
    applied = apply_renames(plans)
    log_path = tmp_path / ".batch_rename_log.json"
    batch1 = record_batch(log_path, tmp_path, pattern.pattern, "a1.txt", applied)

    f2 = tmp_path / "a1.txt"
    plans2 = build_plans([f2], compile_pattern(r"^a1\.txt$"), "a2.txt")
    applied2 = apply_renames(plans2)
    batch2 = record_batch(log_path, tmp_path, "a1.txt", "a2.txt", applied2)

    # Undo the first batch specifically -- should fail because a.txt's
    # renamed form (a1.txt) no longer exists (it became a2.txt).
    with pytest.raises(RenameError, match="no longer exist"):
        undo_batch(log_path, batch_id=batch1.id)

    # Undoing the second (most recent) batch works.
    undone = undo_batch(log_path, batch_id=batch2.id)
    assert undone.id == batch2.id
    assert (tmp_path / "a1.txt").exists()


# --------------------------------------------------------------------------
# CLI smoke tests
# --------------------------------------------------------------------------


def test_cli_preview_does_not_touch_filesystem(tmp_path: Path) -> None:
    _touch(tmp_path / "img_001.jpg")
    result = runner.invoke(
        renamer_module.app,
        ["preview", str(tmp_path), r"^img_(\d+)\.jpg$", r"photo_\1.jpg"],
    )
    assert result.exit_code == 0
    assert "photo_001.jpg" in result.stdout
    assert (tmp_path / "img_001.jpg").exists()
    assert not (tmp_path / "photo_001.jpg").exists()


def test_cli_apply_with_yes_renames_and_records_log(tmp_path: Path) -> None:
    _touch(tmp_path / "img_001.jpg")
    result = runner.invoke(
        renamer_module.app,
        ["apply", str(tmp_path), r"^img_(\d+)\.jpg$", r"photo_\1.jpg", "--yes"],
    )
    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "photo_001.jpg").exists()
    assert (tmp_path / ".batch_rename_log.json").exists()


def test_cli_apply_aborts_on_conflict(tmp_path: Path) -> None:
    _touch(tmp_path / "a1.txt")
    _touch(tmp_path / "a2.txt")
    result = runner.invoke(
        renamer_module.app,
        ["apply", str(tmp_path), r"^a\d\.txt$", "merged.txt", "--yes"],
    )
    assert result.exit_code == 1
    assert (tmp_path / "a1.txt").exists()
    assert (tmp_path / "a2.txt").exists()


def test_cli_undo_round_trip(tmp_path: Path) -> None:
    _touch(tmp_path / "img_001.jpg", "one")
    runner.invoke(
        renamer_module.app,
        ["apply", str(tmp_path), r"^img_(\d+)\.jpg$", r"photo_\1.jpg", "--yes"],
    )
    result = runner.invoke(renamer_module.app, ["undo", str(tmp_path)])
    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "img_001.jpg").exists()


def test_cli_history_lists_batches(tmp_path: Path) -> None:
    _touch(tmp_path / "img_001.jpg")
    runner.invoke(
        renamer_module.app,
        ["apply", str(tmp_path), r"^img_(\d+)\.jpg$", r"photo_\1.jpg", "--yes"],
    )
    result = runner.invoke(renamer_module.app, ["history", str(tmp_path)])
    assert result.exit_code == 0
    assert "active" in result.stdout


def test_recursive_and_include_filters(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    _touch(tmp_path / "top_001.log")
    _touch(tmp_path / "sub" / "nested_002.log")
    _touch(tmp_path / "sub" / "nested_002.txt")

    from renamer import iter_candidate_files

    non_recursive = iter_candidate_files(tmp_path, "*.log", recursive=False)
    assert [p.name for p in non_recursive] == ["top_001.log"]

    recursive = iter_candidate_files(tmp_path, "*.log", recursive=True)
    assert sorted(p.name for p in recursive) == ["nested_002.log", "top_001.log"]
