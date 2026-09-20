"""Batch file renamer with a regex preview step and log-backed undo.

Core design
-----------
Renames are always computed as a *plan* first (``build_plans``) and checked
for conflicts (``find_conflicts``) before anything touches the filesystem.
Applying a plan (``apply_renames``) uses a two-phase temp-rename so that
collisions, case-only changes (``foo.txt`` -> ``FOO.txt`` on a
case-insensitive filesystem like NTFS), and cyclic renames (A -> B, B -> A)
all "just work" without special-casing: every source is first moved to a
unique temporary name in the same directory, then every temp file is moved
to its final destination. Neither phase can collide with an original name.

Every successful ``apply`` batch is appended to a JSON log file
(``.batch_rename_log.json`` by default) so it can be reversed later with
``undo`` -- including cases where the reversal itself needs the same
two-phase trick (undoing a swap is a swap).
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

DEFAULT_LOG_NAME = ".batch_rename_log.json"

console = Console()
app = typer.Typer(
    add_completion=False,
    help="Regex-based batch file renamer with preview, conflict detection, and undo.",
)


class RenameError(RuntimeError):
    """Raised when applying or undoing a batch fails partway through."""


@dataclass
class RenamePlan:
    src: Path
    dst: Path

    @property
    def changed(self) -> bool:
        return self.src.name != self.dst.name


@dataclass
class Conflict:
    kind: str  # "duplicate_target" | "target_exists"
    detail: str
    plan: RenamePlan


@dataclass
class Batch:
    id: str
    timestamp: float
    directory: str
    pattern: str
    replacement: str
    operations: list[dict[str, str]]
    undone: bool = False
    undone_at: float | None = None


def compile_pattern(pattern: str, ignore_case: bool = False) -> re.Pattern[str]:
    flags = re.IGNORECASE if ignore_case else 0
    return re.compile(pattern, flags)


def iter_candidate_files(directory: Path, include: str, recursive: bool) -> list[Path]:
    glob_fn = directory.rglob if recursive else directory.glob
    return sorted(p for p in glob_fn(include) if p.is_file())


def build_plans(
    paths: list[Path], pattern: re.Pattern[str], replacement: str
) -> list[RenamePlan]:
    """Compute (src, dst) pairs for every file whose name matches ``pattern``.

    Files that don't match at all are skipped -- they were never selected by
    the regex, so they shouldn't appear as a no-op rename.
    """
    plans: list[RenamePlan] = []
    for path in paths:
        new_name, n = pattern.subn(replacement, path.name)
        if n == 0:
            continue
        plans.append(RenamePlan(src=path, dst=path.with_name(new_name)))
    return plans


def _is_case_only_change(src: Path, dst: Path) -> bool:
    return (
        src.parent == dst.parent
        and src.name != dst.name
        and src.name.lower() == dst.name.lower()
    )


def find_conflicts(plans: list[RenamePlan]) -> list[Conflict]:
    """Flag destination collisions that ``apply_renames`` cannot safely resolve.

    Two things are *not* conflicts even though a naive check would flag them:
    a case-only rename of a file to itself, and a destination that happens to
    equal another file's source within the same batch (a chain or a swap) --
    the two-phase temp rename handles both correctly.
    """
    conflicts: list[Conflict] = []
    srcs_resolved = {p.src.resolve() for p in plans}

    dest_groups: dict[Path, list[RenamePlan]] = {}
    for plan in plans:
        dest_groups.setdefault(plan.dst.resolve(), []).append(plan)

    for group in dest_groups.values():
        distinct_srcs = {p.src.resolve() for p in group}
        if len(distinct_srcs) > 1:
            for plan in group:
                conflicts.append(
                    Conflict(
                        "duplicate_target",
                        f"{len(distinct_srcs)} files would all rename to {plan.dst.name!r}",
                        plan,
                    )
                )

    for plan in plans:
        dst_resolved = plan.dst.resolve()
        src_resolved = plan.src.resolve()
        if dst_resolved == src_resolved:
            continue  # true no-op (shouldn't happen since build_plans requires a match, but safe)
        if dst_resolved in srcs_resolved:
            continue  # destination is another file's source in this batch -- chain/swap, not a conflict
        if plan.dst.exists() and not _is_case_only_change(plan.src, plan.dst):
            conflicts.append(
                Conflict("target_exists", f"{plan.dst.name!r} already exists", plan)
            )

    return conflicts


def _unique_temp_path(path: Path) -> Path:
    return path.with_name(f".__rename_tmp_{uuid.uuid4().hex}{path.suffix}")


def apply_renames(plans: list[RenamePlan]) -> list[tuple[Path, Path]]:
    """Execute a rename plan using a two-phase temp-name swap.

    Phase 1 moves every source to a unique temp name in the same directory.
    Phase 2 moves every temp name to its final destination. If phase 1 fails
    partway, everything already moved is rolled back and ``RenameError`` is
    raised. If phase 2 fails for some files, the ones that succeeded stay
    renamed (best-effort), the failing ones are left at their temp name for
    manual recovery, and ``RenameError`` reports exactly which failed and why.
    """
    staged: list[tuple[RenamePlan, Path]] = []
    try:
        for plan in plans:
            temp = _unique_temp_path(plan.src)
            plan.src.rename(temp)
            staged.append((plan, temp))
    except OSError as exc:
        for plan, temp in staged:
            if temp.exists():
                temp.rename(plan.src)
        raise RenameError(f"Backup phase failed, rolled back: {exc}") from exc

    applied: list[tuple[Path, Path]] = []
    failures: list[tuple[RenamePlan, OSError]] = []
    for plan, temp in staged:
        try:
            temp.rename(plan.dst)
            applied.append((plan.src, plan.dst))
        except OSError as exc:
            failures.append((plan, exc))

    if failures:
        detail = "; ".join(f"{p.src.name} -> {p.dst.name}: {e}" for p, e in failures)
        raise RenameError(
            f"{len(failures)} of {len(plans)} rename(s) failed after backup; "
            f"{len(applied)} succeeded, failing files left as temp files for recovery: {detail}"
        )
    return applied


# --------------------------------------------------------------------------
# Log / undo
# --------------------------------------------------------------------------


def _log_path_for(directory: Path, log: Path | None) -> Path:
    return log if log is not None else directory / DEFAULT_LOG_NAME


def load_log(log_path: Path) -> list[Batch]:
    if not log_path.exists():
        return []
    raw = json.loads(log_path.read_text(encoding="utf-8"))
    return [Batch(**entry) for entry in raw.get("batches", [])]


def save_log(log_path: Path, batches: list[Batch]) -> None:
    payload = {"batches": [vars(b) for b in batches]}
    log_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def record_batch(
    log_path: Path,
    directory: Path,
    pattern: str,
    replacement: str,
    applied: list[tuple[Path, Path]],
) -> Batch:
    batches = load_log(log_path)
    batch = Batch(
        id=uuid.uuid4().hex[:12],
        timestamp=time.time(),
        directory=str(directory.resolve()),
        pattern=pattern,
        replacement=replacement,
        operations=[{"src": str(s), "dst": str(d)} for s, d in applied],
    )
    batches.append(batch)
    save_log(log_path, batches)
    return batch


def find_batch(batches: list[Batch], batch_id: str | None) -> Batch:
    if batch_id is not None:
        for b in batches:
            if b.id == batch_id:
                return b
        raise RenameError(f"No batch with id {batch_id!r} found in log")
    for b in reversed(batches):
        if not b.undone:
            return b
    raise RenameError("No undoable batch found in log")


def undo_batch(log_path: Path, batch_id: str | None = None) -> Batch:
    batches = load_log(log_path)
    batch = find_batch(batches, batch_id)

    missing = [op["dst"] for op in batch.operations if not Path(op["dst"]).exists()]
    if missing:
        raise RenameError(
            f"Cannot undo batch {batch.id}: {len(missing)} renamed file(s) no longer exist "
            f"at their expected location: {', '.join(missing)}"
        )

    # Reverse the batch: current name (dst) -> original name (src).
    reverse_plans = [
        RenamePlan(src=Path(op["dst"]), dst=Path(op["src"])) for op in batch.operations
    ]
    conflicts = find_conflicts(reverse_plans)
    if conflicts:
        detail = "; ".join(c.detail for c in conflicts)
        raise RenameError(
            f"Cannot undo batch {batch.id}, conflicts would occur: {detail}"
        )

    apply_renames(reverse_plans)

    batch.undone = True
    batch.undone_at = time.time()
    save_log(log_path, batches)
    return batch


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _render_plan_table(plans: list[RenamePlan], conflicts: list[Conflict]) -> Table:
    conflict_by_plan = {id(c.plan): c for c in conflicts}
    table = Table(title=f"{len(plans)} file(s) matched")
    table.add_column("Original", style="cyan")
    table.add_column("")
    table.add_column("Renamed", style="green")
    table.add_column("Status")
    for plan in plans:
        conflict = conflict_by_plan.get(id(plan))
        status = f"[red]{conflict.kind}[/red]" if conflict else "[green]ok[/green]"
        table.add_row(plan.src.name, "->", plan.dst.name, status)
    return table


def _plan_from_args(
    directory: Path,
    pattern: str,
    replacement: str,
    include: str,
    recursive: bool,
    ignore_case: bool,
) -> list[RenamePlan]:
    if not directory.is_dir():
        raise typer.BadParameter(f"{directory} is not a directory")
    compiled = compile_pattern(pattern, ignore_case)
    files = iter_candidate_files(directory, include, recursive)
    return build_plans(files, compiled, replacement)


@app.command()
def preview(
    directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    pattern: Annotated[
        str, typer.Argument(help="Regex applied to each file's basename")
    ],
    replacement: Annotated[
        str,
        typer.Argument(help="Replacement text; supports \\1, \\g<name> backreferences"),
    ],
    include: Annotated[
        str, typer.Option(help="Glob filter applied before the regex")
    ] = "*",
    recursive: Annotated[bool, typer.Option("--recursive", "-r")] = False,
    ignore_case: Annotated[bool, typer.Option("--ignore-case", "-i")] = False,
) -> None:
    """Show what would be renamed without touching the filesystem."""
    plans = _plan_from_args(
        directory, pattern, replacement, include, recursive, ignore_case
    )
    if not plans:
        console.print("[yellow]No files matched.[/yellow]")
        raise typer.Exit(0)
    conflicts = find_conflicts(plans)
    console.print(_render_plan_table(plans, conflicts))
    if conflicts:
        console.print(f"[red]{len(conflicts)} conflict(s) would block apply.[/red]")


@app.command()
def apply(
    directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    pattern: Annotated[str, typer.Argument()],
    replacement: Annotated[str, typer.Argument()],
    include: Annotated[str, typer.Option()] = "*",
    recursive: Annotated[bool, typer.Option("--recursive", "-r")] = False,
    ignore_case: Annotated[bool, typer.Option("--ignore-case", "-i")] = False,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt")
    ] = False,
    log: Annotated[
        Path | None,
        typer.Option(
            help="Path to the undo log (default: <directory>/.batch_rename_log.json)"
        ),
    ] = None,
) -> None:
    """Preview, confirm, then perform the renames (recorded for undo)."""
    plans = _plan_from_args(
        directory, pattern, replacement, include, recursive, ignore_case
    )
    if not plans:
        console.print("[yellow]No files matched.[/yellow]")
        raise typer.Exit(0)

    conflicts = find_conflicts(plans)
    console.print(_render_plan_table(plans, conflicts))
    if conflicts:
        console.print(
            f"[red]{len(conflicts)} conflict(s) found; aborting. Fix the pattern and try again.[/red]"
        )
        raise typer.Exit(1)

    if not yes and not typer.confirm(f"Rename {len(plans)} file(s)?"):
        console.print("Aborted.")
        raise typer.Exit(0)

    try:
        applied = apply_renames(plans)
    except RenameError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    log_path = _log_path_for(directory, log)
    batch = record_batch(log_path, directory, pattern, replacement, applied)
    console.print(
        f"[green]Renamed {len(applied)} file(s). Batch id: {batch.id} "
        f"(undo with `undo {directory} --batch-id {batch.id}`)[/green]"
    )


@app.command()
def undo(
    directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    batch_id: Annotated[
        str | None,
        typer.Option(help="Undo a specific batch; defaults to the most recent one"),
    ] = None,
    log: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Reverse a previously applied batch."""
    log_path = _log_path_for(directory, log)
    try:
        batch = undo_batch(log_path, batch_id)
    except RenameError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(
        f"[green]Undid batch {batch.id} ({len(batch.operations)} file(s) restored).[/green]"
    )


@app.command()
def history(
    directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    log: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """List recorded batches for a directory."""
    log_path = _log_path_for(directory, log)
    batches = load_log(log_path)
    if not batches:
        console.print("[yellow]No history.[/yellow]")
        raise typer.Exit(0)
    table = Table(title="Rename history")
    table.add_column("Batch")
    table.add_column("When")
    table.add_column("Pattern -> Replacement")
    table.add_column("Files")
    table.add_column("Status")
    for b in batches:
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(b.timestamp))
        status = "[dim]undone[/dim]" if b.undone else "[green]active[/green]"
        table.add_row(
            b.id,
            when,
            f"{b.pattern!r} -> {b.replacement!r}",
            str(len(b.operations)),
            status,
        )
    console.print(table)


if __name__ == "__main__":
    app()
