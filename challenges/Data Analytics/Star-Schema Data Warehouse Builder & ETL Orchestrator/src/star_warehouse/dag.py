"""A small task-DAG executor: dependency order, retries with exponential backoff, skip-on-upstream
failure and an event stream for the run log.

It knows nothing about SQL or DuckDB. The pipeline passes it callables and records the events.
Tasks run one at a time (DuckDB has one writer per file), in a deterministic topological order.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any


class DagError(ValueError):
    """The graph is not a DAG: a cycle, an unknown or a duplicate task name."""


class PermanentError(Exception):
    """Retrying will not help (a failed data-quality gate, a malformed input)."""


class TransientError(Exception):
    """A failure worth retrying (a lock timeout, an I/O hiccup, an injected fault in tests)."""


@dataclass(frozen=True)
class Task:
    name: str
    fn: Callable[[], Any]
    deps: tuple[str, ...] = ()
    retries: int = 0
    writes: tuple[str, ...] = ()  # tables written; used only by the lineage report


@dataclass
class Attempt:
    task: str
    number: int  # 1-based
    status: str  # "ok" | "error"
    error: str | None = None


@dataclass
class TaskResult:
    name: str
    status: str  # "succeeded" | "failed" | "skipped"
    attempts: list[Attempt] = field(default_factory=list)
    value: Any = None
    skipped_because: str | None = None


# (event, task, attempt number, status or None, error or None); event is "start" or "end" or "skip"
Observer = Callable[[str, str, int, str | None, str | None], None]


class Dag:
    def __init__(self, tasks: Iterable[Task]):
        self.tasks: dict[str, Task] = {}
        for task in tasks:
            if task.name in self.tasks:
                raise DagError(f"duplicate task {task.name!r}")
            if task.retries < 0:
                raise DagError(f"task {task.name!r} has negative retries")
            self.tasks[task.name] = task
        for task in self.tasks.values():
            for dep in task.deps:
                if dep not in self.tasks:
                    raise DagError(
                        f"task {task.name!r} depends on unknown task {dep!r}"
                    )
        self.order = self._topological_order()

    def _topological_order(self) -> list[str]:
        """Kahn's algorithm; among ready tasks the first-declared one goes first (deterministic)."""
        order: list[str] = []
        done: set[str] = set()
        pending = list(self.tasks)
        while pending:
            ready = next(
                (n for n in pending if set(self.tasks[n].deps) <= done),
                None,
            )
            if ready is None:
                raise DagError(f"cycle among tasks: {', '.join(pending)}")
            order.append(ready)
            done.add(ready)
            pending.remove(ready)
        return order


def run(
    dag: Dag,
    *,
    backoff: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    observer: Observer | None = None,
) -> dict[str, TaskResult]:
    """Run every task once, in order; retry transient failures; skip what an upstream failure blocks.

    A task runs only when every dependency has succeeded. A failed task does not stop unrelated
    branches (partial failure): its own downstream tasks are skipped and the rest still run.
    The wait before retry ``n`` is ``backoff * 2 ** (n - 1)`` seconds.
    """

    def emit(
        event: str, name: str, number: int, status: str | None, error: str | None
    ) -> None:
        if observer is not None:
            observer(event, name, number, status, error)

    results: dict[str, TaskResult] = {}
    for name in dag.order:
        task = dag.tasks[name]
        blocked = [d for d in task.deps if results[d].status != "succeeded"]
        if blocked:
            reason = f"upstream {blocked[0]} {results[blocked[0]].status}"
            results[name] = TaskResult(name, "skipped", skipped_because=reason)
            emit("skip", name, 0, "skipped", reason)
            continue
        result = TaskResult(name, "failed")
        for number in range(1, task.retries + 2):
            emit("start", name, number, None, None)
            try:
                value = task.fn()
            except Exception as exc:  # noqa: BLE001 - every failure is recorded; PermanentError is not retried
                error = f"{type(exc).__name__}: {exc}"
                result.attempts.append(Attempt(name, number, "error", error))
                emit("end", name, number, "error", error)
                if isinstance(exc, PermanentError) or number == task.retries + 1:
                    break
                sleep(backoff * 2 ** (number - 1))
            else:
                result.attempts.append(Attempt(name, number, "ok"))
                emit("end", name, number, "ok", None)
                result.status = "succeeded"
                result.value = value
                break
        results[name] = result
    return results
