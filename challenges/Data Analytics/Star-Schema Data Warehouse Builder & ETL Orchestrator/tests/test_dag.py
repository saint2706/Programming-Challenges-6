"""Property tests for the orchestrator on seeded random DAGs with random transient and permanent faults.

The model below predicts each task's status and attempt count from the graph and the fault plan alone.
The executor must agree with it on every generated case, and must never start a task before every
upstream has succeeded.
"""

from __future__ import annotations

import random

import pytest
from star_warehouse.dag import (
    Dag,
    DagError,
    PermanentError,
    Task,
    TransientError,
    run,
)

CASES = 150  # seeded, so every run checks the same graphs


def _random_graph(
    rng: random.Random,
) -> tuple[list[str], dict[str, tuple[str, ...]], dict[str, int]]:
    """Node names, dependencies (always earlier nodes, so acyclic), and retries per node."""
    n = rng.randint(1, 12)
    names = [f"t{i}" for i in range(n)]
    deps: dict[str, tuple[str, ...]] = {}
    retries: dict[str, int] = {}
    for i, name in enumerate(names):
        deps[name] = tuple(names[j] for j in range(i) if rng.random() < 0.3)
        retries[name] = rng.randint(0, 2)
    declared = names[:]
    rng.shuffle(declared)  # declaration order must not matter, only the dependencies
    return declared, deps, retries


def _plan(rng: random.Random, retries: int) -> tuple[int, bool]:
    """(failing attempts from the start, whether the first failure is permanent)."""
    return rng.randint(0, retries + 2), rng.random() < 0.2


def _predict(order, deps, retries, plans):
    """Expected status and attempt count per task, computed from the graph and the plan alone."""
    status: dict[str, str] = {}
    attempts: dict[str, int] = {}
    for name in order:
        if any(status[d] != "succeeded" for d in deps[name]):
            status[name] = "skipped"
            attempts[name] = 0
            continue
        failing, permanent = plans[name]
        if failing and permanent:
            status[name], attempts[name] = "failed", 1
        elif failing <= retries[name]:
            status[name], attempts[name] = "succeeded", failing + 1
        else:
            status[name], attempts[name] = "failed", retries[name] + 1
    return status, attempts


@pytest.mark.parametrize("seed", range(CASES))
def test_random_dag_matches_model(seed):
    rng = random.Random(seed)
    declared, deps, retries = _random_graph(rng)
    plans = {name: _plan(rng, retries[name]) for name in declared}
    calls: dict[str, int] = {name: 0 for name in declared}
    started_ok: set[str] = set()  # tasks whose final attempt succeeded, in run order
    violations: list[str] = []
    sleeps: list[float] = []

    def make(name: str):
        def fn():
            if any(d not in started_ok for d in deps[name]):
                violations.append(f"{name} ran before its upstreams succeeded")
            calls[name] += 1
            failing, permanent = plans[name]
            if calls[name] <= failing:
                if permanent and calls[name] == 1:
                    raise PermanentError(f"{name} permanent")
                if not permanent:
                    raise TransientError(f"{name} transient")
            return name

        return fn

    tasks = [Task(n, make(n), deps=deps[n], retries=retries[n]) for n in declared]
    dag = Dag(tasks)

    def observe(event, name, number, status, error):
        if event == "end" and status == "ok":
            started_ok.add(name)

    results = run(dag, backoff=0.5, sleep=sleeps.append, observer=observe)

    assert not violations, violations
    expected_status, expected_attempts = _predict(dag.order, deps, retries, plans)
    for name in declared:
        assert results[name].status == expected_status[name], name
        assert len(results[name].attempts) == expected_attempts[name], name

    # Order is a valid topological order, and it is the same on every call.
    position = {name: i for i, name in enumerate(dag.order)}
    for name in declared:
        assert all(position[d] < position[name] for d in deps[name])
    assert Dag(tasks).order == dag.order

    # Backoff doubles before each retry; permanent failures and the last attempt do not sleep.
    expected_sleeps = []
    for name in dag.order:
        result = results[name]
        for attempt in result.attempts:
            if (
                attempt.status == "error"
                and attempt.number <= retries[name]
                and not (plans[name][1] and attempt.number == 1)
            ):
                expected_sleeps.append(0.5 * 2 ** (attempt.number - 1))
    assert sorted(sleeps) == sorted(expected_sleeps)


@pytest.mark.parametrize("seed", range(20))
def test_injected_cycle_is_rejected(seed):
    rng = random.Random(1000 + seed)
    names = [f"t{i}" for i in range(rng.randint(2, 8))]
    a, b = rng.sample(names, 2)  # a depends on b and b depends on a: a two-node cycle
    tasks = [
        Task(n, lambda: None, deps=(b,) if n == a else (a,) if n == b else ())
        for n in names
    ]
    with pytest.raises(DagError, match="cycle"):
        Dag(tasks)


def test_unknown_dependency_and_duplicate_name_are_rejected():
    with pytest.raises(DagError, match="unknown"):
        Dag([Task("a", lambda: None, deps=("missing",))])
    with pytest.raises(DagError, match="duplicate"):
        Dag([Task("a", lambda: None), Task("a", lambda: None)])


def test_skip_names_the_direct_blocker_and_siblings_still_run():
    calls = []
    dag = Dag(
        [
            Task("root", lambda: (_ for _ in ()).throw(PermanentError("bad input"))),
            Task("child", lambda: calls.append("child"), deps=("root",)),
            Task("grandchild", lambda: calls.append("grand"), deps=("child",)),
            Task("sibling", lambda: calls.append("sibling")),
        ]
    )
    results = run(dag, sleep=lambda s: None)
    assert results["root"].status == "failed"
    assert results["child"].skipped_because == "upstream root failed"
    assert results["grandchild"].skipped_because == "upstream child skipped"
    assert results["sibling"].status == "succeeded"
    assert calls == ["sibling"]
