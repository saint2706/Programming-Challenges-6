"""Tests for the prime sieve showdown.

Run with:  uv run --with pytest --with numpy pytest -q
"""

from __future__ import annotations

import json
import subprocess
import sys
from math import isqrt
from pathlib import Path

import pytest

import benchmark
import sieves
from sieves import (
    IMPLEMENTATIONS,
    PI_REFERENCE,
    W30,
    W60,
    atkin,
    eratosthenes_segmented,
    eratosthenes_simple,
    eratosthenes_wheel30,
    iter_primes,
    primes_below,
)

HERE = Path(__file__).parent
PURE = ["era-simple", "era-wheel30", "era-segmented", "atkin"]
ALL_KEYS = list(IMPLEMENTATIONS)


def trial_division_primes(limit: int) -> list[int]:
    """Deliberately naive oracle. Slow, obviously correct, independent of the wheels."""
    out = []
    for n in range(2, limit + 1):
        if all(n % d for d in range(2, isqrt(n) + 1)):
            out.append(n)
    return out


@pytest.fixture(scope="module")
def oracle() -> list[int]:
    return trial_division_primes(5000)


def _available(key: str) -> bool:
    return IMPLEMENTATIONS[key].available()[0]


ALL_AVAILABLE = [k for k in ALL_KEYS if _available(k)]


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ALL_AVAILABLE)
def test_matches_oracle_at_every_small_limit(key, oracle):
    """Every limit from 0 to 300 -- this is where off-by-one wheel bugs live."""
    fn = IMPLEMENTATIONS[key].fn
    counts = [0] * 301
    for p in oracle:
        if p <= 300:
            for n in range(p, 301):
                counts[n] += 1
    for n in range(0, 301):
        assert fn(n) == counts[n], f"{key} disagrees at limit {n}"


@pytest.mark.parametrize("key", ALL_AVAILABLE)
def test_wheel_boundaries(key):
    """Limits sitting exactly on and around the wheel moduli 30 and 60."""
    fn = IMPLEMENTATIONS[key].fn
    reference = eratosthenes_simple
    for base in (30, 60, 210, 2310):
        for n in (base - 2, base - 1, base, base + 1, base + 2):
            assert fn(n) == reference(n), f"{key} at limit {n}"


@pytest.mark.parametrize("key", ALL_AVAILABLE)
@pytest.mark.parametrize("limit", [10**1, 10**2, 10**3, 10**4, 10**5, 10**6])
def test_known_pi_values(key, limit):
    assert IMPLEMENTATIONS[key].fn(limit) == PI_REFERENCE[limit]


@pytest.mark.parametrize("key", ALL_AVAILABLE)
def test_all_implementations_agree(key):
    reference = eratosthenes_simple
    for limit in (7, 49, 121, 169, 1000, 9973, 65536, 100003):
        assert IMPLEMENTATIONS[key].fn(limit) == reference(limit), limit


@pytest.mark.slow
@pytest.mark.parametrize("key", ALL_AVAILABLE)
def test_ten_million(key):
    assert IMPLEMENTATIONS[key].fn(10**7) == PI_REFERENCE[10**7]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ALL_AVAILABLE)
@pytest.mark.parametrize("bad", ["100", None, [], 2.5, complex(1, 2)])
def test_bogus_limits_fail_clearly(key, bad):
    """Not a TypeError from deep inside range() three frames down."""
    with pytest.raises((TypeError, ValueError)) as exc:
        IMPLEMENTATIONS[key].fn(bad)
    assert "limit" in str(exc.value)


@pytest.mark.parametrize("key", ALL_AVAILABLE)
def test_integral_floats_are_accepted(key):
    """``1e6`` is how people write this, and the CLI parses --limit that way."""
    assert IMPLEMENTATIONS[key].fn(1e6) == PI_REFERENCE[10**6]


@pytest.mark.parametrize("key", ALL_AVAILABLE)
@pytest.mark.parametrize("limit", [-1, -10**9])
def test_negative_limits_are_empty_not_an_error(key, limit):
    assert IMPLEMENTATIONS[key].fn(limit) == 0


def test_booleans_are_not_integers_here():
    with pytest.raises(TypeError):
        sieves.eratosthenes_simple(True)


# ---------------------------------------------------------------------------
# Segmentation must not change the answer, whatever the segment size
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("segment", [8, 9, 16, 63, 64, 1000, 1 << 15, 1 << 22])
def test_segment_size_is_irrelevant_to_correctness(segment):
    for limit in (7, 100, 1009, 30030, 100_000):
        assert eratosthenes_segmented(limit, segment) == eratosthenes_simple(limit), (
            limit,
            segment,
        )


def test_segmentation_is_the_memory_win():
    """Compared at the *same* N, so it is a claim about the algorithm."""
    flat = benchmark.measure("era-wheel30", 10**8, 1)
    seg = benchmark.measure("era-segmented", 10**8, 1)
    assert flat.correct and seg.correct
    assert flat.rss_bytes > 16 << 20  # 0.267 bytes/int of 10^8 is ~25 MB
    assert seg.rss_bytes * 2 < flat.rss_bytes


@pytest.mark.slow
def test_segmented_memory_does_not_grow_with_n():
    """A 100x bigger limit must not mean a meaningfully bigger buffer."""
    small = benchmark.measure("era-segmented", 10**7, 1)
    large = benchmark.measure("era-segmented", 10**9, 1)
    assert large.correct and large.count == PI_REFERENCE[10**9]
    assert large.rss_bytes < 3 * max(small.rss_bytes, 4 << 20)


# ---------------------------------------------------------------------------
# Prime lists, not just counts
# ---------------------------------------------------------------------------


def test_primes_below_matches_oracle(oracle):
    assert primes_below(5000) == oracle


def test_primes_below_edges():
    assert primes_below(0) == []
    assert primes_below(1) == []
    assert primes_below(2) == [2]
    assert primes_below(3) == [2, 3]
    assert primes_below(4) == [2, 3]


def test_iter_primes_matches_primes_below():
    for limit in (0, 1, 5, 6, 7, 30, 31, 1000, 100_000):
        assert list(iter_primes(limit)) == primes_below(limit), limit


def test_primes_in_range_against_every_small_window(oracle):
    """Exhaustive: every [lo, hi] with lo < 300 and width < 120."""
    for lo in range(0, 300):
        for hi in range(lo, lo + 120):
            expected = [p for p in oracle if lo <= p <= hi]
            assert sieves.primes_in_range(lo, hi) == expected, (lo, hi)


@pytest.mark.parametrize(
    "lo,hi,expected",
    [
        (10, 5, []),          # inverted
        (0, 1, []),
        (-100, -1, []),
        (-10, 5, [2, 3, 5]),  # negative lower bound
        (2, 2, [2]),
        (5, 7, [5, 7]),
        (7, 7, [7]),
        (6, 6, []),
        (30, 30, []),         # exactly on the wheel modulus
        (29, 31, [29, 31]),
    ],
)
def test_primes_in_range_edges(lo, hi, expected):
    assert sieves.primes_in_range(lo, hi) == expected


def test_primes_in_range_does_not_sieve_from_zero():
    """The point of the exercise: a window at 10^12 without 10^12 of work."""
    assert sieves.primes_in_range(10**12, 10**12 + 100) == [
        1000000000039, 1000000000061, 1000000000063, 1000000000091
    ]


def test_primes_in_range_spans_many_segments():
    """A window wider than one segment must not lose or duplicate primes."""
    lo, hi = 10**7, 10**7 + 3 * (1 << 21)
    got = sieves.primes_in_range(lo, hi)
    assert got == sorted(set(got))
    assert len(got) == sieves.eratosthenes_simple(hi) - sieves.eratosthenes_simple(lo - 1)


def test_iter_primes_is_lazy():
    """Ten primes from a 10^9 range must not sieve 10^9 numbers."""
    stream = iter_primes(10**9)
    first_ten = [next(stream) for _ in range(10)]
    assert first_ten == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    stream.close()


# ---------------------------------------------------------------------------
# Wheel and Atkin invariants
# ---------------------------------------------------------------------------


def test_wheel_residues_are_exactly_the_coprimes():
    from math import gcd

    assert W30 == tuple(r for r in range(30) if gcd(r, 30) == 1)
    assert W60 == tuple(r for r in range(60) if gcd(r, 60) == 1)


def test_wheel30_index_identity():
    """idx(p*(30k+r)) == 8*p*k + idx(p*r) -- the identity the sieve rests on."""
    from sieves import POS30

    def idx(n: int) -> int:
        return (n // 30) * 8 + POS30[n % 30]

    for p in (7, 11, 13, 29, 101):
        for r in W30:
            for k in range(6):
                assert idx(p * (30 * k + r)) == 8 * p * k + idx(p * r)


def test_atkin_form_residues_are_disjoint_and_cover_the_wheel():
    forms = sieves.ATKIN_FORM1 | sieves.ATKIN_FORM2 | sieves.ATKIN_FORM3
    assert not (sieves.ATKIN_FORM1 & sieves.ATKIN_FORM2)
    assert not (sieves.ATKIN_FORM1 & sieves.ATKIN_FORM3)
    assert not (sieves.ATKIN_FORM2 & sieves.ATKIN_FORM3)
    # Between them the three forms partition the sixteen residues coprime to
    # 60. That is the whole reason Atkin can work on a packed wheel-60 array:
    # nothing outside those classes can ever be prime above 5.
    assert forms == set(W60)


def test_atkin_squarefree_pass_actually_matters():
    """Without the p^2 correction, 25, 49, 121 ... would survive the flips."""
    # 49 = 7^2 is representable an odd number of times by 4x^2+y^2 (49 = 4*9+13?
    # no -- the point is only that the final answer excludes it).
    assert atkin(50) == len(trial_division_primes(50))
    assert 49 not in primes_below(50)


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------


def test_measure_reports_correct_counts():
    # 10^7, not 10^6: an RSS delta under a megabyte is allocator noise, so a
    # small run cannot be asserted on.
    m = benchmark.measure("era-wheel30", 10**7, 1)
    assert m.correct and m.count == PI_REFERENCE[10**7]
    assert m.seconds > 0
    assert m.rss_bytes > 1 << 20


def test_measure_skips_rather_than_hanging():
    impl = IMPLEMENTATIONS["atkin"]
    original = impl.max_limit
    impl.max_limit = 1000
    try:
        m = benchmark.measure("atkin", 10**6, 1)
        assert m.skipped and "too slow" in m.skipped
    finally:
        impl.max_limit = original


def test_measurements_are_isolated_between_implementations():
    """A fat sieve must not inflate the next one's peak RSS."""
    benchmark.measure("era-wheel30", 10**8, 1)
    lean = benchmark.measure("era-segmented", 10**8, 1)
    assert lean.rss_bytes < 32 << 20


def test_render_table_markdown_and_text():
    results = [benchmark.measure("era-wheel30", 10**5, 1)]
    md = benchmark.render_table(results, markdown=True)
    assert md.startswith("| Implementation") and md.count("|") > 10
    txt = benchmark.render_table(results, markdown=False)
    assert "Implementation" in txt and "|" not in txt


def test_cli_list():
    proc = subprocess.run(
        [sys.executable, str(HERE / "benchmark.py"), "--list"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0
    for key in ALL_KEYS:
        assert key in proc.stdout


def test_cli_json_roundtrip():
    proc = subprocess.run(
        [sys.executable, str(HERE / "benchmark.py"),
         "--limit", "1e5", "--only", "era-simple,era-segmented", "--json"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert len(payload) == 2
    assert all(row["count"] == PI_REFERENCE[10**5] for row in payload)


def test_cli_rejects_unknown_implementation():
    proc = subprocess.run(
        [sys.executable, str(HERE / "benchmark.py"), "--only", "quantum-sieve"],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert "unknown implementation" in proc.stderr


def test_sieves_module_self_check():
    proc = subprocess.run(
        [sys.executable, str(HERE / "sieves.py")], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "all self-checks passed" in proc.stdout
