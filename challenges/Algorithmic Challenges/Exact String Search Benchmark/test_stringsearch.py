"""Tests for the eleven exact string matchers.

Every algorithm is checked against the same brute-force oracle on the same
inputs, so a bug in one shows up as a disagreement rather than as a plausible
wrong answer.

Run with:  uv run --with pytest pytest -q
Skip the slow ones with:  -m "not slow"
"""

from __future__ import annotations

import itertools
import random
import subprocess
import sys
from pathlib import Path

import pytest

from stringsearch import (
    ALGORITHMS,
    COUNTABLE,
    bitparallel_search,
    boyer_moore_adversary,
    boyer_moore_no_galil_search,
    boyer_moore_search,
    builtin_search,
    count_accesses,
    horspool_search,
    kmp_search,
    main,
    naive_adversary,
    naive_search,
    prefix_function,
    rabin_karp_adversary,
    rabin_karp_randomized_search,
    rabin_karp_search,
    shift_or_search,
    sunday_search,
    two_way_search,
    verify,
)

HERE = Path(__file__).parent
NAMES = list(ALGORITHMS)


def brute(pattern, text):
    m = len(pattern)
    if m == 0:
        return list(range(len(text) + 1))
    return [i for i in range(len(text) - m + 1) if text[i : i + m] == pattern]


def all_agree(pattern, text):
    expected = brute(pattern, text)
    for name in NAMES:
        got = list(ALGORITHMS[name](pattern, text))
        assert got == expected, f"{name} on {pattern!r} in {text!r}: {got} != {expected}"
    return expected


# ---------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_exhaustive_binary(name):
    algo = ALGORITHMS[name]
    patterns = ["".join(p) for k in range(5) for p in itertools.product("ab", repeat=k)]
    texts = ["".join(t) for k in range(9) for t in itertools.product("ab", repeat=k)]
    for pat in patterns:
        for txt in texts:
            assert list(algo(pat, txt)) == brute(pat, txt), (name, pat, txt)


@pytest.mark.parametrize("name", NAMES)
def test_exhaustive_ternary_short(name):
    algo = ALGORITHMS[name]
    patterns = ["".join(p) for k in range(4) for p in itertools.product("abc", repeat=k)]
    texts = ["".join(t) for k in range(7) for t in itertools.product("abc", repeat=k)]
    for pat in patterns:
        for txt in texts:
            assert list(algo(pat, txt)) == brute(pat, txt), (name, pat, txt)


def test_empty_pattern_matches_every_gap():
    for name in NAMES:
        assert list(ALGORITHMS[name]("", "abc")) == [0, 1, 2, 3]
        assert list(ALGORITHMS[name]("", "")) == [0]


def test_empty_text():
    for name in NAMES:
        assert list(ALGORITHMS[name]("a", "")) == []


def test_pattern_longer_than_text():
    for name in NAMES:
        assert list(ALGORITHMS[name]("abcdef", "abc")) == []


def test_pattern_equals_text():
    all_agree("abcabc", "abcabc")


def test_overlapping_matches_all_reported():
    assert all_agree("aa", "aaaa") == [0, 1, 2]
    assert all_agree("aba", "abababa") == [0, 2, 4]
    assert all_agree("aaa", "aaaaaaa") == [0, 1, 2, 3, 4]


def test_single_character_pattern():
    assert all_agree("a", "banana") == [1, 3, 5]
    assert all_agree("z", "banana") == []


def test_periodic_patterns():
    # The Two-Way periodic branch and the Galil rule both live here.
    for pat in ("abab", "aabaab", "abcabcabc", "aaaa", "abaaba"):
        for txt in (pat * 3, pat * 2 + "x" + pat, "x" + pat, pat + "x"):
            all_agree(pat, txt)


def test_random_agreement_across_alphabets():
    rng = random.Random(101)
    for _ in range(600):
        size = rng.choice((1, 2, 3, 8, 26))
        alpha = "abcdefghijklmnopqrstuvwxyz"[:size]
        txt = "".join(rng.choice(alpha) for _ in range(rng.randint(0, 90)))
        if txt and rng.random() < 0.6:
            start = rng.randrange(len(txt))
            pat = txt[start : start + rng.randint(1, 10)]
        else:
            pat = "".join(rng.choice(alpha) for _ in range(rng.randint(0, 10)))
        all_agree(pat, txt)


def test_bytes_inputs():
    rng = random.Random(103)
    for _ in range(300):
        txt = bytes(rng.randrange(256) for _ in range(rng.randint(0, 60)))
        pat = txt[3:9] if len(txt) > 9 and rng.random() < 0.5 else bytes(
            rng.randrange(256) for _ in range(rng.randint(0, 4))
        )
        all_agree(pat, txt)


def test_list_and_tuple_inputs():
    for name in NAMES:
        assert list(ALGORITHMS[name]([1, 2], [3, 1, 2, 1, 2])) == [1, 3]
        assert list(ALGORITHMS[name]((1, 2), (3, 1, 2))) == [1]


def test_unicode_beyond_latin1():
    # Forces bitparallel off its Latin-1 fast path and onto split/join.
    text = "日本語のテキストに日本語がある"
    for name in NAMES:
        assert list(ALGORITHMS[name]("日本語", text)) == [0, 9]
    emoji = "a\U0001f600b\U0001f600"
    for name in NAMES:
        assert list(ALGORITHMS[name]("\U0001f600", emoji)) == [1, 3]


def test_latin1_narrowing_preserves_indices():
    text = "caf\xe9 au lait, caf\xe9 noir"
    for name in NAMES:
        assert list(ALGORITHMS[name]("caf\xe9", text)) == [0, 14]


def test_randomized_rabin_karp_is_still_exact():
    rng = random.Random(107)
    for _ in range(200):
        txt = "".join(rng.choice("abc") for _ in range(rng.randint(0, 60)))
        pat = "".join(rng.choice("abc") for _ in range(rng.randint(0, 5)))
        assert list(rabin_karp_randomized_search(pat, txt, rng=rng)) == brute(pat, txt)


def test_rabin_karp_is_exact_under_a_tiny_modulus():
    # mod 7 collides constantly; verification is what keeps it correct.
    rng = random.Random(109)
    for _ in range(200):
        txt = "".join(rng.choice("abc") for _ in range(rng.randint(0, 60)))
        pat = "".join(rng.choice("abc") for _ in range(rng.randint(1, 5)))
        assert list(rabin_karp_search(pat, txt, base=3, mod=7)) == brute(pat, txt)


# ---------------------------------------------------------------------------
# Instrumentation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", COUNTABLE)
def test_counting_proxy_sees_every_access(name):
    """The proxy must not change behaviour, and must record something."""
    rng = random.Random(211)
    for _ in range(60):
        txt = "".join(rng.choice("ab") for _ in range(rng.randint(1, 120)))
        pat = "".join(rng.choice("ab") for _ in range(rng.randint(1, 6)))
        matches, accesses = count_accesses(ALGORITHMS[name], pat, txt)
        assert matches == brute(pat, txt)
        if len(pat) <= len(txt):
            assert accesses > 0


def test_kmp_never_exceeds_two_n_accesses():
    rng = random.Random(223)
    for _ in range(200):
        txt = "".join(rng.choice("ab") for _ in range(rng.randint(1, 300)))
        pat = "".join(rng.choice("ab") for _ in range(rng.randint(1, 8)))
        _, accesses = count_accesses(kmp_search, pat, txt)
        assert accesses <= 2 * len(txt)


def test_kmp_reads_the_text_exactly_once():
    # KMP indexes text[i] once per position and never backs up.
    txt = "ab" * 500
    _, accesses = count_accesses(kmp_search, "aba", txt)
    assert accesses == len(txt)


def test_boyer_moore_is_sublinear_on_a_large_alphabet():
    rng = random.Random(227)
    txt = "".join(chr(rng.randrange(32, 127)) for _ in range(20_000))
    pat = "".join(chr(rng.randrange(32, 127)) for _ in range(64))
    _, bm = count_accesses(boyer_moore_search, pat, txt)
    _, naive = count_accesses(naive_search, pat, txt)
    assert bm < len(txt) / 8
    assert naive >= len(txt)


def test_shift_or_cost_is_independent_of_content():
    """Bitap reads exactly n characters whatever the text says."""
    for txt in ("a" * 500, "ab" * 250, "".join(random.Random(2).choice("abc") for _ in range(500))):
        _, accesses = count_accesses(shift_or_search, "abc", txt)
        assert accesses == len(txt)


# ---------------------------------------------------------------------------
# Worst cases
# ---------------------------------------------------------------------------


def test_naive_adversary_is_quadratic_and_others_are_not():
    m, n = 24, 2400
    pat, txt = naive_adversary(m, n)
    assert brute(pat, txt) == []
    _, naive = count_accesses(naive_search, pat, txt)
    _, kmp = count_accesses(kmp_search, pat, txt)
    assert naive > 10 * n
    assert kmp <= 2 * n


def test_galil_rule_turns_the_quadratic_case_linear():
    m, n = 32, 3200
    pat, txt = boyer_moore_adversary(m, n)
    assert len(brute(pat, txt)) == n - m + 1
    matches_a, with_galil = count_accesses(boyer_moore_search, pat, txt)
    matches_b, without = count_accesses(boyer_moore_no_galil_search, pat, txt)
    assert matches_a == matches_b == brute(pat, txt)
    assert with_galil <= 2 * n            # linear
    assert without > 10 * n               # quadratic
    assert without / with_galil > m / 4


def test_rabin_karp_hash_flooding():
    m, n, mod = 32, 3200, 127
    pat, txt = rabin_karp_adversary(m, n, mod)
    expected = brute(pat, txt)
    flooded_matches, flooded = count_accesses(
        lambda p, t: rabin_karp_search(p, t, mod=mod), pat, txt
    )
    wide_matches, wide = count_accesses(rabin_karp_search, pat, txt)
    assert flooded_matches == wide_matches == expected
    assert flooded > 4 * wide  # the same text, only the modulus changed


def test_rabin_karp_adversary_rejects_impossible_moduli():
    with pytest.raises(ValueError):
        rabin_karp_adversary(4, 40, (1 << 61) - 1)


def test_adversarial_inputs_do_not_break_any_algorithm():
    for pat, txt in (
        naive_adversary(6, 120),
        boyer_moore_adversary(6, 120),
        rabin_karp_adversary(6, 120, 127),
    ):
        all_agree(pat, txt)


# ---------------------------------------------------------------------------
# Supporting pieces
# ---------------------------------------------------------------------------


def test_prefix_function_matches_definition():
    for k in range(11):
        for bits in itertools.product("ab", repeat=k):
            s = "".join(bits)
            pi = prefix_function(s)
            for i in range(len(s)):
                assert s[: pi[i]] == s[i + 1 - pi[i] : i + 1]
                assert pi[i] < i + 1
                for longer in range(pi[i] + 1, i + 1):
                    assert s[:longer] != s[i + 1 - longer : i + 1]


def test_bitparallel_falls_back_for_non_string_sequences():
    assert list(bitparallel_search([1, 2], [1, 2, 1, 2])) == [0, 2]


def test_bitparallel_short_circuits_on_absent_characters():
    assert list(bitparallel_search("xyz", "aaaaaa")) == []
    assert list(bitparallel_search(b"xyz", b"aaaaaa")) == []


def test_builtin_falls_back_for_non_string_sequences():
    assert list(builtin_search([1, 2], [1, 2, 1, 2])) == [0, 2]


def test_algorithms_are_generators_that_can_stop_early():
    """Every search yields lazily, so `next()` costs one match, not all of them."""
    text = "a" * 100_000
    for name in NAMES:
        it = ALGORITHMS[name]("aa", text)
        assert next(it) == 0
        assert next(it) == 1
        it.close()


@pytest.mark.slow
def test_large_text_agreement():
    rng = random.Random(311)
    text = "".join(rng.choice("acgt") for _ in range(200_000))
    pat = text[123_456:123_456 + 24]
    expected = brute(pat, text)
    for name in ("kmp", "boyer-moore", "horspool", "sunday", "two-way",
                 "rabin-karp", "bitparallel", "builtin"):
        assert list(ALGORITHMS[name](pat, text)) == expected


@pytest.mark.slow
def test_bitparallel_handles_a_million_matches():
    text = "a" * 500_000
    hits = list(bitparallel_search("aaaa", text))
    assert hits == list(range(500_000 - 3))


# ---------------------------------------------------------------------------
# Self-check and CLI
# ---------------------------------------------------------------------------


def test_verify_passes():
    assert verify(trials=40, verbose=False)


def test_main_demo_runs(capsys):
    assert main(["--demo"]) == 0
    out = capsys.readouterr().out
    assert "text accesses" in out
    assert "Galil killer" in out


def test_main_list(capsys):
    assert main(["--list"]) == 0
    out = capsys.readouterr().out
    for name in NAMES:
        assert name in out


def test_main_search(capsys):
    assert main(["--pattern", "ana", "--text", "banana"]) == 0
    out = capsys.readouterr().out
    assert out.count("[1, 3]") == len(NAMES)


def test_main_single_algorithm(capsys):
    assert main(["--pattern", "ana", "--text", "banana", "--algorithm", "kmp"]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and out[0].startswith("kmp")


def test_main_without_arguments_prints_help(capsys):
    assert main([]) == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_module_runs_as_a_script():
    proc = subprocess.run(
        [sys.executable, str(HERE / "stringsearch.py"), "--demo"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "naive killer" in proc.stdout


@pytest.mark.slow
def test_verify_full():
    assert verify(verbose=False)
