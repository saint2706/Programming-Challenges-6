"""Tests for the Z-array, the searches built on it, and the chain matcher.

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

from zalgorithm import (
    AhoCorasick,
    MultiZMatcher,
    ZPatternIndex,
    all_borders,
    count_distinct_substrings,
    count_tandem_repeats,
    longest_border,
    main,
    naive_z_array,
    prefix_from_z,
    prefix_function,
    prefix_occurrence_counts,
    smallest_period,
    string_from_prefix,
    string_power,
    tandem_repeat_runs,
    tandem_repeats,
    verify,
    z_array,
    z_array_counted,
    z_from_prefix,
    z_from_prefix_direct,
    z_match_lengths,
    z_search,
    z_search_concat,
    z_search_stream,
)

HERE = Path(__file__).parent


def binary_strings(max_len: int):
    for length in range(max_len + 1):
        for bits in itertools.product("ab", repeat=length):
            yield "".join(bits)


def brute_search(pattern, text):
    m = len(pattern)
    return [i for i in range(len(text) - m + 1) if text[i : i + m] == pattern]


# ---------------------------------------------------------------------------
# The core array
# ---------------------------------------------------------------------------


def test_z_array_empty():
    assert z_array("") == []
    assert naive_z_array("") == []
    assert z_array_counted("") == ([], 0)


def test_z_array_single_character():
    assert z_array("a") == [1]


def test_z_array_worked_example():
    # The canonical textbook string; every value hand-checkable.
    s = "aabxaayaab"
    assert z_array(s) == [10, 1, 0, 0, 2, 1, 0, 3, 1, 0]


def test_z_zero_is_the_length_not_a_placeholder():
    for s in ("", "a", "abcabc", "x" * 17):
        assert z_array(s)[:1] == ([len(s)] if s else [])


@pytest.mark.parametrize("s", list(binary_strings(11)))
def test_z_array_matches_definition_exhaustively(s):
    z = z_array(s)
    assert z == naive_z_array(s)
    n = len(s)
    for i in range(1, n):
        assert s[: z[i]] == s[i : i + z[i]]
        assert i + z[i] == n or s[z[i]] != s[i + z[i]]


def test_z_array_on_non_string_sequences():
    for seq in ([1, 1, 2, 1, 1, 3], (1, 1, 2, 1, 1, 3), b"aabaab"):
        assert z_array(seq) == naive_z_array(seq)
    assert z_array([1, 1, 2, 1, 1, 2]) == [6, 1, 0, 3, 1, 0]


def test_z_array_no_reserved_characters():
    # Nothing is a sentinel here, so every byte is ordinary input.
    for ch in ("#", "$", "^", "\x00", "|", "￿"):
        s = f"a{ch}b{ch}a{ch}b"
        assert z_array(s) == naive_z_array(s)


@pytest.mark.parametrize(
    "s",
    [
        "a" * 3000,
        "ab" * 1500,
        ("abacaba" * 500),
        "a" * 1500 + "b" + "a" * 1499,
        "".join(random.Random(7).choice("ab") for _ in range(3000)),
    ],
)
def test_extension_loop_is_amortised_linear(s):
    z, extensions = z_array_counted(s)
    assert z == z_array(s)
    assert extensions <= len(s)


def test_counted_and_plain_agree_on_random_input():
    rng = random.Random(11)
    for _ in range(200):
        s = "".join(rng.choice("abc") for _ in range(rng.randint(0, 40)))
        z, _ = z_array_counted(s)
        assert z == z_array(s)


# ---------------------------------------------------------------------------
# Searching
# ---------------------------------------------------------------------------


def test_search_basic():
    assert list(z_search("aba", "abababa")) == [0, 2, 4]
    assert list(z_search("aa", "aaaa")) == [0, 1, 2]
    assert list(z_search("xyz", "abcdef")) == []


def test_empty_pattern_matches_every_gap():
    # Same convention as str.find("") == 0 and re.finditer, and the one that
    # keeps text[j:j+m] valid at every reported j.
    assert list(z_search("", "abc")) == [0, 1, 2, 3]
    assert list(z_search("", "")) == [0]
    assert list(z_search_concat("", "abc")) == [0, 1, 2, 3]
    assert list(z_search_stream("", iter("abc"))) == [0, 1, 2, 3]
    assert list(z_search_stream("", iter(""))) == [0]


def test_empty_text():
    assert list(z_search("a", "")) == []
    assert z_match_lengths("a", "") == []


def test_pattern_longer_than_text():
    assert list(z_search("abcdef", "abc")) == []


def test_pattern_equals_text():
    assert list(z_search("abc", "abc")) == [0]


def test_overlapping_occurrences_are_all_reported():
    assert list(z_search("aaa", "aaaaa")) == [0, 1, 2]
    assert list(z_search("abab", "abababab")) == [0, 2, 4]


@pytest.mark.parametrize("pattern", ["", "a", "b", "aa", "ab", "aab", "aba", "abab"])
def test_search_agrees_with_brute_force_exhaustively(pattern):
    for text in binary_strings(9):
        expected = (
            brute_search(pattern, text) if pattern else list(range(len(text) + 1))
        )
        assert list(z_search(pattern, text)) == expected
        assert list(z_search_concat(pattern, text)) == expected
        assert list(z_search_stream(pattern, iter(text))) == expected


def test_concatenation_without_separator_still_correct():
    # The folklore says P + '#' + T needs a '#' outside the alphabet. It does
    # not: for i >= m, Z[i] >= m is equivalent to a match, separator or none.
    rng = random.Random(3)
    for _ in range(400):
        text = "".join(rng.choice("ab#") for _ in range(rng.randint(0, 30)))
        pat = "".join(rng.choice("ab#") for _ in range(rng.randint(1, 4)))
        expected = brute_search(pat, text)
        assert list(z_search_concat(pat, text, separator=None)) == expected
        # ...and a separator drawn from inside the alphabet is equally harmless.
        assert list(z_search_concat(pat, text, separator="#")) == expected
        assert list(z_search_concat(pat, text, separator="\x00")) == expected


def test_search_on_bytes_and_lists():
    assert list(z_search(b"ab", b"cabab")) == [1, 3]
    assert list(z_search([1, 2], [3, 1, 2, 1, 2])) == [1, 3]
    assert list(z_search((1, 2), (3, 1, 2))) == [1]


def test_match_lengths_is_the_lcp_table():
    rng = random.Random(5)
    for _ in range(300):
        text = "".join(rng.choice("ab") for _ in range(rng.randint(0, 30)))
        pat = "".join(rng.choice("ab") for _ in range(rng.randint(1, 6)))
        got = z_match_lengths(pat, text)
        for j in range(len(text)):
            k = 0
            while k < len(pat) and j + k < len(text) and pat[k] == text[j + k]:
                k += 1
            assert got[j] == k


def test_stream_search_consumes_lazily_and_once():
    """The box only moves right, so the stream is read strictly forward."""
    text = "abcabcabcabx" * 50
    reads = 0

    def counting():
        nonlocal reads
        for ch in text:
            reads += 1
            yield ch

    hits = list(z_search_stream("abcabx", counting()))
    assert hits == brute_search("abcabx", text)
    assert reads == len(text)


def test_stream_search_on_a_generator_of_unknown_length():
    def source():
        yield from "the quick brown fox jumps over the lazy dog"

    assert list(z_search_stream("the", source())) == [0, 31]


def test_stream_search_stops_when_the_pattern_cannot_fit():
    assert list(z_search_stream("abcdef", iter("abc"))) == []


def test_pattern_index_reuses_one_build():
    idx = ZPatternIndex("aba")
    assert idx.find("xxabax") == 2
    assert idx.find("nothing") == -1
    assert idx.count("abababa") == 3
    assert list(idx.search("abababa")) == [0, 2, 4]
    assert len(idx) == 3
    assert idx.borders == [1]
    assert idx.period == 2
    assert idx.match_lengths("abz") == [2, 0, 0]  # partial matches, not just hits
    assert idx.match_lengths("aba") == [3, 0, 1]
    assert list(idx.search_stream(iter("abababa"))) == [0, 2, 4]
    assert "aba" in repr(idx)


# ---------------------------------------------------------------------------
# Z <-> prefix function
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("s", list(binary_strings(11)))
def test_conversions_round_trip_exhaustively(s):
    z = z_array(s)
    pi = prefix_function(s)
    assert prefix_from_z(z) == pi
    assert z_from_prefix(pi) == z
    assert z_from_prefix_direct(pi) == z


def test_conversions_on_a_larger_alphabet():
    rng = random.Random(13)
    for _ in range(400):
        s = "".join(rng.choice("abcdef") for _ in range(rng.randint(0, 60)))
        z, pi = z_array(s), prefix_function(s)
        assert prefix_from_z(z) == pi
        assert z_from_prefix(pi) == z
        assert z_from_prefix_direct(pi) == z


def test_z_from_prefix_needs_the_min():
    # The abbreviated transfer rule z[i+j] = z[i] - j (without the min against
    # z[j]) reports 1 at index 3 of "abab", whose real Z-array ends in 0.
    pi = prefix_function("abab")
    assert z_from_prefix_direct(pi) == [4, 0, 2, 0]

    def without_min(pi):
        n = len(pi)
        z = [0] * n
        for i in range(1, n):
            if pi[i] > 0:
                z[i - pi[i] + 1] = pi[i]
        z[0] = n
        i = 1
        while i < n:
            j = i
            if z[i] > 0:
                for k in range(1, z[i]):
                    if z[i + k] > z[i] - k:
                        break
                    z[i + k] = z[i] - k
                    j = i + k
            i = j + 1
        return z

    assert without_min(pi) == [4, 0, 2, 1] != z_array("abab")


@pytest.mark.parametrize("s", list(binary_strings(10)))
def test_string_from_prefix_round_trips(s):
    pi = prefix_function(s)
    rebuilt = string_from_prefix(pi)
    assert prefix_function(rebuilt) == pi
    assert z_array(rebuilt) == z_array(s)


def test_string_from_prefix_relabels_rather_than_reproduces():
    assert string_from_prefix(prefix_function("aab")) == string_from_prefix(
        prefix_function("xxy")
    )


# ---------------------------------------------------------------------------
# Borders, periods, prefix counts, distinct substrings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("s", list(binary_strings(11)))
def test_borders_and_period_exhaustively(s):
    n = len(s)
    expected = [b for b in range(1, n) if s[:b] == s[n - b :]]
    assert all_borders(s) == expected
    assert longest_border(s) == (expected[-1] if expected else 0)
    p = smallest_period(s)
    assert all(s[i] == s[i + p] for i in range(n - p))
    for q in range(1, p):
        assert any(s[i] != s[i + q] for i in range(n - q))


def test_period_and_power():
    assert smallest_period("abcabcabc") == 3
    assert string_power("abcabcabc") == (3, 3)
    assert smallest_period("aabaa") == 3  # weak period, does not divide 5
    assert string_power("aabaa") == (5, 1)
    assert string_power("") == (0, 0)
    assert string_power("a") == (1, 1)
    assert smallest_period("") == 0
    assert smallest_period("abcdef") == 6
    assert all_borders("") == []


@pytest.mark.parametrize("s", list(binary_strings(10)))
def test_prefix_occurrence_counts_exhaustively(s):
    n = len(s)
    counts = prefix_occurrence_counts(s)
    assert len(counts) == n + 1
    assert counts[0] == n + 1
    for length in range(1, n + 1):
        expected = sum(
            1 for i in range(n - length + 1) if s[i : i + length] == s[:length]
        )
        assert counts[length] == expected


def test_prefix_occurrence_counts_empty():
    assert prefix_occurrence_counts("") == [1]


@pytest.mark.parametrize("s", list(binary_strings(9)))
def test_count_distinct_substrings_exhaustively(s):
    n = len(s)
    expected = len({s[i:j] for i in range(n) for j in range(i + 1, n + 1)})
    assert count_distinct_substrings(s) == expected


def test_count_distinct_substrings_known_values():
    assert count_distinct_substrings("") == 0
    assert count_distinct_substrings("a") == 1
    assert count_distinct_substrings("aaa") == 3
    assert count_distinct_substrings("abc") == 6  # n(n+1)/2, all distinct


# ---------------------------------------------------------------------------
# Multi-pattern
# ---------------------------------------------------------------------------


def brute_multi(patterns, text):
    return sorted(
        (i, p)
        for p, pat in enumerate(patterns)
        for i in (brute_search(pat, text) if pat else range(len(text) + 1))
    )


def test_chain_count_is_the_number_of_maximal_patterns():
    mz = MultiZMatcher(["a", "ab", "abc", "abcd"])
    assert mz.chain_count == 1
    assert mz.chains == [["a", "ab", "abc", "abcd"]]

    mz = MultiZMatcher(["cat", "dog", "bird"])
    assert mz.chain_count == 3

    mz = MultiZMatcher(["a", "ab", "ac"])
    assert mz.chain_count == 2
    assert sorted(len(c) for c in mz.chains) == [1, 2]


def test_chain_count_matches_the_antichain_bound():
    rng = random.Random(17)
    for _ in range(300):
        pats = [
            "".join(rng.choice("ab") for _ in range(rng.randint(1, 5)))
            for _ in range(rng.randint(1, 8))
        ]
        distinct = set(pats)
        maximal = {
            p for p in distinct if not any(q != p and q.startswith(p) for q in distinct)
        }
        assert MultiZMatcher(pats).chain_count == len(maximal)


def test_every_pattern_lands_in_exactly_one_chain():
    rng = random.Random(19)
    for _ in range(200):
        pats = [
            "".join(rng.choice("ab") for _ in range(rng.randint(1, 5)))
            for _ in range(rng.randint(1, 8))
        ]
        flat = [p for chain in MultiZMatcher(pats).chains for p in chain]
        assert sorted(flat) == sorted(set(pats))


def test_multi_matches_brute_force_and_aho_corasick():
    rng = random.Random(23)
    for _ in range(500):
        text = "".join(rng.choice("abc") for _ in range(rng.randint(0, 40)))
        pats = [
            "".join(rng.choice("abc") for _ in range(rng.randint(1, 4)))
            for _ in range(rng.randint(1, 6))
        ]
        expected = brute_multi(pats, text)
        assert (
            sorted((pos, i) for i, pos in MultiZMatcher(pats).search(text)) == expected
        )
        assert sorted((pos, i) for i, pos in AhoCorasick(pats).search(text)) == expected


def test_finditer_is_search_in_text_order():
    rng = random.Random(29)
    for _ in range(200):
        text = "".join(rng.choice("abc") for _ in range(rng.randint(0, 40)))
        pats = [
            "".join(rng.choice("abc") for _ in range(rng.randint(1, 4)))
            for _ in range(rng.randint(1, 6))
        ]
        mz = MultiZMatcher(pats)
        ordered = [(pos, i) for i, pos in mz.finditer(text)]
        assert ordered == sorted((pos, i) for i, pos in mz.search(text))
        assert ordered == sorted(ordered)


def test_multi_with_duplicate_patterns_reports_each_index():
    mz = MultiZMatcher(["ab", "ab", "b"])
    got = sorted((pos, i) for i, pos in mz.search("abab"))
    assert got == [(0, 0), (0, 1), (1, 2), (2, 0), (2, 1), (3, 2)]
    assert mz.chain_count == 2


def test_multi_with_empty_pattern():
    mz = MultiZMatcher(["", "ab"])
    got = sorted((pos, i) for i, pos in mz.search("ab"))
    assert got == [(0, 0), (0, 1), (1, 0), (2, 0)]
    assert mz.count("ab") == 4
    assert [(p, i) for i, p in mz.finditer("ab")] == got


def test_multi_with_only_empty_patterns():
    mz = MultiZMatcher(["", ""])
    assert mz.chain_count == 0
    assert sorted((pos, i) for i, pos in mz.search("ab")) == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
    ]


def test_multi_with_no_patterns():
    mz = MultiZMatcher([])
    assert mz.chain_count == 0
    assert list(mz.search("abc")) == []
    assert list(mz.finditer("abc")) == []
    assert list(AhoCorasick([]).search("abc")) == []


def test_multi_on_non_string_sequences():
    mz = MultiZMatcher([[1, 2], [1, 2, 3]])
    assert sorted((pos, i) for i, pos in mz.search([0, 1, 2, 3, 1, 2])) == [
        (1, 0),
        (1, 1),
        (4, 0),
    ]
    assert mz.chain_count == 1


def test_multi_pattern_longer_than_text():
    mz = MultiZMatcher(["abcdef", "ab"])
    assert sorted((pos, i) for i, pos in mz.search("ab")) == [(0, 1)]


def test_aho_corasick_deep_fail_links():
    # The classic worked example; "hers" needs a fail link through "he".
    ac = AhoCorasick(["he", "she", "his", "hers"])
    got = sorted((pos, i) for i, pos in ac.search("ushers"))
    assert got == [(1, 1), (2, 0), (2, 3)]


def test_aho_corasick_agrees_on_prefix_heavy_dictionaries():
    pats = ["a", "aa", "aaa", "aaaa"]
    text = "aaaaaa"
    expected = brute_multi(pats, text)
    assert sorted((pos, i) for i, pos in AhoCorasick(pats).search(text)) == expected
    assert sorted((pos, i) for i, pos in MultiZMatcher(pats).search(text)) == expected


# ---------------------------------------------------------------------------
# Tandem repeats
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("s", list(binary_strings(11)))
def test_tandem_repeats_exhaustively(s):
    n = len(s)
    expected = sorted(
        (i, w)
        for w in range(1, n // 2 + 1)
        for i in range(n - 2 * w + 1)
        if s[i : i + w] == s[i + w : i + 2 * w]
    )
    assert tandem_repeats(s) == expected
    assert count_tandem_repeats(s) == len(expected)


def test_tandem_repeat_runs_are_contiguous_and_valid():
    rng = random.Random(31)
    for _ in range(200):
        s = "".join(rng.choice("ab") for _ in range(rng.randint(0, 40)))
        for first, count, period in tandem_repeat_runs(s):
            assert count >= 1
            for d in range(count):
                start = first + d
                assert start >= 0 and start + 2 * period <= len(s)
                assert (
                    s[start : start + period] == s[start + period : start + 2 * period]
                )


def test_tandem_repeats_on_a_run_is_the_quadratic_count():
    for n in range(2, 30):
        s = "a" * n
        assert count_tandem_repeats(s) == (n // 2) * (n - n // 2)


def test_tandem_repeats_edge_cases():
    assert tandem_repeat_runs("") == []
    assert tandem_repeat_runs("a") == []
    assert tandem_repeat_runs("ab") == []
    assert tandem_repeats("aa") == [(0, 1)]
    assert tandem_repeats("abcabc") == [(0, 3)]


def test_tandem_repeats_on_non_strings():
    assert tandem_repeats([1, 2, 1, 2]) == [(0, 2)]


@pytest.mark.slow
def test_tandem_repeats_scale():
    s = "a" * 4000
    runs = tandem_repeat_runs(s)
    assert sum(c for _, c, _ in runs) == 2000 * 2000
    assert len(runs) < 40 * len(s)  # O(n log n), not O(n^2)


# ---------------------------------------------------------------------------
# Unicode
# ---------------------------------------------------------------------------


def test_astral_plane_characters_are_ordinary():
    text = "a\U0001f600b\U0001f600a\U0001f600b"
    assert list(z_search("\U0001f600b", text)) == [1, 5]
    assert z_array(text) == naive_z_array(text)


def test_combining_marks_are_codepoints_not_characters():
    # A caller who wants grapheme semantics passes a list of clusters; the
    # sequence-generic core does the rest.
    text = "éxé"
    assert list(z_search("é", text)) == [0, 3]
    clusters = ["é", "x", "é"]
    assert list(z_search(["é"], clusters)) == [0, 2]


# ---------------------------------------------------------------------------
# Self-check and CLI
# ---------------------------------------------------------------------------


def test_verify_passes():
    assert verify(trials=60, verbose=False)


def test_main_demo_runs(capsys):
    assert main(["--demo"]) == 0
    out = capsys.readouterr().out
    assert "tandem repeats" in out
    assert "scan(s)" in out


def test_main_search(capsys):
    assert main(["aba", "abababa"]) == 0
    assert "3 occurrence(s)" in capsys.readouterr().out


def test_main_multi(capsys):
    assert main(["--multi", "he", "she", "hers", "--", "ushers"]) == 0
    out = capsys.readouterr().out
    assert "3 patterns -> 2 Z-scan(s)" in out


def test_main_z_only(capsys):
    assert main(["--z", "aabaab"]) == 0
    assert "[6, 1, 0, 3, 1, 0]" in capsys.readouterr().out


def test_main_no_arguments_prints_help(capsys):
    assert main([]) == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_module_runs_as_a_script():
    proc = subprocess.run(
        [sys.executable, str(HERE / "zalgorithm.py"), "--demo"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "chains" in proc.stdout


@pytest.mark.slow
def test_verify_full():
    assert verify(verbose=False)
