"""Tests for Manacher's algorithm and the palindromic structures around it.

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

from palindromes import (
    Eertree,
    PalindromeIndex,
    all_maximal_palindromes,
    brute_force_longest_palindrome_span,
    count_distinct_palindromes,
    count_palindromic_substrings,
    distinct_palindromes,
    dp_longest_palindrome_span,
    graphemes,
    longest_palindrome,
    longest_palindrome_span,
    longest_palindromic_prefix,
    longest_palindromic_suffix,
    main,
    manacher_odd_even,
    min_palindromic_partition,
    naive_longest_palindrome_span,
    palindrome_radii,
    palindromic_partition,
    relaxed_view,
    shortest_palindrome_by_prepending,
    verify,
)

HERE = Path(__file__).parent

SPAN_METHODS = [
    longest_palindrome_span,
    naive_longest_palindrome_span,
    dp_longest_palindrome_span,
    brute_force_longest_palindrome_span,
]


def all_binary(max_len: int):
    """Every string over {a, b} up to a length. The densest palindrome structure."""
    for length in range(max_len + 1):
        for bits in itertools.product("ab", repeat=length):
            yield "".join(bits)


def random_strings(count: int, seed: int, max_len: int = 30):
    rng = random.Random(seed)
    for _ in range(count):
        alphabet = rng.choice(["a", "ab", "abc", "abcdefghij"])
        yield "".join(rng.choice(alphabet) for _ in range(rng.randint(0, max_len)))


def brute_spans(s: str) -> set[tuple[int, int]]:
    return {(i, j) for i in range(len(s)) for j in range(i + 1, len(s) + 1)
            if s[i:j] == s[i:j][::-1]}


# ---------------------------------------------------------------------------
# The radii arrays are what everything else reads
# ---------------------------------------------------------------------------


def test_d1_d2_match_their_definitions():
    for s in itertools.chain(all_binary(9), random_strings(200, seed=1)):
        d1, d2 = manacher_odd_even(s)
        n = len(s)
        for i in range(n):
            expected = max(r for r in range(1, n + 1)
                           if i - r + 1 >= 0 and i + r <= n
                           and s[i - r + 1:i + r] == s[i - r + 1:i + r][::-1])
            assert d1[i] == expected, (s, i)
            even = 0
            while i - even - 1 >= 0 and i + even + 1 <= n and s[i - even - 1] == s[i + even]:
                candidate = s[i - even - 1:i + even + 1]
                if candidate != candidate[::-1]:
                    break
                even += 1
            assert d2[i] == even, (s, i)


def test_d1_is_at_least_one_everywhere():
    """A single character is always a palindrome, so no radius can be zero."""
    for s in random_strings(200, seed=2):
        d1, _ = manacher_odd_even(s)
        assert all(r >= 1 for r in d1)


def test_d2_starts_at_zero():
    for s in random_strings(100, seed=3):
        if s:
            _, d2 = manacher_odd_even(s)
            assert d2[0] == 0


def test_radii_stay_inside_the_string():
    for s in itertools.chain(all_binary(8), random_strings(200, seed=4)):
        n = len(s)
        d1, d2 = manacher_odd_even(s)
        for i in range(n):
            assert 0 <= i - d1[i] + 1 and i + d1[i] <= n
            assert 0 <= i - d2[i] and i + d2[i] <= n


def test_radii_array_matches_d1_d2():
    for s in random_strings(200, seed=5):
        d1, d2 = manacher_odd_even(s)
        rad = palindrome_radii(s)
        assert len(rad) == 2 * len(s) + 1
        for i in range(len(s)):
            assert rad[2 * i] == 2 * d2[i]
            assert rad[2 * i + 1] == 2 * d1[i] - 1
        assert rad[-1] == 0


def test_radii_recover_the_palindrome_at_every_centre():
    for s in itertools.chain(all_binary(8), random_strings(100, seed=6)):
        rad = palindrome_radii(s)
        for c, r in enumerate(rad):
            chunk = s[(c - r) // 2:(c + r) // 2]
            assert chunk == chunk[::-1], (s, c)
            assert len(chunk) == r


# ---------------------------------------------------------------------------
# Linearity: the claim the whole algorithm rests on
# ---------------------------------------------------------------------------


def instrumented_manacher(s):
    """A copy of the d1/d2 loops that counts brute-force expansion steps.

    Kept in the test file rather than in the module so the production loops
    carry no instrumentation. It has to produce identical output to be a
    valid witness, which is asserted below.
    """
    n = len(s)
    d1, d2 = [0] * n, [0] * n
    steps = 0
    l, r = 0, -1
    for i in range(n):
        k = 1 if i > r else min(d1[l + r - i], r - i + 1)
        while i - k >= 0 and i + k < n and s[i - k] == s[i + k]:
            k += 1
            steps += 1
        d1[i] = k
        if i + k - 1 > r:
            l, r = i - k + 1, i + k - 1
    l, r = 0, -1
    for i in range(n):
        k = 0 if i > r else min(d2[l + r - i + 1], r - i + 1)
        while i - k - 1 >= 0 and i + k < n and s[i - k - 1] == s[i + k]:
            k += 1
            steps += 1
        d2[i] = k
        if i + k - 1 > r:
            l, r = i - k, i + k - 1
    return d1, d2, steps


def test_expansion_work_is_linear_even_on_the_worst_inputs():
    """Total inner-loop steps <= 2n: the amortisation argument, measured.

    Each expansion pushes the right boundary `r` one place right, and `r`
    never decreases and never passes n. So each of the two loops does at most
    n expansions no matter what the string is -- which is why the nested loop
    is O(n) and not O(n^2).
    """
    cases = [
        "a" * 2000,                       # every centre expands maximally
        "ab" * 1000,
        ("a" * 50 + "b") * 40,
        "abacaba" * 300,
        "".join(random.Random(7).choice("ab") for _ in range(2000)),
        "x" * 1000 + "y" + "x" * 1000,    # one giant palindrome
    ]
    for s in cases:
        d1, d2, steps = instrumented_manacher(s)
        assert (d1, d2) == manacher_odd_even(s)
        assert steps <= 2 * len(s), (s[:20], steps, len(s))


def test_naive_really_is_quadratic_where_manacher_is_not():
    """The contrast the brief asks for, as a step count rather than a clock."""
    n = 600
    worst = "a" * n
    naive_steps = 0
    for centre in range(2 * n - 1):
        i, j = centre // 2, centre // 2 + centre % 2
        while i >= 0 and j < n and worst[i] == worst[j]:
            i -= 1
            j += 1
            naive_steps += 1
    _d1, _d2, manacher_steps = instrumented_manacher(worst)
    assert naive_steps > n * n // 4          # quadratic
    assert manacher_steps <= 2 * n           # linear
    assert naive_steps > 100 * manacher_steps


# ---------------------------------------------------------------------------
# The longest palindrome itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", SPAN_METHODS)
def test_known_answers(method):
    cases = {
        "forgeeksskeegfor": "geeksskeeg",
        "babad": "bab",
        "cbbd": "bb",
        "racecar": "racecar",
        "abacaba": "abacaba",
        "a": "a",
        "ac": "a",
        "": "",
        "aaaa": "aaaa",
        "abcde": "a",
        "bananas": "anana",
    }
    for text, expected in cases.items():
        start, end = method(text)
        assert text[start:end] == expected, (text, method.__name__)


@pytest.mark.parametrize("method", SPAN_METHODS[:3])
def test_all_methods_agree_with_the_oracle(method):
    for s in itertools.chain(all_binary(10), random_strings(300, seed=8)):
        expected = brute_force_longest_palindrome_span(s)
        got = method(s)
        assert got[1] - got[0] == expected[1] - expected[0], (s, method.__name__)
        assert s[got[0]:got[1]] == s[got[0]:got[1]][::-1]


def test_ties_are_broken_leftmost():
    """"abaxyzyx" has two length-3... no: use a case with two equal-length ones."""
    assert longest_palindrome("aabb") == "aa"
    assert longest_palindrome("abba" + "cddc") == "abba"
    assert longest_palindrome("xyzzyxabccba") == "xyzzyx"
    # And the naive baseline must break ties the same way.
    for s in ["aabb", "abbacddc", "aba" + "cdc"]:
        assert longest_palindrome_span(s) == naive_longest_palindrome_span(s)


def test_empty_string_yields_the_empty_palindrome():
    assert longest_palindrome("") == ""
    assert longest_palindrome_span("") == (0, 0)
    assert count_palindromic_substrings("") == 0
    assert count_distinct_palindromes("") == 0
    assert min_palindromic_partition("") == 0
    assert palindromic_partition("") == []
    assert list(all_maximal_palindromes("")) == []


def test_longest_palindrome_returns_the_input_type():
    assert isinstance(longest_palindrome("aba"), str)
    assert isinstance(longest_palindrome(["a", "b", "a"]), list)
    assert isinstance(longest_palindrome(("a", "b", "a")), tuple)
    assert isinstance(longest_palindrome(b"aba"), bytes)


def test_non_string_sequences_work():
    """No transformed string means no requirement that the input be text."""
    assert longest_palindrome([1, 2, 3, 2, 1, 9]) == [1, 2, 3, 2, 1]
    assert longest_palindrome((1, 2, 2, 1)) == (1, 2, 2, 1)
    assert longest_palindrome([None, True, None]) == [None, True, None]
    assert count_palindromic_substrings([1, 1, 1]) == 6


# ---------------------------------------------------------------------------
# The characters everyone else's implementation breaks on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sentinel", ["#", "$", "^", "\x00", "|"])
def test_separator_and_sentinel_characters_are_ordinary_input(sentinel):
    """No transformed string and no boundary guards, so no reserved characters.

    The "#"-interleaving trick is in fact safe even when "#" is in the input,
    because Manacher only ever compares equal-parity positions. The unsafe
    variant is the one that pads with "$" and "^" to skip bounds checks. This
    module has neither, so all of these are just characters.
    """
    for text in [sentinel, sentinel * 5, f"a{sentinel}a", f"{sentinel}a{sentinel}",
                 f"ab{sentinel}{sentinel}ba", f"{sentinel}#$^{sentinel}"]:
        expected = brute_force_longest_palindrome_span(text)
        got = longest_palindrome_span(text)
        assert got[1] - got[0] == expected[1] - expected[0], text
        assert count_palindromic_substrings(text) == len(brute_spans(text))


def test_a_string_of_only_separators():
    text = "#####"
    assert longest_palindrome(text) == "#####"
    assert count_distinct_palindromes(text) == 5


def test_astral_plane_characters():
    text = "\U0001f600\U0001f601\U0001f600"
    assert longest_palindrome(text) == text
    # One emoji is one character in Python 3, so the distinct palindromes are
    # the two single emoji and the whole string.
    assert count_distinct_palindromes(text) == 3


def test_combining_marks_need_grapheme_clusters():
    """Reversing codepoints detaches an accent; reversing clusters does not."""
    text = "ab́a"  # a, b + combining acute, a
    assert len(text) == 4 and len(graphemes(text)) == 3
    # Codepoint-wise the string is not a palindrome (the accent is in the way).
    assert longest_palindrome(text) != text
    # Cluster-wise it is.
    assert "".join(longest_palindrome(graphemes(text))) == text


def test_graphemes_keep_zwj_sequences_and_flags_whole():
    family = "\U0001f468‍\U0001f469‍\U0001f467"
    assert len(graphemes(family)) == 1
    assert graphemes("\U0001f1ec\U0001f1e7") == ["\U0001f1ec\U0001f1e7"]
    assert graphemes("a\r\nb") == ["a", "\r\n", "b"]
    assert graphemes("") == []


def test_relaxed_view_finds_the_classic_phrases():
    for phrase in [
        "A man, a plan, a canal: Panama",
        "Was it a car or a cat I saw?",
        "No 'x' in Nixon",
        "Madam, I'm Adam.",
    ]:
        units, _indices = relaxed_view(phrase)
        assert "".join(longest_palindrome(units)) == "".join(units), phrase


def test_relaxed_view_index_map_points_back_at_the_original():
    text = "A man, a plan, a canal: Panama"
    units, indices = relaxed_view(text)
    assert len(units) == len(indices) == 21
    assert "".join(units) == "amanaplanacanalpanama"
    for k, unit in enumerate(units):
        assert text[indices[k]].casefold() == unit


def test_relaxed_view_on_input_with_no_alphanumerics():
    units, indices = relaxed_view("!?? ...")
    assert units == [] and indices == []
    assert longest_palindrome(units) == []


def test_relaxed_view_casefolds_and_normalises():
    units, _ = relaxed_view("Café")
    assert units == ["c", "a", "f", "é"]


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------


def test_occurrence_count_matches_enumeration():
    for s in itertools.chain(all_binary(9), random_strings(200, seed=9)):
        assert count_palindromic_substrings(s) == len(brute_spans(s))


def test_occurrence_count_on_a_run_of_equal_characters():
    """n equal characters give exactly n(n+1)/2 palindromic substrings."""
    for n in range(0, 60):
        assert count_palindromic_substrings("a" * n) == n * (n + 1) // 2


def test_distinct_count_matches_enumeration():
    for s in itertools.chain(all_binary(10), random_strings(200, seed=10)):
        expected = {s[i:j] for i, j in brute_spans(s)}
        assert count_distinct_palindromes(s) == len(expected)
        assert set(distinct_palindromes(s)) == expected


def test_distinct_count_never_exceeds_the_length():
    """Droubay-Justin-Pirillo: a string of length n has at most n distinct
    palindromic substrings, with equality exactly for the "rich" strings."""
    for s in itertools.chain(all_binary(11), random_strings(300, seed=11)):
        assert count_distinct_palindromes(s) <= len(s)


def test_rich_strings_hit_the_bound():
    """These attain the n-distinct-palindromes maximum."""
    for s in ["a", "ab", "aba", "abacaba", "aabb", "a" * 20]:
        assert count_distinct_palindromes(s) == len(s), s


def test_all_maximal_palindromes_are_palindromes_and_cover_everything():
    """The n maximal cores contain every palindromic substring as a nested one."""
    for s in itertools.chain(all_binary(8), random_strings(100, seed=12)):
        maximal = list(all_maximal_palindromes(s))
        for i, j in maximal:
            assert s[i:j] == s[i:j][::-1]
        for i, j in brute_spans(s):
            centre = i + j
            assert any(a + b == centre and a <= i and j <= b for a, b in maximal), (s, i, j)


# ---------------------------------------------------------------------------
# O(1) substring queries
# ---------------------------------------------------------------------------


def test_is_palindrome_is_exact_for_every_span():
    for s in itertools.chain(all_binary(9), random_strings(150, seed=13)):
        idx = PalindromeIndex(s)
        for i in range(len(s) + 1):
            for j in range(i, len(s) + 1):
                assert idx.is_palindrome(i, j) == (s[i:j] == s[i:j][::-1]), (s, i, j)


def test_empty_and_single_spans_are_palindromes():
    idx = PalindromeIndex("abc")
    for i in range(4):
        assert idx.is_palindrome(i, i)
    for i in range(3):
        assert idx.is_palindrome(i, i + 1)


def test_is_palindrome_rejects_out_of_range_spans():
    idx = PalindromeIndex("abc")
    for bad in [(-1, 2), (0, 4), (2, 1), (4, 4)]:
        with pytest.raises(IndexError):
            idx.is_palindrome(*bad)


def test_index_on_the_empty_string():
    idx = PalindromeIndex("")
    assert len(idx) == 0
    assert idx.longest() == ""
    assert idx.count() == 0
    assert idx.is_palindrome(0, 0)


def test_index_longest_at_centre():
    idx = PalindromeIndex("abacaba")
    assert idx.longest_at_centre(7) == (0, 7)   # centred on the middle 'c'
    assert idx.longest_at_centre(0) == (0, 0)   # before the first character


def test_index_agrees_with_the_free_functions():
    for s in random_strings(100, seed=14):
        idx = PalindromeIndex(s)
        assert idx.longest_span() == longest_palindrome_span(s)
        assert idx.count() == count_palindromic_substrings(s)
        assert idx.odd_even() == manacher_odd_even(s)


def test_index_repr():
    assert "PalindromeIndex" in repr(PalindromeIndex("aba"))


# ---------------------------------------------------------------------------
# Prefixes, suffixes, and the shortest-palindrome construction
# ---------------------------------------------------------------------------


def test_longest_palindromic_prefix_and_suffix():
    for s in itertools.chain(all_binary(10), random_strings(200, seed=15)):
        expected_prefix = max((k for k in range(len(s) + 1) if s[:k] == s[:k][::-1]),
                              default=0)
        expected_suffix = max((k for k in range(len(s) + 1) if s[len(s) - k:] ==
                               s[len(s) - k:][::-1]), default=0)
        assert longest_palindromic_prefix(s) == expected_prefix, s
        assert longest_palindromic_suffix(s) == expected_suffix, s


def test_shortest_palindrome_by_prepending_is_a_palindrome_and_minimal():
    for s in itertools.chain(all_binary(9), random_strings(150, seed=16)):
        out = shortest_palindrome_by_prepending(s)
        assert out == out[::-1], s
        assert out.endswith(s), s
        # Nothing shorter works: check every shorter candidate ending in s.
        for k in range(len(out) - len(s)):
            shorter = out[len(out) - len(s) - k:]
            if len(shorter) < len(out):
                assert shorter != shorter[::-1] or shorter == s == s[::-1], s


def test_shortest_palindrome_known_answers():
    assert shortest_palindrome_by_prepending("aacecaaa") == "aaacecaaa"
    assert shortest_palindrome_by_prepending("abcd") == "dcbabcd"
    assert shortest_palindrome_by_prepending("") == ""
    assert shortest_palindrome_by_prepending("a") == "a"
    assert shortest_palindrome_by_prepending("aba") == "aba"


# ---------------------------------------------------------------------------
# Eertree
# ---------------------------------------------------------------------------


def test_eertree_distinct_matches_enumeration():
    for s in itertools.chain(all_binary(11), random_strings(200, seed=17)):
        expected = {s[i:j] for i, j in brute_spans(s)}
        assert set(Eertree(s).distinct()) == expected, s


def test_eertree_occurrence_counts():
    for s in itertools.chain(all_binary(9), random_strings(150, seed=18)):
        expected: dict[str, int] = {}
        for i, j in brute_spans(s):
            expected[s[i:j]] = expected.get(s[i:j], 0) + 1
        assert Eertree(s).occurrences() == expected, s


def test_eertree_is_online():
    """Adding characters one at a time must match building from the whole string."""
    tree = Eertree()
    built = ""
    for c in "abacababa":
        tree.add(c)
        built += c
        assert tree.count_distinct() == count_distinct_palindromes(built)
        assert set(tree.distinct()) == set(Eertree(built).distinct())


def test_eertree_add_reports_new_palindromes():
    """True exactly when the character completes a palindrome not seen before.

    "aaa" is a bad example even though it looks like one: every prefix of a
    run introduces a longer run, so all three adds are new. "abcabc" is the
    honest case -- the second lap repeats palindromes the first lap created.
    """
    assert [Eertree().add(c) for c in "aaa"] != [True, True, False]
    tree = Eertree()
    assert [tree.add(c) for c in "abcabc"] == [True, True, True, False, False, False]
    assert tree.count_distinct() == count_distinct_palindromes("abcabc") == 3


def test_eertree_new_flag_counts_distinct_exactly():
    """Every True from add() is one new distinct palindrome, and nothing else is."""
    for s in itertools.chain(all_binary(9), random_strings(100, seed=19)):
        tree = Eertree()
        news = sum(1 for c in s if tree.add(c))
        assert news == tree.count_distinct() == count_distinct_palindromes(s), s


def test_eertree_on_empty_input():
    tree = Eertree("")
    assert tree.count_distinct() == 0
    assert list(tree.distinct()) == []
    assert tree.occurrences() == {}
    assert tree.longest_span() == (0, 0)
    assert len(tree) == 0


def test_eertree_longest_span():
    for s in random_strings(150, seed=20):
        start, end = Eertree(s).longest_span()
        assert end - start == len(longest_palindrome(s)), s
        assert s[start:end] == s[start:end][::-1]


def test_eertree_works_on_non_strings():
    tree = Eertree([1, 2, 1, 2, 1])
    assert tree.count_distinct() == 5
    assert (1, 2, 1) in set(tree.distinct())


def test_eertree_repr():
    assert "distinct=3" in repr(Eertree("aba"))


# ---------------------------------------------------------------------------
# Palindromic factorisation
# ---------------------------------------------------------------------------


def dp_min_partition(s: str) -> int:
    """O(n^3) reference: no eertree, no radii array, just the definition."""
    n = len(s)
    dp = [0] + [n + 1] * n
    for j in range(1, n + 1):
        for i in range(j):
            if s[i:j] == s[i:j][::-1]:
                dp[j] = min(dp[j], dp[i] + 1)
    return dp[n]


def test_min_partition_matches_the_reference():
    """The O(n log n) series-link algorithm against a definition-level oracle."""
    for s in itertools.chain(all_binary(12), random_strings(300, seed=21, max_len=40)):
        assert min_palindromic_partition(s) == dp_min_partition(s), s


def test_partition_reassembles_and_is_minimal():
    for s in itertools.chain(all_binary(10), random_strings(200, seed=22)):
        pieces = palindromic_partition(s)
        assert "".join(pieces) == s, s
        assert all(p == p[::-1] for p in pieces), s
        assert len(pieces) == min_palindromic_partition(s), s


def test_partition_known_answers():
    assert min_palindromic_partition("aab") == 2          # "aa" + "b"
    assert min_palindromic_partition("abacaba") == 1
    assert min_palindromic_partition("abcde") == 5
    assert min_palindromic_partition("a") == 1
    assert palindromic_partition("aab") == ["aa", "b"]


def test_partition_of_a_run_is_one_piece():
    for n in range(1, 40):
        assert min_palindromic_partition("a" * n) == 1


def test_partition_of_all_distinct_characters_is_n_pieces():
    for n in range(0, 20):
        s = "".join(chr(ord("a") + i) for i in range(n))
        assert min_palindromic_partition(s) == n


def test_partition_on_non_strings():
    assert min_palindromic_partition([1, 1, 2]) == 2
    assert palindromic_partition([1, 1, 2]) == [[1, 1], [2]]


def test_partition_series_links_on_periodic_strings():
    """Periodic strings are where the O(log n) series structure matters most.

    Fine and Wilf's lemma is why the palindromic suffixes of a position fall
    into O(log n) arithmetic progressions; strings like (ab)^k and Fibonacci
    words are the ones that actually have many suffixes to compress.
    """
    fib = ["b", "a"]
    while len(fib[-1]) < 400:
        fib.append(fib[-1] + fib[-2])
    cases = [w for w in fib if w] + ["ab" * 100, "aab" * 60, "a" * 300,
                                     "abc" * 80, ("abacaba" * 30)]
    for s in cases:
        assert min_palindromic_partition(s) == dp_min_partition(s), s[:30]


# ---------------------------------------------------------------------------
# Scale
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_a_million_equal_characters():
    """The worst case for the naive method; linear for Manacher."""
    n = 1_000_000
    s = "a" * n
    assert longest_palindrome_span(s) == (0, n)
    assert count_palindromic_substrings(s) == n * (n + 1) // 2


@pytest.mark.slow
def test_a_million_random_characters():
    s = "".join(random.Random(23).choice("ab") for _ in range(1_000_000))
    start, end = longest_palindrome_span(s)
    assert s[start:end] == s[start:end][::-1]
    assert count_distinct_palindromes(s) <= len(s)


@pytest.mark.slow
def test_eertree_scales_linearly():
    for n in (100_000, 200_000):
        s = "".join(random.Random(n).choice("abc") for _ in range(n))
        assert Eertree(s).count_distinct() <= n


def test_deeply_nested_palindrome():
    """abacaba-style recursion: every prefix length is a palindrome centre."""
    s = "a"
    for c in "bcdefghij":
        s = s + c + s
    assert longest_palindrome(s) == s
    assert min_palindromic_partition(s) == 1


def test_verify_helper_passes():
    assert verify(trials=40, verbose=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_demo(capsys):
    assert main(["--demo"]) == 0
    out = capsys.readouterr().out
    assert "geeksskeeg" in out
    assert "amanaplanacanalpanama" in out


def test_cli_verify(capsys):
    assert main(["--verify"]) == 0
    assert "OK" in capsys.readouterr().out


def test_cli_on_a_string(capsys):
    assert main(["forgeeksskeegfor"]) == 0
    out = capsys.readouterr().out
    assert "'geeksskeeg'" in out
    assert "distinct palindromic substrings" in out


def test_cli_relaxed(capsys):
    assert main(["--relaxed", "A man, a plan, a canal: Panama"]) == 0
    out = capsys.readouterr().out
    assert "amanaplanacanalpanama" in out
    assert "from original index 0" in out


def test_cli_relaxed_with_no_letters(capsys):
    assert main(["--relaxed", "!!!"]) == 0
    assert "no alphanumeric" in capsys.readouterr().out


def test_cli_joins_multiple_words(capsys):
    assert main(["never", "odd", "or", "even"]) == 0
    assert "never odd or even" in capsys.readouterr().out


def test_module_runs_as_a_script():
    result = subprocess.run(
        [sys.executable, "palindromes.py", "--verify"],
        cwd=HERE, capture_output=True, text=True, check=True,
    )
    assert "OK" in result.stdout
