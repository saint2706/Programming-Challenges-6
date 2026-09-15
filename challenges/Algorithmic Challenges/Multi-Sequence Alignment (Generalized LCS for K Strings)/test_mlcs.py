from __future__ import annotations

import random

import pytest
from mlcs import (
    brute_force_mlcs,
    dominant_point_mlcs,
    is_subsequence,
    progressive_mlcs,
)


class TestIsSubsequence:
    def test_basic(self):
        assert is_subsequence("ace", "abcde")
        assert not is_subsequence("aec", "abcde")

    def test_empty_candidate_is_always_a_subsequence(self):
        assert is_subsequence("", "anything")

    def test_empty_sequence_only_matches_empty_candidate(self):
        assert is_subsequence("", "")
        assert not is_subsequence("a", "")


class TestBruteForceMLCS:
    def test_two_strings_matches_textbook_lcs(self):
        assert brute_force_mlcs(["ABCBDAB", "BDCABA"]) in ("BCBA", "BDAB", "BCAB")

    def test_no_common_subsequence(self):
        assert brute_force_mlcs(["AAA", "BBB"]) == ""

    def test_identical_sequences(self):
        assert brute_force_mlcs(["ABC", "ABC", "ABC"]) == "ABC"

    def test_single_sequence(self):
        assert brute_force_mlcs(["HELLO"]) == "HELLO"

    def test_empty_input(self):
        assert brute_force_mlcs([]) == ""

    def test_any_sequence_empty_forces_empty_result(self):
        assert brute_force_mlcs(["ABC", "", "ABC"]) == ""

    def test_result_is_a_valid_common_subsequence(self):
        seqs = ["ABCBDAB", "BDCABA", "AEDBCB", "BADCAB"]
        result = brute_force_mlcs(seqs)
        assert all(is_subsequence(result, s) for s in seqs)


class TestDominantPointMatchesBruteForce:
    @pytest.mark.parametrize("seed", range(60))
    def test_random_instances(self, seed):
        rng = random.Random(seed)
        k = rng.randint(2, 5)
        alphabet = rng.choice(["AB", "ABC", "ABCD"])
        seqs = [
            "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 8)))
            for _ in range(k)
        ]
        exact = brute_force_mlcs(seqs)
        fast = dominant_point_mlcs(seqs)
        assert len(fast) == len(exact), (seqs, exact, fast)
        assert all(is_subsequence(fast, s) for s in seqs)

    def test_empty_input(self):
        assert dominant_point_mlcs([]) == ""

    def test_any_sequence_empty_forces_empty_result(self):
        assert dominant_point_mlcs(["ABC", "", "ABC"]) == ""

    def test_no_common_character_at_all(self):
        assert dominant_point_mlcs(["AAA", "BBB", "CCC"]) == ""

    def test_single_sequence(self):
        assert dominant_point_mlcs(["HELLO"]) == "HELLO"

    def test_four_sequences(self):
        seqs = ["ABCBDAB", "BDCABA", "AEDBCB", "BADCAB"]
        exact = brute_force_mlcs(seqs)
        fast = dominant_point_mlcs(seqs)
        assert len(fast) == len(exact)
        assert all(is_subsequence(fast, s) for s in seqs)


class TestProgressiveMLCSIsAlwaysValidButNotAlwaysOptimal:
    @pytest.mark.parametrize("seed", range(60))
    def test_never_beats_the_true_optimum(self, seed):
        rng = random.Random(seed + 1000)
        k = rng.randint(2, 5)
        seqs = [
            "".join(rng.choice("ABC") for _ in range(rng.randint(0, 8)))
            for _ in range(k)
        ]
        exact = brute_force_mlcs(seqs)
        heuristic = progressive_mlcs(seqs)
        assert all(is_subsequence(heuristic, s) for s in seqs)
        assert len(heuristic) <= len(exact)

    def test_documented_suboptimal_instance(self):
        """A greedy pairwise tie-break locks in 'AA', which then fails against a string with no A's,
        even though the *other* pairwise-optimal choice ('BB') would have succeeded on all three."""
        seqs = ["AABB", "BBAA", "BBBB"]
        exact = brute_force_mlcs(seqs)
        heuristic = progressive_mlcs(seqs)
        assert exact == "BB"
        assert heuristic == ""
        assert len(heuristic) < len(exact)

    def test_matches_optimum_when_pairwise_lcs_is_unambiguous(self):
        seqs = ["ABC", "ABC", "ABC"]
        assert progressive_mlcs(seqs) == "ABC"

    def test_empty_input(self):
        assert progressive_mlcs([]) == ""
