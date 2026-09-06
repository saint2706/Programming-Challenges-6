"""Tests for anagram grouping.

Run with:  uv run --with pytest --with numpy pytest -q
Skip the slow ones with:  -m "not slow"
"""

from __future__ import annotations

import itertools
import random
import string
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pytest

from anagrams import (
    PRIME_TABLE,
    AnagramIndex,
    Normalizer,
    are_anagrams,
    char_value,
    graphemes,
    group_anagrams,
    group_anagrams_external,
    group_anagrams_parallel,
    key_bincount,
    key_counter,
    key_primes,
    key_sorted,
    main,
    multiset_hash,
    shard_of,
    verify,
)

HERE = Path(__file__).parent
EXACT_METHODS = ("sorted", "counter", "hash", "auto")


def canonical(groups) -> set[tuple[str, ...]]:
    return {tuple(sorted(g)) for g in groups}


def oracle(words) -> set[tuple[str, ...]]:
    """Grouping straight from the definition, O(n^2)."""
    buckets: dict[tuple, list[str]] = {}
    for w in words:
        key = tuple(sorted(Counter(unicodedata.normalize("NFC", w)).items()))
        buckets.setdefault(key, []).append(w)
    return canonical(buckets.values())


# ---------------------------------------------------------------------------
# Agreement with the oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", EXACT_METHODS)
def test_random_corpora_match_oracle(method):
    rng = random.Random(1234)
    for _ in range(300):
        alphabet = rng.choice(["ab", "abc", "acgt", string.ascii_lowercase[:10]])
        words = [
            "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 9)))
            for _ in range(rng.randint(0, 50))
        ]
        assert canonical(group_anagrams(words, method=method)) == oracle(words)


def test_prime_method_matches_oracle_on_its_alphabet():
    rng = random.Random(99)
    for _ in range(200):
        words = [
            "".join(
                rng.choice(string.ascii_lowercase) for _ in range(rng.randint(0, 7))
            )
            for _ in range(rng.randint(0, 30))
        ]
        assert canonical(group_anagrams(words, method="primes")) == oracle(words)


def test_external_matches_in_memory_across_chunk_sizes():
    rng = random.Random(7)
    words = [
        "".join(rng.choice("abcd") for _ in range(rng.randint(0, 6)))
        for _ in range(400)
    ]
    expected = oracle(words)
    # chunk_size 1 forces one spilled run per word: the worst case for the merge.
    for chunk in (1, 2, 7, 399, 400, 401, 10_000):
        assert canonical(group_anagrams_external(words, chunk_size=chunk)) == expected


def test_index_matches_grouping():
    rng = random.Random(21)
    words = [
        "".join(rng.choice("abc") for _ in range(rng.randint(0, 5))) for _ in range(300)
    ]
    assert canonical(AnagramIndex(words).groups()) == oracle(words)


def test_verify_helper_passes():
    assert verify(trials=40, verbose=False)


# ---------------------------------------------------------------------------
# Empty, degenerate and duplicate input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", EXACT_METHODS)
def test_empty_corpus(method):
    assert group_anagrams([], method=method) == []


@pytest.mark.parametrize("method", EXACT_METHODS)
def test_empty_strings_are_anagrams_of_each_other(method):
    """The empty multiset is a multiset; "" groups with "" and nothing else."""
    assert canonical(group_anagrams(["", "", "a"], method=method)) == {("", ""), ("a",)}


def test_single_word():
    assert group_anagrams(["solo"]) == [["solo"]]


def test_duplicates_are_kept_by_default():
    assert group_anagrams(["ab", "ab", "ba"]) == [["ab", "ab", "ba"]]


def test_unique_deduplicates_by_surface_form_not_by_class():
    """`unique` drops repeated *words*, not repeated anagram classes."""
    assert group_anagrams(["ab", "ab", "ba"], unique=True) == [["ab", "ba"]]


def test_generators_are_accepted():
    assert group_anagrams(w for w in ["ab", "ba"]) == [["ab", "ba"]]


def test_min_size_filters():
    words = ["ab", "ba", "cd"]
    assert group_anagrams(words, min_size=2) == [["ab", "ba"]]
    assert group_anagrams(words, min_size=3) == []


def test_min_size_must_be_positive():
    with pytest.raises(ValueError, match="min_size"):
        group_anagrams(["a"], min_size=0)


def test_unknown_method_rejected():
    with pytest.raises(ValueError, match="method must be one of"):
        group_anagrams(["a"], method="quantum")


def test_non_string_input_rejected():
    with pytest.raises(TypeError, match="expected str"):
        group_anagrams([b"bytes"])


def test_ordering_is_first_appearance():
    words = ["zz", "ab", "ba", "yy", "az"]
    assert group_anagrams(words) == [["zz"], ["ab", "ba"], ["yy"], ["az"]]


def test_sort_groups_orders_by_size_then_first_word():
    words = ["cd", "b", "a", "dc", "e"]
    assert group_anagrams(words, sort_groups=True) == [
        ["cd", "dc"],
        ["a"],
        ["b"],
        ["e"],
    ]


def test_long_word_takes_the_counter_path_and_still_agrees():
    """`auto` switches key at length 100; both sides must agree on the boundary."""
    for length in (99, 100, 101, 400):
        a = "ab" * (length // 2) + "a" * (length % 2)
        b = "".join(sorted(a, reverse=True))
        assert canonical(group_anagrams([a, b])) == canonical(
            group_anagrams([a, b], method="sorted")
        )
        assert len(group_anagrams([a, b])) == 1


# ---------------------------------------------------------------------------
# Unicode: the part that actually goes wrong in the wild
# ---------------------------------------------------------------------------


def test_utf8_byte_multisets_are_not_injective():
    """The reason `key_sorted` sorts characters and never bytes.

    "ã©" and "é£" have identical UTF-8 byte multisets ({A3, A9, C2, C3})
    and completely different character multisets. Any implementation that
    canonicalises by sorting the encoded bytes merges them.
    """
    a, b = "ã©", "é£"
    assert sorted(a.encode()) == sorted(b.encode())
    assert not are_anagrams(a, b)
    assert len(group_anagrams([a, b])) == 2
    for method in EXACT_METHODS:
        assert len(group_anagrams([a, b], method=method)) == 2


def test_nfc_merges_precomposed_and_decomposed():
    precomposed = "café"
    decomposed = "café"
    assert precomposed != decomposed
    assert group_anagrams([precomposed, decomposed]) == [[precomposed, decomposed]]
    # ...and Normalizer.exact() keeps them apart, as documented.
    assert (
        len(group_anagrams([precomposed, decomposed], normalizer=Normalizer.exact()))
        == 2
    )


def test_graphemes_keep_accents_attached_to_their_base():
    """Under NFD, "éa" and "eá" have equal codepoint multisets but are not anagrams."""
    ea, e_a = "éa", "eá"
    assert len(group_anagrams([ea, e_a], normalizer=Normalizer(form="NFD"))) == 1
    assert (
        len(
            group_anagrams([ea, e_a], normalizer=Normalizer(form="NFD", graphemes=True))
        )
        == 2
    )
    # NFC also fixes it, by never splitting the accent off in the first place.
    assert len(group_anagrams([ea, e_a])) == 2


def test_casefold_beats_lower_on_sharp_s():
    """ "ß".lower() is itself; "ß".casefold() is "ss"."""
    norm = Normalizer(casefold=True)
    assert are_anagrams("straße", "strasse", normalizer=norm)
    assert not are_anagrams("straße", "strasse")


def test_casefold_unifies_both_greek_sigmas():
    norm = Normalizer(casefold=True)
    assert are_anagrams("σς", "ςσ", normalizer=norm)
    assert are_anagrams("Σς", "σσ", normalizer=norm)


def test_phrase_normalizer_handles_the_classics():
    norm = Normalizer.phrase()
    for a, b in [
        ("Dormitory", "Dirty Room"),
        ("The eyes", "They see"),
        ("Astronomer", "Moon starer"),
        ("A gentleman", "Elegant man"),
        ("Conversation", "Voices rant on"),
        ("Slot machines", "Cash lost in 'em"),
    ]:
        assert are_anagrams(a, b, normalizer=norm), (a, b)


def test_default_normalizer_is_case_and_space_sensitive():
    """ "The eyes"/"They see" needs no normaliser (the spaces line up); the rest do."""
    assert are_anagrams("The eyes", "They see")
    for a, b in [
        ("Dormitory", "Dirty Room"),
        ("Astronomer", "Moon starer"),
        ("Slot machines", "Cash lost in 'em"),
    ]:
        assert not are_anagrams(a, b), (a, b)


def test_astral_plane_characters_survive():
    """Codepoints above the BMP are single characters in Python 3, not surrogate pairs."""
    a, b = "\U0001f600\U0001f601", "\U0001f601\U0001f600"
    assert are_anagrams(a, b)
    assert group_anagrams([a, b]) == [[a, b]]


def test_zwj_emoji_sequence_is_one_grapheme():
    family = "\U0001f468‍\U0001f469‍\U0001f467"
    assert len(graphemes(family)) == 1
    assert len(graphemes("a" + family + "b")) == 3


def test_grapheme_flags_pair_up():
    assert graphemes("\U0001f1ec\U0001f1e7") == ["\U0001f1ec\U0001f1e7"]
    # Three regional indicators: a flag, then a lone one.
    assert len(graphemes("\U0001f1ec\U0001f1e7\U0001f1ec")) == 2


def test_grapheme_crlf_stays_together():
    assert graphemes("a\r\nb") == ["a", "\r\n", "b"]


def test_graphemes_of_empty_string():
    assert graphemes("") == []


def test_nfkc_folds_compatibility_forms():
    """Fullwidth "ａ" and ASCII "a" are the same letter under NFKC, not under NFC."""
    assert are_anagrams("ａb", "ab", normalizer=Normalizer(form="NFKC"))
    assert not are_anagrams("ａb", "ab")


def test_words_containing_newlines_survive_the_external_path():
    """Pickled records are self-delimiting; a text format would need escaping."""
    words = ["a\nb", "b\na", "a\tb", "b\ta"]
    assert canonical(group_anagrams_external(words)) == oracle(words)


def test_normalizer_rejects_unknown_form():
    with pytest.raises(ValueError, match="NFC/NFD"):
        Normalizer(form="NFZ")


def test_normalizer_is_hashable_and_frozen():
    n = Normalizer(ignore=" ")
    assert {n, Normalizer(ignore=" ")} == {n}
    with pytest.raises(Exception):
        n.form = "NFD"


def test_ignore_accepts_string_set_or_predicate():
    for ignore in (" -", frozenset(" -"), lambda c: c in " -"):
        norm = Normalizer(ignore=ignore)
        assert are_anagrams("a-b c", "cba", normalizer=norm)


# ---------------------------------------------------------------------------
# Key functions in isolation
# ---------------------------------------------------------------------------


def test_key_sorted_is_canonical_over_all_permutations():
    for perm in itertools.permutations("abcdef"):
        assert key_sorted("".join(perm)) == "abcdef"


def test_key_counter_is_canonical_and_size_bounded_by_distinct_atoms():
    assert key_counter("aaabbc") == key_counter("bacaab")
    assert len(key_counter("a" * 1000)) == 1


def test_key_primes_is_exact_by_unique_factorisation():
    """Different multisets factor differently, so the products differ."""
    seen: dict[int, str] = {}
    for length in range(1, 5):
        for combo in itertools.combinations_with_replacement("abcdef", length):
            word = "".join(combo)
            k = key_primes(word)
            assert seen.setdefault(k, word) == word


def test_key_primes_rejects_letters_outside_its_table():
    with pytest.raises(ValueError, match="outside the prime table"):
        key_primes("hello world")
    with pytest.raises(ValueError, match="outside the prime table"):
        key_primes("\U0001f600")


def test_prime_table_is_a_bijection_onto_the_first_26_primes():
    assert sorted(PRIME_TABLE) == list(string.ascii_lowercase)
    assert sorted(PRIME_TABLE.values()) == [
        2,
        3,
        5,
        7,
        11,
        13,
        17,
        19,
        23,
        29,
        31,
        37,
        41,
        43,
        47,
        53,
        59,
        61,
        67,
        71,
        73,
        79,
        83,
        89,
        97,
        101,
    ]


def test_frequency_ordered_primes_beat_alphabetical_ones():
    """The whole point of the ordering: shorter keys on real text.

    Assigning 2 to "e" rather than to "a" minimises the expected bit length of
    the product, because the key length is sum(count(c) * log2(p_c)).
    """
    alphabetical = dict(zip(string.ascii_lowercase, sorted(PRIME_TABLE.values())))
    text = "the quick brown fox jumps over the lazy dog" * 20
    letters = [c for c in text if c in PRIME_TABLE]
    assert (
        key_primes(letters).bit_length()
        < key_primes(letters, alphabetical).bit_length()
    )


def test_key_bincount_matches_key_counter_grouping():
    pytest.importorskip("numpy")
    words = ["aabbc", "abcab", "abcde", "edcba", "zzz"]
    by_bincount: dict[bytes, list[str]] = {}
    for w in words:
        by_bincount.setdefault(key_bincount(w), []).append(w)
    assert canonical(by_bincount.values()) == oracle(words)


def test_key_bincount_counts_do_not_overflow_uint8():
    """A word with 300 copies of one letter would wrap a byte-wide count vector."""
    pytest.importorskip("numpy")
    assert key_bincount("a" * 300) != key_bincount("a" * 44)


def test_key_bincount_rejects_non_latin1():
    pytest.importorskip("numpy")
    with pytest.raises(UnicodeEncodeError):
        key_bincount("中")


# ---------------------------------------------------------------------------
# The multiset hash
# ---------------------------------------------------------------------------


def test_multiset_hash_is_order_independent():
    for perm in itertools.permutations("abcd"):
        assert multiset_hash("".join(perm)) == multiset_hash("abcd")


def test_multiset_hash_is_a_homomorphism():
    """H(A + B) = H(A) + H(B) mod 2^128 -- the property everything else rests on."""
    mask = (1 << 128) - 1
    rng = random.Random(5)
    for _ in range(200):
        a = "".join(rng.choice("abcdefg") for _ in range(rng.randint(0, 12)))
        b = "".join(rng.choice("abcdefg") for _ in range(rng.randint(0, 12)))
        assert multiset_hash(a + b) == (multiset_hash(a) + multiset_hash(b)) & mask


def test_multiset_hash_supports_deletion():
    mask = (1 << 128) - 1
    assert (multiset_hash("abcd") - char_value("d")) & mask == multiset_hash("abc")


def test_multiset_hash_of_empty_is_zero():
    assert multiset_hash("") == 0


def test_multiset_hash_is_deterministic_across_processes():
    """Shards computed on different machines must agree, so no PYTHONHASHSEED."""
    script = "import anagrams; print(anagrams.multiset_hash('scale'))"
    runs = {
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=HERE,
            capture_output=True,
            text=True,
            env={"PYTHONHASHSEED": str(seed), "PATH": "/usr/bin:/bin"},
            check=True,
        ).stdout.strip()
        for seed in (0, 1, 42)
    }
    assert len(runs) == 1
    assert runs.pop() == str(multiset_hash("scale"))


def test_no_hash_collisions_on_a_large_synthetic_corpus():
    """~120k distinct multisets; a birthday collision would need ~2^64 of them."""
    words = ["".join(p) for p in itertools.product("abcdefgh", repeat=6)]
    assert len({multiset_hash(w) for w in words}) == len({key_sorted(w) for w in words})


def test_hash_method_verifies_rather_than_trusts():
    """Forge a collision by monkeypatching, and check the bucket still splits."""
    import anagrams

    real = anagrams.multiset_hash
    try:
        anagrams.multiset_hash = lambda atoms: 0  # every word collides
        groups = anagrams.group_anagrams(["ab", "ba", "cd", "dc", "e"], method="hash")
    finally:
        anagrams.multiset_hash = real
    assert canonical(groups) == {("ab", "ba"), ("cd", "dc"), ("e",)}


def test_index_reports_forged_collisions_in_stats():
    import anagrams

    real = anagrams.multiset_hash
    try:
        anagrams.multiset_hash = lambda atoms: 0
        idx = anagrams.AnagramIndex(["ab", "ba", "cd"])
        assert idx.stats()["hash_collisions"] == 1
        assert sorted(idx.lookup("ab")) == ["ab", "ba"]
        assert idx.lookup("dc") == ["cd"]
    finally:
        anagrams.multiset_hash = real


# ---------------------------------------------------------------------------
# Sharding and parallelism
# ---------------------------------------------------------------------------


def test_anagrams_always_share_a_shard():
    rng = random.Random(3)
    for shards in (1, 2, 3, 7, 64):
        for _ in range(200):
            word = "".join(rng.choice("abcde") for _ in range(rng.randint(1, 8)))
            shuffled = list(word)
            rng.shuffle(shuffled)
            assert shard_of(word, shards) == shard_of("".join(shuffled), shards)


def test_shards_cover_the_range():
    words = ["".join(p) for p in itertools.product("abcdef", repeat=4)]
    assert {shard_of(w, 8) for w in words} == set(range(8))


def test_shard_count_must_be_positive():
    with pytest.raises(ValueError, match="shards must be at least 1"):
        shard_of("a", 0)


def test_parallel_matches_serial_below_the_pool_threshold():
    rng = random.Random(11)
    words = [
        "".join(rng.choice("abc") for _ in range(rng.randint(0, 5))) for _ in range(500)
    ]
    assert canonical(group_anagrams_parallel(words, workers=4)) == oracle(words)


@pytest.mark.slow
def test_parallel_matches_serial_above_the_pool_threshold():
    rng = random.Random(13)
    words = [
        "".join(rng.choice("abcd") for _ in range(rng.randint(1, 6)))
        for _ in range(60_000)
    ]
    assert canonical(group_anagrams_parallel(words, workers=2)) == oracle(words)


# ---------------------------------------------------------------------------
# AnagramIndex behaviour
# ---------------------------------------------------------------------------


def test_index_lookup_and_membership():
    idx = AnagramIndex(["listen", "silent", "enlist", "google"])
    assert sorted(idx.lookup("tinsel")) == ["enlist", "listen", "silent"]
    assert idx.lookup("nothing") == []
    assert "elgoog" in idx
    assert "nothing" not in idx
    assert len(idx) == 4


def test_index_include_self_excludes_the_query_itself():
    idx = AnagramIndex(["listen", "silent"])
    assert idx.lookup("listen", include_self=False) == ["silent"]
    assert idx.lookup("listen") == ["listen", "silent"]


def test_index_stats_on_the_empty_index():
    stats = AnagramIndex().stats()
    assert stats["words"] == 0
    assert stats["groups"] == 0
    assert stats["largest_group"] == 0
    assert stats["mean_group_size"] == 0.0


def test_index_stats_add_up():
    words = ["ab", "ba", "abc", "cd", "dc", "ef"]
    stats = AnagramIndex(words).stats()
    assert stats["words"] == 6
    assert stats["groups"] == 4
    assert stats["singletons"] == 2
    assert stats["largest_group"] == 2
    assert stats["hash_collisions"] == 0


def test_index_largest_and_incremental_add():
    idx = AnagramIndex()
    for w in ["ab", "ba", "abc"]:
        idx.add(w)
    assert idx.largest(1) == [["ab", "ba"]]
    idx.add("cab")
    assert idx.largest(2) == [["ab", "ba"], ["abc", "cab"]]


def test_index_respects_its_normalizer():
    idx = AnagramIndex(["Dormitory"], normalizer=Normalizer.phrase())
    assert idx.lookup("Dirty Room") == ["Dormitory"]
    assert AnagramIndex(["Dormitory"]).lookup("Dirty Room") == []


def test_index_repr():
    assert "2 words" in repr(AnagramIndex(["ab", "ba"]))


# ---------------------------------------------------------------------------
# External path specifics
# ---------------------------------------------------------------------------


def test_external_is_lazy():
    """Groups come out one at a time; no list of the whole corpus is built."""
    gen = group_anagrams_external(["ab", "ba", "cd"], chunk_size=1)
    assert next(gen) == ["ab", "ba"]
    gen.close()


def test_external_min_size():
    words = ["ab", "ba", "cd"]
    assert list(group_anagrams_external(words, min_size=2)) == [["ab", "ba"]]


def test_external_preserves_input_order_within_a_group():
    words = ["ba", "ab", "ab"]
    assert list(group_anagrams_external(words, chunk_size=1)) == [["ba", "ab", "ab"]]


def test_external_rejects_bad_chunk_size():
    with pytest.raises(ValueError, match="chunk_size"):
        list(group_anagrams_external(["a"], chunk_size=0))


def test_external_cleans_up_its_temp_files(tmp_path):
    list(group_anagrams_external(["ab", "ba"], chunk_size=1, tmpdir=tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_external_on_empty_input():
    assert list(group_anagrams_external([])) == []


# ---------------------------------------------------------------------------
# Mathematical invariants
# ---------------------------------------------------------------------------


def test_groups_partition_the_corpus():
    """Every word appears exactly once, across every method."""
    rng = random.Random(17)
    words = [
        "".join(rng.choice("abc") for _ in range(rng.randint(0, 5))) for _ in range(200)
    ]
    for method in EXACT_METHODS:
        groups = group_anagrams(words, method=method)
        assert Counter(w for g in groups for w in g) == Counter(words)


def test_anagram_relation_is_an_equivalence():
    """Reflexive, symmetric, transitive -- so "group" is even well defined."""
    rng = random.Random(19)
    words = [
        "".join(rng.choice("abc") for _ in range(rng.randint(0, 4))) for _ in range(40)
    ]
    for a in words:
        assert are_anagrams(a, a)
        for b in words:
            assert are_anagrams(a, b) == are_anagrams(b, a)
            if are_anagrams(a, b):
                for c in words:
                    assert are_anagrams(b, c) == are_anagrams(a, c)


def test_group_count_equals_distinct_multiset_count():
    """|groups| = |{multisets}|, the statement the whole module implements."""
    rng = random.Random(23)
    words = [
        "".join(rng.choice("abcd") for _ in range(rng.randint(0, 6)))
        for _ in range(500)
    ]
    distinct = {tuple(sorted(Counter(w).items())) for w in words}
    for method in EXACT_METHODS:
        assert len(group_anagrams(words, method=method)) == len(distinct)


def test_number_of_anagram_classes_of_fixed_length_is_a_multiset_coefficient():
    """Words of length L over an alphabet of size k form C(L+k-1, k-1) classes."""
    from math import comb

    for k, L in [(2, 5), (3, 4), (4, 3), (5, 2)]:
        alphabet = string.ascii_lowercase[:k]
        words = ["".join(p) for p in itertools.product(alphabet, repeat=L)]
        assert len(group_anagrams(words)) == comb(L + k - 1, k - 1)


def test_class_sizes_are_multinomial_coefficients():
    """A class with counts (c1..ck) contains L! / prod(ci!) words."""
    from math import factorial

    words = ["".join(p) for p in itertools.product("abc", repeat=5)]
    for group in group_anagrams(words):
        counts = Counter(group[0])
        expected = factorial(5)
        for c in counts.values():
            expected //= factorial(c)
        assert len(group) == expected


def test_class_sizes_sum_to_the_alphabet_power():
    words = ["".join(p) for p in itertools.product("abcd", repeat=4)]
    assert sum(len(g) for g in group_anagrams(words)) == 4**4


# ---------------------------------------------------------------------------
# Scale
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_half_a_million_words():
    """Grouping stays linear-ish and every method still agrees."""
    rng = random.Random(31)
    words = [
        "".join(rng.choice(string.ascii_lowercase[:12]) for _ in range(7))
        for _ in range(500_000)
    ]
    baseline = len(group_anagrams(words, method="sorted"))
    assert len(group_anagrams(words, method="hash")) == baseline
    assert len(group_anagrams(words, method="counter")) == baseline


@pytest.mark.slow
def test_very_long_words_do_not_blow_up():
    """One 200k-character word: the counter key stays at |alphabet| entries."""
    a = "ab" * 100_000
    b = "ba" * 100_000
    assert len(group_anagrams([a, b], method="counter")) == 1
    assert len(key_counter(a)) == 2


def test_pathological_all_identical_corpus():
    words = ["aaaa"] * 10_000
    groups = group_anagrams(words)
    assert len(groups) == 1 and len(groups[0]) == 10_000


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_demo(capsys):
    assert main(["--demo"]) == 0
    out = capsys.readouterr().out
    assert "listen, silent, enlist" in out


def test_cli_verify(capsys):
    assert main(["--verify"]) == 0
    assert "OK" in capsys.readouterr().out


def test_cli_phrase_mode(capsys):
    assert main(["--demo", "--phrase"]) == 0
    assert "Dormitory, Dirty Room" in capsys.readouterr().out


def test_cli_external_mode(capsys):
    assert main(["--demo", "--external", "--chunk-size", "3"]) == 0
    assert "listen, silent, enlist" in capsys.readouterr().out


def test_cli_reads_a_file(tmp_path, capsys):
    path = tmp_path / "words.txt"
    path.write_text("listen\nsilent\n\ngoogle\n", encoding="utf-8")
    assert main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "listen, silent" in out
    assert "3 words" in out  # the blank line is skipped


def test_cli_requires_input():
    with pytest.raises(SystemExit):
        main([])


def test_module_runs_as_a_script():
    result = subprocess.run(
        [sys.executable, "anagrams.py", "--verify"],
        cwd=HERE,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "OK" in result.stdout
