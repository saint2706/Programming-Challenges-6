from __future__ import annotations

import math
import random

import pytest
from cuckoo_filter import (
    _SEMI_SORT_RANK,
    _SEMI_SORT_TABLE,
    BinaryFuseFilter,
    BloomFilter,
    CountingBloomFilter,
    CuckooFilter,
    SemiSortedCuckooFilter,
    XorFilter,
    _mix_with_seed,
    _splitmix64,
    _stable_hash64,
    recommended_cuckoo_params,
)
from cuckoo_filter import _BitArray as BitArray

# ============================================================================
# _BitArray: the shared packed-storage primitive every filter above sits on.
# If this has a bug, every other test below is meaningless, so it gets its
# own focused fuzz test first.
# ============================================================================


@pytest.mark.parametrize("width", [1, 2, 3, 4, 5, 7, 8, 12, 16, 28, 32])
def test_bitarray_random_round_trip(width: int) -> None:
    rng = random.Random(width * 1000 + 7)
    count = 500
    arr = BitArray(count, width)
    mask = (1 << width) - 1
    shadow = [0] * count
    for _ in range(5000):
        idx = rng.randrange(count)
        value = rng.randrange(1 << width)
        arr.set(idx, value)
        shadow[idx] = value
        # every write must leave every OTHER slot untouched
        check_idx = rng.randrange(count)
        assert arr.get(check_idx) == shadow[check_idx]
    for i in range(count):
        assert arr.get(i) == shadow[i] & mask


def test_bitarray_rejects_oversized_value() -> None:
    arr = BitArray(4, 3)
    with pytest.raises(ValueError):
        arr.set(0, 8)  # 3 bits holds [0, 7]


def test_bitarray_nbytes_matches_theoretical_minimum_order() -> None:
    # 1000 slots of 5 bits = 5000 bits = 625 bytes, rounded up
    arr = BitArray(1000, 5)
    assert arr.nbytes >= math.ceil(1000 * 5 / 8)
    assert arr.nbytes < math.ceil(1000 * 5 / 8) + 16  # small constant slack only


# ============================================================================
# Semi-sort combinatorial table: the rank/unrank scheme SemiSortedCuckooFilter
# depends on. Verified independently of the filter itself.
# ============================================================================


def test_semi_sort_table_size_matches_combinatorial_count() -> None:
    # C(16 + 4 - 1, 4) = C(19, 4) = 3876 non-decreasing 4-tuples over [0, 16)
    assert len(_SEMI_SORT_TABLE) == math.comb(19, 4) == 3876


def test_semi_sort_table_is_sorted_and_bijective_with_rank() -> None:
    assert _SEMI_SORT_TABLE == sorted(_SEMI_SORT_TABLE)
    assert len(_SEMI_SORT_RANK) == len(_SEMI_SORT_TABLE)
    for i, t in enumerate(_SEMI_SORT_TABLE):
        assert _SEMI_SORT_RANK[t] == i


def test_semi_sort_needs_12_bits_to_rank() -> None:
    assert (len(_SEMI_SORT_TABLE) - 1).bit_length() == 12


# ============================================================================
# Hash primitives: determinism (required for reproducible tests/benchmarks)
# and that reseeding actually changes the output (required for peeling
# construction retries to explore different hypergraphs).
# ============================================================================


@pytest.mark.parametrize("item", ["hello", b"bytes-item", 42, -7, 3.14, ("a", "tuple")])
def test_stable_hash64_is_deterministic(item) -> None:
    assert _stable_hash64(item) == _stable_hash64(item)


def test_stable_hash64_differs_across_items() -> None:
    hashes = {_stable_hash64(f"item-{i}") for i in range(2000)}
    assert len(hashes) == 2000  # no collisions expected in a 64-bit space at this scale


def test_mix_with_seed_changes_with_seed() -> None:
    base = _stable_hash64("fixed-item")
    mixed = {_mix_with_seed(base, seed) for seed in range(200)}
    assert len(mixed) == 200


def test_splitmix64_is_a_bijection_on_sample() -> None:
    # SplitMix64 is a bijection on the full 64-bit space; spot-check no
    # collisions on a large random sample as a sanity check on the port.
    rng = random.Random(0)
    sample = [rng.getrandbits(64) for _ in range(5000)]
    assert len({_splitmix64(x) for x in sample}) == len(sample)


# ============================================================================
# BloomFilter: no false negatives, ever; empirical FP rate in the right
# ballpark of the target.
# ============================================================================


def test_bloom_filter_never_false_negative() -> None:
    bf = BloomFilter(capacity=5000, fp_rate=0.01)
    members = [f"member-{i}" for i in range(5000)]
    for m in members:
        bf.add(m)
    assert all(m in bf for m in members)


@pytest.mark.parametrize("fp_rate", [0.1, 0.01, 0.001])
def test_bloom_filter_empirical_fp_rate_near_target(fp_rate: float) -> None:
    n = 4000
    bf = BloomFilter(capacity=n, fp_rate=fp_rate)
    for i in range(n):
        bf.add(f"member-{i}")
    trials = 20_000
    false_positives = sum(1 for i in range(trials) if f"absent-{i}" in bf)
    measured = false_positives / trials
    # Loose bound: within a factor of 3 of the target in either direction.
    # (Bloom filters can modestly overshoot near integer-rounding of k.)
    assert measured < fp_rate * 3 + 0.01


def test_bloom_filter_has_no_delete_method() -> None:
    bf = BloomFilter(100)
    assert not hasattr(bf, "delete")
    assert not hasattr(bf, "remove")


# ============================================================================
# CountingBloomFilter: Bloom's guarantees plus safe deletion of real
# members, and the documented caveat when deleting non-members.
# ============================================================================


def test_counting_bloom_never_false_negative_for_live_members() -> None:
    cbf = CountingBloomFilter(capacity=5000, fp_rate=0.01)
    members = [f"member-{i}" for i in range(5000)]
    for m in members:
        cbf.add(m)
    assert all(m in cbf for m in members)


def test_counting_bloom_delete_removes_real_member() -> None:
    rng = random.Random(1)
    cbf = CountingBloomFilter(
        capacity=2000, fp_rate=0.001
    )  # low fp_rate -> low collision odds
    members = [f"member-{i}" for i in range(2000)]
    for m in members:
        cbf.add(m)
    victim = rng.choice(members)
    cbf.remove(victim)
    assert victim not in cbf
    assert len(cbf) == len(members) - 1
    # every other member must still be present
    still_there = [m for m in members if m != victim]
    assert all(m in cbf for m in still_there)


def test_counting_bloom_uses_about_4x_plain_bloom_space() -> None:
    n, fp = 5000, 0.01
    bf = BloomFilter(n, fp)
    cbf = CountingBloomFilter(n, fp)
    ratio = cbf.nbytes / bf.nbytes
    assert 3.5 < ratio < 4.5  # 4-bit counters vs 1-bit flags, same (m, k) sizing


def test_counting_bloom_counter_does_not_go_negative() -> None:
    cbf = CountingBloomFilter(capacity=100, fp_rate=0.1)
    cbf.add("x")
    cbf.remove("x")
    cbf.remove("x")  # deleting an already-absent item must not corrupt counters
    cbf.add("x")
    assert "x" in cbf


# ============================================================================
# CuckooFilter & SemiSortedCuckooFilter: shared battery of correctness
# properties, parametrized over both classes so the space-optimized variant
# is held to exactly the same bar as the standard one.
# ============================================================================

CUCKOO_CLASSES = [CuckooFilter, SemiSortedCuckooFilter]


@pytest.mark.parametrize("cls", CUCKOO_CLASSES)
def test_cuckoo_no_false_negative_for_present_items(cls) -> None:
    n = 5000
    cf = cls(capacity=n, fp_rate=0.01, seed=42)
    items = [f"item-{i}" for i in range(n)]
    for item in items:
        assert cf.insert(item)
    assert all(item in cf for item in items)


@pytest.mark.parametrize("cls", CUCKOO_CLASSES)
def test_cuckoo_delete_then_absent(cls) -> None:
    n = 3000
    cf = cls(
        capacity=n, fp_rate=0.0001, seed=7
    )  # tiny fp_rate -> collisions vanishingly unlikely
    items = [f"item-{i}" for i in range(n)]
    for item in items:
        cf.insert(item)
    rng = random.Random(3)
    to_delete = rng.sample(items, 500)
    for item in to_delete:
        assert cf.delete(item)
    assert not any(item in cf for item in to_delete)
    remaining = [item for item in items if item not in to_delete]
    assert all(item in cf for item in remaining)
    assert len(cf) == n - 500


@pytest.mark.parametrize("cls", CUCKOO_CLASSES)
def test_cuckoo_duplicate_insert_acts_as_bounded_counter(cls) -> None:
    cf = cls(capacity=1000, fp_rate=0.01, seed=1)
    for _ in range(3):
        assert cf.insert("dup")
    assert "dup" in cf
    assert cf.delete("dup")
    assert "dup" in cf  # two copies still remain
    assert cf.delete("dup")
    assert "dup" in cf  # one copy still remains
    assert cf.delete("dup")
    assert "dup" not in cf  # all copies gone
    assert not cf.delete("dup")  # nothing left to delete


@pytest.mark.parametrize("cls", CUCKOO_CLASSES)
def test_cuckoo_delete_nonexistent_item_returns_false(cls) -> None:
    cf = cls(capacity=1000, fp_rate=0.01, seed=1)
    cf.insert("present")
    assert not cf.delete("never-inserted")
    assert "present" in cf


@pytest.mark.parametrize("cls", CUCKOO_CLASSES)
def test_cuckoo_stress_insert_delete_reinsert_never_loses_live_items(cls) -> None:
    """The core correctness property: every item currently "in" the filter
    (inserted and not yet deleted) must be found, at every checkpoint,
    across a long randomized sequence of inserts and deletes -- including
    once the table is near or past its nominal capacity and the victim-slot
    fallback is exercised.
    """
    rng = random.Random(99)
    n = 2000
    cf = cls(capacity=n, fp_rate=0.01, seed=99)
    universe = [
        f"item-{i}" for i in range(n * 2)
    ]  # 2x nominal capacity worth of candidates
    live: set[str] = set()

    for step in range(6000):
        item = rng.choice(universe)
        if rng.random() < 0.65 and item not in live:
            if cf.insert(item):
                live.add(item)
            # a reported insert failure (rare, near-full table) must not
            # have silently dropped any *other* currently-live item
        elif item in live:
            assert cf.delete(item)
            live.discard(item)

        if step % 500 == 0:
            assert all(x in cf for x in live)

    assert all(x in cf for x in live)
    assert len(cf) == len(live)


@pytest.mark.parametrize("bucket_size,expected_lf", [(2, 0.84), (4, 0.955), (8, 0.98)])
def test_cuckoo_achieves_close_to_paper_load_factor_before_failing(
    bucket_size: int, expected_lf: float
) -> None:
    """Fan et al. 2014 report ~84%/95.5%/98% achievable load factor for
    bucket sizes 2/4/8 before insertion failures become frequent. Insert
    until the *first* failure and check the achieved load factor is in the
    right ballpark (loose bound: a stress test, not a precise reproduction).
    """
    rng = random.Random(bucket_size)
    n = 20_000
    cf = CuckooFilter(
        capacity=n, bucket_size=bucket_size, fp_rate=0.01, seed=bucket_size
    )
    table_capacity = cf.num_buckets * cf.bucket_size
    # Loop relative to the *actual* table capacity, not the nominal n: power-of-2
    # bucket-count rounding can make num_buckets*bucket_size well above n.
    for i in range(int(table_capacity * 1.05)):
        if not cf.insert(f"x-{i}-{rng.random()}"):
            break
    assert (
        cf.load_factor > expected_lf - 0.15
    )  # loose: victim slot delays "true" failure slightly


def test_semi_sorted_cuckoo_uses_fewer_bytes_than_standard() -> None:
    n = 20_000
    standard = CuckooFilter(capacity=n, fp_rate=0.01, seed=1)
    semi = SemiSortedCuckooFilter(capacity=n, fp_rate=0.01, seed=1)
    assert standard.num_buckets == semi.num_buckets  # same sizing inputs
    assert semi.nbytes < standard.nbytes


def test_semi_sorted_cuckoo_rejects_bucket_size_override() -> None:
    ssf = SemiSortedCuckooFilter(capacity=100, bucket_size=8, fp_rate=0.01)
    assert ssf.bucket_size == 4  # forced regardless of the requested value


def test_semi_sorted_cuckoo_rejects_narrow_fingerprint() -> None:
    with pytest.raises(ValueError):
        SemiSortedCuckooFilter(capacity=100, fingerprint_bits=3)


@pytest.mark.parametrize("cls", CUCKOO_CLASSES)
def test_cuckoo_measured_fp_rate_near_target(cls) -> None:
    n = 8000
    fp_rate = 0.02
    cf = cls(capacity=n, fp_rate=fp_rate, seed=5)
    for i in range(n):
        cf.insert(f"item-{i}")
    trials = 20_000
    false_positives = sum(1 for i in range(trials) if f"absent-{i}" in cf)
    measured = false_positives / trials
    assert measured < fp_rate * 3 + 0.01


def test_recommended_cuckoo_params_sane() -> None:
    params = recommended_cuckoo_params(n=100_000, fp_rate=0.001)
    assert params["fingerprint_bits"] >= math.ceil(math.log2(2 * 4 / 0.001))
    assert params["num_buckets"] * params["bucket_size"] >= 100_000


# ============================================================================
# XorFilter & BinaryFuseFilter: build-once structures. No false negatives is
# an *exact* guarantee here (not probabilistic) -- the fingerprint equation
# is satisfied by construction for every peeled key -- so it's tested
# without any tolerance.
# ============================================================================

FUSE_CLASSES = [XorFilter, BinaryFuseFilter]


@pytest.mark.parametrize("cls", FUSE_CLASSES)
def test_fuse_filter_exact_no_false_negatives(cls) -> None:
    n = 10_000
    items = [f"item-{i}" for i in range(n)]
    f = cls(items, fingerprint_bits=8)
    assert all(item in f for item in items)
    assert len(f) == n


@pytest.mark.parametrize("cls", FUSE_CLASSES)
def test_fuse_filter_handles_duplicate_inputs(cls) -> None:
    items = [f"item-{i}" for i in range(1000)] * 3  # every item repeated 3x
    f = cls(items, fingerprint_bits=8)
    assert len(f) == 1000
    assert all(f"item-{i}" in f for i in range(1000))


@pytest.mark.parametrize("cls", FUSE_CLASSES)
def test_fuse_filter_empty_input(cls) -> None:
    f = cls([], fingerprint_bits=8)
    assert len(f) == 0
    assert "anything" not in f


@pytest.mark.parametrize("cls", FUSE_CLASSES)
@pytest.mark.parametrize("n", [1, 2, 3, 5, 10, 50])
def test_fuse_filter_small_n(cls, n: int) -> None:
    items = [f"item-{i}" for i in range(n)]
    f = cls(items, fingerprint_bits=8)
    assert all(item in f for item in items)
    assert len(f) == n


@pytest.mark.parametrize("cls", FUSE_CLASSES)
def test_fuse_filter_measured_fp_rate_near_2_pow_minus_k(cls) -> None:
    n = 15_000
    fingerprint_bits = 8
    items = [f"item-{i}" for i in range(n)]
    f = cls(items, fingerprint_bits=fingerprint_bits)
    trials = 40_000
    false_positives = sum(1 for i in range(trials) if f"absent-{i}" in f)
    measured = false_positives / trials
    target = 1 / (1 << fingerprint_bits)
    assert measured < target * 3 + 0.001


def test_binary_fuse_smaller_than_xor_at_realistic_n() -> None:
    # Binary Fuse's size_factor formula (verified against the reference
    # `xorfilter` implementation) is calibrated against a 1,000,000-item
    # reference scale and only approaches its 1.125 floor as n grows -- the
    # space advantage over Xor (whose overhead is a flat 1.23n + 32) is
    # asymptotic, not universal. See test below for the small-n crossover.
    for n in (20_000, 100_000):
        xf = XorFilter([f"x-{i}" for i in range(n)], fingerprint_bits=8)
        bff = BinaryFuseFilter([f"x-{i}" for i in range(n)], fingerprint_bits=8)
        assert bff.nbytes <= xf.nbytes


def test_binary_fuse_can_be_larger_than_xor_at_small_n() -> None:
    # At small n, Binary Fuse's size_factor (0.875 + 0.25*ln(1e6)/ln(n)) is
    # well above its 1.125 asymptotic floor, while Xor's flat +32-item
    # overhead is comparatively cheap -- so Binary Fuse can legitimately
    # lose the space comparison here. This is a real property of the
    # published formula, not an implementation bug (confirmed by hand
    # against the reference `xorfilter` Go implementation's constants).
    n = 2000
    xf = XorFilter([f"x-{i}" for i in range(n)], fingerprint_bits=8)
    bff = BinaryFuseFilter([f"x-{i}" for i in range(n)], fingerprint_bits=8)
    assert bff.nbytes > xf.nbytes


# ============================================================================
# Cross-family comparison: the brief's actual ask, made explicit as a test
# rather than only a benchmark table. At the same (n, fp_rate), a
# deletion-capable cuckoo filter should be meaningfully smaller than the
# deletion-capable Bloom variant (Counting Bloom), and in the same
# ballpark as -- not wildly larger than -- a plain (non-deleting) Bloom
# filter, which is the "practically better than Bloom" headline result.
# ============================================================================


def test_cuckoo_beats_counting_bloom_for_deletion_capable_structures() -> None:
    n, fp_rate = 20_000, 0.02
    cf = CuckooFilter(capacity=n, fp_rate=fp_rate, seed=1)
    cbf = CountingBloomFilter(capacity=n, fp_rate=fp_rate)
    assert cf.nbytes < cbf.nbytes


def test_cuckoo_is_within_a_small_factor_of_plain_bloom() -> None:
    n, fp_rate = 20_000, 0.02
    cf = CuckooFilter(capacity=n, fp_rate=fp_rate, seed=1)
    bf = BloomFilter(capacity=n, fp_rate=fp_rate)
    assert (
        cf.nbytes < bf.nbytes * 2.5
    )  # cuckoo pays a bounded constant-factor space premium
