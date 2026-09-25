"""Cuckoo Filter: an approximate-membership structure that supports deletion.

A Bloom filter answers "have you seen x?" in O(1) space-efficient bits, but
once a bit is set it can never be safely cleared -- other items' hashes may
share it. The brief asks for a structure that supports deletion and to
compare false-positive rates against Bloom filters. That comparison only
means something once *both* sides of the "deletion" story are on the table,
so this implementation builds the full lineage:

| Structure                 | Deletion | Space (bits/key @ typical eps) | Mutability          |
| -------------------------- | -------- | ------------------------------- | -------------------- |
| `BloomFilter`               | No       | ~1.44 log2(1/eps)                | Insert-only           |
| `CountingBloomFilter`        | Yes      | ~4x plain Bloom (4-bit counters) | Insert + delete        |
| `CuckooFilter`               | Yes      | log2(1/eps) + ~3                 | Insert + delete        |
| `SemiSortedCuckooFilter`     | Yes      | log2(1/eps) + ~2                 | Insert + delete (slower)|
| `XorFilter`                  | No       | ~1.23 * log2(1/eps) (~23% over the info-theoretic bound) | Build-once |
| `BinaryFuseFilter`           | No       | ~1.13 * log2(1/eps) (~13% over the bound) | Build-once, faster to build than Xor |

The brief's literal ask (cuckoo vs Bloom, with deletion) is `CuckooFilter`
vs `BloomFilter`/`CountingBloomFilter`. Going further, the two structures
that have actually superseded cuckoo filters in the literature for the
*static* case -- when you can build once from a known set and never delete
-- are Xor filters (Graf & Lemire, 2020) and Binary Fuse filters (Graf &
Lemire, 2022), both included here: they trade cuckoo's O(1) online
insert/delete for a smaller, faster-to-query structure built by *peeling* a
3-uniform hypergraph rather than by relocating fingerprints. The semi-sorted
cuckoo variant is the paper's own within-family space optimization (Fan et
al. 2014, Section 5.4), included because it directly narrows the space gap
against Bloom filters that the brief's "compare FP rates" is really asking
about.

Every structure above shares the same false-positive mechanism: a `k`-bit
fingerprint standing in for the full key, so two distinct keys collide with
probability roughly `2^-k` (times a small constant for however many
fingerprint slots a single lookup checks). None of them ever produce a false
*negative* for a key that's still a member -- that correctness property is
verified under stress in the test suite, including the classic cuckoo-filter
gotcha: deleting a key must not remove some *other* key's fingerprint that
happens to collide with it.

All six structures share one storage primitive -- :class:`_BitArray` packing
fixed-width unsigned integers into a raw `bytearray` -- so the "bits per
key" numbers reported by the benchmark are real measured bytes, not
Python's per-object overhead.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import random
from collections import deque
from collections.abc import Callable, Hashable, Iterable

_MASK64 = (1 << 64) - 1


def _stable_hash64(item: Hashable) -> int:
    """A deterministic (not process-randomized, unlike Python's ``hash``) 64-bit hash.

    Cuckoo/Xor/Binary-Fuse filters all need a hash that's stable across runs
    (for reproducible tests and benchmarks) and, for Xor/Binary Fuse, cheaply
    re-derivable under many different seeds during construction retries.
    This is computed once per item; :func:`_mix_with_seed` (pure integer
    arithmetic, no hashing) handles the cheap re-seeding.
    """
    if isinstance(item, (bytes, bytearray)):
        data = bytes(item)
    elif isinstance(item, str):
        data = item.encode("utf-8")
    elif isinstance(item, int) and not isinstance(item, bool):
        data = item.to_bytes((item.bit_length() // 8) + 2, "little", signed=True)
    else:
        data = repr(item).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")


def _splitmix64(x: int) -> int:
    """SplitMix64 (Steele, Lea & Flood 2014): a fast avalanche mixer for a single 64-bit word.

    Used both to derive a filter's alternate-bucket / alternate-slot hash
    from a fingerprint alone (the trick that makes partial-key cuckoo
    hashing work without re-hashing the original item) and to cheaply
    re-seed an item's hash across Xor/Binary-Fuse construction retries.
    """
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


def _mix_with_seed(base_hash: int, seed: int) -> int:
    return _splitmix64(base_hash ^ _splitmix64(seed))


def _rotl64(x: int, r: int) -> int:
    return ((x << r) | (x >> (64 - r))) & _MASK64


def _next_pow2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


class _BitArray:
    """A tightly packed array of ``count`` fixed-``width``-bit unsigned integers.

    Backed by a raw ``bytearray`` so ``nbytes`` is the real, measurable
    memory cost -- unlike a Python ``list[int]``, whose per-element object
    overhead (~28 bytes in CPython) would swamp any real difference between
    e.g. an 8-bit and a 12-bit fingerprint. Each ``get``/``set`` only
    touches the handful of bytes spanned by one ``width``-bit field, so
    operations are O(1) regardless of the array's total size (a naive
    single-big-Python-int bitset would instead be O(total size) per op).
    """

    __slots__ = ("_count", "_data", "_width")

    def __init__(self, count: int, width: int) -> None:
        if width <= 0:
            raise ValueError("width must be positive")
        self._count = count
        self._width = width
        total_bits = count * width
        self._data = bytearray(
            (total_bits + 7) // 8 + 8
        )  # +8 bytes of slack for edge reads

    def get(self, index: int) -> int:
        bit_off = index * self._width
        byte_off, shift = divmod(bit_off, 8)
        nbytes = (shift + self._width + 7) // 8
        chunk = int.from_bytes(self._data[byte_off : byte_off + nbytes], "little")
        return (chunk >> shift) & ((1 << self._width) - 1)

    def set(self, index: int, value: int) -> None:
        mask_val = (1 << self._width) - 1
        if value & ~mask_val:
            raise ValueError(f"value {value} does not fit in {self._width} bits")
        bit_off = index * self._width
        byte_off, shift = divmod(bit_off, 8)
        nbytes = (shift + self._width + 7) // 8
        mask = mask_val << shift
        chunk = int.from_bytes(self._data[byte_off : byte_off + nbytes], "little")
        chunk = (chunk & ~mask) | ((value << shift) & mask)
        self._data[byte_off : byte_off + nbytes] = chunk.to_bytes(nbytes, "little")

    @property
    def nbytes(self) -> int:
        return len(self._data) - 8  # exclude the edge-read slack


# ============================================================================
# 1. Bloom filter -- the baseline. No deletion, ever: a bit set by one item
#    may be relied on by others, so clearing it on "delete" can turn a false
#    positive into a false *negative* for a completely different key.
# ============================================================================


class BloomFilter:
    """Standard Bloom filter, sized for ``capacity`` items at false-positive rate ``fp_rate``.

    Uses Kirsch & Mitzenmacher's (2006) double-hashing trick: simulating
    `k` independent hash functions as `g_i(x) = h1(x) + i*h2(x) (mod m)`
    from only two underlying hashes, which is provably as good as `k`
    independent functions for Bloom filters' purposes and avoids `k` separate
    hash computations per operation.
    """

    def __init__(self, capacity: int, fp_rate: float = 0.01) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if not (0 < fp_rate < 1):
            raise ValueError("fp_rate must be in (0, 1)")
        self.num_bits = max(
            8, math.ceil(-capacity * math.log(fp_rate) / math.log(2) ** 2)
        )
        self.num_hashes = max(1, round((self.num_bits / capacity) * math.log(2)))
        self._bits = _BitArray(self.num_bits, 1)
        self._size = 0

    def _positions(self, item: Hashable) -> Iterable[int]:
        h = _stable_hash64(item)
        h1, h2 = h & _MASK64, _splitmix64(h)
        for i in range(self.num_hashes):
            yield (h1 + i * h2) % self.num_bits

    def add(self, item: Hashable) -> None:
        for pos in self._positions(item):
            self._bits.set(pos, 1)
        self._size += 1

    def __contains__(self, item: Hashable) -> bool:
        return all(self._bits.get(pos) for pos in self._positions(item))

    def __len__(self) -> int:
        return self._size

    @property
    def nbytes(self) -> int:
        return self._bits.nbytes


# ============================================================================
# 2. Counting Bloom filter -- Bloom's own answer to deletion (Fan, Cao,
#    Almeida & Broder 1998/2000, "Summary Cache"): widen each bit to a
#    saturating counter. Costs ~4x the space of a plain Bloom filter for the
#    same (n, fp_rate); this is the quantitative "why cuckoo filters" case.
# ============================================================================


class CountingBloomFilter:
    """Bloom filter with 4-bit saturating counters instead of bits, enabling deletion.

    Deleting a key that was genuinely inserted is always safe. Deleting a
    key that was *never* inserted is not: with probability ~fp_rate it will
    still decrement counters that other, real members rely on, silently
    turning them into false negatives later. This mirrors the same caveat
    :class:`CuckooFilter` has, and is exercised directly in the test suite.
    """

    _COUNTER_BITS = 4
    _COUNTER_MAX = (1 << _COUNTER_BITS) - 1

    def __init__(self, capacity: int, fp_rate: float = 0.01) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if not (0 < fp_rate < 1):
            raise ValueError("fp_rate must be in (0, 1)")
        self.num_counters = max(
            8, math.ceil(-capacity * math.log(fp_rate) / math.log(2) ** 2)
        )
        self.num_hashes = max(1, round((self.num_counters / capacity) * math.log(2)))
        self._counters = _BitArray(self.num_counters, self._COUNTER_BITS)
        self._size = 0

    def _positions(self, item: Hashable) -> Iterable[int]:
        h = _stable_hash64(item)
        h1, h2 = h & _MASK64, _splitmix64(h)
        for i in range(self.num_hashes):
            yield (h1 + i * h2) % self.num_counters

    def add(self, item: Hashable) -> None:
        for pos in self._positions(item):
            c = self._counters.get(pos)
            if c < self._COUNTER_MAX:
                self._counters.set(pos, c + 1)
        self._size += 1

    def remove(self, item: Hashable) -> None:
        """Decrement this item's counters. Only safe if `item` was actually inserted."""
        for pos in self._positions(item):
            c = self._counters.get(pos)
            if (
                0 < c < self._COUNTER_MAX
            ):  # never decrement a saturated (overflowed) counter
                self._counters.set(pos, c - 1)
        self._size -= 1

    def __contains__(self, item: Hashable) -> bool:
        return all(self._counters.get(pos) > 0 for pos in self._positions(item))

    def __len__(self) -> int:
        return self._size

    @property
    def nbytes(self) -> int:
        return self._counters.nbytes


# ============================================================================
# 3 & 4. Cuckoo filter family (Fan, Andersen, Kaminsky & Mitzenmacher, 2014).
#
# A compact cuckoo hash table storing f-bit fingerprints instead of full
# keys. Each item has two candidate buckets, `i1` and `i2 = i1 XOR
# hash(fingerprint)` -- crucially, `i2` is a function of `i1` and the
# fingerprint *alone*, not of the original item, so an item found in one
# bucket can be relocated to its alternate without ever re-hashing the
# original key ("partial-key cuckoo hashing"). Insertion that finds both
# candidate buckets full falls back to a random walk: evict a random
# occupant, compute *its* alternate bucket the same way, and try to place it
# there, repeating up to `max_kicks` times.
# ============================================================================

_DEFAULT_LOAD_FACTOR = {1: 0.5, 2: 0.84, 4: 0.955, 8: 0.98}


class _CuckooFilterBase:
    """Shared insert/lookup/delete driver. Subclasses provide bucket storage only.

    A failed insertion (the kick chain exceeds `max_kicks`) does not corrupt
    the table or lose the item: the last displaced fingerprint is parked in
    a single-entry "victim" cache (as in the reference C++ implementation
    that accompanies the original paper) so lookups and deletes still see
    it. Only a *second* simultaneous victim -- astronomically unlikely below
    the recommended load factor -- causes insert() to actually report
    failure.
    """

    def __init__(
        self,
        capacity: int,
        bucket_size: int = 4,
        fingerprint_bits: int | None = None,
        fp_rate: float = 0.01,
        max_kicks: int = 500,
        load_factor: float | None = None,
        seed: int | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if bucket_size <= 0:
            raise ValueError("bucket_size must be positive")
        if fingerprint_bits is None:
            # f >= log2(2b/eps): a lookup checks up to 2*bucket_size fingerprint
            # slots total (both candidate buckets), so by the union bound the
            # false-positive rate is at most 2*bucket_size / 2^f.
            fingerprint_bits = max(1, math.ceil(math.log2(2 * bucket_size / fp_rate)))
        if not (1 <= fingerprint_bits <= 32):
            raise ValueError("fingerprint_bits must be in [1, 32]")
        self.bucket_size = bucket_size
        self.fingerprint_bits = fingerprint_bits
        self._fp_mask = (1 << fingerprint_bits) - 1
        self._max_kicks = max_kicks
        lf = load_factor or _DEFAULT_LOAD_FACTOR.get(bucket_size, 0.9)
        self.num_buckets = _next_pow2(max(1, math.ceil(capacity / (bucket_size * lf))))
        self._bucket_mask = self.num_buckets - 1
        self._size = 0
        self._victim: tuple[int, int] | None = None
        self._rng = random.Random(seed)

    # -- bucket-level primitives implemented by subclasses -----------------
    def _bucket_fingerprints(self, bucket_idx: int) -> list[int]:
        raise NotImplementedError

    def _bucket_write(self, bucket_idx: int, fps: list[int]) -> None:
        raise NotImplementedError

    # -- shared bucket operations, built only on the two primitives above --
    def _try_insert(self, bucket_idx: int, fp: int) -> bool:
        fps = self._bucket_fingerprints(bucket_idx)
        for i, v in enumerate(fps):
            if v == 0:
                fps[i] = fp
                self._bucket_write(bucket_idx, fps)
                return True
        return False

    def _bucket_contains(self, bucket_idx: int, fp: int) -> bool:
        return fp in self._bucket_fingerprints(bucket_idx)

    def _bucket_remove(self, bucket_idx: int, fp: int) -> bool:
        fps = self._bucket_fingerprints(bucket_idx)
        for i, v in enumerate(fps):
            if v == fp:
                fps[i] = 0
                self._bucket_write(bucket_idx, fps)
                return True
        return False

    def _bucket_swap_random(self, bucket_idx: int, fp: int) -> int:
        fps = self._bucket_fingerprints(bucket_idx)
        slot = self._rng.randrange(len(fps))
        old = fps[slot]
        fps[slot] = fp
        self._bucket_write(bucket_idx, fps)
        return old

    # -- fingerprint / index derivation -------------------------------------
    def _fingerprint_and_index1(self, item: Hashable) -> tuple[int, int]:
        h = _stable_hash64(item)
        fp = (h >> 32) & self._fp_mask
        if fp == 0:  # 0 is the empty-slot sentinel; remap the rare collision to 1
            fp = 1
        i1 = h & self._bucket_mask
        return fp, i1

    def _alt_index(self, index: int, fp: int) -> int:
        return (index ^ _splitmix64(fp)) & self._bucket_mask

    # -- public API ----------------------------------------------------------
    def insert(self, item: Hashable) -> bool:
        fp, i1 = self._fingerprint_and_index1(item)
        i2 = self._alt_index(i1, fp)
        if self._try_insert(i1, fp) or self._try_insert(i2, fp):
            self._size += 1
            return True

        i = i1 if self._rng.random() < 0.5 else i2
        cur_fp = fp
        for _ in range(self._max_kicks):
            cur_fp = self._bucket_swap_random(i, cur_fp)
            i = self._alt_index(i, cur_fp)
            if self._try_insert(i, cur_fp):
                self._size += 1
                return True

        if self._victim is None:
            self._victim = (i, cur_fp)
            self._size += 1
            return True
        return False  # table (and its one victim slot) is genuinely full

    def __contains__(self, item: Hashable) -> bool:
        fp, i1 = self._fingerprint_and_index1(item)
        i2 = self._alt_index(i1, fp)
        if self._bucket_contains(i1, fp) or self._bucket_contains(i2, fp):
            return True
        return (
            self._victim is not None
            and self._victim == (i1, fp)
            or self._victim == (i2, fp)
        )

    def delete(self, item: Hashable) -> bool:
        """Remove one copy of `item`. Only safe if `item` was actually inserted.

        Deleting an item that was never inserted can, with probability
        ~false-positive-rate, remove a *different* item's fingerprint that
        happens to collide in the same candidate bucket -- causing a false
        negative for that other item later. This is the same fundamental
        caveat as :class:`CountingBloomFilter`.
        """
        fp, i1 = self._fingerprint_and_index1(item)
        i2 = self._alt_index(i1, fp)
        if self._bucket_remove(i1, fp) or self._bucket_remove(i2, fp):
            self._size -= 1
            self._try_rehome_victim()
            return True
        if self._victim is not None and (
            self._victim == (i1, fp) or self._victim == (i2, fp)
        ):
            self._victim = None
            self._size -= 1
            return True
        return False

    def _try_rehome_victim(self) -> None:
        if self._victim is None:
            return
        vi, vfp = self._victim
        if self._try_insert(vi, vfp):
            self._victim = None

    def __len__(self) -> int:
        return self._size

    @property
    def load_factor(self) -> float:
        return self._size / (self.num_buckets * self.bucket_size)


class CuckooFilter(_CuckooFilterBase):
    """Standard cuckoo filter: each bucket slot stores a raw `fingerprint_bits`-wide fingerprint."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._table = _BitArray(
            self.num_buckets * self.bucket_size, self.fingerprint_bits
        )

    def _bucket_fingerprints(self, bucket_idx: int) -> list[int]:
        base = bucket_idx * self.bucket_size
        return [self._table.get(base + s) for s in range(self.bucket_size)]

    def _bucket_write(self, bucket_idx: int, fps: list[int]) -> None:
        base = bucket_idx * self.bucket_size
        for s, v in enumerate(fps):
            self._table.set(base + s, v)

    @property
    def nbytes(self) -> int:
        return self._table.nbytes


# -- semi-sorted bucket compression (Fan et al. 2014, Sec. 5.4) --------------
#
# A bucket's contents form a *set*, not a sequence: slot order never matters
# for membership. So instead of storing bucket_size independent fingerprints,
# split each into a 4-bit "head" and an (f-4)-bit "tail", sort the
# (head, tail) pairs, and store the sorted heads' *rank* among all
# non-decreasing 4-tuples of nibbles (computed combinatorially here, rather
# than via the paper's hardcoded lookup table, so it generalizes to any
# fingerprint width >= 4 bits) plus the tails in that same sorted order. For
# bucket_size=4, there are C(16+4-1, 4) = 3876 such tuples, needing only 12
# bits to rank instead of the naive 16 -- saving exactly 1 bit per
# fingerprint, regardless of f, at the cost of a sort + rank/unrank on every
# bucket mutation.

_SEMI_SORT_HEAD_BITS = 4
_SEMI_SORT_TABLE: list[tuple[int, int, int, int]] = list(
    itertools.combinations_with_replacement(range(1 << _SEMI_SORT_HEAD_BITS), 4)
)
_SEMI_SORT_RANK: dict[tuple[int, int, int, int], int] = {
    t: i for i, t in enumerate(_SEMI_SORT_TABLE)
}
_SEMI_SORT_CODE_BITS = max(1, (len(_SEMI_SORT_TABLE) - 1).bit_length())


class SemiSortedCuckooFilter(_CuckooFilterBase):
    """Cuckoo filter with semi-sorted (compressed) buckets; fixed at bucket_size=4.

    Saves ~1 bit/fingerprint over :class:`CuckooFilter` at the same
    fingerprint width, moving the space crossover point against Bloom
    filters to a higher false-positive rate -- but every insert/lookup/
    delete now sorts and (un)ranks the whole bucket, so it trades bucket-op
    speed for space. `fingerprint_bits` must be >= 4 (the head alone is 4
    bits; a `fingerprint_bits=4` filter has no tail at all).
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs["bucket_size"] = 4
        super().__init__(*args, **kwargs)
        if self.fingerprint_bits < _SEMI_SORT_HEAD_BITS:
            raise ValueError(f"fingerprint_bits must be >= {_SEMI_SORT_HEAD_BITS}")
        self._tail_bits = self.fingerprint_bits - _SEMI_SORT_HEAD_BITS
        self._bucket_width = _SEMI_SORT_CODE_BITS + 4 * self._tail_bits
        self._table = _BitArray(self.num_buckets, self._bucket_width)

    def _split(self, fp: int) -> tuple[int, int]:
        return fp >> self._tail_bits, fp & ((1 << self._tail_bits) - 1)

    def _bucket_fingerprints(self, bucket_idx: int) -> list[int]:
        code = self._table.get(bucket_idx)
        tail_bits_total = 4 * self._tail_bits
        tail_mask = (1 << tail_bits_total) - 1
        tails_packed = code & tail_mask
        rank = code >> tail_bits_total
        heads = _SEMI_SORT_TABLE[rank]
        tail_mask1 = (1 << self._tail_bits) - 1 if self._tail_bits else 0
        tails = []
        for _ in range(4):
            tails.append(tails_packed & tail_mask1)
            tails_packed >>= self._tail_bits
        tails.reverse()
        return [(h << self._tail_bits) | t for h, t in zip(heads, tails)]

    def _bucket_write(self, bucket_idx: int, fps: list[int]) -> None:
        pairs = sorted(self._split(fp) for fp in fps)
        heads = tuple(p[0] for p in pairs)
        rank = _SEMI_SORT_RANK[heads]
        tails_packed = 0
        for _, t in pairs:
            tails_packed = (tails_packed << self._tail_bits) | t
        code = (rank << (4 * self._tail_bits)) | tails_packed
        self._table.set(bucket_idx, code)

    @property
    def nbytes(self) -> int:
        return self._table.nbytes


def recommended_cuckoo_params(
    n: int, fp_rate: float = 0.01, bucket_size: int = 4
) -> dict[str, int]:
    """Suggested `(fingerprint_bits, num_buckets)` for n items at a target false-positive rate."""
    fingerprint_bits = max(1, math.ceil(math.log2(2 * bucket_size / fp_rate)))
    lf = _DEFAULT_LOAD_FACTOR.get(bucket_size, 0.9)
    num_buckets = _next_pow2(max(1, math.ceil(n / (bucket_size * lf))))
    return {
        "fingerprint_bits": fingerprint_bits,
        "num_buckets": num_buckets,
        "bucket_size": bucket_size,
    }


# ============================================================================
# 5 & 6. Xor / Binary Fuse filters (Graf & Lemire, 2020 / 2022) -- the actual
# state of the art beyond cuckoo filters for the *static* case (build once
# from a known key set, query forever, never insert or delete again). Both
# are 3-uniform "fuse" structures: every key maps to 3 slots, and the filter
# is built by *peeling* -- repeatedly finding a slot only one remaining key
# still touches, recording that key against it, and removing the key's
# influence everywhere else -- rather than by cuckoo's relocate-on-collision
# random walk. Binary Fuse improves on Xor purely by changing *where* a
# key's 3 slots land: overlapping windows instead of three disjoint blocks,
# which needs a smaller array to peel successfully with high probability.
# The peeling/assignment engine below is shared by both; only the
# capacity-sizing and slot-position formulas differ.
# ============================================================================


def _peel_and_assign(
    base_hashes: list[int],
    capacity: int,
    positions_fn: Callable[[int], tuple[int, int, int]],
    fingerprint_bits: int,
    max_attempts: int = 1000,
) -> tuple[_BitArray, int]:
    """Build a 3-peelable fingerprint array over `capacity` slots for the given items.

    Returns `(fingerprints, seed)`; `seed` re-derives each item's positions
    at query time via ``_mix_with_seed(base_hash, seed)`` then
    `positions_fn`. Raises if no attempt peels successfully within
    `max_attempts` -- vanishingly unlikely at the sizing this module uses
    (each attempt succeeds with high probability; see the README).
    """
    n = len(base_hashes)
    fp_mask = (1 << fingerprint_bits) - 1

    for seed in range(max_attempts):
        hashes = [_mix_with_seed(bh, seed) for bh in base_hashes]
        count = [0] * capacity
        xormask = [0] * capacity
        positions: list[tuple[int, int, int]] = [None] * n  # type: ignore[list-item]

        degenerate = False
        for idx, h in enumerate(hashes):
            p0, p1, p2 = positions_fn(h)
            if p0 == p1 or p1 == p2 or p0 == p2:
                degenerate = True
                break
            positions[idx] = (p0, p1, p2)
            for slot in (p0, p1, p2):
                count[slot] += 1
                xormask[slot] ^= h
        if degenerate:
            continue

        queue = deque(s for s in range(capacity) if count[s] == 1)
        peel_order: list[tuple[int, int, tuple[int, int, int]]] = []
        while queue:
            s = queue.popleft()
            if count[s] != 1:
                continue  # stale queue entry; its count changed since it was enqueued
            h = xormask[
                s
            ]  # the sole remaining key mapped here: XOR of one term is itself
            p = positions_fn(h)
            peel_order.append((h, s, p))
            for slot in p:
                count[slot] -= 1
                xormask[slot] ^= h
                if count[slot] == 1:
                    queue.append(slot)

        if len(peel_order) != n:
            continue  # a residual cycle in the hypergraph; retry with a new seed

        fingerprints = _BitArray(capacity, fingerprint_bits)
        # Reverse peel order: a key's "found" slot is provably never any
        # *other* key's found slot, and its two other slots -- if they are
        # some other key's found slot at all -- belong to a key peeled
        # *later* than this one (found only after this slot's count could
        # drop to 1), which in reverse order is processed *earlier*. So by
        # the time this key is assigned, both its other slots already hold
        # their final values.
        for h, s, p in reversed(peel_order):
            value = _splitmix64(h) & fp_mask
            for slot in p:
                if slot != s:
                    value ^= fingerprints.get(slot)
            fingerprints.set(s, value)
        return fingerprints, seed

    raise RuntimeError(
        f"failed to peel a filter for n={n} after {max_attempts} attempts"
    )


class XorFilter:
    """Xor filter (Graf & Lemire, 2020): immutable, ~1.23n slots, 3 disjoint blocks per key.

    About 23% larger than the information-theoretic minimum for its target
    false-positive rate -- worse than :class:`BinaryFuseFilter` -- but the
    simpler of the two to construct and reason about.
    """

    def __init__(
        self,
        items: Iterable[Hashable],
        fingerprint_bits: int = 8,
        max_attempts: int = 1000,
    ) -> None:
        unique = list(dict.fromkeys(items))
        n = len(unique)
        self.fingerprint_bits = fingerprint_bits
        self._n = n
        if n == 0:
            self._capacity = 3
            self._block = 1
            self._fp = _BitArray(3, fingerprint_bits)
            self._seed = 0
            self._positions_fn = lambda h: (0, 0, 0)
            return

        capacity = int(1.23 * n) + 32
        capacity = max(capacity, 3)
        capacity = ((capacity + 2) // 3) * 3  # round up to a multiple of 3
        block = capacity // 3
        self._capacity = capacity
        self._block = block

        def positions(h: int) -> tuple[int, int, int]:
            r0 = ((h & 0xFFFFFFFF) * block) >> 32
            r1 = ((_rotl64(h, 21) & 0xFFFFFFFF) * block) >> 32
            r2 = ((_rotl64(h, 42) & 0xFFFFFFFF) * block) >> 32
            return r0, block + r1, 2 * block + r2

        base_hashes = [_stable_hash64(x) for x in unique]
        self._fp, self._seed = _peel_and_assign(
            base_hashes, capacity, positions, fingerprint_bits, max_attempts
        )
        self._positions_fn = positions

    def __contains__(self, item: Hashable) -> bool:
        if self._n == 0:
            return False
        h = _mix_with_seed(_stable_hash64(item), self._seed)
        p0, p1, p2 = self._positions_fn(h)
        target = _splitmix64(h) & ((1 << self.fingerprint_bits) - 1)
        return (self._fp.get(p0) ^ self._fp.get(p1) ^ self._fp.get(p2)) == target

    def __len__(self) -> int:
        return self._n

    @property
    def nbytes(self) -> int:
        return self._fp.nbytes


class BinaryFuseFilter:
    """Binary Fuse filter (Graf & Lemire, 2022): Xor filter with overlapping segments.

    Same peeling/assignment core as :class:`XorFilter`; the only change is
    *where* a key's 3 slots land -- three overlapping SegmentLength-sized
    windows instead of three disjoint blocks -- which needs a smaller array
    (~1.13n vs ~1.23n) to peel successfully with high probability and, per
    Graf & Lemire, builds faster too. Segment-length and size-factor
    formulas below match the reference `xorfilter` implementation
    (FastFilter/xorfilter, `binaryfusefilter.go`).
    """

    def __init__(
        self,
        items: Iterable[Hashable],
        fingerprint_bits: int = 8,
        max_attempts: int = 1000,
    ) -> None:
        unique = list(dict.fromkeys(items))
        n = len(unique)
        self.fingerprint_bits = fingerprint_bits
        self._n = n
        if n == 0:
            self._fp = _BitArray(4, fingerprint_bits)
            self._seed = 0
            self._positions_fn = lambda h: (0, 0, 0)
            return

        segment_length = self._segment_length(n)
        segment_length_mask = segment_length - 1
        size_factor = self._size_factor(n)
        capacity = round(n * size_factor)
        total_segment_count = (capacity + segment_length - 1) // segment_length
        segment_count = max(total_segment_count - 2, 1)  # arity - 1 = 2 for 3-uniform
        segment_count_length = segment_count * segment_length
        array_length = (segment_count + 2) * segment_length

        def positions(h: int) -> tuple[int, int, int]:
            h0 = (h * segment_count_length) >> 64
            h1 = h0 + segment_length
            h2 = h1 + segment_length
            h1 ^= (h >> 18) & segment_length_mask
            h2 ^= h & segment_length_mask
            return h0, h1, h2

        base_hashes = [_stable_hash64(x) for x in unique]
        self._fp, self._seed = _peel_and_assign(
            base_hashes, array_length, positions, fingerprint_bits, max_attempts
        )
        self._positions_fn = positions

    @staticmethod
    def _segment_length(size: int) -> int:
        if size <= 1:
            return 4
        length = 1 << math.floor(math.log(size) / math.log(3.33) + 2.25)
        return max(4, min(length, 1 << 18))

    @staticmethod
    def _size_factor(size: int) -> float:
        size = max(size, 2)
        return max(1.125, 0.875 + 0.25 * math.log(1_000_000) / math.log(size))

    def __contains__(self, item: Hashable) -> bool:
        if self._n == 0:
            return False
        h = _mix_with_seed(_stable_hash64(item), self._seed)
        p0, p1, p2 = self._positions_fn(h)
        target = _splitmix64(h) & ((1 << self.fingerprint_bits) - 1)
        return (self._fp.get(p0) ^ self._fp.get(p1) ^ self._fp.get(p2)) == target

    def __len__(self) -> int:
        return self._n

    @property
    def nbytes(self) -> int:
        return self._fp.nbytes


def _demo() -> None:
    words = [f"item-{i}" for i in range(20_000)]
    absent = [f"absent-{i}" for i in range(20_000)]

    print("=== Bloom vs Counting Bloom vs Cuckoo family: deletion + space ===")
    bf = BloomFilter(len(words), fp_rate=0.01)
    for w in words:
        bf.add(w)
    print(f"BloomFilter:           {bf.nbytes:>8} bytes, no delete()")

    cbf = CountingBloomFilter(len(words), fp_rate=0.01)
    for w in words:
        cbf.add(w)
    cbf.remove(words[0])
    print(
        f"CountingBloomFilter:   {cbf.nbytes:>8} bytes, "
        f"deleted item still absent: {words[0] not in cbf}"
    )

    cf = CuckooFilter(len(words), fp_rate=0.01)
    for w in words:
        cf.insert(w)
    cf.delete(words[0])
    print(
        f"CuckooFilter:          {cf.nbytes:>8} bytes, "
        f"deleted item still absent: {words[0] not in cf}"
    )

    ssf = SemiSortedCuckooFilter(len(words), fp_rate=0.01)
    for w in words:
        ssf.insert(w)
    ssf.delete(words[0])
    print(
        f"SemiSortedCuckooFilter:{ssf.nbytes:>8} bytes, "
        f"deleted item still absent: {words[0] not in ssf}"
    )

    print(
        "\n=== Xor / Binary Fuse: build-once, no delete, smaller & faster to query ==="
    )
    xf = XorFilter(words, fingerprint_bits=8)
    print(f"XorFilter:             {xf.nbytes:>8} bytes")
    bff = BinaryFuseFilter(words, fingerprint_bits=8)
    print(f"BinaryFuseFilter:      {bff.nbytes:>8} bytes")

    false_positives = sum(1 for x in absent if x in cf)
    print(
        f"\nCuckooFilter measured false-positive rate on {len(absent)} absent keys: "
        f"{false_positives / len(absent):.4%}"
    )


if __name__ == "__main__":
    _demo()
