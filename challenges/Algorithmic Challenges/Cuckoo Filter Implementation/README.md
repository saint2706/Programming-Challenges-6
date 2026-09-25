# Cuckoo Filter Implementation

**Category:** Algorithmic Challenges
**Difficulty:** A (brief: "Support deletion unlike Bloom filters; compare FP rates.")

**Status:** Implemented (Python)

A Bloom filter answers "have you seen x?" in a handful of bits per key, but
once a bit is set it can never be safely cleared -- other keys' hashes may
depend on it too. The brief asks for a structure that supports deletion and
to compare false-positive rates against Bloom filters. That comparison only
means something once *both* sides of the "how do you support deletion"
story are on the table, so this implementation builds out the full lineage
rather than just `CuckooFilter` in isolation:

| Structure                | Deletion | Space @ same false-positive rate | Mutability                           |
| ------------------------ | -------- | -------------------------------- | ------------------------------------ |
| `BloomFilter`            | No       | baseline                         | insert-only                          |
| `CountingBloomFilter`    | Yes      | ~4x `BloomFilter`                | insert + delete                      |
| `CuckooFilter`           | Yes      | ~1.2-1.4x `BloomFilter`          | insert + delete                      |
| `SemiSortedCuckooFilter` | Yes      | ~1.1-1.2x `BloomFilter`          | insert + delete (slower ops)         |
| `XorFilter`              | No       | ~0.75-0.9x `BloomFilter`         | build-once                           |
| `BinaryFuseFilter`       | No       | ~0.7-0.85x `BloomFilter`         | build-once, faster to build than Xor |

The brief's literal ask -- cuckoo vs. Bloom, with deletion -- is
`CuckooFilter` against `BloomFilter`/`CountingBloomFilter`. Going further:
the two structures that have actually superseded cuckoo filters in the
literature for the *static* case (build once from a known key set, never
insert or delete again) are **Xor filters** (Graf & Lemire, 2020) and
**Binary Fuse filters** (Graf & Lemire, 2022), both implemented here too --
they trade cuckoo's O(1) online insert/delete for a smaller, faster-to-query
structure built by *peeling* a 3-uniform hypergraph rather than by
relocating fingerprints. `SemiSortedCuckooFilter` is the original paper's
own within-family space optimization (Fan et al. 2014, Section 5.4),
included because it directly narrows the space gap against Bloom filters
that "compare FP rates" is really asking about.

All six structures share one storage primitive: `_BitArray`, packing
fixed-width unsigned integers into a raw `bytearray`. Every "bytes" and
"bits/key" number in this README is that array's real, measured length --
not `sys.getsizeof` of a Python list of boxed ints, which would be
dominated by ~28-byte-per-object CPython overhead regardless of which
filter it belonged to and would make the whole space comparison meaningless.

## 1. `BloomFilter` and `CountingBloomFilter` -- the baseline, and its answer to deletion

A Bloom filter is `m` bits and `k` hash functions: inserting a key sets `k`
bits, and a lookup answers "maybe present" only if all `k` are set. It can
never have a false negative for a key that was actually inserted, and its
false-positive rate is tunable by `m` and `k` -- but a bit that's set can be
relied on by *any* number of keys, so there is no way to un-set it for one
key without risking a false negative for every other key sharing that bit.
This is the fundamental limitation the brief is about.

`CountingBloomFilter` (Fan, Cao, Almeida & Broder's "Summary Cache", 1998)
is Bloom's own fix: widen each bit to a small saturating counter (4 bits
here) so deletion just decrements. It works, but at a real cost -- 4x the
space of a plain Bloom filter for the same `(n, false-positive rate)`,
confirmed directly in the benchmark below. This is the quantitative reason
cuckoo filters exist: deletion via counters is expensive; deletion via
relocatable fingerprints, below, is much cheaper.

Both use Kirsch & Mitzenmacher's (2006) double-hashing trick: `k`
simulated hash functions from two real ones, `g_i(x) = h1(x) + i*h2(x) (mod
m)`, provably as good as `k` independent functions for a Bloom filter's
purposes and far cheaper than actually computing `k` hashes per operation.

## 2. `CuckooFilter` and `SemiSortedCuckooFilter` (Fan, Andersen, Kaminsky & Mitzenmacher, 2014)

A cuckoo filter is a compact cuckoo hash table storing `f`-bit
**fingerprints** instead of full keys. Every key has two candidate buckets:

```
i1 = hash(x) mod num_buckets
i2 = i1 XOR hash(fingerprint(x))
```

The crucial property is that `i2` depends only on `i1` and the
fingerprint -- **not** on `x` itself (and by the same equation, `i1 = i2
XOR hash(fingerprint(x))` too, since XOR is its own inverse). This is
"partial-key cuckoo hashing": an entry can be relocated from one candidate
bucket to its alternate without ever re-hashing the original key, because
the fingerprint alone is enough to recompute where it could go.

**Insert** first tries both candidate buckets directly. If both are full,
it kicks: evict a random occupant of one candidate bucket, compute *its*
alternate bucket the same way, and try to place it there -- repeating up to
`max_kicks` (500) times. **Lookup** and **delete** just check (or clear) a
matching fingerprint in either candidate bucket. Because relocation never
touches the original key, and deletion only ever removes a fingerprint that
matches, cuckoo filters get true O(1) amortized deletion that counting
Bloom filters can only approximate with 4x the memory.

**The one real correctness subtlety**, handled explicitly here: if the kick
chain exceeds `max_kicks`, the *last* evicted fingerprint might not be the
newly-inserted key at all -- it could be some previously-inserted key that
got displaced deep in the chain. Silently dropping it would be a false
negative for a completely different, older member. This implementation
follows the reference C++ implementation's fix: park that one displaced
fingerprint in a single-entry **victim cache** that lookups and deletes
also check. Only a *second* simultaneous victim -- astronomically unlikely
below the recommended load factor -- makes `insert()` actually report
failure. `test_cuckoo_stress_insert_delete_reinsert_never_loses_live_items`
exercises exactly this: six thousand randomized insert/delete operations
per run, checkpointed so that every currently-live item is provably
findable at every point, including once the table is pushed past its
nominal capacity.

Fingerprint width is sized from the target false-positive rate `eps` and
bucket size `b` by `f >= log2(2b/eps)` -- a lookup checks up to `2b`
fingerprint slots total (both candidate buckets), so by the union bound the
false-positive probability is at most `2b / 2^f`.

### `SemiSortedCuckooFilter`: the paper's own space optimization

A bucket's contents are a *set*, not a sequence -- slot order never affects
membership. Fan et al.'s Section 5.4 exploits exactly this: split each
fingerprint into a 4-bit head and an `(f-4)`-bit tail, sort the
`(head, tail)` pairs, and store the sorted heads' *rank* among all
non-decreasing 4-tuples of nibbles rather than the four heads directly.
There are `C(16+4-1, 4) = 3876` such tuples (`math.comb(19, 4)`,
independently verified in the test suite), needing only 12 bits to rank
instead of the naive 16 -- saving exactly 1 bit per fingerprint, regardless
of `f`. Rather than the paper's hardcoded lookup table, this implementation
computes the rank/unrank table combinatorially at import time via
`itertools.combinations_with_replacement`, which generalizes cleanly to any
`fingerprint_bits >= 4` instead of being locked to the paper's specific
4-bit example.

## 3. `XorFilter` and `BinaryFuseFilter` (Graf & Lemire, 2020 / 2022) -- the actual state of the art for the static case

Cuckoo filters are the right answer when the set changes over time. When
it doesn't -- build once from a known key set, query forever -- the
literature has moved past cuckoo filters entirely. Xor and Binary Fuse
filters are both **3-uniform "fuse" structures**: every key maps to
exactly 3 slots in a fingerprint array, and membership is a single XOR:

```
present  <=>  fp[h0(x)] ^ fp[h1(x)] ^ fp[h2(x)] == fingerprint(x)
```

Construction works by **peeling**: maintain, for every slot, a count of how
many not-yet-resolved keys still map there and the XOR of their hashes.
Whenever a slot's count drops to exactly 1, that XOR value *is* the one
remaining key's hash (XOR of one term is itself) -- push it onto a stack,
then remove its contribution from all three of its slots (which may drop
some of *their* counts to 1, continuing the cascade). If every key gets
peeled, construction succeeded; if a residual cycle is left over (a small
but real probability each attempt), retry with a new hash seed.

Fingerprints are then assigned by **popping the stack in reverse peel
order** and setting `fp[found_slot] = fingerprint(hash) XOR fp[other
slot 1] XOR fp[other slot 2]`. This is the one part of the algorithm that
took real care to get right and is worth stating precisely, since it's easy
to assume the *forward* peel order works just as well (it doesn't): a
key's uniquely-identified "found" slot is, by construction, never any
*other* key's found slot. If one of its two other slots ever *is* some
other key's found slot, that other key must have been peeled **after**
this one (its slot could only reach count 1 once this key's contribution
was already removed) -- which means, in reverse order, that other key is
processed **before** this one. So by the time any key is assigned, both its
other slots already hold their final values. (A full derivation is in the
`_peel_and_assign` docstring; `test_fuse_filter_exact_no_false_negatives`
confirms the consequence -- unlike every probabilistic structure above,
these two never produce a false negative for a member key, by construction,
not by tuning a probability down.)

`BinaryFuseFilter` reuses the *identical* peeling/assignment engine
(`_peel_and_assign` is shared by both classes in `cuckoo_filter.py`) and
changes only *where* a key's three slots land: three **overlapping**
`SegmentLength`-sized windows instead of `XorFilter`'s three disjoint
blocks. The segment-length and size-factor formulas below were pulled
directly from the reference `FastFilter/xorfilter` Go implementation
(`binaryfusefilter.go`) to avoid guessing at published constants:

```python
segment_length = 1 << floor(log(n) / log(3.33) + 2.25)
size_factor = max(1.125, 0.875 + 0.25 * log(1_000_000) / log(n))
capacity = round(n * size_factor)
```

Overlapping segments need a smaller array to peel successfully with high
probability than disjoint blocks do -- Graf & Lemire report Binary Fuse
within ~13% of the information-theoretic space lower bound, against ~23%
for plain Xor filters, and faster construction too.

**A genuine finding from testing this, not just reading the paper:** the
`size_factor` formula above is calibrated against a 1,000,000-item
reference scale and only approaches its 1.125 floor as `n` grows toward
that. At small `n` (verified at `n=2,000` in
`test_binary_fuse_can_be_larger_than_xor_at_small_n`), `size_factor` is
closer to 1.33, and Binary Fuse's array ends up genuinely *larger* than
Xor's flatter `1.23n + 32` overhead -- confirmed directly in the benchmark
table below (§5). Binary Fuse's space advantage is real but asymptotic,
not universal; this implementation deliberately keeps both structures
around rather than treating Binary Fuse as a strict drop-in replacement,
so the crossover point is something this repo actually measured rather
than took on faith.

## Correctness

```
$ uv run --with pytest pytest -q
77 passed in ~5s
```

- **`_BitArray`** (the shared storage primitive everything else depends on)
  is fuzz-tested with 5,000 random read/write operations per width across
  eleven widths from 1 to 32 bits, checking that every write leaves every
  *other* slot untouched.
- **The semi-sort combinatorial table** is checked independently for size
  (`math.comb(19, 4) == 3876`), sortedness, and that its rank map is an
  exact bijection with the table -- before any filter test exercises it.
- **`BloomFilter`/`CountingBloomFilter`**: zero false negatives across
  5,000 inserted members; empirical false-positive rate within a loose
  bound of the theoretical target at three different rates; `CountingBloomFilter`
  measured at ~4x `BloomFilter`'s space for equal `(n, eps)`; deleting an
  already-absent key doesn't corrupt counters (no negative counts).
- **`CuckooFilter`/`SemiSortedCuckooFilter`** (parametrized so the
  space-optimized variant is held to the exact same bar): zero false
  negatives for present items; deletion actually removes a key (checked at
  a false-positive rate low enough that a bucket collision masking the
  result is vanishingly unlikely); duplicate-insert acts as a bounded
  counter (insert x3, delete x3, absent only after the third); the 6,000-op
  randomized insert/delete/re-insert stress test described above; achieved
  load factor near Fan et al.'s own reported figures (84%/95.5%/98% for
  bucket sizes 2/4/8) before the first insertion failure; measured
  false-positive rate near target.
- **`XorFilter`/`BinaryFuseFilter`**: *exact* zero false negatives (not a
  probabilistic bound -- the XOR equation is satisfied by construction);
  correct behavior on duplicate input items (deduplicated before peeling,
  since two identical keys always map to the same three slots and can
  never be peeled apart); empty input; six small-`n` edge cases (`n` = 1
  through 50, where peeling's hypergraph behaves differently than at
  scale); measured false-positive rate near `2^-fingerprint_bits`; Binary
  Fuse smaller than Xor at realistic scale (`n >= 20,000`) *and* the
  opposite at small `n` (`n = 2,000`), both asserted explicitly rather than
  only the direction the paper advertises.
- **Cross-family**: `CuckooFilter` measurably smaller than
  `CountingBloomFilter` and within a small constant factor of plain
  `BloomFilter`, at identical `(n, eps)` -- the brief's comparison, made
  into an assertion rather than left to the benchmark table alone.

## Benchmarks

```
$ uv run python benchmark.py
```

**1. Space vs. measured false-positive rate** (n=50,000, 100,000 absent-key trials per row):

```
structure                   target    measured     bytes  bits/key  delete
--------------------------------------------------------------------------
BloomFilter                10.000%    10.1110%     29954      4.79      no
CountingBloomFilter        10.000%    10.1110%    119814     19.17     yes
CuckooFilter                10.000%     4.6920%     57344      9.18     yes
SemiSortedCuckooFilter     10.000%     4.6920%     49152      7.86     yes
XorFilter                  10.000%     6.1570%     30767      4.92      no
BinaryFuseFilter           10.000%     6.2710%     30720      4.92      no

BloomFilter                 2.000%     1.9600%     50890      8.14      no
CountingBloomFilter         2.000%     1.9600%    203560     32.57     yes
CuckooFilter                 2.000%     1.1450%     73728     11.80     yes
SemiSortedCuckooFilter      2.000%     1.1450%     65536     10.49     yes
XorFilter                   2.000%     1.5060%     46150      7.38      no
BinaryFuseFilter            2.000%     1.5520%     46080      7.37      no

BloomFilter                 1.000%     0.9550%     59907      9.59      no
CountingBloomFilter         1.000%     0.9550%    239627     38.34     yes
CuckooFilter                 1.000%     0.5880%     81920     13.11     yes
SemiSortedCuckooFilter      1.000%     0.5880%     73728     11.80     yes
XorFilter                   1.000%     0.7250%     53842      8.61      no
BinaryFuseFilter            1.000%     0.7490%     53760      8.60      no

BloomFilter                 0.100%     0.1070%     89860     14.38      no
CountingBloomFilter         0.100%     0.1070%    359440     57.51     yes
CuckooFilter                 0.100%     0.0670%    106496     17.04     yes
SemiSortedCuckooFilter      0.100%     0.0670%     98304     15.73     yes
XorFilter                   0.100%     0.0880%     76917     12.31      no
BinaryFuseFilter            0.100%     0.0920%     76800     12.29      no
```

**2. Space among deletion-capable structures only** (n=50,000, fp_rate=1%):

```
structure                    bytes   x plain Bloom
--------------------------------------------------
BloomFilter (no delete)      59907           1.00x
CountingBloomFilter         239627           4.00x
CuckooFilter                 81920           1.37x
SemiSortedCuckooFilter       73728           1.23x
```

**3. Cuckoo filter load factor before the first insert failure**, vs. Fan et al.'s reported 84%/95.5%/98%:

```
 bucket_size  buckets*size  inserted  load_factor
--------------------------------------------------
           2         65536     57544      87.805%
           4         65536     63508      96.906%
           8         65536     64935      99.083%
```

**4. Semi-sorted vs. standard cuckoo buckets**:

```
   fp_rate  standard B  semi-sort B  space saved  insert slowdown
--------------------------------------------------------------------
   10.000%      114688        98304       14.29%            0.84x
    1.000%      163840       147456       10.00%            1.02x
    0.100%      212992       196608        7.69%            0.85x
```

**5. Xor vs. Binary Fuse across n**:

```
         n   xor bytes  bfuse bytes  bfuse/xor   xor build  bfuse build
-----------------------------------------------------------------------
     2,000       2,493        2,816      1.130      0.008s       0.007s
    20,000      24,633       24,576      0.998      0.084s       0.073s
   100,000     123,033      118,784      0.965      0.542s       0.477s
   300,000     369,033      348,160      0.943      2.445s       1.885s
```

**The false-positive comparison is the headline result, and it isn't quite
what the naive "cuckoo beats Bloom" framing suggests.** At every target
rate, `CuckooFilter`'s *measured* rate comes in well under its target --
roughly 2x lower at eps=10%, narrowing toward parity as eps shrinks -- for
a structural reason, not luck: `fingerprint_bits` is an integer, and
`f = ceil(log2(2b/eps))` rounds up to the next whole bit, which can nearly
halve the achieved false-positive rate versus the target. `BloomFilter`
doesn't have this discretization (its `m`/`k` sizing is continuous), so its
measured rate tracks its target far more tightly. Net effect: at a *target*
rate, cuckoo filters often look more expensive per bit than Bloom filters
in this table, but at the same *achieved* rate they're not -- compare
`CuckooFilter` at target 2% (measured 1.145%, 73,728 bytes) against
`BloomFilter` at target 1% (measured 0.955%, 59,907 bytes): once you
account for the rounding, cuckoo's real premium over Bloom for a
*deletable* structure is much smaller than the raw table suggests, and
tiny next to `CountingBloomFilter`'s flat 4x tax. This is precisely Fan et
al.'s "practically better than Bloom" claim, reproduced quantitatively
rather than asserted.

**Load factor confirms the paper's own figures closely**: 87.8%/96.9%/99.1%
achieved against 84%/95.5%/98% reported, for bucket sizes 2/4/8
respectively -- bigger buckets pack tighter before the random-walk kick
chain starts failing, at the cost of a wider fingerprint needed for the
same false-positive rate (`f >= log2(2b/eps)` grows with `b`).

**Semi-sorting saves real space (7.7%-14.3% here) at close to zero
measured throughput cost, not the clear slowdown the "encode/decode on
every mutation" framing predicts.** The extra sort-and-rank work per
bucket operation is real, but buckets are only 4 fingerprints -- cheap
enough in absolute terms, and small enough (`_bucket_width` <= 28 bits for
an 8-bit fingerprint) that the smaller table's better cache locality can
offset the added CPU work entirely. Measured slowdown ranged from a 15%
*speedup* to a 2% slowdown across three fingerprint widths in a single
run -- within ordinary timing noise for a Python microbenchmark, not a
clear cost. Worth stating plainly since it contradicts the intuitive
"more computation per op = slower" assumption this implementation started
with.

**Xor vs. Binary Fuse crosses over between n=2,000 and n=20,000, exactly
where the `size_factor` formula's asymptotic-to-1,000,000 calibration
predicts.** Below the crossover, Binary Fuse's extra segment-overhead
bookkeeping doesn't pay for itself against Xor's flatter constant
overhead; above it, Binary Fuse wins on both space (6.5% smaller by
n=300,000, and still improving) and construction time (~23% faster at
n=300,000, consistent with Graf & Lemire's own construction-speed claim).

## Run it

```bash
cd "challenges/Algorithmic Challenges/Cuckoo Filter Implementation"

uv run python cuckoo_filter.py        # demo: all six filters built, deletion, measured FP rate
uv run python benchmark.py            # all five comparisons above

uv run --with pytest pytest -q           # 77 tests
```

## Where this is used

**Network routers and packet processors.** The original cuckoo filter
paper's motivating use case: per-flow state tables and firewall/ACL rule
matching where entries genuinely churn (flows start and end) and hardware
memory budgets are tight -- exactly where a Bloom filter's insert-only
limitation is a real operational problem, not just a theoretical one.

**Database and storage-engine membership tests.** LSM-tree storage engines
(e.g. RocksDB, Cassandra) use Bloom-family filters to skip disk reads for
keys that provably aren't in an on-disk table (SSTable); a workload with
frequent compaction/tombstone cycles benefits from a filter that can
actually forget deleted keys instead of accumulating stale bits forever.

**Malicious-URL and safe-browsing lists.** Browsers and security products
ship a compact "is this URL bad" filter that's built once from a threat
feed and shipped to millions of clients -- precisely the static,
build-once, query-forever case Xor and Binary Fuse filters target, where
their smaller size directly reduces distribution bandwidth and client
memory, and mutability isn't needed since the whole list is rebuilt and
redistributed on update anyway.

**CDN and cache admission/dedup.** Deciding whether to cache an object
(or whether a blockchain/log entry has already been seen) needs a
disposable per-window membership test that gets rebuilt every rotation --
another static-set case suited to Xor/Binary Fuse rather than a
long-lived mutable Bloom or cuckoo filter.

## Further state of the art beyond this implementation

- **Morton filters** (Breslow & Jayasena, VLDB 2018) restructure cuckoo
  filter buckets into cache-line-aligned, SIMD-friendly blocks with
  compressed occupancy bitmaps, reporting faster lookups and inserts than
  a standard cuckoo filter at the same false-positive rate through better
  memory-system utilization rather than a different core algorithm.
- **Smaller and More Flexible Cuckoo Filters** (Schmitz, Zentgraf &
  Rahmann, 2025) removes the power-of-2 bucket-count restriction this
  implementation still has (`_next_pow2` in `_CuckooFilterBase.__init__`)
  and tightens the per-key space overhead below the `(k+3)`-bit fingerprint
  width used here, at the cost of a more involved bucket-indexing scheme.
- **4-wise Binary Fuse filters** (also Graf & Lemire, 2022) map each key to
  4 slots instead of 3, trading a further space reduction (within ~8% of
  the information-theoretic bound, vs. ~13% for the 3-wise version
  implemented here) for a measured ~30% increase in query time.

## Sources

- [Fan, B., Andersen, D.G., Kaminsky, M. & Mitzenmacher, M., "Cuckoo Filter: Practically Better Than Bloom," *CoNEXT* 2014, pp. 75-88](https://www.cs.cmu.edu/~dga/papers/cuckoo-conext2014.pdf) -- partial-key cuckoo hashing, the kick-based insert algorithm, fingerprint sizing, and the semi-sorting space optimization (Section 5.4) `CuckooFilter`/`SemiSortedCuckooFilter` implement.
- Bloom, B.H., "Space/Time Trade-offs in Hash Coding with Allowable Errors," *Communications of the ACM* 13(7):422-426, 1970 -- the original Bloom filter `BloomFilter` implements.
- Fan, L., Cao, P., Almeida, J. & Broder, A.Z., "Summary Cache: A Scalable Wide-Area Web Cache Sharing Protocol," *IEEE/ACM Transactions on Networking* 8(3):281-293, 2000 -- the counting Bloom filter `CountingBloomFilter` implements.
- Kirsch, A. & Mitzenmacher, M., "Less Hashing, Same Performance: Building a Better Bloom Filter," *ESA* 2006 -- the double-hashing scheme `BloomFilter`/`CountingBloomFilter` use to simulate `k` hash functions from two.
- [Graf, T.M. & Lemire, D., "Xor Filters: Faster and Smaller Than Bloom and Cuckoo Filters," *ACM Journal of Experimental Algorithmics* 25, 2020](https://arxiv.org/abs/1912.08258) -- the peeling-construction 3-uniform filter `XorFilter` implements.
- [Graf, T.M. & Lemire, D., "Binary Fuse Filters: Fast and Smaller Than Xor Filters," *ACM Journal of Experimental Algorithmics*, 2022](https://arxiv.org/abs/2201.01174) -- the overlapping-segment variant `BinaryFuseFilter` implements; segment-length and size-factor formulas cross-checked against the [FastFilter/xorfilter](https://github.com/FastFilter/xorfilter) Go reference implementation (`binaryfusefilter.go`).
- [Schmitz, J.E., Zentgraf, J. & Rahmann, S., "Smaller and More Flexible Cuckoo Filters," arXiv:2505.05847, 2025](https://arxiv.org/abs/2505.05847) -- a further cuckoo-filter-family improvement beyond this implementation's scope, noted above.
- Breslow, A.D. & Jayasena, N.S., "Morton Filters: Faster, Space-Efficient Cuckoo Filters via Biasing, Compression, and Decoupled Logical Sparsity," *PVLDB* 11(9):1041-1055, 2018 -- the cache/SIMD-oriented cuckoo variant noted above.
- Steele, G.L., Lea, D. & Flood, S., "Fast Splittable Pseudorandom Number Generators," *OOPSLA* 2014 -- SplitMix64, the avalanche mixer `_splitmix64` implements, used throughout for alternate-index derivation and construction-retry reseeding.
- Eppstein, D., "Cuckoo Filter: Simplification and Analysis," arXiv:1604.06067, 2016 -- an independent formal analysis of cuckoo filter behavior, consulted for the load-factor claims this README's benchmark §3 cross-checks against.
