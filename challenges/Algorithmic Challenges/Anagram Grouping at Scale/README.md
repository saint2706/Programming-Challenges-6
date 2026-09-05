# Anagram Grouping at Scale

**Category:** Algorithmic Challenges
**Difficulty:** B (brief: "hash-based grouping for very large word lists; benchmark")

**Status:** Implemented (Python)

Two words are anagrams iff their character multisets are equal, so grouping is
grouping by a canonical form of the multiset. That is the whole algorithm, and
every solution to this problem is that one line.

The **scale** is where the problem actually lives. At ten million words the
dictionary of keys weighs more than the corpus does, so the interesting
question is not "which key is asymptotically fastest" but "which key is small,
and can you still be exact once it is". This directory answers that with five
keys, three execution modes, and measurements for all of them.

| Key | Build | Retained size (L=8 / L=64) | Exact? |
| --- | --- | --- | --- |
| `key_sorted` — sorted characters | O(L log L) | 57 B / 113 B | yes |
| `key_counter` — `((atom, count), ...)` | O(L + s log s) | 96 B / 231 B | yes |
| `key_primes` — product of per-letter primes | O(L²/64) bignum | 32 B / 68 B | yes, by unique factorisation |
| `key_bincount` — numpy count vector | O(L) in C | 1 KiB, flat | yes |
| `multiset_hash` — 128-bit additive hash | O(L) | **44 B, flat** | yes, after verification |

Sizes are `sys.getsizeof` of the key object, measured by `benchmark.py --only bytes`.

## The 128-bit multiset hash, and why exactness is not given up

`multiset_hash` is [MSet-Add-Hash](https://people.csail.mit.edu/devadas/pubs/mhashes.pdf)
(Clarke, Devadas, van Dijk, Gassend & Suh, ASIACRYPT 2003) at 128 bits: sum a
random 128-bit value per character, modulo 2¹²⁸. Its defining property is that
it is a **homomorphism from the free commutative monoid over the alphabet into
Z/2¹²⁸**:

```
H(A ⊎ B) = H(A) + H(B)   (mod 2¹²⁸)
```

Three things fall out of that single equation:

- **Order independence for free.** A sum does not care in what order it is
  taken, which is exactly the property a canonical multiset key needs. No
  sorting, no counting, no canonical form to build.
- **O(1) insertion *and deletion*.** Adding a letter is one add; removing one
  is one subtract. So the hash is maintainable over a sliding window, which no
  sorted-string key can be.
- **Free parallelism.** The hash of a concatenation is the sum of the parts,
  so a corpus can be hashed in independent chunks and combined — and since
  anagrams share a hash, hashing also *partitions* the corpus into shards that
  can be grouped with no merge step at all. That is what
  `group_anagrams_parallel` and `shard_of` are.

The key is 16 payload bytes at any word length. The catch is that a 128-bit
hash is not injective, and being additive it is *linear*, so an adversary who
knows the table can construct collisions by solving a subset-sum. Nothing here
depends on that being hard: `group_anagrams(method="hash")` buckets by hash and
then splits each bucket by exact multiset equality. That verification is
O(total corpus length), which is **cheaper than sorting every word**, so the
exact answer comes out of a pipeline that is both smaller and asymptotically
faster than the sorted-key one. `test_hash_method_verifies_rather_than_trusts`
monkeypatches the hash to a constant so that *every* word collides, and checks
the groups still come out right.

For the record, the collision probability with no adversary is birthday-bounded
at about m²/2¹²⁹ — around 5 × 10⁻²¹ at a billion distinct multisets.

## Three things that are less obvious than they look

**Sorting the encoded bytes is wrong.** The tempting speed-up is
`bytes(sorted(word.encode()))`. But UTF-8 byte multisets are not injective on
character multisets:

```
"ã©"  →  C3 A3 C2 A9
"é£"  →  C3 A9 C2 A3      same multiset {A3, A9, C2, C3}, different words
```

Those two are not anagrams, and a byte-sorting key merges them. Every key here
is built from characters. (`test_utf8_byte_multisets_are_not_injective`.)

**Normalisation is part of the problem statement, not a nicety.** `"café"`
written precomposed and decomposed are the same word to a reader and different
multisets to `sorted`. `Normalizer` composes the four Unicode normalisation
forms, `str.casefold` (which maps `ß → ss` and unifies both Greek sigmas, where
`str.lower` does neither), grapheme clustering, and character filtering.

The grapheme case is the sharp one. Under NFD, `"éa"` and `"eá"` have *equal*
codepoint multisets `{e, a, ´}` — but they are plainly not anagrams, because
the accent belongs to a particular letter. NFC fixes it by never splitting the
accent off; `graphemes=True` fixes it for the scripts where NFC cannot
precompose (`"ź"` written as `z` + U+0301 stays two codepoints forever).

**Duplicate words are data.** A corpus containing `"listen"` twice has a group
of three, not two. `unique=True` opts out.

## Beyond one machine's memory

Two exact escape hatches, because "at scale" eventually means "does not fit":

- `group_anagrams_external` — buffer `chunk_size` (key, word) pairs, sort,
  spill a sorted run to disk, then `heapq.merge` the runs and cut the merged
  stream at key boundaries. Peak memory is one chunk, then
  O(runs + largest group). Records are pickled per triple, which is
  self-delimiting: a text format would need escaping for words containing the
  delimiter or a newline, and real corpora contain both.
- `group_anagrams_parallel` — shard on the multiset hash, group each shard in
  its own process, concatenate. No merge, because anagrams cannot cross shards.

Measured at 50k words with `chunk_size=5000`: 2.1 MiB peak in memory versus
0.8 MiB external, for 4.4× the wall time. Halve the chunk size and the ceiling
roughly halves; the corpus size does not enter into it.

## Benchmarks

`uv run --with numpy python benchmark.py`. Key construction, nanoseconds per
word, 26-letter alphabet:

| L | sorted | counter | hash | primes | bincount |
| --- | --- | --- | --- | --- | --- |
| 4 | **290** | 1556 | 1761 | 281 | 2260 |
| 8 | **484** | 2230 | 2509 | 514 | 2271 |
| 32 | **2167** | 5024 | 4968 | 2109 | 2311 |
| 64 | 5053 | 7016 | 6845 | 4771 | **2363** |
| 256 | 26069 | 12713 | 12412 | 21140 | **2694** |
| 4096 | 500884 | 129848 | 117713 | 1690583 | **8379** |

The asymptotics predict the *shape* of that table and get the *ordering*
wrong everywhere it matters. `key_sorted` is O(L log L) and beats three O(L)
keys up to L ≈ 100, because its log factor is bought inside a C sort while the
O(L) keys pay Python overhead per call. `key_bincount` is flat in L — 2.3 µs
from L=2 to L=256 — and takes over around L ≈ 40. `key_primes` tracks
`key_sorted` until its O(L²/64) bignum multiplication takes over past L ≈ 256,
at which point it is 200× slower than `bincount`. `group_anagrams(method="auto")`
switches at the measured crossover, not at the theoretical one.

### Frequency-ordered primes

A prime-product key costs `Σ count(c) · log₂(p_c)` bits, so the key is smallest
when the *most frequent* letters get the *smallest* primes. `PRIME_TABLE`
therefore assigns 2 to `e`, 3 to `t`, 5 to `a`, and so on down English letter
frequency. Predicted against measured:

```
E[log2 p] over a uniform letter:            4.902 bits/char
frequency-ordered, English text: predicted  3.610, measured 3.612
alphabetical,      English text: predicted  4.582, measured 4.596
```

Against a sorted ASCII key's 8 bits/char that is a ~55% key-size win — and it
is still the wrong choice, because you pay O(L²/64) to get it while a 16-byte
hash is free. Worth deriving precisely so the conclusion is a result rather
than an opinion.

## Mathematical checks in the test suite

Rather than only comparing implementations against each other, the suite
checks the grouping against combinatorics:

- Words of length L over an alphabet of size k form exactly `C(L+k-1, k-1)`
  anagram classes — the multiset coefficient.
- A class whose counts are `(c₁ … c_k)` contains exactly `L! / ∏ cᵢ!` words —
  the multinomial coefficient. Checked class by class over all 3⁵ words.
- Class sizes sum to `k^L`.
- The anagram relation is reflexive, symmetric and transitive, so "group" is
  well defined in the first place.
- The multiset hash is a homomorphism, supports deletion, and is 0 on the
  empty multiset.

## Where this is actually used

Grouping words into anagram classes is a puzzle. The two mechanisms built here
to do it at scale are not.

**Incremental multiset hashing is deployed, not theoretical.** The problem
MSet-Add-Hash was invented for is verifying that a collection holds the contents
you expect when the two sides cannot agree on an order — and the homomorphism is
what makes the digest *updatable* rather than recomputed:

- **Bitcoin Core** commits to the UTXO set with a multiset hash (`MuHash`, the
  multiplicative sibling of the additive one here; reachable via
  `gettxoutsetinfo`). Creating or spending an output updates the digest in O(1),
  where recomputing a Merkle root over 100M+ entries every block would not be
  viable.
- **Replica and backup verification**: deciding whether two nodes hold the same
  rows without sorting either side, and without the ordering assumptions a
  Merkle tree imposes.
- **Sliding-window integrity and streaming checksums**, where O(1) *deletion* is
  the whole point — no sorted-string key can retract a symbol.

**Canonical-key grouping is the shape of every `GROUP BY`.** Choose a canonical
form, hash it, partition on the hash: that is how Spark and Hadoop shuffle. And
`group_anagrams_external` is a small readable version of what a database does
for `GROUP BY` on data larger than memory — buffer, sort, spill a run, k-way
merge. If you have ever wanted to see what "external sort" means concretely, it
is about forty lines here.

**The Unicode work is security-relevant, not decorative.** Deciding when two
strings are "the same" under normalization and case folding is the core of
username and email canonicalization (so `Admin` and `admin` cannot both
register), homograph and IDN spoofing defence, deduplicating product catalogs
and address records, and search-index term normalization. The
`str.lower`/`str.casefold` distinction and the NFC-versus-NFD trap are how real
canonicalization bugs happen. The UTF-8 byte-multiset counterexample is a
working demonstration of a plausible shortcut that silently merges distinct
inputs — which, in a canonicalization layer, is an account-takeover primitive.

**Where the literal problem shows up:** Scrabble and Wordle solvers, crossword
tooling, and — over a different alphabet — grouping sequencing reads by k-mer
composition, where the multiset genuinely is the feature you want to index on.

## Files

| File | What it is |
| --- | --- |
| `anagrams.py` | Keys, `Normalizer`, `group_anagrams`, external and parallel modes, `AnagramIndex`, CLI |
| `test_anagrams.py` | 95 tests: oracle agreement, Unicode, hash properties, combinatorial identities, CLI |
| `benchmark.py` | Key timing, end-to-end memory, key sizes, prime bit-lengths, hash quality, out-of-core |

## Running it

```bash
uv run --with numpy python anagrams.py --demo
uv run --with numpy python anagrams.py --verify
uv run --with numpy python anagrams.py /usr/share/dict/words --min-size 4 --top 10
uv run --with numpy python anagrams.py words.txt --phrase --external --chunk-size 100000

uv run --with pytest --with numpy pytest -q            # all 95
uv run --with pytest --with numpy pytest -q -m "not slow"
uv run --with numpy python benchmark.py --quick
```

numpy is optional — only `key_bincount` needs it, and it is imported lazily.

## Sources

- [Incremental Multiset Hash Functions and Their Application to Memory Integrity Checking](https://people.csail.mit.edu/devadas/pubs/mhashes.pdf) — Clarke, Devadas, van Dijk, Gassend & Suh, ASIACRYPT 2003 (MSet-Add-Hash)
- [UAX #29: Unicode Text Segmentation](https://unicode.org/reports/tr29/) — grapheme cluster boundaries
- [UAX #15: Unicode Normalization Forms](https://unicode.org/reports/tr15/)
