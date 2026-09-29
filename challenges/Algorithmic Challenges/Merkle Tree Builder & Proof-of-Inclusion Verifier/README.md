# Merkle Tree Builder & Proof-of-Inclusion Verifier

**Category:** Algorithmic Challenges
**Difficulty:** A (brief: "Build tree over a dataset, generate/verify inclusion proofs")

**Status:** Implemented (Rust)

A Merkle tree commits to a list of records with one 32-byte root; an
*inclusion proof* of O(log n) hashes convinces someone who holds only the
root that a given record is in the list. The brief asks for build + prove +
verify. This implementation follows the actual state of the art for the
problem: **RFC 9162 (Certificate Transparency v2)** -- the tree that
every browser-trusted TLS log runs on -- and adds the pieces real
transparency logs need beyond inclusion: consistency proofs (append-only
guarantee), deduplicated multi-leaf proofs, an O(log n)-memory incremental
tree, and a parallel builder.

| Piece                                      | What it does                                                                        |
| ------------------------------------------ | ----------------------------------------------------------------------------------- |
| `MerkleTree::build`                        | Sequential or rayon-parallel build over any `n` (including 0 and non-powers-of-two) |
| `inclusion_proof` / `verify_inclusion`     | RFC 9162 2.1.3 audit path and its iterative `fn`/`sn` verifier                      |
| `consistency_proof` / `verify_consistency` | RFC 9162 2.1.4: proves tree(n) is an append-only extension of tree(m)               |
| `multi_proof` / `verify_multi`             | One proof for k leaves, shared siblings sent once, no positions transmitted         |
| `CompactTree`                              | Append-only; stores only the O(log n) frontier; same root as the batch build        |
| `Sha256Hasher` / `Blake3Hasher`            | Generic `Hasher` trait; SHA-256 is byte-compatible with CT logs                     |
| `reference`                                | Naive recursive oracle transcribed from the RFC text, used to check everything else |

## Usage

Standalone Cargo crate (edition 2024, no shared workspace):

```bash
cargo nextest run --release      # or: cargo test --release
cargo clippy --all-targets -- -D warnings
cargo bench                      # criterion, ~10 minutes
```

CLI over the lines of a file (commands below were run as written):

```bash
$ merkle root ledger.txt
5 leaves, root f9910ac3a48b50616c67e7868b477bf3805522b21a4d60f427be04bda8dc9fca

$ merkle prove ledger.txt 2
root  f9910ac3...8dc9fca
size  5
proof 15bee132...  65752b87...  9be68c18...

$ merkle verify <root> 2 5 "carol pays dave 1" <proof hashes...>
VALID
$ merkle verify <root> 2 5 "carol pays dave 100" <proof hashes...>
INVALID: proof does not match the expected root      # exit code 1

$ merkle consistency ledger.txt 3       # proof that the first 3 lines are a prefix
$ merkle --hash blake3 root ledger.txt
$ merkle attack
real tree: 4 leaves
  no domain separation: forged 2-leaf tree root == real root ? true
  RFC 9162 (0x00/0x01): forged 2-leaf tree root == real root ? false
```

(`merkle` is `cargo run --release --bin merkle --`.) Library use:

```rust
let t = MerkleTree::<Sha256Hasher>::build(&leaves, Build::Parallel);
let proof = t.inclusion_proof(i, t.len())?;
verify_inclusion_data::<Sha256Hasher>(&leaves[i], i, t.len(), &proof, &t.root())?;
```

## Design notes

**Domain separation is the security-critical detail.** A leaf hashes as
`H(0x00 || data)`, an interior node as `H(0x01 || left || right)`. Without the
prefixes, the 64-byte string `H(a) || H(b)` used as a *leaf* hashes to exactly
the interior node over `a` and `b`, so an attacker can present a shorter tree
whose "leaves" are interior nodes of the real one and get the same root -- a
second-preimage attack. `tests/second_preimage.rs` runs it for real against
`NoDomainSha256` (a deliberately broken hasher in `src/insecure.rs`): forged
root equals the real root and a forged inclusion proof verifies. Under RFC
9162 hashing the same forgery fails.

**Non-power-of-two sizes: promote, don't duplicate.** RFC 9162 splits a range
of `n > 1` leaves at the largest power of two `k < n`. Bitcoin-style trees
instead duplicate an odd last node, which makes `[a,b,c]` and `[a,b,c,c]`
share a root. Built bottom-up, the RFC tree is simply "pair neighbours, carry
an unpaired last node up unchanged" -- so the tree is stored as levels, every
complete subtree is a stored node, and a right-edge subtree is too. Any RFC
subrange hash `MTH(D[s:e])` is then an O(1) lookup, or an O(log n) recombination
for ranges of historical sizes (`root_at(m)`, proofs against an older tree
size).

**Verifiers are stateless and iterative.** They implement the RFC's
`fn`/`sn` scheme: `fn` is the current node's index, `sn` the last node's index
at that level; a node is a right child when `fn` is odd *or* `fn == sn` (the
promoted unpaired node). No tree, no allocation for inclusion proofs. A proof
must consume exactly to `sn == 0`, so both too-short and too-long proofs fail.

**Multi-proofs.** k separate audit paths repeat every sibling near the root.
`multi_proof` walks level by level from the requested leaves and emits a sibling
only if it is not derivable from the other requested leaves; the verifier
replays the same walk from `size` alone, so no positions are sent. For 1,000
contiguous leaves out of 10^6: 16 hashes instead of 20,000.

**Incremental tree.** The RFC tree over `n` leaves is a right-fold of perfect
subtrees, one per set bit of `n`. `CompactTree` keeps just those peaks; append is
a binary-counter increment, the root a fold over the peaks. It can resume from a
stored frontier (log checkpoints). Property tests check its root equals the
batch build at every prefix size.

**Parallel build.** rayon for leaf hashing and for each level of 4,096+ nodes
(narrower levels aren't worth the dispatch). Roots and proofs are bit-identical
to the sequential build (property-tested up to 20,000 leaves).

**A proof does not bind the size.** Path shape depends only on the index's bits
and `size - 1`, so a size-5 audit path for leaf 0 hashes identically when
verified as "size 6". That is inherent to RFC 6962/9162: logs sign
`(root, size)` together in the tree head, and the tests reflect this -- a wrong
size with that size's *own* honest root is always rejected.

## Verification

22 tests, all green (`cargo nextest run --release`), `clippy -D warnings` clean:

- **Known answers** (`rfc_vectors.rs`): the RFC 6962 / CT reference vectors used
  by transparency-dev/merkle -- the empty root, leaf and node primitives, and the
  root for every prefix size 0..=8 of the 8-entry reference dataset (fetched from
  `testonly/constants.go`, not recalled) for sequential, parallel and incremental
  builds. Audit paths for leaves 0 and 5 of 8 are checked against the published
  internal node hashes. That repository's inclusion/consistency proof vectors are
  published only as accumulated digests of a draft-format subtree scheme, so they
  are *not* used; proofs are cross-checked against the oracle instead.
- **Oracle equivalence** (`properties.rs`): exhaustively for every size up to 40
  (SHA-256) / 24 (BLAKE3), every leaf index and every `m <= n` pair: proofs are
  *identical* to the RFC-transcribed naive implementation and verify. proptest adds
  random sizes to 1,500, parallel-vs-sequential, incremental-vs-batch, frontier
  resume and multi-proof (never larger than separate proofs).
- **Tamper detection** (`tamper.rs`): every proof hash bit-flipped (inclusion,
  consistency, multi), flipped leaf/root, wrong index, wrong size, truncated and
  padded proofs, a forked history (rewritten entry 3) failing a consistency
  proof, degenerate sizes.
- **Miri** on the non-rayon tests (known answers, second-preimage): clean. The
  crate has no `unsafe`. The rayon-using known-answer test trips Stacked Borrows
  inside `crossbeam-epoch` (a dependency, not this crate); it passes as a normal
  test. On Windows the blake3 build needs a short `CARGO_TARGET_DIR` (MASM line
  limit), e.g. `CARGO_TARGET_DIR=C:/mt cargo +nightly miri test ...`.
- `cargo fuzz` was not run: it does not support Windows MSVC targets well; the
  exhaustive oracle sweeps play that role here.

## Benchmarks

criterion, release build, 16 hardware threads, 64-byte leaves, median of the
reported interval. Times are from one run on this machine; trust ratios more than
absolutes.

**Building the tree (leaf hashing + all levels)**

| Leaves    | SHA-256 seq | SHA-256 rayon | BLAKE3 seq | BLAKE3 rayon |
| --------- | ----------: | ------------: | ---------: | -----------: |
| 16        |     2.97 us |       12.5 us |          - |            - |
| 256       |     46.5 us |       77.6 us |          - |            - |
| 4,096     |      721 us |        417 us |          - |            - |
| 1,000     |      183 us |        232 us |     282 us |       293 us |
| 10,000    |     1.75 ms |       0.62 ms |    2.65 ms |      0.91 ms |
| 100,000   |     17.6 ms |       2.75 ms |    27.6 ms |      4.04 ms |
| 1,000,000 |      174 ms |       18.2 ms |     276 ms |      29.0 ms |

**Proofs (SHA-256, n = 10^6 unless noted)**

| Operation                                              |                               Time |
| ------------------------------------------------------ | ---------------------------------: |
| Generate inclusion proof (current size)                |   242 ns (n=10^6); 155 ns (n=10^3) |
| Verify inclusion proof (20 hashes)                     | 1.70 us (n=10^6); 0.83 us (n=10^3) |
| Generate against a historical size (not a stored node) |                    792 ns (n=10^6) |
| Verify 100 spread leaves, individually                 |                             174 us |
| Verify the same 100 with one multi-proof               |     121 us (1,314 vs 2,000 hashes) |
| Verify 1,000 contiguous leaves, individually           |            1.74 ms (20,000 hashes) |
| Verify the same 1,000 with one multi-proof             |       88.4 us (16 hashes) -- 19.7x |

Findings worth knowing:

- **BLAKE3 was slower than SHA-256 here (~1.5x), the opposite of the usual
  reputation.** BLAKE3's speed comes from wide SIMD over long inputs and its
  internal tree; a Merkle node is a single 65-byte message, where per-call setup
  dominates, and this CPU has SHA extensions that make SHA-256 hardware-fast. BLAKE3
  would win on large leaves; for tree nodes SHA-256 is both the standard *and* the
  faster choice on this machine.
- **rayon loses at small sizes and wins ~10x at large ones.** 16 leaves: 4x
  *slower* in parallel (fixed dispatch cost swamps 3 us of work); 1,000 leaves:
  25% slower; break-even is around a few thousand leaves; 10^6 leaves: 9.6x
  faster. The 4,096-node per-level threshold keeps the top of the tree
  sequential; leaf hashing is always parallel in `Build::Parallel`, which is why
  the very small cases still pay.
- **Proofs are cheap; building dominates.** 20 hashes to verify a leaf of a
  million-leaf tree (1.7 us). Generation is ~7x cheaper because it only reads
  stored nodes rather than hashing.
- **Multi-proofs pay off exactly where leaves cluster.** Spread-out leaves save
  ~34% of hashes (the shared top of the tree); contiguous ones nearly collapse to
  the siblings of the range's edges.

## Sources

- [RFC 9162 -- Certificate Transparency Version 2.0](https://www.rfc-editor.org/rfc/rfc9162.html) (sections 2.1.1-2.1.5: tree hash, audit paths, consistency proofs, the `fn`/`sn` verification algorithms)
- [RFC 6962 -- Certificate Transparency](https://www.rfc-editor.org/rfc/rfc6962.html) (original 0x00/0x01 hashing and test vectors)
- [transparency-dev/merkle](https://github.com/transparency-dev/merkle) `testonly/constants.go`, `rfc6962/rfc6962_test.go` (known-answer vectors)
- O'Connor, Aumasson, Neves, Wilcox-O'Hearn, [BLAKE3: one function, fast everywhere](https://github.com/BLAKE3-team/BLAKE3-specs/blob/master/blake3.pdf)
- Merkle, "A Digital Signature Based on a Conventional Encryption Function", CRYPTO 1987
