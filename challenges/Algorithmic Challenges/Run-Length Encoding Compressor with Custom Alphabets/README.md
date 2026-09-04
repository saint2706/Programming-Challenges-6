# Run-Length Encoding Compressor with Custom Alphabets

**Category:** Algorithmic Challenges
**Difficulty:** B (brief: "support arbitrary symbol sets, not just text")

**Status:** Implemented (Python)

The interesting constraint here is not the input — any RLE reads any sequence.
It's the **output**.

Textbook RLE writes `(3, 'A')` and quietly assumes the count can be spelled
using characters that aren't in the data. Over `{A, C, G, T}` there is no `'3'`
to write, and no separator either. Every encoder in this module is **closed over
its alphabet**: given symbols from Σ it emits only symbols from Σ, which means
run lengths must be spelled in base |Σ| with a self-delimiting code.

That one rule is what turns the weekend exercise into something with real
design decisions in it — and it makes the compressor genuinely competitive:

```
$ uv run python rle.py analyze genome.txt --alphabet dna
symbols          46,625
alphabet         4 symbols, 2.000 bits each (ideal)
count code       terminated
runs             10,685  (mean 4.36, longest 149)
packing          2.000 bits/symbol -> 11,660 bytes uncompressed
run-length entropy 20,152 bits (lower bound for any run-length coder)

codec         symbols out    ratio   packed bytes
-------------------------------------------------
pair               35,065    0.752          8,770
packbits           21,731    0.466          5,436
escape             23,736    0.509          5,937
adaptive           21,732    0.466          5,436
zlib -9                                     5,506
```

RLE beating zlib is not the usual outcome. It happens because zlib has to spend
bits describing a 4-symbol alphabet it doesn't know about, while this encoder
had it for free.

## The core problem: writing a number with no digits and no separator

"Self-delimiting" is the whole difficulty. Writing 12 as `[1, 2]` is useless if
the decoder can't tell where the number stops — and there's no spare separator,
because every symbol in Σ can legitimately appear in the data.

Three codes, and the alphabet picks between them:

| Code | Works when | Cost | Idea |
| --- | --- | --- | --- |
| **Elias gamma** | \|Σ\| = 2 | 2·log₂v + 1 | (bit_length−1) zeros, then the bits |
| **Terminated** | \|Σ\| ≥ 3 | log_(k−1)v + 1 | Base-(k−1) digits, one symbol reserved as a full stop |
| **Continuation** | \|Σ\| ≥ 4 | log_(k/2)v | LEB128 generalized: the top half of Σ marks the last digit |

The terminator spends a symbol of the alphabet, which sounds wasteful and often
isn't: at |Σ| = 4 the radix drops from 4 to 3, while continuation's flag bit
drops it to 2. Large alphabets flip the answer — at |Σ| = 256 continuation's
radix-128 wins easily.

Rather than guess, `Alphabet` **measures**: at construction it costs both codes
over a spread of realistic run lengths and keeps the cheaper. DNA gets
`terminated`, bytes get `continuation`, and binary gets gamma because every
base-r scheme degenerates to unary when r would be 1.

## Four codecs, each solving something the others don't

### `pair` — and one observation worth the whole codec

`count, symbol` per run. But the symbol after a run can never *equal* the run's
own symbol, so it carries log₂(k−1) bits, not log₂(k). For k > 2 that's a
fractional saving no symbol-oriented encoder can bank.

**For k = 2 it is zero bits.** Runs must alternate, so the symbol is entirely
redundant and simply isn't emitted. What's left — a leading symbol, then nothing
but gamma-coded run lengths — is exactly the classic bi-level / fax run coding,
arrived at by deletion rather than by special-casing. On synthetic scan data:

```
codec         symbols out    ratio   packed bytes
pair                8,436    0.071          1,057
packbits           10,813    0.091          1,354
zlib -9                                     1,082
```

20% better than PackBits on the data faxes were built for, and again ahead of
zlib.

### `packbits` — the one that doesn't blow up

`pair`'s worst case is 2× expansion: every length-1 run costs a count *and* a
symbol. PackBits fixes that with literal blocks, and the classic TIFF version
guarantees "no worse than +1 byte per 128" — a limit forced by its fixed
one-byte control word.

Variable-length counts remove the 128 cap. A control value encodes kind and
length together as `2·(n−1) + is_run`, and the alphabet's count code makes it
self-delimiting. So an *entirely* incompressible input costs **one control
symbol for the whole file**:

```
20,000 random bytes -> 20,003 symbols   (+3, not +156)
```

Run-versus-literal is decided by comparing the two encodings exactly, using the
alphabet's real count lengths, rather than a hard-coded "runs of 3 or more". On
a 256-symbol alphabet that reproduces the classic threshold; on `{0,1}` it
correctly demands longer runs before opening a block.

> The greedy choice is local. A full DP over the run sequence would be exactly
> optimal, but literal blocks merge subadditively so the gap is at most a symbol
> or two per block — not worth the complexity here.

### `escape` — nothing at all for the literal stretches

The PCX / BMP-RLE shape: output is literal, with one reserved symbol
introducing a run. It wins where data is mostly incompressible with rare long
runs, because unlike PackBits it pays **nothing** for literal stretches — no
control word, no block framing.

Two details make it correct rather than merely plausible:

- **The escape symbol must itself be escapable.** `ESC` + a count of 0 means
  "one literal ESC"; non-zero means "read the symbol, repeat it". The obvious
  `ESC ESC` encoding is ambiguous with a run *of* the escape symbol — a bug the
  tests specifically hunt for.
- **The escape defaults to the rarest symbol in the data**, since every literal
  occurrence costs extra.

### `adaptive` — because choosing statically is choosing wrong

Runs every applicable codec, keeps the shortest, tags the winner with one
symbol. Which one wins is genuinely data-dependent: DNA with long homopolymer
tracts goes to `pair`, a noisy byte stream with one long zero fill goes to
`escape`, mixed data goes to `packbits`. One symbol of overhead for a
best-of-N guarantee.

## Getting to actual bytes

A symbol from a 4-letter alphabet is 2 bits of information but a whole byte of
Python object, so a compressor measured only in symbols measures the wrong
thing. `BitPacker` closes the gap.

Naive `ceil(log2 k)` bits per symbol wastes 0.42 bits on every symbol of a
3-letter alphabet. Instead it converts groups of `m` symbols to a `b`-bit
integer, choosing the (m, b) that packs the most symbols per bit subject to
`k^m ≤ 2^b`:

| Alphabet | Packing | bits/symbol | Ideal |
| --- | --- | --- | --- |
| binary | 8 per byte | 1.000 | 1.000 |
| DNA | 4 per byte | 2.000 | 2.000 |
| 3 symbols | 5 per byte | 1.600 | 1.585 |
| 10 digits | 12 per 40 bits | 3.333 | 3.322 |

So "DNA at 2 bits per base" is a real file on disk, not a talking point.

## Honest baselines

`analyze` reports two things that keep the result truthful:

- **zlib -9** on the same bit-packed data, so the comparison is like for like
  rather than RLE-in-symbols against zlib-on-ASCII.
- **The entropy of the run-length distribution** — a genuine lower bound for any
  coder that models only run lengths. It says how much of the remaining gap is
  the codec's fault and how much is the data's.

## Symbols are anything hashable

Nothing in the module assumes text. Characters, byte values, enum members,
tuples of note-and-duration:

```python
notes = [("C", 4), ("E", 8), ("G", 4), ("A", 2)]
alpha = Alphabet(notes)
compress([notes[0]] * 30 + [notes[1]] * 12, alpha, "packbits")
```

The *file container* is narrower on purpose: it stores `str` and `int` symbols
and raises a clear error for anything else, because `repr`-ing a tuple would
produce a file that decodes to the wrong data. Failing loudly beats corrupting
quietly.

Containers are self-describing — a JSON header carries the alphabet, so a reader
that has never seen `♠♡♢♣` still decodes the file.

## Run it

```bash
cd "challenges/Algorithmic Challenges/Run-Length Encoding Compressor with Custom Alphabets"

uv run python rle.py --self-check
uv run python rle.py analyze genome.txt --alphabet dna
uv run python rle.py compress genome.txt genome.rle --alphabet dna
uv run python rle.py decompress genome.rle genome.out
uv run python rle.py analyze scan.bin --binary --alphabet bytes

uv run --with pytest pytest -q      # 126 tests
```

Standard library only (`zlib` is stdlib). Named alphabets: `binary`, `bits`,
`dna`, `rna`, `protein`, `digits`, `lower`, `bytes` — or `auto` to infer from
the data, or pass any `Alphabet` you build yourself.

## Tests

The load-bearing property is one line, fuzzed hard: for every codec and every
alphabet, `decompress(compress(x)) == x` **and every symbol of the compressed
form is itself in the alphabet**. That second half is the one the naive
implementation fails, so it's asserted on every round trip, at alphabet sizes
2, 3, 4, 5, 7, 16 and 256, over both run-heavy and pure-noise inputs.

The rest covers the pieces that are easy to get subtly wrong: count codes
decoding correctly when *concatenated* with no separator, truncated streams
raising rather than returning garbage, runs of the escape symbol, and the
expansion bound holding on genuinely incompressible input.
