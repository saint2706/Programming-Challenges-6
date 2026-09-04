"""Run-length encoding over arbitrary symbol alphabets.

The brief is "support arbitrary symbol sets, not just text", and the constraint
that makes that interesting is not the *input* -- any RLE reads any sequence.
It is the **output**. Textbook RLE writes ``(3, 'A')`` and quietly assumes the
count can be spelled with characters that are not in the data's alphabet. Over
``{A, C, G, T}`` there is no '3' to write, and no separator either.

So everything here is closed over the alphabet: an encoder given symbols from
Sigma emits only symbols from Sigma, which means run lengths have to be spelled
in base |Sigma| with a self-delimiting code. That single rule is what turns a
weekend exercise into something with real design decisions in it.

Four codecs, all closed, all round-trip exact:

    pair       (count, symbol) per run. Simple; up to 2x expansion on noise.
               On a *binary* alphabet it degrades gracefully into Elias-gamma
               run coding -- the classic fax scheme -- because the symbol after
               a run is not free information, it is no information at all.
    packbits   Literal/run blocks, TIFF-style. Worst case +1 symbol *total*,
               not +1 per 128, because the counts are variable-length.
    escape     Mostly-literal output with an escape symbol introducing runs.
               Right when data is incompressible with rare long runs.
    adaptive   Tries each of the above per block and tags the winner. Costs one
               symbol; guarantees best-of.

Plus :class:`BitPacker`, which turns a symbol sequence into actual bytes at
close to log2(|Sigma|) bits per symbol -- so "DNA at 2 bits per base" is a real
file on disk rather than a talking point.

Run directly for a self-check and a worked example:

    uv run python rle.py --self-check
    uv run python rle.py analyze genome.txt --alphabet dna
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import zlib
from dataclasses import dataclass
from itertools import groupby
from typing import Hashable, Iterable, Iterator, Sequence

__all__ = [
    "Alphabet",
    "NAMED_ALPHABETS",
    "CountCode",
    "GammaCount",
    "TerminatedCount",
    "ContinuationCount",
    "choose_count_code",
    "Codec",
    "PairCodec",
    "PackBitsCodec",
    "EscapeCodec",
    "AdaptiveCodec",
    "CODECS",
    "BitPacker",
    "runs",
    "compress",
    "decompress",
    "analyze",
    "Analysis",
    "CodecReport",
    "pack_file",
    "unpack_file",
]


# ---------------------------------------------------------------------------
# Counting in base |Sigma|
# ---------------------------------------------------------------------------


class CountCode:
    """A self-delimiting code for non-negative integers over |Sigma| symbols.

    "Self-delimiting" is the whole problem. Writing 12 as ``[1, 2]`` is useless
    if the decoder cannot tell where the number stops -- and there is no spare
    separator symbol, because every symbol in Sigma can legitimately appear in
    the data.
    """

    name = "abstract"

    def encode(self, value: int, k: int) -> list[int]:
        raise NotImplementedError

    def decode(self, idx: Sequence[int], pos: int, k: int) -> tuple[int, int]:
        raise NotImplementedError

    def length(self, value: int, k: int) -> int:
        return len(self.encode(value, k))


class GammaCount(CountCode):
    """Elias gamma, the only sensible choice when |Sigma| = 2.

    ``v+1`` written as (bit_length-1) zeros then its own bits: 2*log2(v) + 1
    symbols, and no radix to waste. This is what makes a binary alphabet work
    at all -- every base-r scheme degenerates to unary when r would be 1.
    """

    name = "gamma"

    def encode(self, value: int, k: int) -> list[int]:
        if value < 0:
            raise ValueError("counts must be non-negative")
        v = value + 1
        bits = v.bit_length()
        out = [0] * (bits - 1)
        out.extend((v >> i) & 1 for i in range(bits - 1, -1, -1))
        return out

    def decode(self, idx: Sequence[int], pos: int, k: int) -> tuple[int, int]:
        zeros = 0
        while pos < len(idx) and idx[pos] == 0:
            zeros += 1
            pos += 1
        if pos >= len(idx):
            raise ValueError("truncated gamma code")
        v = 1
        pos += 1
        for _ in range(zeros):
            if pos >= len(idx):
                raise ValueError("truncated gamma code")
            v = (v << 1) | idx[pos]
            pos += 1
        return v - 1, pos

    def length(self, value: int, k: int) -> int:
        return 2 * (value + 1).bit_length() - 1


class TerminatedCount(CountCode):
    """Base-(k-1) digits followed by a reserved terminator symbol.

    Spends one symbol of the alphabet as a full stop, which sounds wasteful and
    usually is not: at |Sigma| = 4 the radix only drops from 4 to 3, while the
    alternative (splitting the alphabet in half for continuation flags) drops it
    to 2. Small alphabets prefer the terminator; large ones prefer continuation.
    """

    name = "terminated"

    def encode(self, value: int, k: int) -> list[int]:
        if value < 0:
            raise ValueError("counts must be non-negative")
        radix = k - 1
        out: list[int] = []
        while True:
            out.append(value % radix)
            value //= radix
            if value == 0:
                break
        out.append(k - 1)  # terminator
        return out

    def decode(self, idx: Sequence[int], pos: int, k: int) -> tuple[int, int]:
        radix = k - 1
        value = 0
        scale = 1
        while True:
            if pos >= len(idx):
                raise ValueError("truncated count")
            d = idx[pos]
            pos += 1
            if d == radix:
                return value, pos
            value += d * scale
            scale *= radix


class ContinuationCount(CountCode):
    """Base-(k//2) digits; the top half of the alphabet marks the last digit.

    The generalization of LEB128 to an arbitrary radix. No symbol is reserved,
    so nothing is spent on a terminator, but the effective radix halves. Needs
    |Sigma| >= 4 for the two halves to be distinguishable.
    """

    name = "continuation"

    def encode(self, value: int, k: int) -> list[int]:
        if value < 0:
            raise ValueError("counts must be non-negative")
        radix = k // 2
        digits: list[int] = []
        while True:
            digits.append(value % radix)
            value //= radix
            if value == 0:
                break
        out = digits[:-1]
        out.append(radix + digits[-1])
        return out

    def decode(self, idx: Sequence[int], pos: int, k: int) -> tuple[int, int]:
        radix = k // 2
        value = 0
        scale = 1
        while True:
            if pos >= len(idx):
                raise ValueError("truncated count")
            d = idx[pos]
            pos += 1
            if d >= radix:
                return value + (d - radix) * scale, pos
            value += d * scale
            scale *= radix


# Counts that a real run-length distribution actually produces, used to pick
# between the two schemes at alphabet-construction time.
_COUNT_PROBE = [1, 2, 3, 5, 8, 16, 40, 100, 500, 4000, 100_000, 10**7]


def choose_count_code(k: int) -> CountCode:
    """Pick the cheapest self-delimiting count code for an alphabet of size k."""
    if k < 2:
        raise ValueError("an alphabet needs at least two symbols")
    if k == 2:
        return GammaCount()
    candidates: list[CountCode] = [TerminatedCount()]
    if k >= 4:
        candidates.append(ContinuationCount())
    return min(candidates, key=lambda c: sum(c.length(v, k) for v in _COUNT_PROBE))


# ---------------------------------------------------------------------------
# Alphabets
# ---------------------------------------------------------------------------


class Alphabet:
    """An ordered set of symbols, and the arithmetic that goes with it.

    Symbols may be anything hashable -- characters, byte values, enum members,
    tuples of note-and-duration. Nothing in this module assumes text.
    """

    __slots__ = ("symbols", "index", "count_code", "_size")

    def __init__(
        self, symbols: Iterable[Hashable], count_code: CountCode | None = None
    ) -> None:
        seen: dict[Hashable, int] = {}
        for s in symbols:
            if s in seen:
                raise ValueError(f"duplicate symbol in alphabet: {s!r}")
            seen[s] = len(seen)
        if len(seen) < 2:
            raise ValueError("an alphabet needs at least two distinct symbols")
        self.symbols: tuple[Hashable, ...] = tuple(seen)
        self.index = seen
        self._size = len(seen)
        self.count_code = count_code or choose_count_code(self._size)

    def __len__(self) -> int:
        return self._size

    def __contains__(self, symbol: Hashable) -> bool:
        return symbol in self.index

    def __repr__(self) -> str:
        head = "".join(map(str, self.symbols[:8]))
        tail = "..." if self._size > 8 else ""
        return f"<Alphabet size={self._size} [{head}{tail}] counts={self.count_code.name}>"

    @property
    def bits_per_symbol(self) -> float:
        """Information-theoretic floor for one symbol, ignoring symbol statistics."""
        return math.log2(self._size)

    def to_indices(self, data: Iterable[Hashable]) -> list[int]:
        idx = self.index
        try:
            return [idx[s] for s in data]
        except KeyError as exc:
            raise ValueError(f"symbol not in alphabet: {exc.args[0]!r}") from None

    def to_symbols(self, indices: Iterable[int]) -> list[Hashable]:
        syms = self.symbols
        return [syms[i] for i in indices]

    def encode_count(self, value: int) -> list[int]:
        return self.count_code.encode(value, self._size)

    def decode_count(self, idx: Sequence[int], pos: int) -> tuple[int, int]:
        return self.count_code.decode(idx, pos, self._size)

    def count_length(self, value: int) -> int:
        return self.count_code.length(value, self._size)

    # -- constructors --------------------------------------------------------

    @classmethod
    def of(cls, data: Iterable[Hashable], **kwargs) -> "Alphabet":
        """Infer the alphabet from the data, in order of first appearance."""
        seen: dict[Hashable, None] = {}
        for s in data:
            seen.setdefault(s, None)
        return cls(seen, **kwargs)

    @classmethod
    def named(cls, name: str) -> "Alphabet":
        try:
            return NAMED_ALPHABETS[name]()
        except KeyError:
            raise ValueError(
                f"unknown alphabet {name!r}; choose from {', '.join(NAMED_ALPHABETS)}"
            ) from None


NAMED_ALPHABETS = {
    "binary": lambda: Alphabet("01"),
    "bits": lambda: Alphabet([0, 1]),
    "dna": lambda: Alphabet("ACGT"),
    "rna": lambda: Alphabet("ACGU"),
    "protein": lambda: Alphabet("ACDEFGHIKLMNPQRSTVWY"),
    "digits": lambda: Alphabet("0123456789"),
    "lower": lambda: Alphabet("abcdefghijklmnopqrstuvwxyz"),
    "bytes": lambda: Alphabet(range(256)),
}


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def runs(data: Sequence[Hashable]) -> Iterator[tuple[Hashable, int]]:
    """``[A, A, A, C]`` -> ``(A, 3), (C, 1)``."""
    for symbol, group in groupby(data):
        yield symbol, sum(1 for _ in group)


# ---------------------------------------------------------------------------
# Codecs
# ---------------------------------------------------------------------------


class Codec:
    """Encode and decode index sequences, closed over the alphabet."""

    name = "abstract"

    def encode(self, idx: Sequence[int], alpha: Alphabet) -> list[int]:
        raise NotImplementedError

    def decode(self, idx: Sequence[int], alpha: Alphabet) -> list[int]:
        raise NotImplementedError

    def applicable(self, alpha: Alphabet) -> bool:
        return True


class PairCodec(Codec):
    """``count, symbol`` per run -- and one observation that matters.

    The symbol after a run can never equal the run's own symbol, so it carries
    log2(k-1) bits, not log2(k). For k > 2 that is a fractional saving no
    symbol-oriented encoder can bank. For **k = 2 it is zero bits**: the runs
    must alternate, so the symbol is entirely redundant and simply is not
    emitted. What is left -- a leading symbol and then nothing but gamma-coded
    run lengths -- is exactly the classic bi-level/fax run coding, arrived at
    by deletion rather than by special-casing.

    Worst case: all runs of length 1 gives 2 symbols per input symbol (k > 2),
    which is why :class:`PackBitsCodec` exists.
    """

    name = "pair"

    def encode(self, idx: Sequence[int], alpha: Alphabet) -> list[int]:
        if not idx:
            return []
        binary = len(alpha) == 2
        out: list[int] = []
        first = True
        for symbol, count in runs(idx):
            out.extend(alpha.encode_count(count))
            if first or not binary:
                out.append(symbol)
                first = False
        return out

    def decode(self, idx: Sequence[int], alpha: Alphabet) -> list[int]:
        if not idx:
            return []
        binary = len(alpha) == 2
        out: list[int] = []
        pos = 0
        symbol = -1
        while pos < len(idx):
            count, pos = alpha.decode_count(idx, pos)
            if symbol < 0 or not binary:
                if pos >= len(idx):
                    raise ValueError("truncated pair stream: missing symbol")
                symbol = idx[pos]
                pos += 1
            else:
                symbol ^= 1  # binary runs alternate; nothing was stored
            out.extend([symbol] * count)
        return out


class PackBitsCodec(Codec):
    """Literal and run blocks, TIFF PackBits generalized to any alphabet.

    A control value encodes both the block kind and its length as one number,
    ``2*(n-1) + is_run``, which the alphabet's own count code makes
    self-delimiting.

    The reason this is the default: PackBits' famous guarantee is "no worse than
    +1 byte per 128", forced by its fixed one-byte control word. Variable-length
    counts remove the 128 cap, so an *entirely* incompressible input costs
    **one control symbol in total** -- +1 on the whole file, not +1 per block.

    Run-versus-literal is decided by comparing the two encodings exactly, using
    the alphabet's real count lengths, rather than by a hard-coded "runs of 3 or
    more". On a 256-symbol alphabet that reproduces the classic threshold; on
    ``{0,1}`` it correctly demands longer runs before a block is worth opening.
    """

    name = "packbits"

    def encode(self, idx: Sequence[int], alpha: Alphabet) -> list[int]:
        if not idx:
            return []
        out: list[int] = []
        literal: list[int] = []

        def flush() -> None:
            if literal:
                out.extend(alpha.encode_count(2 * (len(literal) - 1)))
                out.extend(literal)
                literal.clear()

        for symbol, count in runs(idx):
            # A run block costs control + 1 symbol. Keeping it literal costs
            # `count` symbols, plus a fresh control word for whatever follows
            # if this split a literal run in two.
            run_cost = alpha.count_length(2 * (count - 1) + 1) + 1
            split_penalty = alpha.count_length(0) if literal else 0
            if run_cost + split_penalty < count:
                flush()
                out.extend(alpha.encode_count(2 * (count - 1) + 1))
                out.append(symbol)
            else:
                literal.extend([symbol] * count)
        flush()
        return out

    def decode(self, idx: Sequence[int], alpha: Alphabet) -> list[int]:
        out: list[int] = []
        pos = 0
        while pos < len(idx):
            control, pos = alpha.decode_count(idx, pos)
            n = (control >> 1) + 1
            if control & 1:
                if pos >= len(idx):
                    raise ValueError("truncated packbits stream: missing run symbol")
                out.extend([idx[pos]] * n)
                pos += 1
            else:
                if pos + n > len(idx):
                    raise ValueError("truncated packbits stream: short literal block")
                out.extend(idx[pos : pos + n])
                pos += n
        return out


class EscapeCodec(Codec):
    """Literal output with one symbol reserved to introduce a run.

    The PCX/BMP-RLE shape. It wins where most of the data is incompressible but
    rare long runs exist, because unlike PackBits it pays *nothing at all* for
    the literal stretches -- no control word, no block framing.

    Two details make it correct rather than merely plausible:

    * The escape symbol must be escapable. ``ESC`` followed by a count of 0
      means "one literal ESC"; a non-zero count means "read the symbol and
      repeat it". Encoding the literal case as ``ESC ESC`` instead would be
      ambiguous with a run *of* the escape symbol.
    * The escape symbol defaults to the rarest one in the data, since every
      literal occurrence of it costs extra.
    """

    name = "escape"

    def __init__(self, escape: int | None = None) -> None:
        self.escape = escape

    def _pick_escape(self, idx: Sequence[int], alpha: Alphabet) -> int:
        if self.escape is not None:
            return self.escape
        freq = [0] * len(alpha)
        for i in idx:
            freq[i] += 1
        return min(range(len(alpha)), key=lambda i: freq[i])

    def encode(self, idx: Sequence[int], alpha: Alphabet) -> list[int]:
        if not idx:
            return []
        esc = self._pick_escape(idx, alpha)
        out = [esc]  # header: which symbol is the escape
        for symbol, count in runs(idx):
            run_cost = 1 + alpha.count_length(count) + 1
            literal_cost = count * (3 if symbol == esc else 1)
            if run_cost < literal_cost:
                out.append(esc)
                out.extend(alpha.encode_count(count))
                out.append(symbol)
            elif symbol == esc:
                for _ in range(count):
                    out.append(esc)
                    out.extend(alpha.encode_count(0))
            else:
                out.extend([symbol] * count)
        return out

    def decode(self, idx: Sequence[int], alpha: Alphabet) -> list[int]:
        if not idx:
            return []
        esc = idx[0]
        out: list[int] = []
        pos = 1
        while pos < len(idx):
            symbol = idx[pos]
            pos += 1
            if symbol != esc:
                out.append(symbol)
                continue
            count, pos = alpha.decode_count(idx, pos)
            if count == 0:
                out.append(esc)
                continue
            if pos >= len(idx):
                raise ValueError("truncated escape stream: missing run symbol")
            out.extend([idx[pos]] * count)
            pos += 1
        return out

    def applicable(self, alpha: Alphabet) -> bool:
        # On a two-symbol alphabet, reserving one for escapes leaves one symbol
        # of literal data -- technically valid, always terrible.
        return len(alpha) > 2


class AdaptiveCodec(Codec):
    """Run every applicable codec and keep the shortest, tagged by one symbol.

    One symbol of overhead for a guarantee of best-of-N. Which codec wins is
    genuinely data-dependent -- runs of DNA go to ``pair``, a noisy byte stream
    with one long zero fill goes to ``escape``, mixed data goes to
    ``packbits`` -- so choosing statically is choosing wrong some of the time.
    """

    name = "adaptive"

    def __init__(self, members: Sequence[Codec] | None = None) -> None:
        self.members = list(members) if members else [
            PairCodec(), PackBitsCodec(), EscapeCodec()
        ]

    def _usable(self, alpha: Alphabet) -> list[Codec]:
        usable = [c for c in self.members if c.applicable(alpha)]
        if len(usable) > len(alpha):
            raise ValueError("alphabet too small to tag this many codecs")
        return usable

    def encode(self, idx: Sequence[int], alpha: Alphabet) -> list[int]:
        if not idx:
            return []
        usable = self._usable(alpha)
        best_tag, best_out = 0, None
        for tag, codec in enumerate(usable):
            out = codec.encode(idx, alpha)
            if best_out is None or len(out) < len(best_out):
                best_tag, best_out = tag, out
        return [best_tag] + (best_out or [])

    def decode(self, idx: Sequence[int], alpha: Alphabet) -> list[int]:
        if not idx:
            return []
        usable = self._usable(alpha)
        tag = idx[0]
        if tag >= len(usable):
            raise ValueError(f"unknown codec tag {tag}")
        return usable[tag].decode(idx[1:], alpha)

    def chosen(self, idx: Sequence[int], alpha: Alphabet) -> str:
        """Which member codec would win. Diagnostics only."""
        if not idx:
            return "none"
        usable = self._usable(alpha)
        return min(usable, key=lambda c: len(c.encode(idx, alpha))).name


CODECS: dict[str, Codec] = {
    c.name: c for c in (PairCodec(), PackBitsCodec(), EscapeCodec(), AdaptiveCodec())
}


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------


def _resolve_codec(codec: "Codec | str") -> Codec:
    if isinstance(codec, Codec):
        return codec
    try:
        return CODECS[codec]
    except KeyError:
        raise ValueError(
            f"unknown codec {codec!r}; choose from {', '.join(CODECS)}"
        ) from None


def compress(
    data: Sequence[Hashable],
    alphabet: "Alphabet | str | None" = None,
    codec: "Codec | str" = "adaptive",
) -> tuple[list[Hashable], Alphabet]:
    """Compress a symbol sequence; the output uses the same alphabet."""
    alpha = _resolve_alphabet(alphabet, data)
    out = _resolve_codec(codec).encode(alpha.to_indices(data), alpha)
    return alpha.to_symbols(out), alpha


def decompress(
    data: Sequence[Hashable],
    alphabet: Alphabet,
    codec: "Codec | str" = "adaptive",
) -> list[Hashable]:
    """Inverse of :func:`compress`."""
    idx = alphabet.to_indices(data)
    return alphabet.to_symbols(_resolve_codec(codec).decode(idx, alphabet))


def _resolve_alphabet(
    alphabet: "Alphabet | str | None", data: Sequence[Hashable]
) -> Alphabet:
    if isinstance(alphabet, Alphabet):
        return alphabet
    if alphabet is None or alphabet == "auto":
        return Alphabet.of(data)
    return Alphabet.named(alphabet)


# ---------------------------------------------------------------------------
# Getting to actual bytes
# ---------------------------------------------------------------------------


class BitPacker:
    """Pack symbols into bytes at close to log2(|Sigma|) bits each.

    A symbol from a 4-letter alphabet is 2 bits of information but a whole byte
    of Python object, so a compressor measured only in symbols is measuring the
    wrong thing. This closes the gap.

    Rather than ceil(log2 k) bits per symbol -- which wastes 0.42 bits on every
    symbol of a 3-letter alphabet -- it converts groups of ``m`` symbols to a
    ``b``-bit integer, choosing the (m, b) pair that packs the most symbols per
    bit subject to k^m <= 2^b. For DNA that is 4 symbols per byte exactly; for a
    3-letter alphabet, 5 symbols per byte (1.600 bits each against an ideal
    1.585).
    """

    def __init__(self, alphabet: Alphabet) -> None:
        self.alphabet = alphabet
        k = len(alphabet)
        best = None
        for bits in (8, 16, 24, 32, 40, 48, 56, 64):
            m = 0
            while k ** (m + 1) <= (1 << bits):
                m += 1
            if m and (best is None or m / bits > best[0] / best[1]):
                best = (m, bits)
        self.group, self.bits = best  # type: ignore[misc]
        self.nbytes = self.bits // 8

    @property
    def bits_per_symbol(self) -> float:
        return self.bits / self.group

    def pack(self, symbols: Sequence[Hashable]) -> bytes:
        idx = self.alphabet.to_indices(symbols)
        k = len(self.alphabet)
        out = bytearray(_leb128(len(idx)))
        for start in range(0, len(idx), self.group):
            chunk = idx[start : start + self.group]
            value = 0
            for d in reversed(chunk):
                value = value * k + d
            out.extend(value.to_bytes(self.nbytes, "big"))
        return bytes(out)

    def unpack(self, blob: bytes) -> list[Hashable]:
        n, pos = _un_leb128(blob, 0)
        k = len(self.alphabet)
        idx: list[int] = []
        while len(idx) < n:
            value = int.from_bytes(blob[pos : pos + self.nbytes], "big")
            pos += self.nbytes
            for _ in range(min(self.group, n - len(idx))):
                idx.append(value % k)
                value //= k
        return self.alphabet.to_symbols(idx)


def _leb128(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def _un_leb128(blob: bytes, pos: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if pos >= len(blob):
            raise ValueError("truncated length header")
        byte = blob[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


@dataclass
class CodecReport:
    name: str
    symbols: int
    ratio: float
    packed_bytes: int
    error: str = ""


@dataclass
class Analysis:
    alphabet_size: int
    count_code: str
    n_symbols: int
    n_runs: int
    mean_run: float
    max_run: int
    bits_per_symbol: float
    packed_bits_per_symbol: float
    raw_packed_bytes: int
    zlib_bytes: int
    run_entropy_bits: float
    codecs: list[CodecReport]

    def render(self) -> str:
        lines = [
            f"symbols          {self.n_symbols:,}",
            f"alphabet         {self.alphabet_size} symbols, "
            f"{self.bits_per_symbol:.3f} bits each (ideal)",
            f"count code       {self.count_code}",
            f"runs             {self.n_runs:,}  "
            f"(mean {self.mean_run:.2f}, longest {self.max_run:,})",
            f"packing          {self.packed_bits_per_symbol:.3f} bits/symbol "
            f"-> {self.raw_packed_bytes:,} bytes uncompressed",
            f"run-length entropy {self.run_entropy_bits:,.0f} bits "
            f"(lower bound for any run-length coder)",
            "",
            f"{'codec':<12}{'symbols out':>13}{'ratio':>9}{'packed bytes':>15}",
            "-" * 49,
        ]
        for c in self.codecs:
            if c.error:
                lines.append(f"{c.name:<12}{c.error:>37}")
            else:
                lines.append(
                    f"{c.name:<12}{c.symbols:>13,}{c.ratio:>9.3f}{c.packed_bytes:>15,}"
                )
        lines.append(f"{'zlib -9':<12}{'':>13}{'':>9}{self.zlib_bytes:>15,}")
        return "\n".join(lines)


def analyze(
    data: Sequence[Hashable], alphabet: "Alphabet | str | None" = None
) -> Analysis:
    """Measure every codec on this data, with honest baselines."""
    alpha = _resolve_alphabet(alphabet, data)
    idx = alpha.to_indices(data)
    run_list = list(runs(idx))
    lengths = [c for _, c in run_list] or [0]
    packer = BitPacker(alpha)

    # Entropy of the run-length distribution: no run-length coder can beat this
    # without modelling something other than run lengths.
    total = sum(lengths)
    freq: dict[int, int] = {}
    for length in lengths:
        freq[length] = freq.get(length, 0) + 1
    n = len(lengths)
    entropy = -sum(c / n * math.log2(c / n) for c in freq.values()) * n if n else 0.0

    reports = []
    for name, codec in CODECS.items():
        if not codec.applicable(alpha):
            reports.append(CodecReport(name, 0, 0.0, 0, "n/a for this alphabet"))
            continue
        out = codec.encode(idx, alpha)
        reports.append(
            CodecReport(
                name,
                len(out),
                len(out) / len(idx) if idx else 0.0,
                len(packer.pack(alpha.to_symbols(out))),
            )
        )

    raw = packer.pack(data)
    return Analysis(
        alphabet_size=len(alpha),
        count_code=alpha.count_code.name,
        n_symbols=len(idx),
        n_runs=len(run_list),
        mean_run=total / len(run_list) if run_list else 0.0,
        max_run=max(lengths),
        bits_per_symbol=alpha.bits_per_symbol,
        packed_bits_per_symbol=packer.bits_per_symbol,
        raw_packed_bytes=len(raw),
        zlib_bytes=len(zlib.compress(raw, 9)),
        run_entropy_bits=entropy,
        codecs=reports,
    )


# ---------------------------------------------------------------------------
# File container
# ---------------------------------------------------------------------------

MAGIC = b"RLE1"


def pack_file(data: Sequence[Hashable], alpha: Alphabet, codec_name: str) -> bytes:
    """Self-describing container: magic, JSON header, then bit-packed payload.

    The header carries the alphabet, so a ``.rle`` file needs nothing else to
    be decoded -- including a custom alphabet the reader has never seen.

    Only ``str`` and ``int`` symbols can travel in a file; the in-memory API
    accepts any hashable, but a tuple or an enum member has no portable
    serialization and silently ``repr``-ing it would produce a container that
    decodes to the wrong data.
    """
    bad = next((s for s in alpha.symbols if not isinstance(s, (str, int))), None)
    if bad is not None:
        raise ValueError(
            f"symbol {bad!r} of type {type(bad).__name__} cannot be stored in a "
            "container; the file format supports str and int symbols only"
        )
    encoded, _ = compress(data, alpha, codec_name)
    body = BitPacker(alpha).pack(encoded)
    header = json.dumps(
        {"codec": codec_name, "symbols": list(alpha.symbols)}, separators=(",", ":")
    ).encode("utf-8")
    return MAGIC + _leb128(len(header)) + header + body


def unpack_file(blob: bytes) -> list[Hashable]:
    if not blob.startswith(MAGIC):
        raise ValueError("not an RLE1 container")
    size, pos = _un_leb128(blob, len(MAGIC))
    meta = json.loads(blob[pos : pos + size].decode("utf-8"))
    pos += size
    alpha = Alphabet(meta["symbols"])
    encoded = BitPacker(alpha).unpack(blob[pos:])
    return decompress(encoded, alpha, meta["codec"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _read_symbols(path: str, binary: bool) -> list[Hashable]:
    if path == "-":
        raw = sys.stdin.buffer.read()
    else:
        with open(path, "rb") as fh:
            raw = fh.read()
    return list(raw) if binary else list(raw.decode("utf-8"))


def _self_check() -> int:
    cases: list[tuple[str, Sequence[Hashable], "Alphabet | str"]] = [
        ("empty", "", Alphabet("ab")),
        ("single symbol", "a", Alphabet("ab")),
        ("one long run", "A" * 10_000, "dna"),
        ("no runs at all", "ACGT" * 500, "dna"),
        ("mixed", "A" * 300 + "CGCGCG" + "T" * 5000 + "ACGT", "dna"),
        ("binary sparse", "0" * 5000 + "1" + "0" * 5000, "binary"),
        ("binary alternating", "01" * 2000, "binary"),
        ("bytes with a fill", None, "bytes"),
        ("text", "aaabbbcccaaa" * 100, "auto"),
        ("unicode", "🙂🙂🙂🙃" * 50, "auto"),
        ("tuple symbols", None, None),
    ]
    failures = 0
    for label, data, alpha_spec in cases:
        if label == "bytes with a fill":
            data = list(bytes(range(256))) + [0] * 4000 + list(bytes(range(256)))
            alpha_spec = "bytes"
        elif label == "tuple symbols":
            # Symbols do not have to be characters.
            data = [("C", 4)] * 30 + [("E", 8)] * 12 + [("G", 4)] * 30
            alpha_spec = Alphabet([("C", 4), ("E", 8), ("G", 4), ("A", 2)])

        alpha = _resolve_alphabet(alpha_spec, data)
        ok = True
        for name in CODECS:
            if not CODECS[name].applicable(alpha):
                continue
            encoded, _ = compress(data, alpha, name)
            if any(s not in alpha for s in encoded):
                print(f"  [FAIL] {label}/{name}: output escaped the alphabet")
                ok = False
                continue
            back = decompress(encoded, alpha, name)
            if back != list(data):
                print(f"  [FAIL] {label}/{name}: round trip differs")
                ok = False
        if ok:
            best = min(
                (len(compress(data, alpha, n)[0]) for n in CODECS
                 if CODECS[n].applicable(alpha)),
                default=0,
            )
            ratio = f"{best / len(data):.3f}x" if data else "n/a"
            print(f"  [ok  ] {label:<20} |Sigma|={len(alpha):<4} best {ratio}")
        else:
            failures += 1

    # The bounded-expansion promise, on genuinely incompressible input.
    import random

    rng = random.Random(20260904)
    alpha = Alphabet.named("bytes")
    noise = [rng.randrange(256) for _ in range(20_000)]
    grown = len(compress(noise, alpha, "packbits")[0]) - len(noise)
    if grown > 4:
        print(f"  [FAIL] packbits grew incompressible data by {grown} symbols")
        failures += 1
    else:
        print(f"  [ok  ] packbits on 20k random bytes: +{grown} symbols total")

    # Bit packing must be exact and lossless.
    for name in ("binary", "dna", "digits", "bytes"):
        a = Alphabet.named(name)
        packer = BitPacker(a)
        sample = [a.symbols[rng.randrange(len(a))] for _ in range(1000)]
        if packer.unpack(packer.pack(sample)) != sample:
            print(f"  [FAIL] BitPacker round trip failed for {name}")
            failures += 1
    print("  [ok  ] BitPacker round-trips binary / dna / digits / bytes")

    print("all self-checks passed" if not failures else f"{failures} failure(s)")
    return 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="rle", description="Run-length encoding over arbitrary alphabets."
    )
    sub = ap.add_subparsers(dest="command")

    def add_common(p):
        p.add_argument("input", help="input file, or '-' for stdin")
        p.add_argument("--binary", action="store_true",
                       help="treat the file as bytes rather than UTF-8 text")
        p.add_argument("--alphabet", default="auto",
                       help=f"auto, or one of: {', '.join(NAMED_ALPHABETS)}")

    c = sub.add_parser("compress", help="write an .rle container")
    add_common(c)
    c.add_argument("output")
    c.add_argument("--codec", default="adaptive", choices=list(CODECS))

    d = sub.add_parser("decompress", help="read an .rle container")
    d.add_argument("input")
    d.add_argument("output")

    a = sub.add_parser("analyze", help="compare every codec on this data")
    add_common(a)
    a.add_argument("--json", action="store_true")

    ap.add_argument("--self-check", action="store_true", help="run the built-in demo")
    args = ap.parse_args(argv)

    if args.self_check or args.command is None:
        return _self_check()

    if args.command == "decompress":
        with open(args.input, "rb") as fh:
            symbols = unpack_file(fh.read())
        out = (bytes(symbols) if all(isinstance(s, int) for s in symbols)
               else "".join(map(str, symbols)).encode("utf-8"))
        with open(args.output, "wb") as fh:
            fh.write(out)
        print(f"{args.input} -> {args.output}: {len(symbols):,} symbols")
        return 0

    data = _read_symbols(args.input, args.binary)
    if not data:
        ap.error("input is empty")
    try:
        alpha = _resolve_alphabet(args.alphabet, data)
    except ValueError as exc:
        ap.error(str(exc))

    if args.command == "compress":
        blob = pack_file(data, alpha, args.codec)
        with open(args.output, "wb") as fh:
            fh.write(blob)
        raw = len(BitPacker(alpha).pack(data))
        print(f"{args.input} -> {args.output}")
        print(f"  {len(data):,} symbols, |Sigma| = {len(alpha)}, codec = {args.codec}")
        print(f"  {raw:,} bytes packed -> {len(blob):,} bytes "
              f"({len(blob) / raw:.3f}x, container included)")
        return 0

    report = analyze(data, alpha)
    if args.json:
        json.dump(
            {**report.__dict__, "codecs": [c.__dict__ for c in report.codecs]},
            sys.stdout, indent=2,
        )
        print()
    else:
        print(report.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
