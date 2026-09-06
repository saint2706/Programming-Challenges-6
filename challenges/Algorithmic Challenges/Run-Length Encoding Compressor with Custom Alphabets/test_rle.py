"""Tests for the run-length compressor.

The load-bearing property is simple to state and easy to get wrong: for every
codec and every alphabet, ``decompress(compress(x)) == x`` **and** every symbol
of the compressed form is itself in the alphabet. Most tests here are that,
fuzzed hard.

Run with:  uv run --with pytest pytest -q
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

import rle
from rle import (
    CODECS,
    DecompressionBomb,
    NAMED_ALPHABETS,
    AdaptiveCodec,
    Alphabet,
    BitPacker,
    ContinuationCount,
    EscapeCodec,
    GammaCount,
    TerminatedCount,
    analyze,
    compress,
    decompress,
    main,
    pack_file,
    unpack_file,
)

HERE = Path(__file__).parent
CODEC_NAMES = list(CODECS)


def roundtrip(data, alpha, codec):
    encoded, _ = compress(data, alpha, codec)
    assert all(s in alpha for s in encoded), f"{codec} emitted a symbol outside Sigma"
    assert decompress(encoded, alpha, codec) == list(data)
    return encoded


# ---------------------------------------------------------------------------
# Count codes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code,k",
    [
        (GammaCount(), 2),
        *[(TerminatedCount(), k) for k in (3, 4, 5, 10, 256)],
        *[(ContinuationCount(), k) for k in (4, 5, 10, 256)],
    ],
)
def test_count_codes_round_trip(code, k):
    for value in [0, 1, 2, 3, 7, 8, 100, 255, 256, 10**6, 10**15]:
        encoded = code.encode(value, k)
        assert all(0 <= d < k for d in encoded), "count code escaped the alphabet"
        assert code.length(value, k) == len(encoded)
        got, pos = code.decode(encoded, 0, k)
        assert (got, pos) == (value, len(encoded))


@pytest.mark.parametrize(
    "code,k",
    [
        (GammaCount(), 2),
        (TerminatedCount(), 4),
        (ContinuationCount(), 256),
    ],
)
def test_count_codes_are_self_delimiting_when_concatenated(code, k):
    """The whole point: a decoder must find the boundaries with no separator."""
    values = [0, 5, 1, 900, 3, 77777, 2]
    stream = [d for v in values for d in code.encode(v, k)]
    out, pos = [], 0
    while pos < len(stream):
        value, pos = code.decode(stream, pos, k)
        out.append(value)
    assert out == values


def test_count_codes_reject_negatives():
    for code, k in (
        (GammaCount(), 2),
        (TerminatedCount(), 4),
        (ContinuationCount(), 8),
    ):
        with pytest.raises(ValueError):
            code.encode(-1, k)


def test_count_codes_reject_truncated_input():
    for code, k in (
        (GammaCount(), 2),
        (TerminatedCount(), 4),
        (ContinuationCount(), 8),
    ):
        stream = code.encode(10**9, k)[:-1]
        with pytest.raises(ValueError):
            code.decode(stream, 0, k)


def test_binary_alphabets_use_gamma():
    assert Alphabet("01").count_code.name == "gamma"


def test_count_code_choice_is_the_cheaper_one():
    """Small alphabets should prefer the terminator, big ones continuation."""
    assert Alphabet("ACGT").count_code.name == "terminated"
    assert Alphabet(range(256)).count_code.name == "continuation"


# ---------------------------------------------------------------------------
# Alphabets
# ---------------------------------------------------------------------------


def test_alphabet_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicate symbol"):
        Alphabet("AABC")


def test_alphabet_needs_two_symbols():
    with pytest.raises(ValueError):
        Alphabet("A")
    with pytest.raises(ValueError):
        Alphabet("")


def test_alphabet_rejects_foreign_symbols():
    with pytest.raises(ValueError, match="not in alphabet"):
        Alphabet("ACGT").to_indices("ACGTX")


def test_alphabet_inferred_in_first_appearance_order():
    assert Alphabet.of("banana").symbols == ("b", "a", "n")


def test_named_alphabets_all_construct():
    for name in NAMED_ALPHABETS:
        alpha = Alphabet.named(name)
        assert len(alpha) >= 2


def test_unknown_named_alphabet():
    with pytest.raises(ValueError, match="unknown alphabet"):
        Alphabet.named("klingon")


def test_symbols_need_not_be_characters():
    """Tuples, ints, anything hashable."""
    notes = [("C", 4), ("E", 8), ("G", 4), ("A", 2)]
    alpha = Alphabet(notes)
    song = [notes[0]] * 30 + [notes[1]] * 12 + [notes[2]] * 30
    for codec in CODEC_NAMES:
        if CODECS[codec].applicable(alpha):
            roundtrip(song, alpha, codec)


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------


SAMPLES = {
    "empty": ("", "ab"),
    "one symbol": ("a", "ab"),
    "one run": ("A" * 5000, "ACGT"),
    "no runs": ("ACGT" * 400, "ACGT"),
    "alternating binary": ("01" * 1000, "01"),
    "sparse binary": ("0" * 3000 + "1" + "0" * 3000, "01"),
    "mixed": ("A" * 200 + "CGCGCG" + "T" * 4000 + "ACGT", "ACGT"),
    "text": ("aaabbbcccaaa" * 60, "abc"),
    "runs at the edges": ("A" * 50 + "CGT" + "A" * 50, "ACGT"),
}


@pytest.mark.parametrize("codec", CODEC_NAMES)
@pytest.mark.parametrize("label", list(SAMPLES))
def test_round_trip(codec, label):
    data, letters = SAMPLES[label]
    alpha = Alphabet(letters)
    if CODECS[codec].applicable(alpha):
        roundtrip(data, alpha, codec)


@pytest.mark.parametrize("codec", CODEC_NAMES)
@pytest.mark.parametrize("k", [2, 3, 4, 5, 7, 16, 256])
def test_fuzz_round_trip(codec, k):
    """Random data at every alphabet size, biased to produce real runs."""
    rng = random.Random(1000 + k)
    alpha = Alphabet(range(k))
    if not CODECS[codec].applicable(alpha):
        pytest.skip("not applicable")
    for _ in range(60):
        data: list[int] = []
        while len(data) < rng.randint(0, 400):
            data.extend([rng.randrange(k)] * rng.choice([1, 1, 1, 2, 3, 9, 40]))
        roundtrip(data, alpha, codec)


@pytest.mark.parametrize("codec", CODEC_NAMES)
def test_fuzz_pure_noise(codec):
    """Incompressible input is where expansion bugs show up."""
    rng = random.Random(4242)
    alpha = Alphabet(range(256))
    for _ in range(30):
        data = [rng.randrange(256) for _ in range(rng.randint(0, 500))]
        roundtrip(data, alpha, codec)


def test_output_never_leaves_the_alphabet_even_for_huge_runs():
    alpha = Alphabet("ACGT")
    encoded, _ = compress("A" * 10**6, alpha, "pair")
    assert set(encoded) <= set("ACGT")
    assert decompress(encoded, alpha, "pair") == ["A"] * 10**6


# ---------------------------------------------------------------------------
# Compression quality and the guarantees
# ---------------------------------------------------------------------------


def test_packbits_expansion_is_bounded_by_a_few_symbols():
    """PackBits' classic +1-per-128 becomes +1 per *file* with varint counts."""
    rng = random.Random(9)
    alpha = Alphabet(range(256))
    for n in (100, 1000, 20_000):
        noise = [rng.randrange(256) for _ in range(n)]
        grown = len(compress(noise, alpha, "packbits")[0]) - n
        assert 0 < grown <= 4, (n, grown)


def test_pair_can_double_on_noise_which_is_why_packbits_exists():
    rng = random.Random(9)
    alpha = Alphabet(range(256))
    noise = [rng.randrange(256) for _ in range(5000)]
    assert len(compress(noise, alpha, "pair")[0]) > 1.5 * len(noise)
    assert len(compress(noise, alpha, "packbits")[0]) < 1.01 * len(noise)


def test_adaptive_is_best_of_plus_one():
    alpha = Alphabet("ACGT")
    data = "A" * 300 + "CGCGCG" + "T" * 2000 + "ACGT"
    best = min(
        len(compress(data, alpha, name)[0])
        for name in ("pair", "packbits", "escape")
        if CODECS[name].applicable(alpha)
    )
    assert len(compress(data, alpha, "adaptive")[0]) == best + 1


def test_adaptive_picks_different_codecs_for_different_data():
    adaptive = AdaptiveCodec()
    dna = Alphabet("ACGT")
    byte = Alphabet(range(256))
    long_runs = dna.to_indices("A" * 4000 + "C" * 4000)
    noisy_with_fill = [i % 251 for i in range(3000)] + [0] * 6000
    assert adaptive.chosen(long_runs, dna) == "pair"
    assert adaptive.chosen(noisy_with_fill, byte) in {"packbits", "escape"}


def test_binary_pair_codec_omits_the_redundant_symbol():
    """Runs alternate on a 2-symbol alphabet, so the symbol carries no bits."""
    alpha = Alphabet("01")
    data = "0" * 100 + "1" * 100 + "0" * 100
    encoded = compress(data, alpha, "pair")[0]
    # 1 leading symbol + three gamma codes, and nothing else.
    expected = 1 + sum(alpha.count_length(100) for _ in range(3))
    assert len(encoded) == expected


def test_binary_run_coding_beats_packbits_on_bilevel_data():
    """The fax scheme should win on the data faxes were built for."""
    alpha = Alphabet("01")
    rng = random.Random(3)
    scan = "".join(rng.choice("01") * rng.randint(30, 300) for _ in range(400))
    assert len(compress(scan, alpha, "pair")[0]) < len(
        compress(scan, alpha, "packbits")[0]
    )


def test_escape_codec_survives_runs_of_the_escape_symbol():
    """The ambiguity that a naive ESC-ESC scheme gets wrong."""
    alpha = Alphabet("ABC")
    codec = EscapeCodec(escape=0)
    for data in ["A" * 50, "A" * 50 + "B" * 50, "ABABA", "A", "AAB" * 30, "BAAAAAAAAB"]:
        idx = alpha.to_indices(data)
        assert codec.decode(codec.encode(idx, alpha), alpha) == idx


def test_escape_codec_is_refused_on_binary_alphabets():
    assert not EscapeCodec().applicable(Alphabet("01"))


# ---------------------------------------------------------------------------
# Bit packing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [2, 3, 4, 5, 6, 7, 8, 10, 16, 100, 256])
def test_bit_packer_round_trip(k):
    alpha = Alphabet(range(k))
    packer = BitPacker(alpha)
    rng = random.Random(k)
    for n in (0, 1, 2, 7, 8, 9, 100, 1000):
        data = [rng.randrange(k) for _ in range(n)]
        assert packer.unpack(packer.pack(data)) == data


def test_bit_packer_is_near_the_information_theoretic_floor():
    for k, expected in [(2, 1.0), (4, 2.0), (16, 4.0), (256, 8.0)]:
        packer = BitPacker(Alphabet(range(k)))
        assert packer.bits_per_symbol == expected  # exact for powers of two
    # 3 symbols: 5 per byte is 1.600 bits against an ideal 1.585.
    packer = BitPacker(Alphabet(range(3)))
    assert packer.group == 5 and packer.bits == 8
    assert packer.bits_per_symbol < 1.61


def test_dna_packs_to_two_bits():
    alpha = Alphabet("ACGT")
    packed = BitPacker(alpha).pack("ACGT" * 1000)
    assert len(packed) <= 4000 // 4 + 4  # + the length header


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def test_analyze_reports_every_codec():
    report = analyze("A" * 500 + "CGT" * 100, "dna")
    assert {c.name for c in report.codecs} == set(CODEC_NAMES)
    assert report.n_symbols == 800
    assert report.alphabet_size == 4
    assert report.max_run == 500
    assert "codec" in report.render()


def test_analyze_marks_inapplicable_codecs():
    report = analyze("0" * 100 + "1" * 100, "binary")
    escape = next(c for c in report.codecs if c.name == "escape")
    assert escape.error


def test_run_entropy_is_a_real_lower_bound():
    """No run-length coder can beat the entropy of its own run lengths."""
    rng = random.Random(11)
    data = "".join(rng.choice("ACGT") * rng.randint(1, 60) for _ in range(2000))
    report = analyze(data, "dna")
    best_bits = min(c.packed_bytes * 8 for c in report.codecs if not c.error)
    assert best_bits >= report.run_entropy_bits


# ---------------------------------------------------------------------------
# Hostile input
# ---------------------------------------------------------------------------


def _bomb_stream(alpha: Alphabet, claimed: int) -> list[int]:
    """A handful of symbols claiming an astronomical run length."""
    return alpha.count_code.encode(claimed, len(alpha)) + [0]


@pytest.mark.parametrize("codec", ["pair", "packbits", "escape", "adaptive"])
def test_decoders_refuse_a_decompression_bomb(codec):
    alpha = Alphabet(range(256))
    if not CODECS[codec].applicable(alpha):
        pytest.skip("not applicable")
    stream = _bomb_stream(alpha, 10**13)
    if codec == "escape":
        stream = [0, 0] + alpha.count_code.encode(10**13, 256) + [1]
    elif codec == "adaptive":
        stream = [0] + stream
    with pytest.raises(DecompressionBomb):
        CODECS[codec].decode(stream, alpha, max_output=10**6)


def test_a_bomb_is_eight_symbols_claiming_ten_trillion():
    """Documenting the threat model, not just guarding it."""
    alpha = Alphabet(range(256))
    stream = _bomb_stream(alpha, 10**13)
    assert len(stream) < 16
    with pytest.raises(DecompressionBomb, match="10,000,000,000,000"):
        CODECS["pair"].decode(stream, alpha, max_output=1000)


def test_unbounded_decode_is_still_available():
    alpha = Alphabet("ACGT")
    encoded, _ = compress("A" * 500_000, alpha, "pair")
    assert len(decompress(encoded, alpha, "pair", max_symbols=None)) == 500_000
    assert len(decompress(encoded, alpha, "pair", max_symbols=500_000)) == 500_000
    with pytest.raises(DecompressionBomb):
        decompress(encoded, alpha, "pair", max_symbols=499_999)


def test_container_blocks_a_bomb_by_default():
    alpha = Alphabet(range(256))
    payload = alpha.to_symbols(_bomb_stream(alpha, 10**13))
    header = json.dumps(
        {"codec": "pair", "symbols": list(alpha.symbols)}, separators=(",", ":")
    ).encode("utf-8")
    blob = (
        rle.MAGIC + rle._leb128(len(header)) + header + BitPacker(alpha).pack(payload)
    )
    assert len(blob) < 2000
    with pytest.raises(DecompressionBomb):
        unpack_file(blob)


def test_container_still_allows_a_genuinely_huge_ratio():
    """A 54-byte file that expands 37,000x is legitimate, and must survive."""
    alpha = Alphabet("ACGT")
    blob = pack_file(list("A" * 2_000_000), alpha, "pair")
    assert len(blob) < 200
    assert len(unpack_file(blob)) == 2_000_000


def test_truncated_packed_data_raises_instead_of_inventing_symbols():
    alpha = Alphabet("ACGT")
    packed = BitPacker(alpha).pack(list("ACGT" * 100))
    with pytest.raises(ValueError, match="truncated packed data"):
        BitPacker(alpha).unpack(packed[:-5])


def test_truncated_container_header_raises():
    with pytest.raises(ValueError, match="truncated container"):
        unpack_file(rle.MAGIC + b"\xff\xff\xff\x7f")


@pytest.mark.parametrize("codec", CODEC_NAMES)
def test_random_garbage_never_crashes_unexpectedly(codec):
    """Fuzzing the decoder: only ValueError (or a clean decode) is acceptable."""
    rng = random.Random(31337)
    alpha = Alphabet(range(16))
    for _ in range(500):
        stream = [rng.randrange(16) for _ in range(rng.randint(0, 40))]
        try:
            out = CODECS[codec].decode(stream, alpha, max_output=10**5)
            assert all(0 <= s < 16 for s in alpha.to_indices(out))
        except ValueError:
            pass  # truncated / bomb / unknown tag are all fine


def test_one_distinct_symbol_gives_actionable_advice():
    """RLE's best case is also the one auto-inference cannot handle."""
    with pytest.raises(ValueError, match="at least two symbols"):
        Alphabet.of("AAAA")
    # ...and the workaround in the message actually works.
    assert compress("AAAA", Alphabet("ACGT"), "pair")[0]


def test_empty_data_cannot_infer_an_alphabet_either():
    with pytest.raises(ValueError, match="cannot infer"):
        Alphabet.of("")


# ---------------------------------------------------------------------------
# Container format
# ---------------------------------------------------------------------------


def test_container_round_trip():
    for name, data in [
        ("dna", "ACGT" * 100 + "A" * 900),
        ("binary", "0" * 500 + "1" * 500),
        ("digits", "1234567890" * 50),
    ]:
        alpha = Alphabet.named(name)
        blob = pack_file(list(data), alpha, "adaptive")
        assert unpack_file(blob) == list(data)


def test_container_carries_a_custom_alphabet():
    """A reader that has never seen this alphabet must still decode the file."""
    alpha = Alphabet("♠♡♢♣")
    data = list("♠" * 200 + "♣" * 50)
    assert unpack_file(pack_file(data, alpha, "packbits")) == data


def test_container_rejects_unserializable_symbols():
    alpha = Alphabet([("a", 1), ("b", 2)])
    with pytest.raises(ValueError, match="container"):
        pack_file([("a", 1)] * 5, alpha, "pair")


def test_container_rejects_foreign_data():
    with pytest.raises(ValueError, match="not an RLE1 container"):
        unpack_file(b"PK\x03\x04nope")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_self_check():
    assert main(["--self-check"]) == 0


def test_cli_compress_decompress(tmp_path, capsys):
    src = tmp_path / "genome.txt"
    src.write_text("ACGT" * 200 + "A" * 3000 + "GGGG")
    out = tmp_path / "genome.rle"
    back = tmp_path / "genome.out"

    assert main(["compress", str(src), str(out), "--alphabet", "dna"]) == 0
    assert main(["decompress", str(out), str(back)]) == 0
    assert back.read_text() == src.read_text()
    assert out.stat().st_size < src.stat().st_size


def test_cli_binary_mode(tmp_path):
    src = tmp_path / "blob.bin"
    src.write_bytes(bytes(range(256)) + bytes(2000))
    out = tmp_path / "blob.rle"
    back = tmp_path / "blob.out"
    assert (
        main(["compress", str(src), str(out), "--binary", "--alphabet", "bytes"]) == 0
    )
    assert main(["decompress", str(out), str(back)]) == 0
    assert back.read_bytes() == src.read_bytes()


def test_cli_analyze_json(tmp_path, capsys):
    src = tmp_path / "x.txt"
    src.write_text("A" * 400 + "CGT" * 40)
    assert main(["analyze", str(src), "--alphabet", "dna", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_symbols"] == 520
    assert len(payload["codecs"]) == len(CODEC_NAMES)


def test_cli_rejects_unknown_alphabet(tmp_path):
    src = tmp_path / "x.txt"
    src.write_text("abc")
    with pytest.raises(SystemExit):
        main(["analyze", str(src), "--alphabet", "klingon"])


def test_module_runs_as_a_script():
    proc = subprocess.run(
        [sys.executable, str(HERE / "rle.py"), "--self-check"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "all self-checks passed" in proc.stdout
