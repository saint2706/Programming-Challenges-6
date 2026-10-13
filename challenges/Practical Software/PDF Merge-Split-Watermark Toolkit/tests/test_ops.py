import pytest
from helpers import make_pdf, outline_titles, page_texts, with_outline
from pdftoolkit import ops
from pdftoolkit.errors import PdfToolError
from pdftoolkit.files import Source
from pypdf import PdfReader


def S(path, pages=None):
    return Source(path, pages)


def test_merge_keeps_order_and_selection(tmp_path, three):
    other = make_pdf(tmp_path / "other.pdf", 4)
    out = tmp_path / "m.pdf"
    assert ops.merge([S(three), S(other, "end-1")], out) == 3 + 4
    assert page_texts(out) == [
        "three 1",
        "three 2",
        "three 3",
        "other 4",
        "other 3",
        "other 2",
        "other 1",
    ]


def test_merge_outline_modes(tmp_path, three):
    other = with_outline(make_pdf(tmp_path / "other.pdf", 2), ["Intro", "Body"])
    srcs = [S(three), S(other)]
    for mode, expected in {
        "nested": ["three", "other", ["Intro", "Body"]],
        "files": ["three", "other"],
        "original": ["Intro", "Body"],
        "none": [],
    }.items():
        out = tmp_path / f"{mode}.pdf"
        ops.merge(srcs, out, outline=mode)
        assert outline_titles(out) == expected, mode


def test_merge_drops_bookmarks_of_unselected_pages(tmp_path):
    src = with_outline(make_pdf(tmp_path / "s.pdf", 3), ["One", "Two", "Three"])
    out = tmp_path / "o.pdf"
    ops.merge([S(src, "2")], out, outline="original")
    assert outline_titles(out) == ["Two"]


def test_merge_bad_outline_mode_and_missing_selection(tmp_path, three):
    with pytest.raises(PdfToolError, match="outline"):
        ops.merge([S(three)], tmp_path / "o.pdf", outline="weird")
    with pytest.raises(PdfToolError, match="out of range"):
        ops.merge([S(three, "9")], tmp_path / "o.pdf")
    assert not (tmp_path / "o.pdf").exists()


def test_merge_metadata(tmp_path, three):
    out = tmp_path / "m.pdf"
    ops.merge([S(three)], out, title="Quarterly", author="Ada")
    meta = PdfReader(out).metadata
    assert (meta.title, meta.author) == ("Quarterly", "Ada")


def test_merge_does_not_overwrite_without_force(tmp_path, three):
    out = tmp_path / "m.pdf"
    ops.merge([S(three)], out)
    with pytest.raises(PdfToolError, match="already exists"):
        ops.merge([S(three)], out)
    ops.merge([S(three, "1")], out, force=True)
    assert page_texts(out) == ["three 1"]


def test_split_every(tmp_path, five):
    files = ops.split(S(five), tmp_path / "o", every=2)
    assert [f.name for f in files] == ["five_p1-2.pdf", "five_p3-4.pdf", "five_p5.pdf"]
    assert [page_texts(f) for f in files] == [
        ["five 1", "five 2"],
        ["five 3", "five 4"],
        ["five 5"],
    ]


def test_split_ranges_and_non_contiguous(tmp_path, five):
    files = ops.split(S(five), tmp_path / "o", ranges="1-2,5,odd")
    assert [f.name for f in files] == [
        "five_p1-2.pdf",
        "five_p5.pdf",
        "five_part003.pdf",
    ]
    assert page_texts(files[2]) == ["five 1", "five 3", "five 5"]


def test_split_each(tmp_path, three):
    assert len(ops.split(S(three), tmp_path / "o", each=True)) == 3


def test_split_needs_exactly_one_mode(tmp_path, three):
    for kwargs in ({}, {"every": 1, "each": True}):
        with pytest.raises(PdfToolError, match="exactly one"):
            ops.split(S(three), tmp_path / "o", **kwargs)
    with pytest.raises(PdfToolError, match="at least 1"):
        ops.split(S(three), tmp_path / "o", every=0)


def test_split_clash_writes_nothing(tmp_path, five):
    out = tmp_path / "o"
    ops.split(S(five), out, every=5)
    with pytest.raises(PdfToolError, match="already exists"):
        ops.split(S(five), out, every=5)
    with pytest.raises(PdfToolError, match="already exists"):
        ops.split(
            S(five), out, ranges="1-5,1"
        )  # first target exists; nothing new written
    assert [f.name for f in out.iterdir()] == ["five_p1-5.pdf"]


def test_split_same_name_twice(tmp_path, five):
    with pytest.raises(PdfToolError, match="same output"):
        ops.split(S(five), tmp_path / "o", ranges="2,2")


def test_extract_and_reorder(tmp_path, five):
    out = tmp_path / "e.pdf"
    assert ops.extract(S(five), out, pages="4-5,1") == 3
    assert page_texts(out) == ["five 4", "five 5", "five 1"]
    out2 = tmp_path / "r.pdf"
    ops.reorder(S(five), out2, order="end-1")
    assert page_texts(out2) == ["five 5", "five 4", "five 3", "five 2", "five 1"]


def test_reorder_must_be_a_permutation(tmp_path, three):
    for order in ("1-2", "1,1,2,3", "1-3,1"):
        with pytest.raises(PdfToolError, match="exactly once"):
            ops.reorder(S(three), tmp_path / "r.pdf", order=order)


def test_rotate(tmp_path, three):
    out = tmp_path / "r.pdf"
    assert ops.rotate(S(three), out, degrees=90, pages="2") == 1
    assert [p.rotation for p in PdfReader(out).pages] == [0, 90, 0]
    ops.rotate(S(out), tmp_path / "r2.pdf", degrees=-90, pages="2")
    assert [p.rotation for p in PdfReader(tmp_path / "r2.pdf").pages] == [0, 0, 0]
    with pytest.raises(PdfToolError, match="multiple of 90"):
        ops.rotate(S(three), tmp_path / "x.pdf", degrees=45)


def test_rotate_duplicate_pages_turn_once(tmp_path, three):
    out = tmp_path / "r.pdf"
    ops.rotate(S(three), out, degrees=90, pages="1,1")
    assert PdfReader(out).pages[0].rotation == 90


def test_encrypt_decrypt_round_trip(tmp_path, three):
    enc = tmp_path / "enc.pdf"
    ops.encrypt(S(three), enc, user_password="s3cret")
    assert PdfReader(enc).is_encrypted
    assert b"/V 5" in enc.read_bytes()  # AES-256
    with pytest.raises(PdfToolError, match="password required"):
        ops.info(enc)
    with pytest.raises(PdfToolError, match="wrong password"):
        ops.info(enc, "nope")
    assert ops.info(enc, "s3cret")["pages"] == 3
    dec = tmp_path / "dec.pdf"
    ops.decrypt(S(enc), dec, password="s3cret")
    assert not PdfReader(dec).is_encrypted
    assert page_texts(dec) == ["three 1", "three 2", "three 3"]


def test_encrypt_permissions_and_owner_password(tmp_path, three):
    from pypdf.constants import UserAccessPermissions as P

    enc = tmp_path / "enc.pdf"
    ops.encrypt(
        S(three),
        enc,
        user_password="u",
        owner_password="o",
        allow_print=False,
        allow_copy=False,
        allow_modify=False,
    )
    reader = PdfReader(enc)
    reader.decrypt("u")
    perms = reader.user_access_permissions
    assert not perms & P.PRINT
    assert not perms & P.MODIFY
    assert not perms & P.EXTRACT
    assert PdfReader(enc).decrypt("o")  # the owner password opens it too


def test_encrypt_permissions_allowed_by_default(tmp_path, three):
    from pypdf.constants import UserAccessPermissions as P

    enc = tmp_path / "enc.pdf"
    ops.encrypt(S(three), enc, user_password="u")
    reader = PdfReader(enc)
    reader.decrypt("u")
    assert reader.user_access_permissions & P.PRINT
    assert reader.user_access_permissions & P.EXTRACT


def test_encrypt_rejects_empty_password_and_decrypt_plain(tmp_path, three):
    with pytest.raises(PdfToolError, match="non-empty"):
        ops.encrypt(S(three), tmp_path / "e.pdf", user_password="")
    with pytest.raises(PdfToolError, match="not encrypted"):
        ops.decrypt(S(three), tmp_path / "d.pdf", password="x")


def test_encrypted_input_flows_through_other_ops(tmp_path, three):
    enc = tmp_path / "enc.pdf"
    ops.encrypt(S(three), enc, user_password="pw")
    out = tmp_path / "e.pdf"
    ops.extract(S(enc), out, pages="2", password="pw")
    assert page_texts(out) == ["three 2"]
    assert not PdfReader(out).is_encrypted  # results are written unencrypted


def test_metadata_set_and_clear(tmp_path, three):
    a = tmp_path / "a.pdf"
    ops.set_metadata(S(three), a, title="T", author="A", subject="S", keywords="k1, k2")
    meta = PdfReader(a).metadata
    assert (meta.title, meta.author, meta.subject) == ("T", "A", "S")
    assert meta["/Keywords"] == "k1, k2"
    b = tmp_path / "b.pdf"
    ops.set_metadata(S(a), b, clear=True, title="Only")
    meta = PdfReader(b).metadata
    assert meta.title == "Only" and meta.author is None


def test_info(tmp_path, three):
    ops.rotate(S(three), tmp_path / "r.pdf", degrees=90, pages="1")
    data = ops.info(tmp_path / "r.pdf")
    assert (data["pages"], data["encrypted"], data["bookmarks"]) == (3, False, 0)
    assert data["page_sizes"][0] == {"width": 300.0, "height": 400.0, "rotation": 90}
    assert data["version"].startswith("1.")
