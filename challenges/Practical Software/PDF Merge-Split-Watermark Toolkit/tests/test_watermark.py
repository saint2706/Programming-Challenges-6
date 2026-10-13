from pathlib import Path

import pytest
from helpers import darkness, ink, make_blank, make_pdf, page_texts, render
from pdftoolkit import ops
from pdftoolkit.errors import PdfToolError
from pdftoolkit.files import Source
from pdftoolkit.watermark import watermark
from PIL import Image
from pypdf import PdfWriter
from reportlab.pdfgen.canvas import Canvas


def S(path, pages=None):
    return Source(path, pages)


@pytest.fixture
def blank(tmp_path):
    return make_blank(tmp_path / "blank.pdf")


def mark(src, tmp_path, name="w.pdf", **kwargs):
    out = tmp_path / name
    watermark(S(src), out, **kwargs)
    return out


def test_text_watermark_adds_ink_and_keeps_text(tmp_path, three):
    out = mark(three, tmp_path, text="CONFIDENTIAL")
    assert ink(render(out)) > ink(render(three)) * 3
    assert page_texts(out)[0].startswith("three 1")  # original content untouched


def test_opacity_controls_darkness(tmp_path, blank):
    light = darkness(render(mark(blank, tmp_path, "l.pdf", text="X", opacity=0.2)))
    heavy = darkness(render(mark(blank, tmp_path, "h.pdf", text="X", opacity=1.0)))
    assert heavy > light * 2


def test_color_is_applied(tmp_path, blank):
    out = mark(
        blank, tmp_path, text="MMMM", opacity=1, color="#ff0000", angle=0, size=120
    )
    reds = sum(
        1
        for r, g, b in render(out).get_flattened_data()
        if r > 200 and g < 60 and b < 60
    )
    assert reds > 100


def test_center_mark_sits_in_the_middle(tmp_path, blank):
    img = render(mark(blank, tmp_path, text="MM", angle=0, size=80, opacity=1))
    w, h = img.size
    assert ink(img, (w // 4, h // 3, 3 * w // 4, 2 * h // 3)) > 500
    assert ink(img, (0, 0, w, h // 6)) == 0


@pytest.mark.parametrize(
    ("position", "box"),
    [
        ("top-left", lambda w, h: (0, 0, w // 2, h // 2)),
        ("top-right", lambda w, h: (w // 2, 0, w, h // 2)),
        ("bottom-left", lambda w, h: (0, h // 2, w // 2, h)),
        ("bottom-right", lambda w, h: (w // 2, h // 2, w, h)),
    ],
)
def test_corner_positions(tmp_path, blank, position, box):
    out = mark(
        blank, tmp_path, text="HI", angle=0, size=40, opacity=1, position=position
    )
    img = render(out)
    w, h = img.size
    total = ink(img)
    assert total > 100
    assert ink(img, box(w, h)) == total  # all ink in the requested quadrant


def test_tile_covers_more_than_center(tmp_path, blank):
    one = ink(render(mark(blank, tmp_path, "c.pdf", text="DRAFT", size=30, opacity=1)))
    tiled = mark(
        blank, tmp_path, "t.pdf", text="DRAFT", size=30, opacity=1, position="tile"
    )
    assert ink(render(tiled)) > one * 4


def test_tile_too_small_is_refused(tmp_path, blank):
    with pytest.raises(PdfToolError, match="too small"):
        mark(blank, tmp_path, text=".", size=1, position="tile")


def test_under_puts_mark_behind_content(tmp_path):
    src = tmp_path / "solid.pdf"
    c = Canvas(str(src), pagesize=(300, 400))
    c.setFillColorRGB(0, 0, 1)
    c.rect(0, 0, 300, 400, fill=1, stroke=0)
    c.save()
    kwargs = {"text": "MMMM", "color": "#ff0000", "opacity": 1, "angle": 0, "size": 120}
    over = render(mark(src, tmp_path, "o.pdf", **kwargs))
    under = render(mark(src, tmp_path, "u.pdf", under=True, **kwargs))

    def red(im):
        return sum(1 for r, _, b in im.get_flattened_data() if r > 200 and b < 60)

    assert red(over) > 500
    assert red(under) == 0  # fully hidden beneath the opaque page


def test_page_subset(tmp_path, blank):
    out = mark(blank, tmp_path, text="MM", pages="2", opacity=1, size=80)
    assert [ink(render(out, i)) > 0 for i in range(3)] == [False, True, False]


def test_source_page_selection_is_honoured(tmp_path, blank):
    out = tmp_path / "w.pdf"
    watermark(S(blank, "3"), out, text="MM", size=80, opacity=1)
    assert [ink(render(out, i)) > 0 for i in range(3)] == [False, False, True]


def test_rotated_pages(tmp_path, blank):
    turned = tmp_path / "turned.pdf"
    ops.rotate(S(blank), turned, degrees=90)
    out = mark(turned, tmp_path, text="MM", size=80, opacity=1, angle=0)
    img = render(out)
    assert img.size == (400, 300)  # still landscape as displayed
    w, h = img.size
    assert ink(img, (w // 4, h // 4, 3 * w // 4, 3 * h // 4)) > 500


def test_mixed_page_sizes_each_get_a_fitting_overlay(tmp_path):
    w = PdfWriter()
    w.add_blank_page(200, 200)
    w.add_blank_page(500, 300)
    src = tmp_path / "mixed.pdf"
    w.write(src)
    out = mark(src, tmp_path, text="MM", size=50, opacity=1, angle=0)
    for i, (pw, ph) in enumerate([(200, 200), (500, 300)]):
        img = render(out, i)
        assert img.size == (pw, ph)
        assert ink(img, (pw // 3, ph // 3, 2 * pw // 3, 2 * ph // 3)) > 100


def test_page_with_offset_mediabox(tmp_path):
    from pypdf.generic import RectangleObject

    w = PdfWriter()
    page = w.add_blank_page(300, 300)
    page.mediabox = RectangleObject((100, 100, 400, 400))
    src = tmp_path / "offset.pdf"
    w.write(src)
    img = render(mark(src, tmp_path, text="MM", size=60, opacity=1, angle=0))
    w_, h_ = img.size
    assert ink(img, (w_ // 4, h_ // 4, 3 * w_ // 4, 3 * h_ // 4)) > 200


def test_image_watermark_with_transparency(tmp_path, blank):
    png = tmp_path / "logo.png"
    im = Image.new("RGBA", (100, 50), (0, 0, 0, 0))
    for x in range(50):  # opaque left half, transparent right half
        for y in range(50):
            im.putpixel((x, y), (0, 128, 0, 255))
    im.save(png)
    img = render(mark(blank, tmp_path, image=png, opacity=1, angle=0, scale=0.5))
    w, h = img.size
    left = (w // 4, h // 2 - 40, w // 2, h // 2 + 40)
    right = (w // 2 + 5, h // 2 - 40, 3 * w // 4, h // 2 + 40)
    assert ink(img, left) > 1000
    assert ink(img, right) == 0
    faint = mark(blank, tmp_path, "faint.pdf", image=png, opacity=0.2)
    solid = mark(blank, tmp_path, "solid.pdf", image=png, opacity=1)
    assert darkness(render(faint)) < darkness(render(solid))


def test_pdf_stamp(tmp_path, blank):
    stamp = make_pdf(tmp_path / "stamp.pdf", 2, label="APPROVED", size=(200, 100))
    out = mark(blank, tmp_path, stamp=stamp, angle=0, scale=0.5, stamp_page=2)
    assert "APPROVED 2" in page_texts(out)[0]
    assert ink(render(out)) > 0
    corner = mark(
        blank, tmp_path, "c.pdf", stamp=stamp, angle=0, scale=0.3, position="top-left"
    )
    img = render(corner)
    w, h = img.size
    assert ink(img) > 0
    assert ink(img, (w // 2, 0, w, h)) == 0
    with pytest.raises(PdfToolError, match="stamp page 5"):
        mark(blank, tmp_path, "bad.pdf", stamp=stamp, stamp_page=5)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "exactly one"),
        ({"text": "a", "image": Path("x.png")}, "exactly one"),
        ({"text": "a", "opacity": 1.5}, "opacity"),
        ({"text": "a", "scale": 0}, "scale"),
        ({"text": "a", "position": "middle"}, "position"),
        ({"text": "  "}, "empty"),
        ({"text": "a", "color": "notacolor"}, "color"),
        ({"text": "Привет"}, "TrueType"),
    ],
)
def test_invalid_arguments(tmp_path, three, kwargs, message):
    with pytest.raises(PdfToolError, match=message):
        mark(three, tmp_path, **kwargs)
    assert not (tmp_path / "w.pdf").exists()


def test_bad_image_and_font(tmp_path, three):
    junk = tmp_path / "junk.png"
    junk.write_bytes(b"nope")
    with pytest.raises(PdfToolError, match="cannot read image"):
        mark(three, tmp_path, image=junk)
    with pytest.raises(PdfToolError, match="cannot load font"):
        mark(three, tmp_path, text="a", font=junk)


def test_truetype_font_allows_unicode(tmp_path, blank):
    fonts = (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/segoeui.ttf"))
    font = next((p for p in fonts if p.exists()), None)
    if font is None:
        pytest.skip("no system TrueType font")
    out = mark(blank, tmp_path, text="Привет", font=font, opacity=1, size=60)
    assert ink(render(out)) > 200
