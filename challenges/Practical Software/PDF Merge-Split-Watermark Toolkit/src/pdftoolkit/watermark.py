"""Text, image and PDF-stamp watermarks.

Every mark is reduced to "draw this item centred at these points, rotated by ``angle``", so the
same placement logic serves all three kinds. Text and images are drawn by reportlab into a
one-page overlay per page size; PDF stamps are placed by pypdf directly (no opacity: a stamp
keeps whatever transparency its own content has).
"""

import hashlib
import math
import struct
from io import BytesIO
from pathlib import Path
from typing import Literal

from PIL import Image, UnidentifiedImageError
from pypdf import PageObject, PdfReader, PdfWriter, Transformation
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFError, TTFont
from reportlab.pdfgen.canvas import Canvas

from pdftoolkit.errors import PdfToolError
from pdftoolkit.files import Source, open_reader, write_pdf
from pdftoolkit.ranges import parse_pages

Position = Literal[
    "center", "tile", "top-left", "top-right", "bottom-left", "bottom-right"
]
POSITIONS = ("center", "tile", "top-left", "top-right", "bottom-left", "bottom-right")
_MARGIN = 36.0
_MAX_TILES = 2000
_MAX_IMAGE_PIXELS = 50_000_000


def _centers(
    w: float, h: float, iw: float, ih: float, angle: float, position: str
) -> list[tuple[float, float]]:
    """Centre points for an ``iw`` x ``ih`` item on a ``w`` x ``h`` page."""
    if position == "center":
        return [(w / 2, h / 2)]
    if position != "tile":
        vertical, _, horizontal = position.partition("-")
        x = _MARGIN + iw / 2 if horizontal == "left" else w - _MARGIN - iw / 2
        y = h - _MARGIN - ih / 2 if vertical == "top" else _MARGIN + ih / 2
        return [(x, y)]
    # Lay the grid out in the rotated frame, over the page diagonal so no corner is left bare.
    step_x, step_y = iw * 1.5, ih * 3.0
    half = math.hypot(w, h) / 2
    nx, ny = int(half // step_x) + 1, int(half // step_y) + 1
    if (2 * nx + 1) * (2 * ny + 1) > _MAX_TILES:
        raise PdfToolError("watermark is too small to tile; increase its size")
    cos, sin = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    points = []
    for i in range(-nx, nx + 1):
        for j in range(-ny, ny + 1):
            gx, gy = i * step_x, j * step_y
            x, y = w / 2 + gx * cos - gy * sin, h / 2 + gx * sin + gy * cos
            if -iw <= x <= w + iw and -ih <= y <= h + ih:
                points.append((x, y))
    return points


class _TextMark:
    def __init__(self, text: str, size: float, font: Path | None) -> None:
        self.text, self.size = text, size
        if font is None:
            self.font = "Helvetica-Bold"
            try:
                text.encode("cp1252")
            except UnicodeEncodeError as exc:
                raise PdfToolError(
                    "text has characters the built-in font cannot draw; pass a TrueType font"
                ) from exc
        else:
            # Keyed on the full path: two fonts in different folders may share a file name.
            digest = hashlib.sha1(str(font.resolve()).encode()).hexdigest()[:12]
            self.font = f"pdftool-{digest}"
            try:
                if self.font not in pdfmetrics.getRegisteredFontNames():
                    pdfmetrics.registerFont(TTFont(self.font, str(font)))
            # A corrupt TrueType file surfaces as any of these from reportlab's parser.
            except (TTFError, OSError, struct.error, IndexError, KeyError) as exc:
                raise PdfToolError(f"cannot load font {font}: {exc}") from exc
        self.width = pdfmetrics.stringWidth(text, self.font, size)
        self.height = size

    def draw(self, c: Canvas) -> None:
        c.setFont(self.font, self.size)
        c.drawCentredString(0, -self.size * 0.3, self.text)


class _ImageMark:
    def __init__(self, path: Path, scale: float, page_w: float) -> None:
        try:
            with Image.open(path) as im:
                if im.width * im.height > _MAX_IMAGE_PIXELS:
                    raise PdfToolError(
                        f"image {path.name} is too large (over 50 megapixels)"
                    )
                im.load()
                self.image = im.convert("RGBA")
        except Image.DecompressionBombError as exc:
            raise PdfToolError(f"image {path.name} is too large") from exc
        except (OSError, UnidentifiedImageError) as exc:
            raise PdfToolError(f"cannot read image {path}: {exc}") from exc
        self.width = page_w * scale
        self.height = self.width * self.image.height / self.image.width
        self.reader = ImageReader(self.image)

    def draw(self, c: Canvas) -> None:
        c.drawImage(
            self.reader,
            -self.width / 2,
            -self.height / 2,
            self.width,
            self.height,
            mask="auto",
        )


def _overlay(mark: _TextMark | _ImageMark, w: float, h: float, o: dict) -> PageObject:
    buf = BytesIO()
    c = Canvas(buf, pagesize=(w, h))
    if isinstance(mark, _TextMark):
        c.setFillColor(o["color"])
    # After the colour: setFillColor resets alpha to the colour's own (opaque) alpha.
    c.setFillAlpha(o["opacity"])
    for x, y in _centers(w, h, mark.width, mark.height, o["angle"], o["position"]):
        c.saveState()
        c.translate(x, y)
        c.rotate(o["angle"])
        mark.draw(c)
        c.restoreState()
    c.save()
    buf.seek(0)
    return PdfReader(buf).pages[0]


def watermark(
    src: Source,
    dst: Path,
    *,
    text: str | None = None,
    image: Path | None = None,
    stamp: Path | None = None,
    stamp_page: int = 1,
    size: float = 64,
    color: str = "#808080",
    opacity: float = 0.3,
    angle: float = 45,
    position: Position = "center",
    scale: float = 0.5,
    under: bool = False,
    font: Path | None = None,
    pages: str | None = None,
    password: str | None = None,
    force: bool = False,
) -> int:
    """Mark the selected pages (all by default) and return how many were marked.

    Give exactly one of ``text``, ``image`` or ``stamp``. ``scale`` is the image/stamp width as
    a fraction of the page width; ``under`` puts the mark beneath the page content.
    """
    if sum(x is not None for x in (text, image, stamp)) != 1:
        raise PdfToolError("choose exactly one of text, image, stamp")
    if not all(math.isfinite(v) for v in (size, angle, opacity, scale)):
        raise PdfToolError("size, angle, opacity and scale must be finite numbers")
    if size <= 0:
        raise PdfToolError("size must be greater than 0")
    if not 0 <= opacity <= 1:
        raise PdfToolError("opacity must be between 0 and 1")
    if not 0 < scale <= 2:
        raise PdfToolError("scale must be greater than 0 and at most 2")
    if position not in POSITIONS:
        raise PdfToolError(f"position must be one of {', '.join(POSITIONS)}")
    if text is not None and not text.strip():
        raise PdfToolError("watermark text is empty")
    try:
        rgb = HexColor(color)
    except (ValueError, TypeError) as exc:
        raise PdfToolError(f"bad color {color!r}; use #rrggbb") from exc

    reader = open_reader(src.path, password)
    writer = PdfWriter(clone_from=reader)
    spec = pages or src.pages
    total = len(reader.pages)
    targets = sorted(set(parse_pages(spec, total))) if spec else list(range(total))
    opts = {"opacity": opacity, "color": rgb, "angle": angle, "position": position}

    stamp_page_obj = _load_stamp(stamp, stamp_page, password) if stamp else None
    overlays: dict[tuple[float, float], PageObject] = {}
    for i in targets:
        page = writer.pages[i]
        # Bake /Rotate into the content so the overlay's axes match what the reader sees.
        page.transfer_rotation_to_content()
        left, bottom = float(page.mediabox.left), float(page.mediabox.bottom)
        w, h = float(page.mediabox.width), float(page.mediabox.height)
        if stamp_page_obj is not None:
            _place_stamp(page, stamp_page_obj, w, h, left, bottom, scale, opts, under)
            continue
        key = (round(w, 2), round(h, 2))
        if key not in overlays:
            mark = (
                _TextMark(text, size, font)
                if text is not None
                else _ImageMark(image, scale, w)
            )
            overlays[key] = _overlay(mark, w, h, opts)
        page.merge_transformed_page(
            overlays[key], Transformation().translate(left, bottom), over=not under
        )
    write_pdf(writer, dst, force=force)
    return len(targets)


def _load_stamp(path: Path, number: int, password: str | None) -> PageObject:
    # Cloned into a writer: pypdf deprecates rotating pages that belong to no writer.
    doc = PdfWriter(clone_from=open_reader(path, password))
    if not 1 <= number <= len(doc.pages):
        raise PdfToolError(f"stamp page {number} out of range ({len(doc.pages)} pages)")
    page = doc.pages[number - 1]
    page.transfer_rotation_to_content()
    return page


def _place_stamp(
    page: PageObject,
    stamp: PageObject,
    w: float,
    h: float,
    left: float,
    bottom: float,
    scale: float,
    o: dict,
    under: bool,
) -> None:
    sw, sh = float(stamp.mediabox.width), float(stamp.mediabox.height)
    k = (w * scale) / sw
    centers = _centers(w, h, sw * k, sh * k, o["angle"], o["position"])
    for x, y in centers:
        ctm = (
            Transformation()
            .translate(
                -float(stamp.mediabox.left) - sw / 2,
                -float(stamp.mediabox.bottom) - sh / 2,
            )
            .scale(k, k)
            .rotate(o["angle"])
            .translate(left + x, bottom + y)
        )
        page.merge_transformed_page(stamp, ctm, over=not under)
