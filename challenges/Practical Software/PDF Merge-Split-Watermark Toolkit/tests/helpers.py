"""Build small PDFs and inspect results, so tests never depend on checked-in binaries."""

from pathlib import Path

import pypdfium2 as pdfium
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen.canvas import Canvas


def make_pdf(
    path: Path, pages: int = 3, *, label: str | None = None, size=(300, 400)
) -> Path:
    """A PDF whose page k says ``<label> <k>``; the label defaults to the file stem."""
    label = label or path.stem
    path.parent.mkdir(parents=True, exist_ok=True)
    c = Canvas(str(path), pagesize=size)
    for k in range(1, pages + 1):
        c.setFont("Helvetica", 24)
        c.drawString(40, size[1] - 80, f"{label} {k}")
        c.showPage()
    c.save()
    return path


def page_texts(path: Path, password: str | None = None) -> list[str]:
    reader = PdfReader(path)
    if reader.is_encrypted:
        reader.decrypt(password or "")
    return [p.extract_text().strip() for p in reader.pages]


def with_outline(path: Path, titles: list[str]) -> Path:
    """Add one top-level bookmark per page."""
    writer = PdfWriter(clone_from=path)
    for i, title in enumerate(titles):
        writer.add_outline_item(title, i)
    writer.write(path)
    return path


def outline_titles(path: Path) -> list:
    def walk(items):
        return [walk(i) if isinstance(i, list) else i.title for i in items]

    return walk(PdfReader(path).outline)


def render(path: Path, page: int = 0, scale: float = 1.0):
    """Rasterise one page to an RGB PIL image."""
    doc = pdfium.PdfDocument(str(path))
    try:
        return doc[page].render(scale=scale).to_pil().convert("RGB")
    finally:
        doc.close()


def ink(image, box=None) -> int:
    """Count non-white pixels (optionally within a (left, top, right, bottom) box)."""
    region = image.crop(box) if box else image
    return sum(1 for px in region.get_flattened_data() if px != (255, 255, 255))


def darkness(image, box=None) -> int:
    """Total ink weight: unlike ``ink`` it grows with opacity, so it can compare transparency."""
    region = image.crop(box) if box else image
    return sum(765 - sum(px) for px in region.get_flattened_data())


def make_blank(path: Path, pages: int = 3, size=(300, 400)) -> Path:
    """Pages with no content at all, so every pixel of ink in a test comes from the mark."""
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(*size)
    writer.write(path)
    return path
