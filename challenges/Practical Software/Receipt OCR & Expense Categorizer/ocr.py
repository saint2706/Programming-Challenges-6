"""Thin wrapper around PaddleOCR for text extraction from receipt images.

This is the only file allowed to import paddleocr. The OCR model is lazily
loaded on first use and reused across calls.
"""

from __future__ import annotations

from pathlib import Path

import paddleocr


_ocr_instance: paddleocr.PaddleOCR | None = None


def _get_ocr() -> paddleocr.PaddleOCR:
    """Get or create the shared OCR instance."""
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = paddleocr.PaddleOCR(use_angle_cls=True, lang="en")
    return _ocr_instance


def extract_text(image_path: str | Path) -> list[str]:
    """Extract text from an image file and return as a list of lines.

    Args:
        image_path: Path to the receipt image file

    Returns:
        List of text lines extracted from the image
    """
    ocr = _get_ocr()
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    result = ocr.ocr(str(image_path), cls=True)

    lines: list[str] = []
    for block in result:
        if block is None:
            continue
        for line in block:
            if line and len(line) > 0:
                text = line[1][0] if isinstance(line[1], (tuple, list)) else str(line[1])
                if text.strip():
                    lines.append(text.strip())

    return lines
