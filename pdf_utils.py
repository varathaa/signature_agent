import pymupdf as fitz
import numpy as np

import config


def is_scanned(path) -> bool:
    """True when the PDF has no extractable text (image-only PDF)."""
    with fitz.open(path) as doc:
        return not any(p.get_text().strip() for p in doc)


def page_count(path) -> int:
    with fitz.open(path) as doc:
        return len(doc)


def render_page(path, page_no: int, clip: fitz.Rect | None = None, dpi: int | None = None) -> np.ndarray:
    """Render a page (or a clip of it) to a BGR image array."""
    with fitz.open(path) as doc:
        pix = doc[page_no].get_pixmap(dpi=dpi or config.RENDER_DPI, clip=clip, colorspace=fitz.csRGB)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return img[:, :, ::-1].copy()
