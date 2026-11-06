"""Create image-only (scanned) copies of the sample PDFs in samples_scanned/ to test the vision path."""
import pymupdf as fitz

import config

OUT = config.BASE_DIR / "samples_scanned"
OUT.mkdir(exist_ok=True)

for src in config.SAMPLES_DIR.glob("*.pdf"):
    with fitz.open(src) as doc, fitz.open() as out:
        for p in doc:
            page = out.new_page(width=p.rect.width, height=p.rect.height)
            page.insert_image(page.rect, pixmap=p.get_pixmap(dpi=150))
        out.save(OUT / src.name)
print("Created", OUT)
