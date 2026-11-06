import base64
import json
import logging
import re
from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np
import pymupdf as fitz

import config
import pdf_utils

log = logging.getLogger("sigagent.extract")

_CIF_RE = re.compile(r"CIF\s*(?:No\.?|Number)?\s*[:\-]?\s*(\d{8})\b", re.I)
_VLM_DPI = 150


@dataclass
class Record:
    cif: str | None
    name: str | None
    image: np.ndarray | None  # signature crop (BGR)
    page: int = 0


# ---------- text PDFs (geometry from the PDF text layer) ----------

def _text_master(path) -> list[Record]:
    records = []
    with fitz.open(path) as doc:
        for pno, page in enumerate(doc):
            words = page.get_text("words")
            starts = []
            for w in words:
                if w[4].upper() != "CIF":
                    continue
                num = [x for x in words if abs(x[1] - w[1]) < 3 and x[0] > w[0] and re.fullmatch(r"\d{8}", x[4])]
                if num:
                    starts.append((w[1], num[0][4]))
            starts.sort()
            for i, (y0, cif) in enumerate(starts):
                top = y0 - 4
                bottom = starts[i + 1][0] - 4 if i + 1 < len(starts) else page.rect.height
                card = fitz.Rect(0, top, page.rect.width, bottom)
                cw = [w for w in words if fitz.Rect(w[:4]).intersects(card)]
                label = next((w for w in cw if w[4].lower().startswith("signature")), None)
                name_w = [w for w in cw if abs(w[1] - next((n[1] for n in cw if n[4].lower() == "name"), -99)) < 3]
                name = " ".join(w[4] for w in name_w if w[4] not in ("Name", ":")) or None
                if not label or not name_w:
                    records.append(Record(cif, name, None, pno))
                    continue
                # Signature sits right of the label, below the name line, above the card border.
                clip = fitz.Rect(label[2] + 4, max(w[3] for w in name_w) + 2, page.rect.width - 56, label[3] + 12)
                img = pdf_utils.render_page(path, pno, clip=clip)
                records.append(Record(cif, name, img, pno))
    return records


def _text_letter(path) -> Record:
    with fitz.open(path) as doc:
        page = doc[0]
        m = _CIF_RE.search(page.get_text())
        cif = m.group(1) if m else None
        words = page.get_text("words")
        close = [w for w in words if w[4].lower().startswith("faithfully")]
        if not close:
            return Record(cif, None, None)
        last_y0 = max(w[1] for w in words)
        name = " ".join(w[4] for w in words if abs(w[1] - last_y0) < 3)
        clip = fitz.Rect(30, close[0][3] + 4, page.rect.width - 30, last_y0 - 1)
    return Record(cif, name, pdf_utils.render_page(path, 0, clip=clip))


# ---------- scanned PDFs (vision model) ----------

def _ask_vlm(img: np.ndarray, prompt: str, stage: str):
    from langchain_core.messages import HumanMessage
    from langchain_ollama import ChatOllama

    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Could not encode scanned page for vision model.")
    b64 = base64.b64encode(buf.tobytes()).decode()
    llm = ChatOllama(model=config.OLLAMA_VISION_MODEL, base_url=config.OLLAMA_BASE_URL, temperature=0, format="json")
    msg = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": f"data:image/png;base64,{b64}"},
    ])
    started = perf_counter()
    log.info("Vision request started: %s (model=%s)", stage, config.OLLAMA_VISION_MODEL)
    try:
        text = llm.invoke([msg]).content
    except Exception:
        log.exception("Vision request failed: %s (elapsed=%.1fs)", stage, perf_counter() - started)
        raise
    log.info("Vision request completed: %s (elapsed=%.1fs)", stage, perf_counter() - started)
    return json.loads(text)


def _crop(img: np.ndarray, bbox) -> np.ndarray | None:
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    h, w = img.shape[:2]
    if max(x1, y1, x2, y2) <= 1.5:  # normalised coordinates
        x1, x2, y1, y2 = x1 * w, x2 * w, y1 * h, y2 * h
    x1, x2 = sorted((int(max(0, x1)), int(min(w, x2))))
    y1, y2 = sorted((int(max(0, y1)), int(min(h, y2))))
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    return img[y1:y2, x1:x2]


_MASTER_PROMPT = (
    "This page is a bank master signature card register. For EVERY card return JSON "
    '{"cards":[{"cif":"<8 digit CIF>","name":"<customer name>","signature_bbox":[x1,y1,x2,y2]}]}. '
    "signature_bbox is the pixel box of the handwritten specimen signature only (no printed text or borders) "
    "in this image. Return only JSON."
)
_LETTER_PROMPT = (
    "This is a customer letter to a bank. Return JSON "
    '{"cif":"<8 digit CIF>","name":"<signer name>","signature_bbox":[x1,y1,x2,y2]}. '
    "signature_bbox is the pixel box of the handwritten signature only (not printed text) in this image. "
    "Return only JSON."
)


def _scanned_master(path) -> list[Record]:
    records = []
    total_pages = pdf_utils.page_count(path)
    for pno in range(total_pages):
        log.info("Rendering scanned master page %d/%d", pno + 1, total_pages)
        img = pdf_utils.render_page(path, pno, dpi=_VLM_DPI)
        page_label = f"master page {pno + 1}/{total_pages}"
        for c in _ask_vlm(img, _MASTER_PROMPT, page_label).get("cards", []):
            records.append(Record(str(c.get("cif", "")).strip() or None, c.get("name"), _crop(img, c.get("signature_bbox")), pno))
    return records


def _scanned_letter(path) -> Record:
    log.info("Rendering scanned letter page 1/1")
    img = pdf_utils.render_page(path, 0, dpi=_VLM_DPI)
    d = _ask_vlm(img, _LETTER_PROMPT, "letter page 1/1")
    return Record(str(d.get("cif", "")).strip() or None, d.get("name"), _crop(img, d.get("signature_bbox")))


# ---------- public API ----------

def extract_master(path) -> list[Record]:
    scanned = pdf_utils.is_scanned(path)
    log.info("Master %s scanned=%s", path, scanned)
    return _scanned_master(path) if scanned else _text_master(path)


def extract_letter(path) -> Record:
    scanned = pdf_utils.is_scanned(path)
    log.info("Letter %s scanned=%s", path, scanned)
    rec = _scanned_letter(path) if scanned else _text_letter(path)
    if rec.cif and not re.fullmatch(r"\d{8}", rec.cif):
        rec.cif = None
    return rec
