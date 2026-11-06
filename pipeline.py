import csv
import logging
from pathlib import Path

import cv2

import compare
import config
import extract

log = logging.getLogger("sigagent.pipeline")

COLUMNS = ["cif", "name", "letter_file", "similarity_score", "result", "remarks"]


def build_master_store(master_pdf) -> dict:
    store = {}
    for r in extract.extract_master(master_pdf):
        if not r.cif or r.image is None:
            log.warning("Skipping master card cif=%s (no signature found)", r.cif)
            continue
        cv2.imwrite(str(config.MASTER_STORE / f"{r.cif}.png"), r.image)
        store[r.cif] = {"name": r.name, "image": r.image}
    log.info("Master store: %d signatures", len(store))
    return store


def verify_letter(letter_pdf, store: dict, include_images: bool = False) -> dict:
    letter_pdf = Path(letter_pdf)
    rec = extract.extract_letter(letter_pdf)
    row = {"cif": rec.cif or "", "name": rec.name or "", "letter_file": letter_pdf.name,
           "similarity_score": "", "result": "", "remarks": ""}
    if include_images:
        row["_master_signature_image"] = store.get(rec.cif or "", {}).get("image")
        row["_letter_signature_image"] = rec.image
    if not rec.cif:
        row.update(result="REVIEW", remarks="CIF not found in letter")
    elif rec.cif not in store:
        row.update(result="NOT_FOUND", remarks="No master signature for CIF")
    elif rec.image is None:
        row.update(result="REVIEW", remarks="Signature not found in letter")
    else:
        cv2.imwrite(str(config.LETTER_CROPS / f"{letter_pdf.stem}.png"), rec.image)
        score = compare.similarity(store[rec.cif]["image"], rec.image)
        row.update(similarity_score="" if score is None else score, result=compare.verdict(score),
                   name=store[rec.cif]["name"] or row["name"],
                   remarks="" if score is not None else "No ink detected")
    return row


def write_csv(rows: list[dict], out_csv) -> Path:
    out_csv = Path(out_csv)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return out_csv


def run(master_pdf, letters_dir, out_csv) -> list[dict]:
    master_pdf, letters_dir = Path(master_pdf), Path(letters_dir)
    store = build_master_store(master_pdf)
    letters = sorted(p for p in letters_dir.glob("*.pdf") if p.name != master_pdf.name and "master" not in p.name.lower())
    rows = [verify_letter(p, store) for p in letters]
    write_csv(rows, out_csv)
    return rows
