"""Deterministic batch run: python run.py [master.pdf] [letters_dir] [out.csv]"""
import csv
import sys
from pathlib import Path

import config
import pipeline


def _accuracy(rows, expected_csv: Path):
    if not expected_csv.exists():
        return
    with open(expected_csv, encoding="utf-8") as f:
        exp = {r["letter_file"]: r["expected_result"] for r in csv.DictReader(f)}
    hits = sum(1 for r in rows if exp.get(r["letter_file"]) == r["result"])
    print(f"Agreement with expected_results.csv: {hits}/{len(rows)}")


def main():
    master = Path(sys.argv[1]) if len(sys.argv) > 1 else config.SAMPLES_DIR / "master_signature.pdf"
    letters = Path(sys.argv[2]) if len(sys.argv) > 2 else config.SAMPLES_DIR
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else config.OUTPUT_DIR / "cif_list.csv"
    rows = pipeline.run(master, letters, out)
    for r in rows:
        print(f"{r['letter_file']:22} cif={r['cif']:9} score={r['similarity_score']!s:7} {r['result']:9} {r['remarks']}")
    print("Written:", out)
    _accuracy(rows, config.SAMPLES_DIR / "expected_results.csv")


if __name__ == "__main__":
    main()
