import argparse
import csv
import difflib
import io
import os
import re
import subprocess
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import List

import pymupdf

from .pdf_to_text import Line, convert_pages, write_outputs

DPI = 300
MIN_CONFIDENCE = 30       # Tesseract word confidence (0-100)
MIN_X_PT = 54             # ignore marks in the left 0.75in (hole punches)
HEADER_ZONE_PT = 72       # top inch of the page
HEADER_MIN_SHARE = 0.3    # a top line repeated on this share of pages is a running header
HEADER_SIMILARITY = 0.5   # OCR variants of the header still match at this ratio
HEADER_MAX_ROWS = 2       # rows from the top of each page checked against the header


def _correct(text: str) -> str:
    """Systematic Courier OCR confusions ('m' read as 'ny')."""
    return re.sub(r"\b([Nn])y(self)?\b", lambda m: ("M" if m.group(1) == "N" else "m") + "y" + (m.group(2) or ""), text)


def ocr_page(png: str) -> List[Line]:
    tsv = subprocess.run(
        ["tesseract", png, "stdout", "-l", "eng", "--psm", "4", "tsv"],
        capture_output=True, text=True, check=True,
    ).stdout

    words = {}
    for w in csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE):
        text = (w["text"] or "").strip()
        # Revision bars in the margin; low-confidence words are kept if they contain letters
        if not text or text in {"|", "||", "l|", "|l"}:
            continue
        if float(w["conf"]) < MIN_CONFIDENCE and not re.search(r"[A-Za-z]{2,}", text):
            continue
        key = (w["block_num"], w["par_num"], w["line_num"])
        words.setdefault(key, []).append((int(w["left"]), int(w["top"]), text))

    scale = 72.0 / DPI
    lines = []
    for ws in words.values():
        ws.sort()
        x, y = ws[0][0] * scale, min(t for _, t, _ in ws) * scale
        text = _correct(" ".join(t for _, _, t in ws))
        if x < MIN_X_PT and len(text) < 4:
            continue
        lines.append((round(y, 1), round(x, 1), text, False))
    return sorted(lines)


def _header_key(text: str) -> str:
    return re.sub(r"[\d\W]+", " ", text.lower()).strip()


def drop_running_headers(pages: List[List[Line]]) -> List[List[Line]]:
    """Remove the repeated page header; OCR varies it slightly, so match approximately."""
    tops = Counter()
    for lines in pages:
        tops.update({_header_key(t) for y, _, t, _ in lines if y < HEADER_ZONE_PT})
    headers = [k for k, n in tops.items() if k and n >= HEADER_MIN_SHARE * len(pages)]

    def is_header(text: str) -> bool:
        key = _header_key(text)
        if not key or re.search(r"\d+\.?\s*$", text) and len(key.split()) <= 8:
            return True
        return any(difflib.SequenceMatcher(None, key, h).ratio() > HEADER_SIMILARITY for h in headers)

    # Skewed scans push the header below the zone, so also check each page's first rows
    out = []
    for lines in pages:
        top_ys = sorted({round(y) for y, _, _, _ in lines})[:HEADER_MAX_ROWS]
        out.append([ln for ln in lines
                    if not ((ln[0] < HEADER_ZONE_PT or round(ln[0]) in top_ys) and is_header(ln[2]))])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf")
    ap.add_argument("name", help="output stem, e.g. little_women_1994")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        # Render sequentially (PyMuPDF is not thread-safe), OCR in parallel
        pngs = []
        for page in pymupdf.open(args.pdf):
            pngs.append(os.path.join(tmp, f"p{page.number:04d}.png"))
            page.get_pixmap(dpi=DPI, colorspace=pymupdf.csGRAY).save(pngs[-1])
        with ThreadPoolExecutor(args.workers) as ex:
            pages = list(ex.map(ocr_page, pngs))
    pages = drop_running_headers(pages)

    text, headings = convert_pages(pages)
    write_outputs(args.name, text, headings)


if __name__ == "__main__":
    main()
