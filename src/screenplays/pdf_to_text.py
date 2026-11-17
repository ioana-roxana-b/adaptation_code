import argparse
import json
import os
import re
from typing import List, Tuple

import pymupdf

from . import corpus
from .parser import _is_heading, _clean_heading

CHAR_WIDTH = 7.2          # Courier 12pt
INDENT_OFFSET = 5         # so the usual 1.5in action margin (108pt) becomes 10 spaces
LINE_HEIGHT = 12.0
DUAL_SPLIT_X = 300        # right column of dual dialogue starts past this x
CUE_X, PAREN_X, DIALOGUE_X = 252, 212, 180   # standard columns used for unfolded dual dialogue

Line = Tuple[float, float, str, bool]   # y, x, text, is_red


def _is_red(color: int) -> bool:
    r, g, b = (color >> 16) & 255, (color >> 8) & 255, color & 255
    return r > 150 and g < 80 and b < 80


def page_lines(page) -> List[Line]:
    seen, lines = set(), []
    for block in page.get_text("dict")["blocks"]:
        for l in block.get("lines", []):
            text = "".join(s["text"] for s in l["spans"]).strip()
            if not text:
                continue
            x, y = round(l["bbox"][0], 1), round(l["bbox"][1], 1)
            key = (round(y), round(x), text)
            if key in seen:
                continue
            seen.add(key)
            words = [s for s in l["spans"] if s["text"].strip()]
            red = sum(len(s["text"]) for s in words if _is_red(s["color"])) > sum(len(s["text"]) for s in words) / 2
            lines.append((y, x, text, red))
    return sorted(lines)


def _rows(lines: List[Line]) -> List[List[Line]]:
    """Group lines that share a baseline."""
    rows: List[List[Line]] = []
    for ln in lines:
        if rows and abs(rows[-1][0][0] - ln[0]) < 2:
            rows[-1].append(ln)
        else:
            rows.append([ln])
    return rows


def _format(x: float, text: str) -> str:
    return " " * max(0, round(x / CHAR_WIDTH) - INDENT_OFFSET) + text


def convert_page(lines: List[Line]) -> Tuple[List[str], List[Tuple[str, bool]]]:
    """Text lines for one page, plus (scene heading, is_red) pairs in order."""
    out, headings = [], []
    rows = _rows(lines)
    i, prev_y = 0, None
    while i < len(rows):
        row = rows[i]
        y = row[0][0]
        if prev_y is not None and y - prev_y > 1.5 * LINE_HEIGHT:
            out.append("")

        # Dual dialogue starts with two cues on one baseline, one in each column
        if len(row) == 2 and row[0][1] < DUAL_SPLIT_X < row[1][1] and all(t.isupper() for _, _, t, _ in row):
            block = [row]
            j = i + 1
            while j < len(rows) and rows[j][0][0] - block[-1][0][0] <= 1.5 * LINE_HEIGHT:
                block.append(rows[j])
                j += 1
            for side in (lambda x: x < DUAL_SPLIT_X, lambda x: x >= DUAL_SPLIT_X):
                col = [ln for r in block for ln in r if side(ln[1])]
                for k, (_, _, text, _) in enumerate(col):
                    x = CUE_X if k == 0 else PAREN_X if text.startswith("(") else DIALOGUE_X
                    out.append(_format(x, text))
                out.append("")
            prev_y = block[-1][0][0]
            i = j
            continue

        for _, x, text, red in row:
            out.append(_format(x, text))
            if _is_heading(text):
                headings.append((_clean_heading(text), red))
        prev_y = y
        i += 1
    return out, headings


def convert_pages(pages: List[List[Line]]):
    """Join converted pages into one text, skipping front matter before the first heading."""
    text_lines, headings = [], []
    for lines in pages:
        if not lines:
            continue
        page_text, page_headings = convert_page(lines)
        # Front matter (title page, author's note) precedes the first scene heading
        if not headings and not page_headings:
            continue
        text_lines += page_text + [""]
        headings += page_headings
    return "\n".join(text_lines), headings


def convert(pdf_path: str):
    return convert_pages([page_lines(page) for page in pymupdf.open(pdf_path)])


def write_outputs(name: str, text: str, headings: List[Tuple[str, bool]]) -> None:
    os.makedirs(corpus.MANUAL_DIR, exist_ok=True)
    out_txt = os.path.join(corpus.MANUAL_DIR, f"{name}.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Wrote {out_txt} ({len(text.split())} words, {len(headings)} scene headings)")

    # Scene colours are only worth keeping when the script uses them
    n_red = sum(red for _, red in headings)
    if n_red:
        out_json = os.path.join(corpus.MANUAL_DIR, f"{name}.scenes.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump([{"heading": h, "red": red} for h, red in headings], f, indent=1, ensure_ascii=False)
        print(f"Wrote {out_json} ({n_red} of {len(headings)} headings in red)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf")
    ap.add_argument("name", help="output stem, e.g. little_women_2019")
    args = ap.parse_args()

    text, headings = convert(args.pdf)
    if len(re.findall(r"\w+", text)) < 1000:
        raise SystemExit(f"{args.pdf}: almost no text layer; use ocr_to_text instead.")
    write_outputs(args.name, text, headings)


if __name__ == "__main__":
    main()
