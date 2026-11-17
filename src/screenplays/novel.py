import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from preprocess import load_text

from . import corpus

DIALOGUE, NARRATION, EMBEDDED = "dialogue", "narration", "embedded"
COMPONENTS = (DIALOGUE, NARRATION, EMBEDDED)

DOUBLE = ("“", "”")
SINGLE = ("‘", "’")
STRAIGHT = '"'
STRAIGHT_SINGLE = ("'", "'")
_STRAIGHT_SINGLE_OPEN_RE = re.compile(r"(?:^|(?<=[\s(\[—-]))'(?=\w)", re.M)
# Single quotes mark speech if they open more than this many times as often as double ones
SINGLE_QUOTE_RATIO = 2

# A quotation spanning at least this many paragraphs and words is embedded narration
EMBEDDED_MIN_PARAGRAPHS = 3
EMBEDDED_MIN_WORDS = 300

# Gutenberg cleaning: an unquoted paragraph of at most this many words that is
# (almost) all capitals is a heading
HEADING_MAX_WORDS = 12
HEADING_UPPER_SHARE = 0.8
_CHAPTER_RE = re.compile(r"^(?:CHAPTER|Chapter|BOOK|Book|PART|Part)\s+[IVXLCDM\d]+\b"
                         r"|^[IVXLC]+\.\s|^(?:Prologue|Epilogue|Preface|Conclusion)\.?$")
# Chapter epigraph: a short quotation followed by its source ("...”—Shakespeare)
_EPIGRAPH_RE = re.compile(r"^[“\"].{0,400}[”\"]\s*—\s*[A-Z][\w .’']{1,40}$")
_NOTE_RE = re.compile(r"^\[(?:Illustration|Footnote)|^\{\d+\}")   # {3}: Edgeworth's notes
_NOTE_MARK_RE = re.compile(r"^\[\d+\]$")


@dataclass
class Segment:
    kind: str
    text: str
    paragraph: int


@dataclass
class NovelParts:
    segments: List[Segment] = field(default_factory=list)

    def text(self, components=(DIALOGUE, NARRATION, EMBEDDED)) -> str:
        """Paragraph-separated text of the selected components, in reading order.

        Segments of one paragraph are joined, so each output paragraph keeps
        the source paragraph's position for windowing.
        """
        wanted = set(components)
        paras: Dict[int, List[str]] = {}
        for s in self.segments:
            if s.kind in wanted:
                paras.setdefault(s.paragraph, []).append(s.text)
        return "\n\n".join(" ".join(p) for _, p in sorted(paras.items()))

    def stats(self) -> Dict[str, float]:
        words = Counter()
        for s in self.segments:
            words[s.kind] += len(s.text.split())
        total = sum(words.values()) or 1
        return {"words": total, **{f"{k}_pct": round(100 * words[k] / total, 1) for k in COMPONENTS}}


_HEADING_RE = re.compile(
    r"^(?:PART|BOOK|VOLUME|CHAPTER|Chapter|LETTER|Letter)\b[\w .,:;'’-]{0,60}$|^[IVXLC]+\.?$"
)


def _paragraphs(text: str) -> List[str]:
    paras = [re.sub(r"\s+", " ", p).strip().lstrip("﻿") for p in re.split(r"\n\s*\n", text)]
    paras = [p for p in paras if p]
    # The body starts at the first heading followed by prose; earlier headings are a contents list
    start = next(
        (i for i in range(len(paras) - 1) if _HEADING_RE.match(paras[i]) and not _HEADING_RE.match(paras[i + 1])),
        0,
    )
    return [p for p in paras[start:] if not _HEADING_RE.match(p)]


def _is_heading(p: str) -> bool:
    if p[:1] in DOUBLE + SINGLE + (STRAIGHT,):
        return False
    if _CHAPTER_RE.match(p) and len(p.split()) <= 3 * HEADING_MAX_WORDS:
        return True
    letters = [c for c in p if c.isalpha()]
    return (not letters or len(p.split()) <= HEADING_MAX_WORDS
            and sum(c.isupper() for c in letters) / len(letters) > HEADING_UPPER_SHARE)


def _gutenberg_paragraphs(text: str, start: Optional[str], end: Optional[str]) -> List[str]:
    """Body paragraphs of a Gutenberg download, without headings, notes or markup."""
    if start:
        paras = [re.sub(r"\s+", " ", p).strip().lstrip("﻿") for p in re.split(r"\n\s*\n", text)]
        paras = [p for p in paras if p]
        first = next(i for i, p in enumerate(paras) if re.match(start, p))
        paras = paras[first:]
    else:
        paras = _paragraphs(text)
    if end:
        paras = paras[:next(i for i, p in enumerate(paras) if re.match(end, p))]
    out, skip_next = [], False
    for p in paras:
        if skip_next:                  # footnote text after a bare "[1]" marker (Cooper)
            skip_next = False
            continue
        if _NOTE_MARK_RE.match(p):
            skip_next = True
            continue
        if _NOTE_RE.match(p) or _EPIGRAPH_RE.match(p) or _is_heading(p):
            continue
        p = re.sub(r"_([^_]+)_", r"\1", re.sub(r"\[\d+\]|\{\d+\}", "", p)).strip()
        if p:
            out.append(p)
    return out


def body_bounds(path: str) -> Optional[Tuple[Optional[str], Optional[str]]]:
    """(start, end) regexes for a Gutenberg download, None for a hand-cleaned text."""
    if os.path.dirname(os.path.normpath(path)) != os.path.normpath(corpus.GUTENBERG_DIR):
        return None
    m = re.match(r"pg(\d+)\.txt$", os.path.basename(path))
    return corpus.NOVEL_BODY.get(int(m.group(1)), (None, None)) if m else (None, None)


def _quote_pair(text: str) -> Tuple[str, str]:
    """The curly quote pair that marks speech in this text."""
    doubles = text.count(DOUBLE[0]) + text.count(STRAIGHT) / 2
    if text.count(SINGLE[0]) > SINGLE_QUOTE_RATIO * doubles:
        return SINGLE
    curly = text.count(DOUBLE[0]) + text.count(SINGLE[0])
    if not curly and len(_STRAIGHT_SINGLE_OPEN_RE.findall(text)) > SINGLE_QUOTE_RATIO * doubles:
        return STRAIGHT_SINGLE
    return DOUBLE


def _quote_spans(para: str, open_at_start: bool, pair: Tuple[str, str] = DOUBLE):
    """Yield (is_quoted, text) pieces; returns whether a quote is still open at the end."""
    pieces, buf, quoted = [], [], open_at_start
    open_q, close_q = pair
    curly = open_q in para or close_q in para
    straight = pair == DOUBLE and not curly
    for i, ch in enumerate(para):
        opens = (ch == open_q and not (pair == STRAIGHT_SINGLE and i > 0 and para[i - 1].isalnum())) \
            or (straight and ch == STRAIGHT and not quoted)
        closes = (ch == close_q and not (pair != DOUBLE and para[i + 1:i + 2].isalpha())) \
            or (straight and ch == STRAIGHT and quoted)
        if opens and not quoted:
            pieces.append((False, "".join(buf)))
            buf, quoted = [], True
        elif closes and quoted:
            pieces.append((True, "".join(buf)))
            buf, quoted = [], False
        else:
            buf.append(ch)
    pieces.append((quoted, "".join(buf)))
    return [(q, t.strip()) for q, t in pieces if t.strip()], quoted


def _starts_with_quote(para: str, pair: Tuple[str, str]) -> bool:
    return para[:1] == pair[0] or (pair == DOUBLE and para[:1] == STRAIGHT)


def _nested_dialogue(segments: List[Segment], pair: Tuple[str, str]) -> List[Segment]:
    """Split embedded narration into the teller's narration and the speech it quotes."""
    inner = SINGLE if pair == DOUBLE else DOUBLE
    out = []
    for s in segments:
        if s.kind != EMBEDDED:
            out.append(s)
            continue
        pieces, _ = _quote_spans(s.text, open_at_start=False, pair=inner)
        out += [Segment(DIALOGUE if q else EMBEDDED, t, s.paragraph) for q, t in pieces]
    return out


def split(text: str, body: Optional[Tuple[Optional[str], Optional[str]]] = None) -> NovelParts:
    """body: Gutenberg bounds (see body_bounds); None for a hand-cleaned text."""
    paras = _paragraphs(text) if body is None else _gutenberg_paragraphs(text, *body)
    pair = _quote_pair(text)
    segments: List[Segment] = []
    # Track runs of paragraphs inside one continued quotation
    run: List[int] = []          # indices into segments belonging to the current quotation
    run_paras = 0
    still_open = False

    def close_run():
        nonlocal run, run_paras
        words = sum(len(segments[i].text.split()) for i in run)
        if run_paras >= EMBEDDED_MIN_PARAGRAPHS and words >= EMBEDDED_MIN_WORDS:
            for i in run:
                segments[i].kind = EMBEDDED
        run, run_paras = [], 0

    for pi, para in enumerate(paras):
        # An open quote carries over only if this paragraph reopens it
        continuing = still_open and _starts_with_quote(para, pair)
        if not continuing:
            close_run()
        start = para[1:] if continuing else para
        pieces, still_open = _quote_spans(start, open_at_start=continuing, pair=pair)

        if continuing or (pieces and pieces[0][0]) or still_open:
            run_paras += 1
        for quoted, piece in pieces:
            segments.append(Segment(DIALOGUE if quoted else NARRATION, piece, pi))
            if quoted:
                run.append(len(segments) - 1)
        if not still_open:
            close_run()
    close_run()
    return NovelParts(_nested_dialogue(segments, pair))


def load(path: str) -> NovelParts:
    return split(load_text(path, strip_gutenberg=True), body_bounds(path))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    for path in sys.argv[1:]:
        parts = load(path)
        print(f"\n== {path}")
        print("  " + "  ".join(f"{k}={v}" for k, v in parts.stats().items()))
        for kind in COMPONENTS:
            sample = [s.text for s in parts.segments if s.kind == kind][:3]
            for s in sample:
                print(f"    {kind:9s} {s[:100]}")


if __name__ == "__main__":
    main()
