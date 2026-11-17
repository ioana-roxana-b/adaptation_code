import json
import os
import re
import sys
import tarfile
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from . import corpus

HEADING, ACTION, DIALOGUE, PARENTHETICAL, TRANSITION = (
    "heading", "action", "dialogue", "parenthetical", "transition"
)
COMPONENTS = (HEADING, ACTION, DIALOGUE)

_HEADING_RE = re.compile(
    r"^(?:\d+[A-Z]?\.?\s+)?"
    r"(?:(?:[A-Z’']+\s+)?(?:POV|ANGLE|CLOSE|WIDE|INSERT|MONTAGE)[^a-z]{0,10}?[-:~=]?\s*)?"   # "JO'S POV - EXT: ..."
    r"(?:INT|EXT|I/E|INT\.?\s*/\s*EXT|EXT\.?\s*/\s*INT)(?:\.|:|\s|/|-|$)", re.I
)
_SCENE_NUM_RE = re.compile(r"^\d+[A-Z]?\.?\s+|\s+\d+[A-Z]?\.?$|\s+[IVX]+\.?$")
_TRANSITION_RE = re.compile(
    r"^(?:FADE (?:IN|OUT|TO BLACK)|CUT TO|SMASH CUT(?: TO)?|MATCH CUT(?: TO)?|DISSOLVE TO|"
    r"WIPE TO|JUMP CUT(?: TO)?|INTERCUT(?: WITH)?|BACK TO|CUT BACK TO|TIME CUT|FADE TO)\b.*$|^.*\bTO:$"
)
_NOISE_RE = re.compile(
    r"^(?:\d+[A-Z]?\.?|\(?CONTINUED\)?:?|\(?MORE\)?|CONT(?:INUED|'D)|THE END\.?|"
    r"[\W_\d]*|Page \d+.*)$", re.I
)
_CUE_EXT_RE = re.compile(r"\s*\((?:V\.?O\.?|O\.?S\.?|O\.?C\.?|CONT'?D|CONTINUING|INTO PHONE|ON PHONE|"
                         r"OVER RADIO|SUBTITLED?|PRE-?LAP|FILTERED)\)", re.I)
_PAREN_RE = re.compile(r"^\(.*\)?$")
_BRACKET_RE = re.compile(r"^\[.*\]?$")
# Upper-case slugs that look like cues but are camera directions or on-screen text
_NOT_SPEAKER_RE = re.compile(
    r"\b(?:ANGLE|CLOSE|SHOT|SHOTS|POV|VIEW|INSERT|MONTAGE|FLASHBACK|LATER|CONTINUOUS|BACK TO|"
    r"SERIES|INTERCUT|CAMERA|WIDE|PAN|TRACKING|MUSIC|SOUND|SFX|TITLE|TITLES|SUPER|SUPERIMPOSE|"
    r"CAPTION|CARD|INTERTITLE|SUBTITLE|ON SCREEN|END CREDITS|CREDITS|BLACK|SCREEN|DAY|NIGHT|"
    r"MORNING|EVENING|DAWN|DUSK|SAME|MOMENTS|SCENE|REVERSE|OMITTED)\b"
)

# Revision marks before a line: bullets, asterisks, or a stray letter before a scene number
_REVISION_MARK_RE = re.compile(r"^(?:[•*_]+|[A-Z](?=\s+\d+[A-Z]?\.?\s+(?:INT|EXT)))\s*")

MAX_CUE_WORDS = 5
MAX_CUE_CHARS = 40
# Script without a heading before this many words of text: treat the text as a title page
MAX_TITLE_PAGE_WORDS = 30


@dataclass
class Element:
    kind: str
    text: str
    scene: int
    character: Optional[str] = None   # speaker, for dialogue and parentheticals


@dataclass
class Screenplay:
    title: str
    elements: List[Element]
    scene_headings: List[str] = field(default_factory=list)
    # Per-heading metadata from pdf_to_text (e.g. {"red": True}), aligned with scene_headings
    scene_meta: List[dict] = field(default_factory=list)

    def characters(self, min_lines: int = 2) -> Counter:
        """Speaking characters with at least min_lines dialogue blocks."""
        c = Counter(e.character for e in self.elements if e.kind == DIALOGUE)
        return Counter({k: v for k, v in c.items() if v >= min_lines})

    def text(self, components: Iterable[str] = COMPONENTS) -> str:
        """Plain text of the selected components, one element per paragraph.

        components: any of "heading", "action", "dialogue" (e.g. ("dialogue",)
        for the dialogue-only ablation). Parentheticals count as action.
        """
        wanted = set(components)
        if ACTION in wanted:
            wanted.add(PARENTHETICAL)
        return "\n\n".join(e.text for e in self.elements if e.kind in wanted)

    def scenes(self, components: Iterable[str] = COMPONENTS) -> List[str]:
        """Text per scene; scene 0 holds anything before the first heading."""
        wanted = set(components)
        if ACTION in wanted:
            wanted.add(PARENTHETICAL)
        out = [[] for _ in range(len(self.scene_headings) + 1)]
        for e in self.elements:
            if e.kind in wanted:
                out[e.scene].append(e.text)
        return ["\n\n".join(s) for s in out if s]

    def stats(self) -> Dict[str, object]:
        words = Counter()
        for e in self.elements:
            words[e.kind] += len(e.text.split())
        total = sum(words.values()) or 1
        return {
            "scenes": len(self.scene_headings),
            "speakers": len(self.characters()),
            "dialogue_blocks": sum(e.kind == DIALOGUE for e in self.elements),
            "words": total,
            **{f"{k}_pct": round(100 * words[k] / total, 1)
               for k in (HEADING, ACTION, DIALOGUE, PARENTHETICAL)},
        }


# =====================================================================
# Line classification
# =====================================================================

def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _is_heading(s: str) -> bool:
    return bool(_HEADING_RE.match(s))


def _clean_heading(s: str) -> str:
    return _SCENE_NUM_RE.sub("", s).strip(" .")


def _cue_name(s: str) -> Optional[str]:
    """Normalised speaker name if s looks like a cue line, else None."""
    name = _CUE_EXT_RE.sub("", s).strip()
    name = re.sub(r"\s*\(.*?\)\s*$", "", name).strip(" :.")
    if not name or len(name) > MAX_CUE_CHARS or len(name.split()) > MAX_CUE_WORDS:
        return None
    letters = [c for c in name if c.isalpha()]
    if len(letters) < 2 or any(c.islower() for c in letters) or name[-1] in "!?,;-":
        return None
    if _is_heading(name) or _TRANSITION_RE.match(name) or _NOISE_RE.match(name):
        return None
    if _NOT_SPEAKER_RE.search(name) or name.startswith(("[", "(")):
        return None
    return name


def _percentile(values: List[int], p: float, default: int) -> int:
    return sorted(values)[min(int(p * len(values)), len(values) - 1)] if values else default


def _strip_title_page(lines: List[str], action_indent: int) -> List[str]:
    """Drop the centred title block (title, 'Written by', draft dates) before the script starts."""
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        if _is_heading(s) or _TRANSITION_RE.match(s) or _indent(line) <= action_indent + 2:
            return lines[i:]
    return lines


def _clean_line(line: str) -> str:
    """Undo UTF-8 read as cp1252 (e.g. 'â€¢' for '•') and drop leading revision marks."""
    if "â€" in line or "Ã" in line:
        try:
            line = line.encode("cp1252").decode("utf-8")
        except UnicodeError:
            pass
    stripped = line.lstrip(" ")
    cleaned = _REVISION_MARK_RE.sub("", stripped)
    if cleaned != stripped:
        # Keep the original indent so column detection is unaffected
        return line[: len(line) - len(stripped)] + cleaned
    return line


def _mode(values: List[int], default: int) -> int:
    return Counter(values).most_common(1)[0][0] if values else default


def parse(text: str, title: str = "") -> Screenplay:
    lines = [_clean_line(l.replace("\t", " " * 8).rstrip()) for l in text.replace("\r", "").split("\n")]
    nonblank = [(i, l) for i, l in enumerate(lines) if l.strip()]

    # Pass 1: candidate cues (followed by text within one blank line) and their frequency
    cue_counts = Counter()
    for k, (i, l) in enumerate(nonblank):
        name = _cue_name(l.strip())
        if name and k + 1 < len(nonblank) and nonblank[k + 1][0] - i <= 2:
            cue_counts[name] += 1

    def is_cue(line: str, next_gap: int) -> Optional[str]:
        name = _cue_name(line.strip())
        if name and (next_gap == 1 or cue_counts[name] >= 2):
            return name
        return None

    # Margins and column widths. The first line after a cue is reliably dialogue;
    # when dialogue shares the action margin, its narrower column separates the two.
    first_dialogue = [
        nonblank[k + 1][1] for k, (i, l) in enumerate(nonblank[:-1])
        if is_cue(l, nonblank[k + 1][0] - i) and not _PAREN_RE.match(nonblank[k + 1][1].strip())
    ]
    action_indent = _mode([_indent(l) for _, l in nonblank], 0)
    dialogue_indent = _mode([_indent(l) for l in first_dialogue], action_indent)
    indent_split = dialogue_indent > action_indent + 2

    dialogue_width = _percentile([len(l.strip()) for l in first_dialogue], 0.95, 0)
    action_width = _percentile(
        [len(l.strip()) for _, l in nonblank if _indent(l) == action_indent], 0.90, 0
    )
    width_split = not indent_split and action_width > dialogue_width + 8

    lines = _strip_title_page(lines, action_indent)

    elements: List[Element] = []
    headings: List[str] = []
    buf: List[str] = []
    buf_kind: Optional[str] = None
    speaker: Optional[str] = None
    in_dialogue = False
    pending_blank = False

    def flush():
        nonlocal buf, buf_kind
        if buf:
            elements.append(Element(buf_kind, " ".join(buf), len(headings),
                                    speaker if buf_kind in (DIALOGUE, PARENTHETICAL) else None))
        buf, buf_kind = [], None

    def add(kind, s):
        nonlocal buf_kind
        if buf_kind != kind:
            flush()
            buf_kind = kind
        buf.append(s)

    for idx, line in enumerate(lines):
        s = line.strip()
        if not s:
            # One blank line is allowed between a cue and its parenthetical/dialogue
            if in_dialogue and buf_kind is None and not pending_blank:
                pending_blank = True
                continue
            flush()
            in_dialogue = False
            pending_blank = False
            continue
        pending_blank = False

        if _NOISE_RE.match(s):
            continue
        if _is_heading(s):
            flush()
            in_dialogue = False
            headings.append(_clean_heading(s))
            elements.append(Element(HEADING, _clean_heading(s), len(headings)))
            continue
        # Upper-case check guards against action such as "Back to the house..."; short
        # lines are exempt because OCR sometimes miscases transitions ("cuT TO:")
        if _TRANSITION_RE.match(s.upper()) and (s.isupper() or len(s) <= 15 and s.endswith(":")):
            flush()
            in_dialogue = False
            elements.append(Element(TRANSITION, s, len(headings)))
            continue

        next_gap = next((j - idx for j in range(idx + 1, len(lines)) if lines[j].strip()), 99)
        name = is_cue(line, next_gap)
        if name:
            flush()
            speaker, in_dialogue = name, True
            continue

        if _BRACKET_RE.match(s):
            flush()
            in_dialogue = False
            add(ACTION, s.strip("[] "))
            flush()
            continue

        if in_dialogue:
            if indent_split and _indent(line) <= action_indent + 2:
                in_dialogue = False
            elif width_split and buf_kind == DIALOGUE and len(s) > dialogue_width + 4:
                in_dialogue = False
            elif _PAREN_RE.match(s) or buf_kind == PARENTHETICAL and not buf[-1].endswith(")"):
                add(PARENTHETICAL, s)
                continue
            else:
                add(DIALOGUE, s)
                continue

        add(ACTION, s)

    flush()
    elements = [e for e in elements if e.kind != TRANSITION]

    # A few short lines before the first heading are a title page, not action
    pre = [e for e in elements if e.scene == 0]
    if headings and sum(len(e.text.split()) for e in pre) < MAX_TITLE_PAGE_WORDS:
        elements = elements[len(pre):]
    return Screenplay(title=title, elements=elements, scene_headings=headings)


# =====================================================================
# Loading
# =====================================================================

def read_scriptbase(archive: str) -> str:
    with tarfile.open(archive) as t:
        member = next(m for m in t.getmembers() if m.name.endswith("/script.txt"))
        return t.extractfile(member).read().decode("utf-8", "replace")


def load(film: str) -> Screenplay:
    """Parse a curated screenplay by its corpus.ADAPTATIONS film name."""
    a = corpus.get(film)
    path = a.script_path if a else os.path.join(corpus.SCRIPTBASE_DIR, f"{film}.tar.gz")
    if path.endswith(".tar.gz"):
        return parse(read_scriptbase(path), title=film)

    with open(path, encoding="utf-8", errors="replace") as f:
        sp = parse(f.read(), title=film)
    meta_path = path[: -len(".txt")] + ".scenes.json"
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if [m["heading"] for m in meta] == sp.scene_headings:
            sp.scene_meta = meta
        else:
            print(f"  WARNING: {meta_path} does not match the parsed headings; ignored")
    return sp


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    films = sys.argv[1:] or [a.film for a in corpus.included(pilot_only=True)]
    for film in films:
        a = corpus.get(film)
        if a and not os.path.exists(a.script_path):
            print(f"\n== {film}: missing ({a.script_path})")
            continue
        sp = load(film)
        print(f"\n== {film}")
        print("  " + "  ".join(f"{k}={v}" for k, v in sp.stats().items()))
        print("  top speakers: " + ", ".join(f"{n} ({c})" for n, c in sp.characters().most_common(10)))
        print("  first elements:")
        for e in sp.elements[:12]:
            who = f" [{e.character}]" if e.character else ""
            print(f"    {e.kind:13s}{who} {e.text[:90]}")


if __name__ == "__main__":
    main()
