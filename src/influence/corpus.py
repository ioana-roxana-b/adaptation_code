import os
from dataclasses import dataclass
from itertools import zip_longest
from typing import Dict, List, Tuple

import main as iccc
from screenplays import corpus as sp_corpus

INFLUENCE, TRANSLATION, CONTROL, CIRCLE = "influence", "translation", "control", "circle"
REVERSED = "reversed"  # an influence pair scored target -> source (directionality check)
MODERNIST = "modernist_eng"

# key -> (Gutenberg ebook number, year, tradition). Bennett and Maugham are
# Edwardian realists (Woolf's "Mr Bennett and Mrs Brown" sets Bennett against
# the modernists), so they go with the realist tradition, not the movement.
GUTENBERG_CONTROLS: Dict[str, Tuple[int, int, str]] = {
    # Modernist, no documented link to Woolf or Mansfield
    "conrad_jim":          (5658, 1900, MODERNIST),
    "stein_lives":         (15408, 1909, MODERNIST),
    "ford_soldier":        (2775, 1915, MODERNIST),
    "lewis_tarr":          (58340, 1918, MODERNIST),
    "anderson_winesburg":  (416, 1919, MODERNIST),
    "sinclair_olivier":    (9366, 1919, MODERNIST),
    # Woolf's and Mansfield's circle
    "forster_howards":     (2891, 1910, MODERNIST),
    "lawrence_sons":       (217, 1913, MODERNIST),
    "lawrence_rainbow":    (28948, 1915, MODERNIST),
    "richardson_roofs":    (3019, 1915, MODERNIST),
    # Realist
    "gaskell_north":       (4276, 1855, "victorian_eng"),
    "trollope_barchester": (2432, 1857, "victorian_eng"),
    "collins_moonstone":   (155, 1868, "victorian_eng"),
    "gissing_grub":        (1709, 1891, "victorian_eng"),
    "bennett_wives":       (5247, 1908, "victorian_eng"),
    "maugham_bondage":     (351, 1915, "victorian_eng"),
    # Romantic-era
    "burney_evelina":      (6053, 1778, "romantic_eng"),
    "edgeworth_rackrent":  (1424, 1800, "romantic_eng"),
    # Continental, in English translation
    "balzac_goriot":       (1237, 1835, "continental"),
    "zola_germinal":       (56528, 1885, "continental"),
    # Classical epic other than Homer (Dryden)
    "virgil_aeneid":       (228, 1697, "classical"),
}

# Held-out test set for direction scores (src/direction): one-way, documented
# relations, mostly with large time gaps, never scored during development.
# Not part of pairs(); the texts are used only by src/direction.
TEST_TEXTS: Dict[str, Tuple[int, int, str]] = {
    "malory_arthur":      (1251, 1485, "early_eng"),      # vol. 1
    "milton_paradise":    (26, 1667, "early_eng"),
    "bunyan_pilgrim":     (131, 1678, "early_eng"),
    "defoe_crusoe":       (521, 1719, "eighteenth_eng"),
    "richardson_pamela":  (6124, 1740, "eighteenth_eng"),
    "fielding_andrews":   (9611, 1742, "eighteenth_eng"),  # vol. 1
    "fielding_tom":       (6593, 1749, "eighteenth_eng"),
    "radcliffe_udolpho":  (3268, 1794, "romantic_eng"),
    "austen_northanger":  (121, 1817, "romantic_eng"),
    "ballantyne_coral":   (646, 1857, "victorian_eng"),
    "alcott_women":       (514, 1868, "victorian_eng"),
    "twain_yankee":       (86, 1889, "victorian_eng"),
}
TEST_PAIRS = [
    ("malory_arthur", "twain_yankee"),           # Twain's burlesque quotes Malory at length
    ("milton_paradise", "shelley_frankenstein"),  # epigraph; the creature reads Paradise Lost
    ("bunyan_pilgrim", "alcott_women"),           # chapters framed as stages of the Pilgrim's Progress
    ("defoe_crusoe", "ballantyne_coral"),         # a Robinsonade
    ("richardson_pamela", "fielding_andrews"),    # written against Pamela; Joseph is Pamela's brother
    ("fielding_tom", "dickens_coperfield"),       # David reads Tom Jones (ch. 4); Dickens's acknowledged model
    ("radcliffe_udolpho", "austen_northanger"),   # Austen's heroine reads and misreads Udolpho
]

CIRCLE_AUTHORS = {"forster", "lawrence", "richardson"}
MIN_CONTROLS = 20
TRANSLATIONS = [("homer_oddyssey_pope", "homer_oddyssey_butler")]
INFLUENCE_PAIRS = [p for p in iccc.INFLUENCE_PAIRS if p not in TRANSLATIONS]


@dataclass(frozen=True)
class Text:
    key: str
    path: str
    year: int
    tradition: str

    @property
    def author(self) -> str:
        return self.key.split("_")[0]


def texts(include_test: bool = False) -> Dict[str, Text]:
    out = {k: Text(k, os.path.join(iccc.DATASET_DIR, f), y, iccc._get_tradition(k))
           for k, (f, y) in iccc.CORPUS.items()}
    extra = {**GUTENBERG_CONTROLS, **(TEST_TEXTS if include_test else {})}
    for k, (pg, y, trad) in extra.items():
        out[k] = Text(k, os.path.join(sp_corpus.GUTENBERG_DIR, f"pg{pg}.txt"), y, trad)
    return out


def pair_type(a: Text, b: Text) -> str:
    if (a.key, b.key) in INFLUENCE_PAIRS:
        return INFLUENCE
    if (a.key, b.key) in TRANSLATIONS:
        return TRANSLATION
    circle = {a.author, b.author} & CIRCLE_AUTHORS
    other = b if a.author in CIRCLE_AUTHORS else a
    return CIRCLE if circle and other.tradition == MODERNIST else CONTROL


def pairs() -> List[Tuple[Text, Text, str]]:
    """(source, target, pair type), references first."""
    T = texts()
    documented = {frozenset(p) for p in INFLUENCE_PAIRS + TRANSLATIONS}
    sources = {a for a, _ in INFLUENCE_PAIRS} | set(iccc.ANCHORS)
    targets = {b for _, b in INFLUENCE_PAIRS}
    out = [(T[a], T[b], pair_type(T[a], T[b])) for a, b in INFLUENCE_PAIRS + TRANSLATIONS]

    def eligible(a: str, b: str) -> bool:
        return a != b and T[a].author != T[b].author and frozenset((a, b)) not in documented

    chosen = [(a, b) for a in T for b in T
              if (a in sources or b in targets) and eligible(a, b) and T[a].year <= T[b].year]
    # Anachronistic top-up for influence sources (side 0) and targets (side 1) with few controls
    for side, keys in ((0, sorted({a for a, _ in INFLUENCE_PAIRS})), (1, sorted(targets))):
        for k in keys:
            have = sum(p[side] == k and pair_type(T[p[0]], T[p[1]]) == CONTROL for p in chosen)
            others = [o for o in T if (eligible(k, o) if side == 0 else eligible(o, k))]
            late = [o for o in others if (T[o].year < T[k].year if side == 0 else T[o].year > T[k].year)
                    and pair_type(*((T[k], T[o]) if side == 0 else (T[o], T[k]))) == CONTROL]
            late.sort(key=lambda o: (abs(T[o].year - T[k].year), o))
            by_trad: Dict[str, List[str]] = {}
            for o in late:
                by_trad.setdefault(T[o].tradition, []).append(o)
            order = [o for group in zip_longest(*by_trad.values()) for o in group if o]
            for o in order[:max(0, MIN_CONTROLS - have)]:
                chosen.append((k, o) if side == 0 else (o, k))

    for a, b in sorted(set(chosen)):
        out.append((T[a], T[b], pair_type(T[a], T[b])))
    return out


def reversed_pairs() -> List[Tuple[Text, Text, str]]:
    """The influence pairs backwards (target, source, REVERSED)."""
    T = texts()
    return [(T[b], T[a], REVERSED) for a, b in INFLUENCE_PAIRS]
