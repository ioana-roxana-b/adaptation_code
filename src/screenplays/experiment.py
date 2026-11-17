import argparse
import gc
import hashlib
import os
import re
import sys
from collections import Counter
from itertools import product
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import corpus, novel, parser

OUT_DIR = "out_screenplays"
COMPONENT_DIR = os.path.join(corpus.EXTERNAL_DIR, "components")

NOVEL_PARTS = {
    "dialogue": (novel.DIALOGUE,),
    "narration": (novel.NARRATION, novel.EMBEDDED),
    "full": novel.COMPONENTS,
}
SCRIPT_PARTS = {
    "dialogue": (parser.DIALOGUE,),
    "action": (parser.ACTION,),
    "full": parser.COMPONENTS,
}
MASK_TOKEN = "Name"
# A cue token is a name only if at most this share of its occurrences are lowercase
LOWERCASE_MAX_SHARE = 0.1
# Feature z-scores are clipped so one heavy-tailed feature (topic_cosine, whose
# control values sit near 1e-4) cannot dominate its channel mean
Z_CLIP = 5.0
# Honorifics and generic cue words kept when masking names
NOT_NAME = {"MR", "MRS", "MISS", "MS", "DR", "LADY", "LORD", "SIR", "AUNT", "UNCLE", "MADAM",
            "THE", "OLD", "YOUNG", "LITTLE", "VOICE", "MAN", "WOMAN", "GIRL", "BOY", "FATHER",
            "MOTHER", "SERVANT", "MAID", "CAPTAIN", "COLONEL", "OFFICER", "DOCTOR", "PROFESSOR",
            "GRANDFATHER", "SAILOR", "LOOKOUT", "CREATURE", "MONSTER", "STUDENT", "WOMEN", "MEN",
            "HERR", "FRAU", "GERMAN", "ARAB", "AUSTRALIAN", "BRITISH", "EGYPTIAN", "FRENCH",
            "FRENCHMAN", "PRUSSIAN", "BARMAID", "CABBY", "CONSTABLE", "DOORMAN", "MOTORIST",
            "STATIONMASTER", "SENESCHAL", "DERVISH", "ABBOT", "PROF", "SGT", "EXODUS"}
# Full design: control novels per screenplay
CONTROLS_PER_SCRIPT = 5


def pilot_pairs() -> List[Tuple[corpus.Adaptation, corpus.Adaptation, str]]:
    """(novel source, screenplay, pair type) for the pilot: references plus cross-novel controls."""
    scripts = corpus.included(pilot_only=True)
    novels = list({a.novel_key: a for a in reversed(scripts)}.values())
    pairs = [(a, a, "influence") for a in scripts]
    pairs += [(n, s, "control") for n in novels for s in scripts if s.novel_key != n.novel_key]
    return pairs


def _author(novel_key: str) -> str:
    return novel_key.split("_")[0]


def full_pairs(k: int = CONTROLS_PER_SCRIPT, seed: int = 0) -> List[Tuple[corpus.Adaptation, corpus.Adaptation, str]]:
    """References plus k control novels per screenplay, by other authors.

    Controls are drawn so that every novel is used about equally often, which
    lets the scoring estimate a novel effect and a screenplay effect from the
    controls (see v2.calibrate.Additive).
    """
    scripts = corpus.included()
    novels = list({a.novel_key: a for a in reversed(scripts)}.values())
    rng = np.random.default_rng(seed)
    used = Counter()
    pairs = [(a, a, "influence") for a in scripts]
    for s in [scripts[i] for i in rng.permutation(len(scripts))]:
        eligible = [n for n in novels if _author(n.novel_key) != _author(s.novel_key)]
        rng.shuffle(eligible)
        for n in sorted(eligible, key=lambda n: used[n.novel_key])[:k]:
            used[n.novel_key] += 1
            pairs.append((n, s, "control"))
    return pairs


DESIGNS = {"pilot": pilot_pairs, "full": full_pairs}
# Component cells per design: the pilot crosses all parts, the full corpus keeps
# the screenplay ablation (full novel) and the two matching-modality cells
CELLS = {
    "pilot": list(product(NOVEL_PARTS, SCRIPT_PARTS)),
    "full": [("full", "full"), ("full", "dialogue"), ("full", "action"),
             ("dialogue", "dialogue"), ("narration", "action")],
}


def variant_name(design: str, masked: bool) -> str:
    """Output folder under OUT_DIR; the pilot keeps its original names."""
    base = "masked" if masked else "unmasked"
    return base if design == "pilot" else f"{design}_{base}"


def parse_cells(spec: str, design: str) -> List[Tuple[str, str]]:
    """'full:full,full:dialogue' -> cells; empty -> the design's default cells."""
    if not spec:
        return CELLS[design]
    cells = [tuple(c.split(":")) for c in spec.split(",")]
    bad = [c for c in cells if len(c) != 2 or c[0] not in NOVEL_PARTS or c[1] not in SCRIPT_PARTS]
    if bad:
        sys.exit(f"Unknown cells {bad}; use novel_part:script_part with {list(NOVEL_PARTS)} x {list(SCRIPT_PARTS)}")
    return cells


def character_names(films: List[str], novel_paths: List[str]) -> List[str]:
    """Cue tokens that behave as proper names in the pilot texts.

    A cue token is kept when it occurs capitalised in running text and its lowercase form is rare in the novels and
    screenplays (fewer than LOWERCASE_MAX_SHARE of its occurrences), which drops
    role words used as cues (EXECUTIONER, VILLAGERS) but keeps names (JO, DARCY).
    """
    texts = [novel.load(p).text() for p in novel_paths]
    cues = set()
    for film in films:
        sp = parser.load(film)
        texts.append(sp.text())
        for cue in sp.characters():
            for tok in re.findall(r"[A-Z][A-Z’'\-]+", cue.upper()):
                tok = re.sub(r"[’'](S)?$", "", tok)
                if len(tok) > 2 and tok not in NOT_NAME:
                    cues.add(tok.capitalize())
    counts = Counter(re.findall(r"[A-Za-z][a-z’'\-]+", "\n".join(texts)))
    names = {t for t in cues
             if counts[t] and counts[t.lower()] <= LOWERCASE_MAX_SHARE * (counts[t] + counts[t.lower()])}
    return sorted(names, key=len, reverse=True)


def mask(text: str, names: List[str]) -> str:
    if not names:
        return text
    pattern = re.compile(r"\b(?:" + "|".join(map(re.escape, names)) + r")(?:[’']s)?\b", re.I)
    return pattern.sub(MASK_TOKEN, text)


def write_component(key: str, part: str, text: str) -> Tuple[str, str]:
    """Write a component text; the label carries a content hash so stale caches are never reused."""
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    label = f"{key}__{part}__{digest}"
    path = os.path.join(COMPONENT_DIR, f"{label}.txt")
    if not os.path.exists(path):
        os.makedirs(COMPONENT_DIR, exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"   # renamed into place, so parallel workers never read a partial file
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    return label, path


def build_components(pairs, names: List[str]) -> Dict[Tuple[str, str, str], Tuple[str, str]]:
    """(side, key, part) -> (label, path) for every novel and screenplay component."""
    suffix = "__masked" if names else ""
    comps = {}
    for src in {p[0].novel_key: p[0] for p in pairs}.values():
        parts = novel.load(src.novel_path)
        for part, kinds in NOVEL_PARTS.items():
            comps[("novel", src.novel_key, part)] = write_component(
                f"novel_{src.novel_key}{suffix}", part, mask(parts.text(kinds), names))
    for scr in {p[1].film: p[1] for p in pairs}.values():
        sp = parser.load(scr.film)
        key = re.sub(r"\W+", "_", scr.film).strip("_").lower()
        for part, kinds in SCRIPT_PARTS.items():
            comps[("script", scr.film, part)] = write_component(
                f"script_{key}{suffix}", part, mask(sp.text(kinds), names))
    return comps


def run(pairs, comps, raw_dir: str, cells: List[Tuple[str, str]]) -> List[str]:
    import main as pipeline
    from features_runner import run_pair_preprocessed
    from preprocess import ensure_nltk, load_or_build_preprocessed

    ensure_nltk()
    config = dict(pipeline.CONFIG, strategies=["paragraph"], strip_gutenberg=False)
    os.makedirs(raw_dir, exist_ok=True)
    paths = []
    # Cell by cell, so the first cells are complete for every pair before the next starts
    for (n_part, s_part), (src, scr, kind) in product(cells, pairs):
        a_label, a_path = comps[("novel", src.novel_key, n_part)]
        b_label, b_path = comps[("script", scr.film, s_part)]
        pair_key = f"{a_label}_vs_{b_label}"
        out = os.path.join(raw_dir, f"{pair_key}.csv")
        paths.append(out)
        if os.path.exists(out):
            continue
        print(f"Processing [{kind}] {src.novel_key}:{n_part} -> {scr.film}:{s_part}")
        pre_a = load_or_build_preprocessed(a_label, a_path, config)
        pre_b = load_or_build_preprocessed(b_label, b_path, config)
        df = run_pair_preprocessed(
            pre_a, pre_b, a_label, b_label, pair_key,
            strategies=config["strategies"], families=pipeline.FEATURE_FAMILIES,
            semantic_enabled=True, semantic_cfg={},
            syntax_use_spacy=True, syntax_spacy_model=config["syntax_spacy_model"],
        )
        df["pair_key"] = pair_key
        df["pair_type"] = kind
        df["novel"], df["novel_part"] = src.novel_key, n_part
        df["screenplay"], df["script_part"] = scr.film, s_part
        df.to_csv(out + ".tmp", index=False)
        os.replace(out + ".tmp", out)
        del pre_a, pre_b, df
        gc.collect()
    return paths


def score(raw_paths: List[str]) -> pd.DataFrame:
    """Channel z-scores and channelled score per pair and component cell."""
    import main as pipeline

    df = pd.concat([pd.read_csv(p) for p in raw_paths if os.path.exists(p)], ignore_index=True)
    feats = [f for f in pipeline.ALL_DFW_FEATURES if f in df.columns]
    for f in feats:
        df[f] = pd.to_numeric(df[f], errors="coerce")

    rows = []
    for (n_part, s_part), cell in df.groupby(["novel_part", "script_part"]):
        ctrl = cell[cell["pair_type"] == "control"]
        for pk, sub in cell.groupby("pair_key"):
            z = {}
            for f in feats:
                sd = ctrl[f].std()
                if not sd > pipeline.EPS:
                    continue
                v = float(np.clip((sub[f].mean() - ctrl[f].mean()) / sd, -Z_CLIP, Z_CLIP))
                z[f] = -v if f in pipeline.DIVERGENCE_FEATURES else v
            channels = {
                ch: float(np.mean([z[f] for f in fs if f in z])) if any(f in z for f in fs) else 0.0
                for ch, fs in pipeline.DFW_CHANNELS.items()
            }
            peak_ch = max(channels, key=channels.get)
            n_neg = sum(v < pipeline.DIVERGENCE_THRESHOLD for v in channels.values())
            rows.append({
                "novel": sub["novel"].iloc[0], "screenplay": sub["screenplay"].iloc[0],
                "pair_type": sub["pair_type"].iloc[0], "novel_part": n_part, "script_part": s_part,
                **{f"z_{ch}": v for ch, v in channels.items()},
                "peak_channel": peak_ch,
                "channelled": max(0.0, channels[peak_ch]) * (1 + pipeline.CHANNELLING_ALPHA * n_neg),
                "n_windows": len(sub),
            })
    return pd.DataFrame(rows)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mask-names", action="store_true")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--design", choices=list(DESIGNS), default="pilot")
    ap.add_argument("--cells", default="", help="e.g. full:full,full:dialogue (default: the design's cells)")
    ap.add_argument("--dry-run", action="store_true", help="list the pairs and the number of runs, then stop")
    ap.add_argument("--shard", default="", help="i/n: run every n-th pair from the i-th (0-based), for parallel "
                                                "workers; scoring is left to a final unsharded --score-only")
    args = ap.parse_args()

    pairs = DESIGNS[args.design]()
    cells = parse_cells(args.cells, args.design)
    if args.dry_run:
        for kind in ("influence", "control"):
            print(f"{kind}: {sum(p[2] == kind for p in pairs)} pairs")
        print("control uses per novel:", dict(Counter(p[0].novel_key for p in pairs if p[2] == "control")))
        print(f"cells: {cells}\nruns: {len(pairs) * len(cells)}")
        return
    names = (character_names(sorted({p[1].film for p in pairs}), sorted({p[0].novel_path for p in pairs}))
             if args.mask_names else [])
    if names:
        print(f"Masking {len(names)} names: {', '.join(names[:25])}...")
    variant = variant_name(args.design, bool(names))
    raw_dir = os.path.join(OUT_DIR, variant, "raw")

    comps = build_components(pairs, names)
    if args.score_only:
        import main as pipeline  # noqa: F401  (import check only)
        raw_paths = [os.path.join(raw_dir, f) for f in sorted(os.listdir(raw_dir))]
    elif args.shard:
        i, n = map(int, args.shard.split("/"))
        run(pairs[i::n], comps, raw_dir, cells)
        return
    else:
        raw_paths = run(pairs, comps, raw_dir, cells)

    res = score(raw_paths)
    out = os.path.join(OUT_DIR, variant, "component_scores.csv")
    res.to_csv(out, index=False)
    print(f"\nSaved {out}")

    for kind in ("influence", "control"):
        sub = res[res["pair_type"] == kind]
        print(f"\nMean channelled score, {kind} pairs (rows: novel part, cols: screenplay part)")
        print(sub.pivot_table(index="novel_part", columns="script_part", values="channelled").round(2))


if __name__ == "__main__":
    main()
