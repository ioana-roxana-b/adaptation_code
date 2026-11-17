import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import experiment, novel, parser

PASSAGE_WORDS = 200
PIECE_WORDS = 180       # all-MiniLM-L6-v2 truncates at 256 word pieces
MIN_SCENE_WORDS = 20
MIN_SCENES = 30
CHUNK_WORDS = 250


def _pieces(text: str, size: int = PIECE_WORDS) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)] or [""]


def embed(model, texts: List[str]) -> np.ndarray:
    """Mean-pooled, normalised embeddings of possibly long texts."""
    pieces, owner = [], []
    for i, t in enumerate(texts):
        for p in _pieces(t):
            pieces.append(p)
            owner.append(i)
    vecs = model.encode(pieces, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    owner = np.array(owner)
    out = np.stack([vecs[owner == i].mean(axis=0) for i in range(len(texts))])
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def novel_passages(text: str) -> List[str]:
    """Consecutive paragraphs grouped into passages of about PASSAGE_WORDS words."""
    out, buf, n = [], [], 0
    for para in text.split("\n\n"):
        buf.append(para)
        n += len(para.split())
        if n >= PASSAGE_WORDS:
            out.append(" ".join(buf))
            buf, n = [], 0
    if buf:
        out.append(" ".join(buf))
    return out


def screenplay_scenes(sp: parser.Screenplay, components) -> List[Tuple[int, str]]:
    """(scene number, text) for scenes with at least MIN_SCENE_WORDS words in the components."""
    wanted = set(components)
    if parser.ACTION in wanted:
        wanted.add(parser.PARENTHETICAL)
    scenes: Dict[int, List[str]] = {}
    for e in sp.elements:
        if e.kind in wanted and e.scene > 0:
            scenes.setdefault(e.scene, []).append(e.text)
    return [(s, " ".join(t)) for s, t in sorted(scenes.items()) if len(" ".join(t).split()) >= MIN_SCENE_WORDS]


def screenplay_chunks(sp: parser.Screenplay, components) -> List[Tuple[int, str]]:
    """(chunk number, text): consecutive elements grouped into about CHUNK_WORDS words."""
    wanted = set(components)
    if parser.ACTION in wanted:
        wanted.add(parser.PARENTHETICAL)
    out, buf, n = [], [], 0
    for e in sp.elements:
        if e.kind not in wanted:
            continue
        buf.append(e.text)
        n += len(e.text.split())
        if n >= CHUNK_WORDS:
            out.append((len(out) + 1, " ".join(buf)))
            buf, n = [], 0
    if n >= MIN_SCENE_WORDS:
        out.append((len(out) + 1, " ".join(buf)))
    return out


def screenplay_units(sp: parser.Screenplay, components) -> Tuple[str, List[Tuple[int, str]]]:
    """Scenes if the headings parse into enough of them, else chunks."""
    if len(sp.scene_headings) >= MIN_SCENES:
        return "scene", screenplay_scenes(sp, components)
    return "chunk", screenplay_chunks(sp, components)


def match(scene_vecs: np.ndarray, passage_vecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Best novel passage per scene (relative position in [0, 1]) and its raw similarity."""
    sim = scene_vecs @ passage_vecs.T
    best = (sim - sim.mean(axis=0, keepdims=True)).argmax(axis=1)
    return best / max(len(passage_vecs) - 1, 1), sim[np.arange(len(best)), best]


def order_features(positions: np.ndarray) -> Dict[str, float]:
    if len(positions) < 3:
        return {"order_rho": np.nan, "order_local": np.nan}
    rho = spearmanr(np.arange(len(positions)), positions)[0]
    return {"order_rho": float(rho), "order_local": float(np.mean(np.diff(positions) >= 0))}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mask-names", action="store_true")
    ap.add_argument("--design", choices=list(experiment.DESIGNS), default="pilot")
    ap.add_argument("--cells", default="", help="e.g. full:full,full:dialogue (default: the design's cells)")
    args = ap.parse_args()

    from features.semantic import get_sbert_model

    pairs = experiment.DESIGNS[args.design]()
    cells = experiment.parse_cells(args.cells, args.design)
    films = sorted({p[1].film for p in pairs})
    names = (experiment.character_names(films, sorted({p[0].novel_path for p in pairs}))
             if args.mask_names else [])
    variant = experiment.variant_name(args.design, bool(names))
    model = get_sbert_model("all-MiniLM-L6-v2")

    # Embed every novel and screenplay component once
    passages: Dict[Tuple[str, str], np.ndarray] = {}
    for src in {p[0].novel_key: p[0] for p in pairs}.values():
        parts = novel.load(src.novel_path)
        for part, kinds in experiment.NOVEL_PARTS.items():
            texts = [experiment.mask(t, names) for t in novel_passages(parts.text(kinds))]
            passages[(src.novel_key, part)] = embed(model, texts)
    scenes: Dict[Tuple[str, str], Tuple[List[int], np.ndarray]] = {}
    screenplays = {f: parser.load(f) for f in films}
    units = {}
    for film, sp in screenplays.items():
        for part, kinds in experiment.SCRIPT_PARTS.items():
            units[film], sc = screenplay_units(sp, kinds)
            scenes[(film, part)] = ([s for s, _ in sc], embed(model, [experiment.mask(t, names) for _, t in sc]))

    rows, matches = [], []
    for src, scr, kind in pairs:
        for n_part, s_part in cells:
            nums, svecs = scenes[(scr.film, s_part)]
            pos, sim = match(svecs, passages[(src.novel_key, n_part)])
            rows.append({"novel": src.novel_key, "screenplay": scr.film, "pair_type": kind,
                         "novel_part": n_part, "script_part": s_part, "unit": units[scr.film],
                         "n_scenes": len(nums), **order_features(pos)})
            meta = screenplays[scr.film].scene_meta if units[scr.film] == "scene" else []
            for s, p, v in zip(nums, pos, sim):
                red = meta[s - 1].get("red") if meta and s - 1 < len(meta) else None
                matches.append({"novel": src.novel_key, "screenplay": scr.film, "pair_type": kind,
                                "novel_part": n_part, "script_part": s_part,
                                "scene": s, "novel_pos": round(float(p), 4), "sim": round(float(v), 4),
                                "timeline": {True: "past", False: "present"}.get(red)})

    out_dir = os.path.join(experiment.OUT_DIR, variant)
    os.makedirs(out_dir, exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, "sequence.csv"), index=False)
    m = pd.DataFrame(matches)
    m.to_csv(os.path.join(out_dir, "sequence_matches.csv"), index=False)
    print(f"Saved {out_dir}/sequence.csv and sequence_matches.csv")

    full = res[(res.novel_part == "full") & (res.script_part == "full")]
    print("\nOrder, full novel x full screenplay")
    print(full[["pair_type", "novel", "screenplay", "order_rho", "order_local", "unit", "n_scenes"]].round(2).to_string(index=False))

    # Timeline check: in Gerwig's script, scenes of the 1861 timeline (red) should
    # match earlier novel positions than present-day scenes, whatever their place in the film
    t = m[(m.pair_type == "influence") & m.timeline.notna() & (m.novel_part == "full") & (m.script_part == "full")]
    for film, g in t.groupby("screenplay"):
        print(f"\n{film}: matched novel position by timeline")
        print(g.groupby("timeline").novel_pos.describe()[["count", "mean", "50%"]].round(2).to_string())
        for tl, h in g.groupby("timeline"):
            print(f"  order_rho within {tl} scenes: {order_features(h.novel_pos.to_numpy())['order_rho']:.2f}")


if __name__ == "__main__":
    main()
