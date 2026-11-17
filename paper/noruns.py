import argparse
import os
import re
import sys
from functools import lru_cache

sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), "src")]
from src.screenplays import corpus, experiment, novel, parser, sequence  # noqa: E402

K = 6
# NORUNS_POOLED=1: windows embedded by pooling 180-word pieces (paper/sensitivity.py), into
# full_masked_noruns_pooled/, with the order scores copied from full_masked_noruns/
POOLED = os.environ.get("NORUNS_POOLED") == "1"
VARIANT = "full_masked_noruns_pooled" if POOLED else "full_masked_noruns"
CELLS = [("dialogue", "dialogue"), ("narration", "action")]
RAW_DIR = os.path.join(experiment.OUT_DIR, VARIANT, "raw")
WORD = re.compile(r"[a-z']+")
STATS = {}
_original_load = parser.load


def _words(text):
    return WORD.findall(text.lower().replace("’", "'"))


def strip_elements(elements, source_text):
    """Delete, in place, every token of the elements inside a run of >= K words also found in the source."""
    src = _words(source_text)
    grams = {tuple(src[i:i + K]) for i in range(len(src) - K + 1)}
    tokens = [e.text.split() for e in elements]
    seq = [(ei, ti, w) for ei, ts in enumerate(tokens) for ti, t in enumerate(ts) for w in _words(t)]
    words = [w for _, _, w in seq]
    covered = set()
    for i in range(len(words) - K + 1):
        if tuple(words[i:i + K]) in grams:
            covered.update(range(i, i + K))
    drop = {(seq[i][0], seq[i][1]) for i in covered}
    for ei, (e, ts) in enumerate(zip(elements, tokens)):
        e.text = " ".join(t for ti, t in enumerate(ts) if (ei, ti) not in drop)
    return len(covered), len(words)


@lru_cache(maxsize=None)
def stripped_load(film):
    sp = _original_load(film)
    parts = novel.load(corpus.get(film).novel_path)
    dia = [e for e in sp.elements if e.kind == parser.DIALOGUE]
    act = [e for e in sp.elements if e.kind in (parser.ACTION, parser.PARENTHETICAL)]
    STATS[film] = (strip_elements(dia, parts.text(experiment.NOVEL_PARTS["dialogue"])),
                   strip_elements(act, parts.text(experiment.NOVEL_PARTS["narration"])))
    return sp


def setup():
    pairs = experiment.full_pairs()
    # Names from the original texts, so the novels are masked exactly as in the main run
    names = experiment.character_names(sorted({p[1].film for p in pairs}), sorted({p[0].novel_path for p in pairs}))
    parser.load = stripped_load
    return pairs, names


def stats(pairs):
    for film in sorted({p[1].film for p in pairs}):
        stripped_load(film)
        (dc, dn), (ac, an) = STATS[film]
        print(f"{dc / max(dn, 1):6.3f} of {dn:6d} dialogue words, {ac / max(an, 1):6.3f} of {an:6d} action words  {film}")


def preprocess(pairs, names):
    import main as pipeline
    from preprocess import ensure_nltk, load_or_build_preprocessed
    ensure_nltk()
    config = dict(pipeline.CONFIG, strategies=["paragraph"], strip_gutenberg=False)
    comps = experiment.build_components(pairs, names)
    needed = {comps[("script", p[1].film, s)] for p in pairs for _, s in CELLS}
    needed |= {comps[("novel", p[0].novel_key, n)] for p in pairs for n, _ in CELLS}
    for i, (label, path) in enumerate(sorted(needed), 1):
        print(f"[{i}/{len(needed)}] {label}", flush=True)
        load_or_build_preprocessed(label, path, config)


def shard(pairs, names, spec):
    if POOLED:
        import shutil
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import sensitivity
        sensitivity.patch("pooled")
        os.makedirs(RAW_DIR, exist_ok=True)
        seq = os.path.join(experiment.OUT_DIR, VARIANT, "sequence.csv")
        if not os.path.exists(seq):
            shutil.copy(os.path.join(experiment.OUT_DIR, "full_masked_noruns", "sequence.csv"), seq)
    i, n = map(int, spec.split("/"))
    comps = experiment.build_components(pairs, names)
    experiment.run(pairs[i::n], comps, RAW_DIR, CELLS)


def order(pairs, names):
    import pandas as pd
    from features.semantic import get_sbert_model

    model = get_sbert_model("all-MiniLM-L6-v2")
    passages = {}
    for src in {p[0].novel_key: p[0] for p in pairs}.values():
        parts = novel.load(src.novel_path)
        for part, _ in CELLS:
            texts = [experiment.mask(t, names) for t in sequence.novel_passages(parts.text(experiment.NOVEL_PARTS[part]))]
            passages[(src.novel_key, part)] = sequence.embed(model, texts)
    scenes = {}
    for film in sorted({p[1].film for p in pairs}):
        sp = parser.load(film)
        for _, part in CELLS:
            unit, sc = sequence.screenplay_units(sp, experiment.SCRIPT_PARTS[part])
            scenes[(film, part)] = sequence.embed(model, [experiment.mask(t, names) for _, t in sc])
    rows = []
    for src, scr, kind in pairs:
        for n_part, s_part in CELLS:
            pos, _ = sequence.match(scenes[(scr.film, s_part)], passages[(src.novel_key, n_part)])
            rows.append({"novel": src.novel_key, "screenplay": scr.film, "pair_type": kind,
                         "novel_part": n_part, "script_part": s_part, **sequence.order_features(pos)})
    out = os.path.join(experiment.OUT_DIR, VARIANT, "sequence.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved {out}")


def compare():
    import numpy as np
    import pandas as pd
    from scipy.stats import wilcoxon

    chans = ["z_lexical", "z_semantic", "z_conceptual", "z_structural", "z_narrative"]

    def holm(p):
        p = np.asarray(p, float)
        o = np.argsort(p)
        adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[o]).clip(max=1)
        out = np.empty(len(p))
        out[o] = adj
        return out

    lines = []
    for name in (("full_masked_pooled", VARIANT) if POOLED else ("full_masked", VARIANT)):
        d = pd.read_csv(os.path.join("out_v2", f"screenplays_{name}.csv"))
        a = d[d.pair_type != "control"]
        cell = lambda n, s: a[(a.novel_part == n) & (a.script_part == s)].set_index("screenplay")
        dd, na = cell("dialogue", "dialogue"), cell("narration", "action")
        for level, (x, y) in {"22 films": (dd, na),
                              "19 sources": (dd.groupby("novel")[chans + ["profile"]].mean(),
                                             na.groupby("novel")[chans + ["profile"]].mean())}.items():
            diffs = [(x[c] - y[c].reindex(x.index)).dropna() for c in chans]
            adj = holm([wilcoxon(v).pvalue for v in diffs])
            pr = (x["profile"] - y["profile"].reindex(x.index)).dropna()
            lines.append(f"{name:20s} {level:10s} " + "  ".join(
                f"{c[2:]} {v.mean():+.2f} (p {h:.3f})" for c, v, h in zip(chans, diffs, adj))
                + f"  profile {pr.mean():+.2f} (p {wilcoxon(pr).pvalue:.3f})")
        y = (d.pair_type != "control")
        from sklearn.metrics import roc_auc_score
        for (n, s), g in d.groupby(["novel_part", "script_part"]):
            if (n, s) in CELLS:
                lines.append(f"{name:20s} AUC {n} x {s}: profile {roc_auc_score(g.pair_type != 'control', g.profile):.3f}, "
                             f"semantic {roc_auc_score(g.pair_type != 'control', g.z_semantic):.3f}")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join("paper", "noruns_pooled_results.txt" if POOLED else "noruns_results.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--preprocess", action="store_true")
    ap.add_argument("--shard", default="")
    ap.add_argument("--sequence", action="store_true")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    if args.compare:
        return compare()
    pairs, names = setup()
    if args.stats:
        stats(pairs)
    if args.preprocess:
        preprocess(pairs, names)
    if args.shard:
        shard(pairs, names, args.shard)
    if args.sequence:
        order(pairs, names)


if __name__ == "__main__":
    main()
