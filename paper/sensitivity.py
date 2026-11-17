import argparse
import hashlib
import os
import shutil
import sys

sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), "src")]
import numpy as np  # noqa: E402

from src.screenplays import experiment, sequence  # noqa: E402

CELLS = [("full", "full"), ("dialogue", "dialogue"), ("narration", "action")]
VARIANTS = ("pooled", "free")
_CACHE = {}


def pooled_embeddings(windows):
    from align import _window_text_for_embedding
    from features.semantic import get_sbert_model
    texts = [_window_text_for_embedding(w) for w in windows]
    key = hashlib.sha1("\x00".join(texts).encode("utf-8")).hexdigest()
    if key not in _CACHE:
        _CACHE[key] = sequence.embed(get_sbert_model("all-MiniLM-L6-v2"), texts)
    return _CACHE[key]


def encode_pooled(wA, wB):
    return pooled_embeddings(wA), pooled_embeddings(wB)


def free_align(wA, wB, embeddings_A=None, embeddings_B=None, **_):
    """Each target window matched to its most similar source window, with no order constraint."""
    if not wA or not wB:
        return []
    A = embeddings_A / np.linalg.norm(embeddings_A, axis=1, keepdims=True)
    B = embeddings_B / np.linalg.norm(embeddings_B, axis=1, keepdims=True)
    best = (B @ A.T).argmax(axis=1)
    return [(int(i), j) for j, i in enumerate(best)]


def patch(variant):
    import features_runner
    features_runner._encode_sbert_cached = encode_pooled
    if variant == "free":
        features_runner.align_windows = free_align


def shard(variant, spec, cells=CELLS, unmasked=False):
    """unmasked: no name masking, into full_unmasked_<variant>; cells: e.g. all five cells of the design."""
    patch(variant)
    pairs = experiment.full_pairs()
    names = [] if unmasked else experiment.character_names(sorted({p[1].film for p in pairs}),
                                                          sorted({p[0].novel_path for p in pairs}))
    base = "full_unmasked" if unmasked else "full_masked"
    out = os.path.join(experiment.OUT_DIR, f"{base}_{variant}")
    os.makedirs(out, exist_ok=True)
    seq = os.path.join(out, "sequence.csv")
    if not os.path.exists(seq):
        shutil.copy(os.path.join(experiment.OUT_DIR, base, "sequence.csv"), seq)
    comps = experiment.build_components(pairs, names)
    i, n = map(int, spec.split("/"))
    experiment.run(pairs[i::n], comps, os.path.join(out, "raw"), cells)


def compare():
    import pandas as pd
    from scipy.stats import spearmanr, wilcoxon

    chans = ["z_lexical", "z_semantic", "z_conceptual", "z_structural", "z_narrative"]

    def holm(p):
        p = np.asarray(p, float)
        o = np.argsort(p)
        adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[o]).clip(max=1)
        r = np.empty(len(p))
        r[o] = adj
        return r

    lines = []
    load = lambda v: pd.read_csv(os.path.join("out_v2", f"screenplays_{v}.csv"))
    base = load("full_masked")
    for name in ["full_masked"] + [f"full_masked_{v}" for v in VARIANTS]:
        if not os.path.exists(os.path.join("out_v2", f"screenplays_{name}.csv")):
            continue
        d = load(name)
        a = d[d.pair_type != "control"]
        cell = lambda df, n, s: df[(df.novel_part == n) & (df.script_part == s)].set_index("screenplay")
        lines.append(f"\n===== {name}")
        dd, na = cell(a, "dialogue", "dialogue"), cell(a, "narration", "action")
        for level, (x, y) in {"22 films": (dd, na),
                              "19 sources": (dd.groupby("novel")[chans + ["profile"]].mean(),
                                             na.groupby("novel")[chans + ["profile"]].mean())}.items():
            diffs = [(x[c] - y[c].reindex(x.index)).dropna() for c in chans]
            adj = holm([wilcoxon(v).pvalue for v in diffs])
            pr = (x["profile"] - y["profile"].reindex(x.index)).dropna()
            lines.append(f"  routes {level:10s} " + "  ".join(
                f"{c[2:5]} {v.mean():+.2f} (p {h:.3f})" for c, v, h in zip(chans, diffs, adj))
                + f"  profile {pr.mean():+.2f} (p {wilcoxon(pr).pvalue:.3f})")
        ff = cell(a, "full", "full")
        corr = ff[chans].corr(method="spearman")
        lines.append(f"  full x full correlations: lex-sem {corr.loc['z_lexical', 'z_semantic']:.2f}, "
                     f"lex-str {corr.loc['z_lexical', 'z_structural']:.2f}, narrative with others "
                     f"{corr.loc['z_narrative', chans[:4]].min():.2f} to {corr.loc['z_narrative', chans[:4]].max():.2f}")
        lines.append("  full x full mean channel z: " + ", ".join(f"{c[2:]} {ff[c].mean():+.2f}" for c in chans)
                     + f", profile {ff.profile.mean():+.2f}")
        b = cell(base[base.pair_type != "control"], "full", "full")
        lines.append("  agreement with original across films (Spearman): " + ", ".join(
            f"{c[2:]} {spearmanr(ff[c], b[c].reindex(ff.index))[0]:.2f}" for c in chans + ["profile"]))
        for film in ("The Last of the Mohicans (1992 film)", "The Wizard of Oz (1939 film)", "little_women_2019"):
            if film in ff.index:
                lines.append(f"  {film}: " + ", ".join(f"{c[2:5]} {ff.loc[film, c]:+.1f}" for c in chans))
    text = "\n".join(lines)
    print(text)
    with open(os.path.join("paper", "sensitivity_results.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=VARIANTS)
    ap.add_argument("--shard", default="")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--cells", default="", help="e.g. full:dialogue,full:action (default: full and the two routes)")
    ap.add_argument("--unmasked", action="store_true")
    args = ap.parse_args()
    if args.compare:
        compare()
    elif args.variant and args.shard:
        cells = experiment.parse_cells(args.cells, "full") if args.cells else CELLS
        shard(args.variant, args.shard, cells, args.unmasked)
    else:
        ap.error("give --compare, or --variant with --shard")


if __name__ == "__main__":
    main()
