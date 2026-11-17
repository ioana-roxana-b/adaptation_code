import argparse
import os
import sys

sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), "src")]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.screenplays import experiment  # noqa: E402

CELLS = [("dialogue", "dialogue"), ("narration", "action")]
MAIN = os.path.join(experiment.OUT_DIR, "full_masked_pooled")
POOL = os.path.join(experiment.OUT_DIR, "control_pool")
CHANS = ["z_lexical", "z_semantic", "z_conceptual", "z_structural", "z_narrative"]


def setup():
    pairs = experiment.full_pairs()
    names = experiment.character_names(sorted({p[1].film for p in pairs}), sorted({p[0].novel_path for p in pairs}))
    return pairs, names


def all_controls(pairs):
    """Every (novel, screenplay) pair with the novel by another author than the screenplay's source."""
    scripts = [p[1] for p in pairs if p[2] == "influence"]
    novels = list({a.novel_key: a for a in reversed(scripts)}.values())
    return [(n, s, "control") for s in scripts for n in novels
            if experiment._author(n.novel_key) != experiment._author(s.novel_key)]


def raw_name(comps, src, scr, n_part, s_part):
    return f"{comps[('novel', src.novel_key, n_part)][0]}_vs_{comps[('script', scr.film, s_part)][0]}.csv"


def shard(spec):
    import sensitivity
    sensitivity.patch("pooled")
    pairs, names = setup()
    comps = experiment.build_components(pairs, names)
    todo = [p for p in all_controls(pairs)
            if not all(os.path.exists(os.path.join(MAIN, "raw", raw_name(comps, p[0], p[1], n, s))) for n, s in CELLS)]
    i, n = map(int, spec.split("/"))
    print(f"{len(todo)} control pairs not in the main run; this shard: {len(todo[i::n])}")
    experiment.run(todo[i::n], comps, os.path.join(POOL, "raw"), CELLS)


def order():
    """Order scores of every control pair for the two cells, as in sequence.main."""
    from features.semantic import get_sbert_model
    from src.screenplays import novel, parser, sequence
    pairs, names = setup()
    ctrl = all_controls(pairs)
    model = get_sbert_model("all-MiniLM-L6-v2")
    passages, scenes, units = {}, {}, {}
    for src in {p[0].novel_key: p[0] for p in ctrl}.values():
        parts = novel.load(src.novel_path)
        for part, _ in CELLS:
            texts = [experiment.mask(t, names) for t in sequence.novel_passages(parts.text(experiment.NOVEL_PARTS[part]))]
            passages[(src.novel_key, part)] = sequence.embed(model, texts)
    for film in sorted({p[1].film for p in ctrl}):
        sp = parser.load(film)
        for _, part in CELLS:
            units[film], sc = sequence.screenplay_units(sp, experiment.SCRIPT_PARTS[part])
            scenes[(film, part)] = (len(sc), sequence.embed(model, [experiment.mask(t, names) for _, t in sc]))
    rows = []
    for src, scr, kind in ctrl:
        for n_part, s_part in CELLS:
            n_sc, vecs = scenes[(scr.film, s_part)]
            pos, _ = sequence.match(vecs, passages[(src.novel_key, n_part)])
            rows.append({"novel": src.novel_key, "screenplay": scr.film, "pair_type": kind, "novel_part": n_part,
                         "script_part": s_part, "unit": units[scr.film], "n_scenes": n_sc,
                         **sequence.order_features(pos)})
    os.makedirs(POOL, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(POOL, "sequence.csv"), index=False)
    print(f"Saved {len(rows)} order rows")


def score_seed(seed, comps, seq):
    from src.v2 import calibrate, run
    pairs = experiment.full_pairs(seed=seed)
    paths = []
    for src, scr, _ in pairs:
        for n, s in CELLS:
            name = raw_name(comps, src, scr, n, s)
            p = os.path.join(MAIN, "raw", name)
            paths.append(p if os.path.exists(p) else os.path.join(POOL, "raw", name))
    keep = ["pair_key", "pair_type", *run.SCREENPLAY_KEYS, *run.POSITION_COLS]
    rows = pd.concat([pd.read_csv(p, usecols=lambda c: c in keep + run.FEATURES) for p in paths], ignore_index=True)
    rows = rows.merge(seq, on=run.SCREENPLAY_KEYS, how="left")
    return calibrate.score(rows, run.CHANNELS, cell_cols=["novel_part", "script_part"],
                           keep_cols=["novel", "screenplay"], reference=calibrate.additive(["novel", "screenplay"]),
                           n_boot=0)


def route_test(res):
    from scipy.stats import wilcoxon
    a = res[res.pair_type != "control"].copy()
    a["four"] = a[CHANS[:4]].mean(axis=1)
    cols = CHANS + ["profile", "four"]
    cell = lambda n, s: a[(a.novel_part == n) & (a.script_part == s)].groupby("novel")[cols].mean()
    x, y = cell("dialogue", "dialogue"), cell("narration", "action")
    d = (x - y.reindex(x.index)).dropna()
    raw = [wilcoxon(d[c]).pvalue for c in CHANS]
    o = np.argsort(raw)
    adj = np.empty(len(raw))
    adj[o] = np.maximum.accumulate((len(raw) - np.arange(len(raw))) * np.asarray(raw)[o]).clip(max=1)
    out = {f"{c[2:]}": d[c].mean() for c in CHANS}
    out.update({f"{c[2:]}_p": p for c, p in zip(CHANS, adj)})
    for c in ("profile", "four"):
        out[c], out[f"{c}_p"] = d[c].mean(), wilcoxon(d[c]).pvalue
    return out


def score(n_seeds):
    from src.v2 import run
    pairs, names = setup()
    comps = experiment.build_components(pairs, names)
    seqs = [os.path.join(MAIN, "sequence.csv"), os.path.join(POOL, "sequence.csv")]
    seq = pd.concat([pd.read_csv(s) for s in seqs if os.path.exists(s)])
    seq = seq.drop_duplicates(subset=run.SCREENPLAY_KEYS)[run.SCREENPLAY_KEYS + ["order_rho"]]
    rows = []
    for seed in range(n_seeds + 1):
        rows.append({"seed": seed, **route_test(score_seed(seed, comps, seq))})
        print({k: round(v, 3) for k, v in rows[-1].items()}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join("paper", "control_seeds.csv"), index=False)
    alt = df[df.seed > 0]
    lines = [f"Route difference across sources, dialogue minus narration-to-action; seed 0 = paper, "
             f"{len(alt)} other control draws"]
    for c in ["profile", "four"] + [c[2:] for c in CHANS]:
        lines.append(f"  {c:11s} seed 0 {df[c].iloc[0]:+.2f} (p {df[c + '_p'].iloc[0]:.3f}) | other seeds mean "
                     f"{alt[c].mean():+.2f}, range {alt[c].min():+.2f} to {alt[c].max():+.2f}; positive in "
                     f"{(alt[c] > 0).sum()}/{len(alt)}, p<.05 in {(alt[c + '_p'] < .05).sum()}/{len(alt)}, "
                     f"max p {alt[c + '_p'].max():.3f}")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join("paper", "control_seeds_results.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", default="")
    ap.add_argument("--sequence", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()
    if args.shard:
        shard(args.shard)
    if args.sequence:
        order()
    if args.score:
        score(args.seeds)


if __name__ == "__main__":
    main()
