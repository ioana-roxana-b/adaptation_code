import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

from v2 import calibrate

from . import corpus, experiment, novel, parser, sequence

OUT_DIR = os.path.join(experiment.OUT_DIR, "baselines")
MFW = 150
N_BOOT = 2000
SEED = 0
FULL = experiment.SCRIPT_PARTS["full"]
_TOKEN = re.compile(r"[a-z]+(?:[’'][a-z]+)?")


def delta_matrix(docs):
    """Pairwise Burrows's Delta over the MFW most frequent words of all documents (the mask token excluded)."""
    counts = {k: Counter(_TOKEN.findall(t.lower())) for k, t in docs.items()}
    total = sum(counts.values(), Counter())
    total.pop(experiment.MASK_TOKEN.lower(), None)
    vocab = [w for w, _ in total.most_common(MFW)]
    keys = list(docs)
    F = np.array([[counts[k][w] / max(1, sum(counts[k].values())) for w in vocab] for k in keys])
    Z = (F - F.mean(axis=0)) / F.std(axis=0, ddof=1)
    return {(a, b): float(np.abs(Z[i] - Z[j]).mean()) for i, a in enumerate(keys) for j, b in enumerate(keys)}


def compute() -> pd.DataFrame:
    from features.semantic import get_sbert_model

    pairs = experiment.full_pairs()
    names = experiment.character_names(sorted({p[1].film for p in pairs}), sorted({p[0].novel_path for p in pairs}))
    novels = {p[0].novel_key: experiment.mask(novel.load(p[0].novel_path).text(), names) for p in pairs}
    scripts = {p[1].film: parser.load(p[1].film) for p in pairs}
    script_text = {f: experiment.mask(sp.text(), names) for f, sp in scripts.items()}

    docs = {**{f"n:{k}": t for k, t in novels.items()}, **{f"s:{k}": t for k, t in script_text.items()}}
    delta = delta_matrix(docs)
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2, max_df=0.9)
    keys = list(docs)
    X = vec.fit_transform([docs[k] for k in keys])
    pos = {k: i for i, k in enumerate(keys)}

    model = get_sbert_model("all-MiniLM-L6-v2")
    nv = {k: sequence.embed(model, sequence.novel_passages(t)) for k, t in novels.items()}
    sv = {f: sequence.embed(model, [experiment.mask(t, names) for _, t in sequence.screenplay_units(sp, FULL)[1]]) for f, sp in scripts.items()}
    unit = lambda v: v.mean(axis=0) / np.linalg.norm(v.mean(axis=0))

    rows = []
    for src, scr, kind in pairs:
        a, b = f"n:{src.novel_key}", f"s:{scr.film}"
        rows.append({"novel": src.novel_key, "screenplay": scr.film, "pair_type": kind,
                     "delta": -delta[(a, b)], "tfidf": float((X[pos[a]] @ X[pos[b]].T).toarray()[0, 0]),
                     "sbert_doc": float(unit(nv[src.novel_key]) @ unit(sv[scr.film])),
                     "best_match": float((sv[scr.film] @ nv[src.novel_key].T).max(axis=1).mean())})
    out = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(os.path.join(OUT_DIR, "scores.csv"), index=False)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    path = os.path.join(OUT_DIR, "scores.csv")
    b = pd.read_csv(path) if os.path.exists(path) else compute()
    sw = pd.read_csv(os.path.join(experiment.OUT_DIR, "sw_baseline", "masked.csv"))[["novel", "screenplay", "retention"]]
    ours = pd.read_csv("out_v2/screenplays_full_masked.csv")
    ours = ours[(ours.novel_part == "full") & (ours.script_part == "full")]
    d = b.merge(sw, on=["novel", "screenplay"]).merge(ours.drop(columns="pair_type"), on=["novel", "screenplay"])
    d["key"] = d.novel + "_vs_" + d.screenplay
    d = d.set_index("key")
    is_ctrl = d.pair_type == "control"
    methods = {"Burrows's Delta": "delta", "TF-IDF cosine": "tfidf", "SBERT, whole document": "sbert_doc",
               "SBERT, best match per scene": "best_match", "SW + SBERT retention (Pial et al.)": "retention"}
    chans = {c: {c: +1} for c in methods.values()}
    z = calibrate.score_cell(d[list(chans)], is_ctrl, calibrate.Additive(d, is_ctrl, ["novel", "screenplay"]), chans, n_boot=0)
    d = d.join(z[[f"z_{c}" for c in chans]], rsuffix="_cal")
    y = (d.pair_type != "control").to_numpy()
    rng = np.random.default_rng(SEED)
    # Resample literary sources: all pairs of a source move together, since they are not independent
    groups = {k: v.to_numpy() for k, v in d.reset_index(drop=True).groupby("novel").groups.items()}
    keys = list(groups)
    draws = []
    while len(draws) < N_BOOT:
        i = np.concatenate([groups[k] for k in rng.choice(keys, len(keys))])
        if 0 < y[i].sum() < len(i):
            draws.append(i)
    profile = d.profile.to_numpy()
    print(f"{int(y.sum())} adaptations, {int((~y).sum())} controls, {len(keys)} sources; AUC (95% interval, sources resampled); "
          "difference = profile minus method, 95% interval")
    rows = []
    for label, c in methods.items():
        for kind, col in (("raw", c), ("calibrated", f"z_{c}")):
            s = d[col].to_numpy()
            au = [roc_auc_score(y[i], s[i]) for i in draws]
            df = [roc_auc_score(y[i], profile[i]) - a for i, a in zip(draws, au)]
            rows.append({"method": label, "version": kind, "AUC": roc_auc_score(y, s), "lo": np.percentile(au, 2.5), "hi": np.percentile(au, 97.5),
                         "diff_lo": np.percentile(df, 2.5), "diff_hi": np.percentile(df, 97.5)})
    au = [roc_auc_score(y[i], profile[i]) for i in draws]
    rows.append({"method": "Five-channel profile (ours)", "version": "calibrated", "AUC": roc_auc_score(y, profile),
                 "lo": np.percentile(au, 2.5), "hi": np.percentile(au, 97.5)})
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
